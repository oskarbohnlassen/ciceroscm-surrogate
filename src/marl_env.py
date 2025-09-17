from gymnasium import spaces
from ray.rllib.env import MultiAgentEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

import torch.nn as nn
import sys
import os
import numpy as np
import torch
import pickle
import time
import copy
import pandas as pd
import importlib
import matplotlib.pyplot as plt
import ray

sys.path.insert(0,os.path.join(os.getcwd(), '../ciceroscm/', 'src')) ## Make ciceroscm importable
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURR_DIR)  # make 'train.py' and 'model.py' importable

from marl_env_step import CICEROSCMEngine, CICERONetEngine


class LSTMSurrogate(nn.Module):
    def __init__(self, n_gas, hidden=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_gas,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
        )
        self.out = nn.Sequential(
            nn.Linear(hidden + n_gas, 64),
            nn.GELU(),
            nn.Linear(64, 1)  # ΔT_air only
        )

    def forward(self, x):                 # x: (B, 51, G)
        hist, action = x[:, :-1, :], x[:, -1, :]  # (B,50,G), (B,G)
        seq_out, _ = self.lstm(hist)      # seq_out: (B,50,H)
        h_last = seq_out[:, -1, :]        # last timestep hidden: (B,H)
        y_hat = self.out(torch.cat([h_last, action], dim=-1)).squeeze(1)
        return y_hat


class ClimateMARL(MultiAgentEnv):
    """
    N agents (countries). Each year they pick a deviation a_i from their baseline share.
    Total emissions E_t (40 gases) feed a climate engine (CICERO-NET or CICERO-SCM).
    Observation (same for all agents): [T_t, year_norm, B_t(40)].
    Reward_i = xxx
    """

    def __init__(self, env_config, emission_data, economics_config, actions_config):
        super().__init__()
        
        # --- config dicts ---
        rng = np.random.default_rng(env_config['random_seed'])
      
        # --- core sizes/time ---
        self.N = int(env_config["N"])
        self.G = int(env_config["G"])
        
        self.hist_end   = int(env_config["hist_end"])
        self.future_end = int(env_config["future_end"])
        self.horizon    = int(env_config["horizon"])
        self.engine_kind = str(env_config['engine']).lower()
        
        # --- provided data ---
        self._hist_emissions_np = np.asarray(emission_data["historical_emissions"], dtype=np.float32).copy()
        self._hist_emissions_pd = emission_data["historical_emissions"].copy()
        self.baseline_emissions      = np.asarray(emission_data["baseline_emissions"], dtype=np.float32)
        self.emission_shares         = np.asarray(emission_data["emission_shares"], dtype=np.float32)

        # --- heterogeneous economics/impacts ---
        self.climate_disaster_sensitivity     = np.asarray(economics_config['climate_disaster_sensitivity'], dtype=np.float32)     # (N,)
        self.emission_reduction_sensitivity   = np.asarray(economics_config['emission_reduction_sensitivity'], dtype=np.float32)   # (N,)
        self.climate_prevention_sensitivity   = np.asarray(economics_config['climate_prevention_sensitivity'], dtype=np.float32)   # (N,)

        self.prevention_decay = 0.95  # Each year the prevention stock decays by 5%
        self.max_prevention_benefit = 0  # Maximum prevention can reduce 30% of disaster costs
        self.damage_exponent = 3  # Damages scale with T^damage_exponent

        self.climate_base_cost = 200.0
        self.prevention_base_cost = 500.0
        self.reduction_base_cost = 1.0
        
        # --- actions ---
        self.emission_actions = np.asarray(actions_config['emission_actions'], dtype=np.float32)  # -5% to +5% emission changes
        self.prevention_actions = np.asarray(actions_config['prevention_actions'], dtype=np.float32)  # 0%, 2%, 5%, 8% of damage reduction
        self.prevention_stock = np.zeros(self.N, dtype=np.float32) # Current prevention stock (decays over time)
        self.prevention_stock_obs = self.prevention_stock.copy()
        self._controllable_gases = 4  # first four indices
        self.last_agent_emissions = np.zeros(
            (self.N, self._controllable_gases), dtype=np.float32)
        self.cumulative_emission_delta = np.zeros(
            (self.N, self._controllable_gases), dtype=np.float32)  # running (actual - baseline)
        
        self.last_emission_change = np.zeros(self.N, dtype=np.float32)
        self.last_prevention_rate = np.zeros(self.N, dtype=np.float32)
        
        self._act_space = spaces.MultiDiscrete([len(self.emission_actions), len(self.prevention_actions)])

        # --- agents  ---
        self.agents = [f"country_{i}" for i in range(self.N)]

        # --- spaces ---
        cumulative_bound = 5000.0  # generous cap for accumulated deviations from baseline
        obs_low = np.concatenate([
            np.array([0.0, 0.0], dtype=np.float32),
            np.zeros(self.N * self._controllable_gases, dtype=np.float32),
            np.full(self.N * self._controllable_gases, -cumulative_bound, dtype=np.float32),
            np.zeros(self.N, dtype=np.float32),
            np.full(self.N, -0.05, dtype=np.float32),  # emission deltas min
            np.zeros(self.N, dtype=np.float32),         # prevention deltas min
        ])
        obs_high = np.concatenate([
            np.array([3.0, 40.0], dtype=np.float32),
            np.full(self.N * self._controllable_gases, 1000.0, dtype=np.float32),
            np.full(self.N * self._controllable_gases, cumulative_bound, dtype=np.float32),
            np.ones(self.N, dtype=np.float32),          # prevention stock cap
            np.full(self.N, 0.05, dtype=np.float32),    # emission deltas max
            np.full(self.N, 0.08, dtype=np.float32),    # prevention deltas max
        ])
        self._obs_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        self.observation_spaces = {a: self._obs_space for a in self.agents}
        self.action_spaces      = {a: self._act_space for a in self.agents}

        # --- model parameters ---
        self.net_params = env_config["net_params"]
        self.scm_params = env_config["scm_params"]

        self.mu_gas = self.net_params['mu']
        self.std_gas = self.net_params['std']

        # --- timer ---
        self._engine_time = 0.0

        # internal state
        self.reset()

    def observation_space(self, agent_id=None):
        return self.observation_spaces[self.agents[0] if agent_id is None else agent_id]

    def action_space(self, agent_id=None):
        return self.action_spaces[self.agents[0] if agent_id is None else agent_id]

    def _make_scm_engine(self):

        p = self.scm_params
        
        return CICEROSCMEngine(
            historical_emissions=self._hist_emissions_pd.copy(),
            gaspam_data=p["gaspam_data"],
            conc_data=p["conc_data"],
            nat_ch4_data=p["nat_ch4_data"],
            nat_n2o_data=p["nat_n2o_data"],
            pamset_udm=p["pamset_udm"],
            pamset_emiconc=p["pamset_emiconc"],
            em_data_start=p["em_data_start"],
            em_data_policy=p["em_data_policy"],
            udir=p["udir"],
            idtm=p["idtm"],
            scenname=p["scenname"],
        )

    def _make_net_engine(self):            
        p = self.net_params
        device = p["device"]

        # Recreate a torch state dict from numpy:
        state = {k: torch.from_numpy(arr).to(device) for k, arr in p["state_dict_np"].items()}

        # Initialize model
        m = LSTMSurrogate(n_gas=40, hidden=128, num_layers=1).to(device)
        m.load_state_dict(state, strict=True)
        m.eval()
        for param in m.parameters():
            param.requires_grad_(False)

        return CICERONetEngine(
            historical_emissions=self._hist_emissions_np.copy(),
            model=m,
            device=device,
            mu=np.asarray(p["mu"], np.float32),
            std=np.asarray(p["std"], np.float32),
            autocast = p["autocast"],
            use_half = p["use_half"]
        )

    def _current_observation(self):
        temp = float(self.engine.T)
        year_idx = float(self.year_idx - (self.hist_end + 1))
        emissions_flat = self.last_agent_emissions.flatten()
        cumulative_flat = self.cumulative_emission_delta.flatten()
        prevention_stock = self.prevention_stock
        last_emission_change = self.last_emission_change.flatten()
        last_prevention_rate = self.last_prevention_rate.flatten()
        obs_vec = np.concatenate([
            np.array([temp, year_idx], dtype=np.float32),
            emissions_flat.astype(np.float32),
            cumulative_flat.astype(np.float32),
            prevention_stock.astype(np.float32),
            last_emission_change.astype(np.float32),
            last_prevention_rate.astype(np.float32)
            ])
        
        #print(f"Obs vec: {obs_vec} \n")
        return obs_vec
    
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            _ = np.random.default_rng(seed)
        
        #print(f"Last emission changes: {self.last_emission_change}, Last prevention rates: {self.last_prevention_rate}")

        self.t = 0
        self.year_idx = self.hist_end + 1  # 2016
        self.prevention_stock.fill(0.0)
        self.last_emission_change.fill(0.0)
        self.last_prevention_rate.fill(0.0)
        self.cumulative_emission_delta.fill(0.0)

        baseline_global = self.baseline_emissions[0]
        baseline_agent = self.emission_shares * baseline_global[None, :]
        self.last_agent_emissions[...] = baseline_agent[:, :self._controllable_gases]

        if self.engine_kind == "scm":
            self.engine = self._make_scm_engine()
        elif self.engine_kind == "net":
            self.engine = self._make_net_engine()

        #self.net_engine = self._make_net_engine()

        obs_vec = self._current_observation()
        return {agent: obs_vec.copy() for agent in self.agents}, {agent: {} for agent in self.agents}


    def step(self, action_dict):
        
        # Parse separate actions for emission and mitigation
        emission_indices = np.array([action_dict[ag][0] for ag in self.agents], dtype=np.int32)
        mitigation_indices = np.array([action_dict[ag][1] for ag in self.agents], dtype=np.int32)

        # Convert to actual values
        emission_changes = self.emission_actions[emission_indices]  # -5% to +5% emission changes
        #emission_changes = self.emission_actions[[0]*self.N]  # test
        # test where emissions are reduced max
        prevention_rates = self.prevention_actions[mitigation_indices]  # 0%, 2%, 5%, 8% of damage reduction
        #print(f"Emission changes: {emission_changes}, Prevention rates: {prevention_rates}")

        self.last_emission_change[...] = emission_changes
        self.last_prevention_rate[...] = prevention_rates
        #print(f"Last emission changes: {self.last_emission_change}, Last prevention rates: {self.last_prevention_rate}")

        # Update prevention stock (decays over time)
        self.prevention_stock = (self.prevention_stock * self.prevention_decay) + prevention_rates
        self.prevention_stock = np.clip(self.prevention_stock, 0.0, self.max_prevention_benefit)
        self.prevention_stock_obs = self.prevention_stock.copy()
        #print(f"Prevention stock: {self.prevention_stock}")

        # Baseline emissions allocation
        baseline_global_emissions = self.baseline_emissions[self.year_idx - (self.hist_end + 1)]  # (40,)
        baseline_agent_emissions = self.emission_shares * baseline_global_emissions[None, :]  # (N,40)
        #print(f"Baseline emissions global: {baseline_global_emissions[:5]}, \n Base emissions per agent: {baseline_agent_emissions[:, :5]}")

        # Apply emission changes
        emission_agents = baseline_agent_emissions.copy()

        # Only allow actions to modify the first four gas species.
        gas_slice = slice(0, self._controllable_gases)
        emission_agents[:, gas_slice] = np.clip(
            (1.0 + emission_changes[:, None]) * baseline_agent_emissions[:, gas_slice], 0.0, None)
        emission_global = emission_agents.sum(axis=0)

        self.last_agent_emissions[...] = emission_agents[:, gas_slice]
        delta_from_baseline = (
            emission_agents[:, gas_slice] - baseline_agent_emissions[:, gas_slice]
        ).astype(np.float32)
        self.cumulative_emission_delta += delta_from_baseline
        #print(f"Actual emissions per agent: {emission_agents[:, :5]}, \n Actual emissions global: {emission_global[:5]}")

        # Step climate engine
        T_next = self.engine.step(emission_global)
        #print(f"SCM engine temperature: {T_next}")
        #T_next = self.net_engine.step(emission_global)
        #print(f"Net engine temperature: {T_next}")
        
        # (A) Climate disasters cost per agent
        base_disaster_cost = self.climate_base_cost * ((T_next+1) ** self.damage_exponent) # (T^3) scaling of damages
        agent_disaster_cost = base_disaster_cost * self.climate_disaster_sensitivity * (1-self.prevention_stock) # (N,)
        #print(f"Disaster costs per agent: {agent_disaster_cost}")

        # (B) Climate reduction cost per agent
        emission_reduction_action = np.maximum(0.0, -emission_changes) # fraction cut (0..0.05)
        emission_reduction_action += (emission_reduction_action != 0).astype(float)
        emission_reduction_cost = self.reduction_base_cost * self.emission_reduction_sensitivity * emission_reduction_action
        #print(f"Reduction costs per agent: {emission_reduction_cost}")

        # (C) Climate prevention cost per agent
        prevention_rates += (prevention_rates != 0).astype(float)
        prevention_cost = self.prevention_base_cost * self.climate_prevention_sensitivity * prevention_rates  # (N,)
        #print(f"Prevention costs per agent: {prevention_cost}")

        # Total reward
        r = - (agent_disaster_cost + emission_reduction_cost + prevention_cost)
        #print(f"Rewards per agent: {r}")
        #print("-----------------------------------------------------")

        # Update time 
        self.t += 1
        self.year_idx += 1
        done = (self.t >= self.horizon) or (self.year_idx > self.future_end)
        
        # Update observation
        obs_vec = self._current_observation()
        obs_d = {agent: obs_vec.copy() for agent in self.agents}
        
        # Additional info
        rew_d   = {a: float(x) for a, x in zip(self.agents, r)}
        term_d  = {a: done for a in self.agents}
        trunc_d = {a: False for a in self.agents}
        info_d = {}
        for idx, agent in enumerate(self.agents):
            info_d[agent] = {
                "temperature": float(self.engine.T),
                "year": int(self.year_idx),
                "emission_delta": float(emission_changes[idx]),
                "emission_level": float(emission_agents[idx, gas_slice].mean())}        
        term_d["__all__"]  = done
        trunc_d["__all__"] = False

        return obs_d, rew_d, term_d, trunc_d, info_d
