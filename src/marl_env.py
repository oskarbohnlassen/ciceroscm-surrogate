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
        self.baseline_emission_growth = np.asarray(emission_data["baseline_emission_growth"], dtype=np.float32)
        self.emission_shares         = np.asarray(emission_data["emission_shares"], dtype=np.float32)
        self.baseline_emissions_agent = self.emission_shares[:, None, :] * self.baseline_emissions[None, :, :]

        # --- heterogeneous economics/impacts ---
        self.climate_disaster_sensitivity     = np.asarray(economics_config['climate_disaster_sensitivity'], dtype=np.float32)     # (N,)
        self.emission_reduction_sensitivity   = np.asarray(economics_config['emission_reduction_sensitivity'], dtype=np.float32)   # (N,)
        self.climate_prevention_sensitivity   = np.asarray(economics_config['climate_prevention_sensitivity'], dtype=np.float32)   # (N,)

        self.prevention_decay = 0.95  # Each year the prevention stock decays by 5%
        self.max_prevention_benefit = 0.4  # Maximum prevention 
        self.damage_exponent = 3  # Damages scale with T^damage_exponent

        self.climate_base_cost = 40.0
        self.prevention_base_cost = 2000.0
        self.reduction_base_cost = 10.0
        
        self._controllable_gases = 4  # first four indices
        self.gas_slice = slice(0, self._controllable_gases)
        self.last_agent_emissions = np.zeros((self.N, self.G), dtype=np.float32)
        self.cumulative_emission_delta = np.zeros((self.N, self._controllable_gases), dtype=np.float32)

        # --- actions ---
        self.emission_actions = np.asarray(actions_config['emission_actions'], dtype=np.float32)  # -5% to +5% emission changes
        self.prevention_actions = np.asarray(actions_config['prevention_actions'], dtype=np.float32)  # 0%, 2%, 5%, 8% of damage reduction
        self.prevention_stock = np.zeros(self.N, dtype=np.float32) # Current prevention stock (decays over time)
        self._act_space = spaces.MultiDiscrete([len(self.emission_actions), len(self.prevention_actions)])

        baseline_emissions_first_year = float(np.mean(self.baseline_emissions[0][:self._controllable_gases]))
        self._cum_scale = max(1.0, baseline_emissions_first_year * max(1, self.horizon))
        self._emission_scale = max(1.0, float(np.mean(self.baseline_emissions[0][:self._controllable_gases])))

        # --- agents  ---
        self.agents = [f"country_{i}" for i in range(self.N)]

        # --- spaces ---
        obs_low = np.concatenate([
            np.array([0.0, 0.0], dtype=np.float32),
            np.zeros(self.N * self._controllable_gases, dtype=np.float32), # last emissions
            np.full(self.N * self._controllable_gases, -100.0, dtype=np.float32), # large cumulative emissions
            np.zeros(self.N, dtype=np.float32), # prevention stock cap
        ])
        obs_high = np.concatenate([
            np.array([3.0, 1.0], dtype=np.float32),
            np.full(self.N * self._controllable_gases, 100.0, dtype=np.float32), # last emissions
            np.full(self.N * self._controllable_gases, 100.0, dtype=np.float32), # large cumulative emissions
            np.ones(self.N, dtype=np.float32), # prevention stock cap
        ])
        self._obs_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        self.observation_spaces = {a: self._obs_space for a in self.agents}
        self.action_spaces      = {a: self._act_space for a in self.agents}

        # --- model parameters ---
        self.net_params = env_config["net_params"]
        self.scm_params = env_config["scm_params"]

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
        year_idx_norm = year_idx / 35 # normalize to [0..1] over 2016-2050

        self.last_controlled_emissions = self.last_agent_emissions[:, self.gas_slice]

        emissions_flat = (self.last_controlled_emissions / self._emission_scale).flatten().astype(np.float32)
        cumulative_flat = (self.cumulative_emission_delta / self._cum_scale).flatten().astype(np.float32)
        prevention_stock = self.prevention_stock.astype(np.float32)

        obs_vec = np.concatenate([
            np.array([temp, year_idx_norm], dtype=np.float32),
            emissions_flat,
            cumulative_flat,
            prevention_stock,
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
        self.cumulative_emission_delta.fill(0.0)

        baseline_global = self.baseline_emissions[0]
        baseline_agent = self.emission_shares * baseline_global[None, :]
        self.last_agent_emissions[...] = baseline_agent

        if self.engine_kind == "scm":
            self.engine = self._make_scm_engine()
        elif self.engine_kind == "net":
            self.engine = self._make_net_engine()

        #self.net_engine = self._make_net_engine()

        obs_vec = self._current_observation()
        return {agent: obs_vec.copy() for agent in self.agents}, {agent: {} for agent in self.agents}


    def step(self, action_dict):
        # Year
        idx = self.year_idx - (self.hist_end + 1)   # 2016 → 0, …, 2050 → 34

        # Parse separate actions for emission and prevention
        emission_indices = np.array([action_dict[ag][0] for ag in self.agents], dtype=np.int32)
        prevention_indices = np.array([action_dict[ag][1] for ag in self.agents], dtype=np.int32)
        emission_changes = self.emission_actions[emission_indices]  # -4% to +4% emission changes
        prevention_rates = self.prevention_actions[prevention_indices]  # 0%, 2%, 5%, 8% of damage reduction    
        
        # Prevention
        self.prevention_stock = (self.prevention_stock * self.prevention_decay) + prevention_rates
        self.prevention_stock = np.clip(self.prevention_stock, 0.0, self.max_prevention_benefit)
    
        # Emission growth rate
        baseline_growth_year = self.baseline_emission_growth[idx] # 2016..2050 as baseline growth starts from 2016
        baseline_growth_per_agent = np.broadcast_to(baseline_growth_year, (self.N, self.G)).copy()
        baseline_growth_per_agent[:, self.gas_slice] *= (1.0 + emission_changes)[:, None]

        # Emissions actual
        emission_agents = self.last_agent_emissions * baseline_growth_per_agent 
        emission_agents[:, self.gas_slice] = np.clip(emission_agents[:, self.gas_slice], 0.0, None)

        emission_global = emission_agents.sum(axis=0)  # (G,)
        self.last_agent_emissions[...] = emission_agents

        # Cummulative delta emissions
        base_agent_year = self.baseline_emissions_agent[:, idx+1, :]  # (N, G) → year = 2016+idx
        delta_from_baseline = (emission_agents[:, self.gas_slice] - base_agent_year[:, self.gas_slice]).astype(np.float32)
        self.cumulative_emission_delta += delta_from_baseline

        # Step climate engine
        T_next = self.engine.step(emission_global)
      #  print(f"SCM engine temperature: {T_next}")
       # T_next = self.net_engine.step(emission_global)
        #print(f"Net engine temperature: {T_next}")
    
        # (A) Climate disasters cost per agent
        base_disaster_cost = self.climate_base_cost * ((T_next+1) ** self.damage_exponent) # (T^3) scaling of damages
        agent_disaster_cost = base_disaster_cost * self.climate_disaster_sensitivity * (1-self.prevention_stock) # (N,)

        # (B) Climate reduction cost per agent
        cut = np.maximum(0.0, -emission_changes)  
        emission_reduction_cost = self.reduction_base_cost * self.emission_reduction_sensitivity * cut

        # (C) Climate prevention cost per agent
        prevention_cost = self.prevention_base_cost * self.climate_prevention_sensitivity * prevention_rates  # (N,)

        #print(f"All reward components: Disaster: {agent_disaster_cost}, Emission reduction: {emission_reduction_cost}, Prevention: {prevention_cost}")

        # Total reward
        r = - (agent_disaster_cost + emission_reduction_cost + prevention_cost)
        r *= 1e-3
        #print(f"Rewards per agent: {r}")
        #print("-----------------------------------------------------")

        # Update time 
        self.t += 1
        self.year_idx += 1
        done = (self.t >= self.horizon) or (self.year_idx > self.future_end)
        
        # Update observation
        obs_vec = self._current_observation()
        #print(f"Obs vec: {obs_vec} \n")
        obs_d = {agent: obs_vec.copy() for agent in self.agents}
        
        # Additional info
        rew_d   = {a: float(x) for a, x in zip(self.agents, r)}
        term_d  = {a: done for a in self.agents}
        trunc_d = {a: False for a in self.agents}
        info_d = {}

        term_d["__all__"]  = done
        trunc_d["__all__"] = False

        return obs_d, rew_d, term_d, trunc_d, info_d
