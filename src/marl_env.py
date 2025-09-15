from gymnasium import spaces
from ray.rllib.env import MultiAgentEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

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

from train import load_processed_data, format_data
from model import LSTMSurrogate
from data_generation import load_core_data
from marl_env_step import CICEROSCMEngine, CICERONetEngine


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
        self.year0_hist = int(env_config["year0_hist"])
        self.hist_end   = int(env_config["hist_end"])
        self.future_end = int(env_config["future_end"])
        self.horizon    = int(env_config["horizon"])
        self.engine_kind = str(env_config['engine']).lower()
        
        # --- provided data ---
        self.historical_emissions_pd = emission_data['historical_emissions']
        self.historical_emissions    = np.asarray(emission_data['historical_emissions'], dtype=np.float32)
        self.baseline_emissions      = np.asarray(emission_data["baseline_emissions"], dtype=np.float32)
        self.emission_shares         = np.asarray(emission_data["emission_shares"], dtype=np.float32)

        # --- heterogeneous economics/impacts ---
        self.climate_disaster_cost         = np.asarray(economics_config['climate_disaster_cost'], dtype=np.float32)         # (N,)
        self.climate_investment_cost       = np.asarray(economics_config['climate_investment_cost'], dtype=np.float32)       # (N,)
        self.economic_growth_sensitivity   = np.asarray(economics_config['economic_growth_sensitivity'], dtype=np.float32)   # (N,)

        # --- actions ---
        self.emission_actions = np.asarray(actions_config['m_levels'], dtype=np.float32)  # -5% to +5% emission changes
        self.adaptation_actions = np.array([0.0, 0.02, 0.05, 0.08], dtype=np.float32)   # 0%, 2%, 5%, 8% GDP for adaptation

        self._act_space = spaces.MultiDiscrete([len(self.emission_actions), len(self.adaptation_actions)])
        self.adaptation_stock = np.zeros(self.N, dtype=np.float32)
        self.max_adaptation_benefit = 0.5

        # --- agents  ---
        self.agents = [f"country_{i}" for i in range(self.N)]

        # --- spaces ---
        obs_low  = np.array([0, 0] + [0.0]*self.G + [0.0]*self.N, dtype=np.float32)
        obs_high = np.array([3.0, 40] + [3000]*self.G + [1.0]*self.N, dtype=np.float32)  # Adaptation normalized 0-1
        self._obs_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        self.observation_spaces = {a: self._obs_space for a in self.agents}
        self.action_spaces      = {a: self._act_space for a in self.agents}

        # --- model parameters ---
        self.net_params = env_config.get("net_params", {})
        self.scm_params = env_config.get("scm_params", {})

        # --- timer ---
        self._engine_time = 0.0

        # internal state
        self.reset()

    
    def observation_space(self, agent_id=None):
        return self.observation_spaces[self.agents[0] if agent_id is None else agent_id]

    def action_space(self, agent_id=None):
        return self.action_spaces[self.agents[0] if agent_id is None else agent_id]

    def _make_engine(self):
        if self.engine_kind == "net":
            
            p = self.net_params
            mod = importlib.import_module(p["model_module"])
            ModelCls = getattr(mod, p["model_class"])
            device = p["device"]

            m = ModelCls(**p["model_kwargs"]).to(device)
            # Recreate a torch state dict from numpy:
            state = {k: torch.from_numpy(arr).to(device) for k, arr in p["state_dict_np"].items()}
            m.load_state_dict(state, strict=True)
            m.eval()

            for param in m.parameters():
                param.requires_grad_(False)

        
            return CICERONetEngine(
                historical_emissions=self.historical_emissions,
                model=m,
                device=device,
                window=p["window"],
                mu=np.asarray(p["mu"], np.float32),
                std=np.asarray(p["std"], np.float32),
                autocast = p["autocast"],
                use_half = p["use_half"]
            )
            
        elif self.engine_kind == "scm":
            p = self.net_params
            
            return CICEROSCMEngine(
                historical_emissions=self.historical_emissions_pd,
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
        else:
            raise ValueError(self.engine_kind)

    def _obs(self):
        temp = float(self.engine.T)
        year_idx = float(self.year_idx - (self.hist_end + 1))  # 0..35
        emissions = self.historical_emissions[-1]              # (G,)
        adaptation_levels = self.adaptation_stock.copy()

        return np.asarray([temp, year_idx] + list(emissions) + list(adaptation_levels), dtype=np.float32)
    
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            _ = np.random.default_rng(seed)
        self.t = 0
        self.year_idx = self.hist_end + 1  # 2016
        self.engine = self._make_engine()
        obs = self._obs()
        return {a: obs for a in self.agents}, {a: {} for a in self.agents}

    def step(self, action_dict):
        
        # Parse separate actions for emission and adaptation
        emission_indices = np.array([action_dict[ag][0] for ag in self.agents], dtype=np.int32)
        adaptation_indices = np.array([action_dict[ag][1] for ag in self.agents], dtype=np.int32)

        # Convert to actual values
        emission_changes = self.emission_actions[emission_indices]  # -5% to +5% emission changes
        adaptation_rates = self.adaptation_actions[adaptation_indices]  # 0%, 2%, 5%, 8% GDP

        B_t = self.baseline_emissions[self.year_idx - (self.hist_end + 1)]  # (40,)
        b_alloc = self.emission_shares * B_t[None, :]  # (N,40)
        base_agent_sum = b_alloc.sum(axis=1)  # (N,) GDP per country

        # Apply emission changes
        e_agents = (1.0 + emission_changes[:, None]) * b_alloc
        e_agents = np.clip(e_agents, 0.0, None)
        
        E_t = e_agents.sum(axis=0)  # (40,) Total global emissions
        self.historical_emissions = np.vstack([self.historical_emissions, E_t[None, :]])
        
        T_next = self.engine.step(E_t)

        # Update adaptation stock (decays over time)
        adaptation_investments = adaptation_rates * base_agent_sum
        self.adaptation_stock = self.adaptation_stock * 0.95 + adaptation_investments  # 5% annual decay

        # Calculate rewards with economic incentives
        emiss_agent_sum = e_agents.sum(axis=1)   # (N,) Actual emissions per country
        delta_emissions = emission_changes * base_agent_sum   # (N,) Emission change amounts

        # (A) Climate disasters: capped at 50% reduction from adaptation
        base_disaster_cost = self.climate_disaster_cost * (T_next ** 2)
        adaptation_effectiveness = np.clip(self.adaptation_stock / (base_agent_sum + 1e-6), 0.0, self.max_adaptation_benefit)
        disaster_cost = base_disaster_cost * (1.0 - adaptation_effectiveness)

        # (B) Climate investment cost: only when mitigating (a < 0).
        # Scale by the size of the baseline so deeper cuts cost more.
        emission_reduction_cost = self.climate_investment_cost * np.maximum(0, -emission_changes) * base_agent_sum  # Only for reductions
        adaptation_cost = adaptation_rates * base_agent_sum

        total_investment_cost = emission_reduction_cost + adaptation_cost

        # (C) Economic growth incentives
        # Countries with low disaster costs benefit more from emissions (industrial advantage)
        emission_benefit_factor = 1.0 / (self.climate_disaster_cost + 0.1)  # Inverse of vulnerability
        emission_growth_bonus = emission_benefit_factor * np.maximum(0, delta_emissions)  # Bonus for increasing emissions

        # Standard growth from economic activity
        economic_growth = self.economic_growth_sensitivity * delta_emissions

        # Total growth = standard growth + emission bonus - investment crowding out
        investment_crowding = total_investment_cost * 0.3  # Investments reduce available resources for growth
        total_growth = economic_growth + emission_growth_bonus - investment_crowding

        r = total_growth - disaster_cost

         # Strategy information for debugging
        emission_action_names = [f"{self.emission_actions[idx]*100:.0f}%" for idx in emission_indices]
        adaptation_action_names = [f"{self.adaptation_actions[idx]*100:.0f}%" for idx in adaptation_indices]
        
        self.t += 1
        self.year_idx += 1
        done = (self.t >= self.horizon) or (self.year_idx > self.future_end)
        
        obs = self._obs()
        obs_d = {a: obs for a in self.agents}
        
        rew_d   = {a: float(x) for a, x in zip(self.agents, r)}
        term_d  = {a: done for a in self.agents}
        trunc_d = {a: False for a in self.agents}
        info_d  = {a: {
            "E_sum": float(E_t.sum()),
            "emission_change": emission_action_names[i],
            "adaptation_investment": adaptation_action_names[i],
            "adaptation_stock": float(self.adaptation_stock[i]),
            "adaptation_effectiveness": float(adaptation_effectiveness[i]),
            "disaster_cost": float(disaster_cost[i]),
            "growth_benefit": float(total_growth[i]),
            "emission_bonus": float(emission_growth_bonus[i])
        } for i, a in enumerate(self.agents)}
        term_d["__all__"]  = done
        trunc_d["__all__"] = False

        return obs_d, rew_d, term_d, trunc_d, info_d