from gymnasium import spaces
from ray.rllib.env import MultiAgentEnv

import sys
import os
from pathlib import Path
import numpy as np
import copy
import pandas as pd

sys.path.insert(0,os.path.join(os.getcwd(), '../ciceroscm/', 'src')) ## Make ciceroscm importable
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURR_DIR)  # make 'train.py' and 'model.py' importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from marl_env_step import CICEROSCMEngine, CICERONetEngine
from src.utils.model_utils import instantiate_model, load_state_dict, numpy_state_to_torch


class ClimateMARL(MultiAgentEnv):

    def __init__(self, env_config, emission_data, economics_config, actions_config):
        super().__init__()

        self.env_config = env_config

        # --- core sizes/time ---
        self.N = int(env_config["N"])
        self.horizon = int(env_config["horizon"])
        self.hist_end = int(env_config["hist_end"])
        self.future_end = int(env_config["future_end"])
        self.window_size = int(env_config["window_size"])
        self.engine_kind = str(env_config["engine"]).lower()
        self.rollout_length = int(env_config["rollout_length"])

        # --- emissions data ---
        self._hist_emissions_pd = emission_data["historical_emissions"].copy()
        self._hist_emissions_np = np.asarray(self._hist_emissions_pd, dtype=np.float32)
        self.baseline_emissions = np.asarray(emission_data["baseline_emissions"], dtype=np.float32)
        self.baseline_emission_growth = np.asarray(
            emission_data["baseline_emission_growth"], dtype=np.float32
        )

        self.gas_names = list(emission_data["gas_names"])
        self.G = len(self.gas_names)

        name_to_index = {name: idx for idx, name in enumerate(self.gas_names)}
        self.controlled_gases = list(env_config["controlled_gases"])
        self.control_indices = np.array(
            [name_to_index[name] for name in self.controlled_gases], dtype=np.int32
        )
        self._controllable_gases = len(self.control_indices)


        emission_shares = np.asarray(emission_data["emission_shares"], dtype=np.float32)
        if emission_shares.shape != (self.N, self.G):
            raise ValueError(
                f"emission_shares must have shape ({self.N}, {self.G}); got {emission_shares.shape}"
            )
        self.emission_shares = emission_shares
        self.baseline_emissions_agent = (
            self.emission_shares[:, None, :] * self.baseline_emissions[None, :, :]
        )

        baseline_control = self.baseline_emissions[0, self.control_indices]
        baseline_control_mean = float(np.mean(baseline_control))
        self._cum_scale = max(1.0, baseline_control_mean * max(1, self.horizon))
        self._emission_scale = max(1.0, baseline_control_mean)

        self.last_agent_emissions = np.zeros((self.N, self.G), dtype=np.float32)
        self.cumulative_emission_delta = np.zeros(
            (self.N, self._controllable_gases), dtype=np.float32
        )

        # --- actions ---
        self.lever_names = list(actions_config["lever_names"])
        if not self.lever_names:
            raise ValueError("At least one mitigation lever must be specified")

        self.lever_count = len(self.lever_names)
        lever_levels_cfg = actions_config["lever_levels"]
        self.lever_levels = []  # list of np.ndarray per lever (ascending order)
        for name in self.lever_names:
            levels = np.asarray(lever_levels_cfg[name], dtype=np.float32)
            if levels.ndim != 1 or levels.size == 0:
                raise ValueError(f"Lever '{name}' levels must be a non-empty 1D array")
            self.lever_levels.append(levels)

        self.policy_matrix = np.asarray(actions_config["policy_matrix"], dtype=np.float32)
        if self.policy_matrix.shape != (self.lever_count, self._controllable_gases):
            raise ValueError(
                "policy_matrix must have shape (num_levers, num_controlled_gases); "
                f"got {self.policy_matrix.shape}, expected ({self.lever_count}, {self._controllable_gases})"
            )

        self.adaptation_levels = np.asarray(
            actions_config["adaptation_levels"], dtype=np.float32
        )
        if self.adaptation_levels.ndim != 1 or self.adaptation_levels.size == 0:
            raise ValueError("Adaptation levels must be a non-empty 1D array")

        self.action_sizes = [len(levels) for levels in self.lever_levels]
        self.action_sizes.append(int(self.adaptation_levels.size))

        # --- heterogeneous economics/impacts ---
        costs_cfg = economics_config["costs"]
        self.climate_damage_costs = np.asarray(
            costs_cfg["climate_damage_costs"], dtype=np.float32
        )
        if self.climate_damage_costs.shape != (self.N,):
            raise ValueError(
                "economics.costs.climate_damage_costs must have length equal to num_agents"
            )

        self.adaptation_costs = np.asarray(
            costs_cfg["adaptation_costs"], dtype=np.float32
        )
        if self.adaptation_costs.shape != (self.N,):
            raise ValueError(
                "economics.costs.adaptation_costs must have length equal to num_agents"
            )

        lever_costs = np.zeros((self.N, self.lever_count), dtype=np.float32)
        for idx, name in enumerate(self.lever_names):
            key = f"{name}_costs"
            if key not in costs_cfg:
                raise ValueError(f"Missing economics cost vector for lever '{name}'")
            arr = np.asarray(costs_cfg[key], dtype=np.float32)
            if arr.shape != (self.N,):
                raise ValueError(
                    f"economics.costs.{key} must have length {self.N}, got {len(arr)}"
                )
            lever_costs[:, idx] = arr
        self.lever_costs = lever_costs

        self.prevention_decay = float(economics_config.get("prevention_decay", 0.95))
        self.max_prevention_benefit = float(
            economics_config.get("max_prevention_benefit", 0.5)
        )

        self.prevention_stock = np.zeros(self.N, dtype=np.float32)
        self._act_space = spaces.MultiDiscrete(self.action_sizes)

        # --- agents  ---
        self.agents = [f"country_{i}" for i in range(self.N)]

        # --- observation space ---
        obs_low = np.concatenate([
            np.array([0.0, 0.0], dtype=np.float32),
            np.zeros(self.N * self._controllable_gases, dtype=np.float32),
            np.full(self.N * self._controllable_gases, -100.0, dtype=np.float32),
            np.zeros(self.N, dtype=np.float32),
        ])
        obs_high = np.concatenate([
            np.array([3.0, 1.0], dtype=np.float32),
            np.full(self.N * self._controllable_gases, 100.0, dtype=np.float32),
            np.full(self.N * self._controllable_gases, 100.0, dtype=np.float32),
            np.ones(self.N, dtype=np.float32),
        ])
        self._obs_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        self.observation_spaces = {a: self._obs_space for a in self.agents}
        self.action_spaces = {a: self._act_space for a in self.agents}

        # --- engine parameters ---
        self.net_params = env_config.get("net_params")
        self.scm_params = env_config.get("scm_params")

        # internal state
        self.reset()

    def observation_space(self, agent_id=None):
        return self.observation_spaces[self.agents[0] if agent_id is None else agent_id]

    def action_space(self, agent_id=None):
        return self.action_spaces[self.agents[0] if agent_id is None else agent_id]

    def _make_scm_engine(self):

        p = self.scm_params
        if p is None:
            raise ValueError("scm_params must be provided when engine='scm'")
        
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
        if p is None:
            raise ValueError("net_params must be provided when engine='net'")

        device = p["device"]
        model = instantiate_model(
            p["model_type"],
            p["n_gas"],
            p["hidden"],
            p["num_layers"],
            kernel_size=p.get("kernel_size", 3),
            device=device,
        )
        state = numpy_state_to_torch(p["state_dict_np"], device)
        load_state_dict(model, state)

        hist_subset = self._hist_emissions_np[:, self.control_indices]

        return CICERONetEngine(
            historical_emissions=hist_subset,
            model=model,
            device=device,
            mu=np.asarray(p["mu"], np.float32),
            std=np.asarray(p["std"], np.float32),
            autocast=p.get("autocast", False),
            use_half=p.get("use_half", False),
            window_size=self.window_size,
        )

    def _current_observation(self):
        temp = float(self.engine.T)
        year_idx = float(self.year_idx - (self.hist_end + 1))
        year_idx_norm = year_idx / self.horizon # normalize to [0..1]

        self.last_controlled_emissions = self.last_agent_emissions[:, self.control_indices]

        emissions_flat = (self.last_controlled_emissions / self._emission_scale).flatten().astype(np.float32)
        cumulative_flat = (self.cumulative_emission_delta / self._cum_scale).flatten().astype(np.float32)
        prevention_stock = self.prevention_stock.astype(np.float32)

        obs_vec = np.concatenate([
            np.array([temp, year_idx_norm], dtype=np.float32),
            emissions_flat,
            cumulative_flat,
            prevention_stock,
            ])
        
        return obs_vec
    
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            _ = np.random.default_rng(seed)
        
        self._temp_log = []
        self._E_global_log = []               # (G,) per step
        self._years_log = []                  # years per MARL step (no terminal rollout)

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
        else:
            raise ValueError(f"Unsupported climate engine kind: {self.engine_kind}")

        #self.net_engine = self._make_net_engine()

        obs_vec = self._current_observation()
        return {agent: obs_vec.copy() for agent in self.agents}, {agent: {} for agent in self.agents}


    def step(self, action_dict):
        # Year
        idx = self.year_idx - (self.hist_end + 1)  

        # Parse lever and adaptation actions
        action_matrix = np.vstack([
            np.asarray(action_dict[ag], dtype=np.int32) for ag in self.agents
        ])
        if action_matrix.shape[1] != self.lever_count + 1:
            raise ValueError(
                f"Expected action dimension {self.lever_count + 1}, got {action_matrix.shape[1]}"
            )

        lever_indices = action_matrix[:, : self.lever_count]
        adaptation_indices = action_matrix[:, self.lever_count]

        lever_efforts = np.zeros((self.N, self.lever_count), dtype=np.float32)
        for j, levels in enumerate(self.lever_levels):
            lever_efforts[:, j] = levels[np.clip(lever_indices[:, j], 0, len(levels) - 1)]

        adaptation_selected = self.adaptation_levels[
            np.clip(adaptation_indices, 0, len(self.adaptation_levels) - 1)
        ]

        # Adaptation stock update
        self.prevention_stock = (
            self.prevention_stock * self.prevention_decay
        ) + adaptation_selected
        self.prevention_stock = np.clip(
            self.prevention_stock, 0.0, self.max_prevention_benefit
        )

        # Emission growth rate
        baseline_growth_year = self.baseline_emission_growth[idx] # 2016..2050 as baseline growth starts from 2016
        #print(f"Baseline growth year {self.year_idx}: {baseline_growth_year}")
        baseline_growth_per_agent = np.broadcast_to(baseline_growth_year, (self.N, self.G)).copy()
        delta_growth = lever_efforts @ self.policy_matrix  # (N, controlled_gases)
        baseline_growth_per_agent[:, self.control_indices] *= (1.0 + delta_growth)
        #print(f"Baseline growth altered with lever efforts: {delta_growth} yielding: {baseline_growth_per_agent}")
        # Emissions actual
        emission_agents = self.last_agent_emissions * baseline_growth_per_agent 
        emission_agents[:, self.control_indices] = np.clip(
            emission_agents[:, self.control_indices], 0.0, None
        )

        emission_global = emission_agents.sum(axis=0)  # (G,)
        self.last_agent_emissions[...] = emission_agents

        # Cummulative delta emissions
        base_agent_year = self.baseline_emissions_agent[:, idx+1, :]  # (N, G) → year = 2016+idx
        delta_from_baseline = (
            emission_agents[:, self.control_indices] - base_agent_year[:, self.control_indices]
        ).astype(np.float32)
        self.cumulative_emission_delta += delta_from_baseline

        # Step climate engine
        engine_input = (
            emission_global[self.control_indices]
            if self.engine_kind == "net"
            else emission_global
        )
        T_next = self.engine.step(engine_input)
        self._temp_log.append(T_next)
        self._years_log.append(int(self.year_idx))
        self._E_global_log.append(emission_global.astype(np.float32))
    
        # (A) Climate disasters cost per agent
        phi = 0.003
        global_climate_cost = phi * (T_next ** 4)
        agent_disaster_cost = global_climate_cost * self.climate_damage_costs * (1.0 - self.prevention_stock)

        # (B) Mitigation lever costs per agent (convex in effort)
        lever_cost = np.sum(self.lever_costs * (lever_efforts ** 2), axis=1)

        # (C) Adaptation investment cost per agent
        adaptation_cost = self.adaptation_costs * adaptation_selected

        # Total reward
        r = - (agent_disaster_cost + lever_cost + adaptation_cost)
        r *= 1e-1

        # print(f"Reward components at year {self.year_idx}:")
        # print(f"  - Climate disaster cost: {agent_disaster_cost}")
        # print(f"  - Mitigation lever cost: {lever_cost}")
        # print(f"  - Adaptation investment cost: {adaptation_cost}")

        # Update time
        self.t += 1
        self.year_idx += 1
        done = (self.t >= self.horizon) or (self.year_idx > self.future_end)    

        # Update observation
        obs_vec = self._current_observation()
        obs_d = {agent: obs_vec.copy() for agent in self.agents}      
        info_d = {a: {} for a in self.agents}

        if done:
            terminal_damage = np.zeros(self.N, dtype=np.float32)

            for year_ahead_idx in range(self.rollout_length):
                idx = self.year_idx - (self.hist_end + 1)
                baseline_growth_year = self.baseline_emission_growth[idx] 
                baseline_growth_per_agent = np.broadcast_to(baseline_growth_year, (self.N, self.G)).copy()

                emission_agents = self.last_agent_emissions * baseline_growth_per_agent 
                emission_agents[:, self.control_indices] = np.clip(
                    emission_agents[:, self.control_indices], 0.0, None
                )
                emission_global = emission_agents.sum(axis=0)  # (G,)
                self.last_agent_emissions[...] = emission_agents

                engine_input = (
                    emission_global[self.control_indices]
                    if self.engine_kind == "net"
                    else emission_global
                )
                T_next = self.engine.step(engine_input)
                self._temp_log.append(T_next)
                self._years_log.append(int(self.year_idx))
                self._E_global_log.append(emission_global.astype(np.float32))

                # Base curvature (covers all temps)
                phi = 0.003
                global_climate_cost = phi * (T_next ** 4)
                agent_disaster_cost = global_climate_cost * self.climate_damage_costs * (1.0 - self.prevention_stock)

                terminal_damage += agent_disaster_cost * 1e-1
                self.year_idx += 1


            r = r - terminal_damage

            for a in self.agents:
                info_d[a]["Temperature_trajectory"] = self._temp_log

            if self.env_config["log_episode_trajectories"]:
                years_arr = np.asarray(self._years_log, dtype=np.int16)                # shape (T,)
                E_global_arr = np.stack(self._E_global_log, axis=0)                    # shape (T, G)
                T_arr = np.asarray(self._temp_log, dtype=np.float32)  # shape (T,)
                log_dir = Path(self.env_config["log_dir"])
                results_dir = Path(self.env_config["output_dir"])
                log_dir = results_dir / log_dir
                if log_dir:
                    os.makedirs(log_dir, exist_ok=True)
                    # Simple deterministic-ish filename
                    ep_id = f"ep_{np.random.randint(1e9)}"
                    np.savez_compressed(
                        os.path.join(log_dir, f"{ep_id}.npz"),
                        years=years_arr,                 # (T,)
                        E_global=E_global_arr,           # (T, G)
                        T=T_arr,                         # (T,)
                        gas_names=np.array(self.gas_names, dtype=object),
                    )


        # Additional info
        rew_d   = {a: float(x) for a, x in zip(self.agents, r)}
        term_d  = {a: done for a in self.agents}
        trunc_d = {a: False for a in self.agents}


        term_d["__all__"]  = done
        trunc_d["__all__"] = False

        return obs_d, rew_d, term_d, trunc_d, info_d
