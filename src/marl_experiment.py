from email import policy
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
from collections import defaultdict
import json


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ray.*")
warnings.filterwarnings("ignore", message=".*Logger.*deprecated.*", category=DeprecationWarning)

sys.path.insert(0,os.path.join(os.getcwd(), '../ciceroscm/', 'src')) ## Make ciceroscm importable
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURR_DIR)  # make 'train.py' and 'model.py' importable

from ciceroscm import CICEROSCM
from ciceroscm.parallel.cscmparwrapper import run_ciceroscm_parallel
import ciceroscm.input_handler as input_handler
from train import load_processed_data, format_data
from model import LSTMSurrogate
from data_generation import load_core_data
from marl_env import ClimateMARL


class ClimateMarlExperiment():
    def __init__(self,
                env_config, 
                economics_config, 
                actions_config,
                emission_data):
        
        # Env config
        self.env_config = env_config

        # Emission data
        self.emission_data = emission_data
        self.historical_emissions = emission_data['historical_emissions']
        self.baseline_emissions = emission_data['baseline_emissions']
        self.gaspam_data = emission_data['gaspam_data']
        self.conc_data = emission_data['conc_data']
        self.nat_ch4_data = emission_data['nat_ch4_data']
        self.nat_n2o_data = emission_data['nat_n2o_data']

        # Economics
        self.economics_config = economics_config

        # Actions
        self.actions_config = actions_config
        
        
    def compute_training_stats(self, result):

        for key, value in result.items():
            print(f"{key}: {value} \n")
            print("-"*40)
            

        return None

    def load_surrogate(self):
        # Load model
        run_dir="/home/obola/repositories/cicero-scm-surrogate/data/20250805_152136"
        device="cuda:0"
        weights_name="model_lstm.pth"

        # Get standardizer
        run_path = "/home/obola/repositories/cicero-scm-surrogate/data/20250805_152136"
        data_path = os.path.join(run_path, "processed")
        X_train = np.load(os.path.join(data_path, "X_train.npy"))

        model = LSTMSurrogate(n_gas=40, hidden=128, num_layers=1).to(device)
        wpath = os.path.join(run_dir, weights_name)
        state = torch.load(wpath, map_location=device, weights_only=False)
        model.load_state_dict(state)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        self.cicero_net = model
        self.gas_mu  = X_train.reshape(-1, 40).mean(axis=0) 
        self.gas_std = X_train.reshape(-1, 40).std(axis=0) + 1e-6

    
    def make_env_step_configs(self):
        '''
        Make env step configs
        '''
        # Surrogate net params
        state_dict_cpu = {k: v.cpu() for k, v in self.cicero_net.state_dict().items()}
        state_dict_np = {k: v.numpy() for k, v in state_dict_cpu.items()}

        net_params = {
            "state_dict_np": state_dict_np,       # pure numpy
            "device": "cuda:0",
            "mu": self.gas_mu.tolist(),
            "std": self.gas_std.tolist(),
            "autocast": False,
            "use_half": False,
        }
        self.env_config["net_params"] = net_params

        # SCM params
        cfg = [
            {
                "pamset_udm": {
                    "rlamdo": 15.08357,
                    "akapa": 0.6568376339229769,
                    "cpi": 0.2077266,
                    "W": 2.205919,
                    "beto": 6.89822,
                    "lambda": 0.6062529,
                    "mixed": 107.2422,
                },
                "pamset_emiconc": {
                    "qbmb": 0.0,
                    "qo3": 0.5,
                    "qdirso2": -0.3562,
                    "qindso2": -0.96609,
                    "qbc": 0.1566,
                    "qoc": -0.0806,
                },
                "Index": "13555_old_NR_improved",
            }
        ]
        
        scm_params = {
            "gaspam_data": self.gaspam_data,
            "conc_data": self.conc_data,
            "nat_ch4_data": self.nat_ch4_data,
            "nat_n2o_data": self.nat_n2o_data,
            "pamset_udm": cfg[0]["pamset_udm"],
            "pamset_emiconc": cfg[0]["pamset_emiconc"],
            "em_data_start": 1900,
            "em_data_policy": 2015,
            "udir": "/home/obola/repositories/cicero-scm-surrogate/ciceroscm/tests/test-data",
            "idtm": 24,
            "scenname": "scm_marl_test"
        }
        self.env_config["scm_params"] = scm_params

    def rollout_greedy_actions(self, algo, policy_mapping_fn, seed=None):
        """
        Run one greedy (deterministic) episode and return actions + rewards.

        Returns:
            trajectory: list of {agent_id -> action(list[int])} per step
            per_agent_return: dict[agent_id -> float]
            total_return: float
            episode_len: int
        """
        env = ClimateMARL(
            self.env_config,
            self.emission_data,
            self.economics_config,
            self.actions_config,
        )

        obs, _ = env.reset(seed=seed) if seed is not None else env.reset()

        done = False
        t = 0
        trajectory = []
        per_agent_return = {f"country_{i}": 0.0 for i in range(self.env_config["N"])}
        total_return = 0.0

        # --- RNN state per agent (and per policy) ---
        # RLlib LSTM expects a list [h, c] for TorchPolicy by default.
        rnn_state = {}
        for agent_id in obs.keys():
            pol_id = policy_mapping_fn(agent_id)
            pol = algo.get_policy(pol_id)
            rnn_state[agent_id] = pol.get_initial_state()  # usually [h0, c0]

        while not done:
            actions, step_log = {}, {}

            for agent_id, agent_obs in obs.items():
                pol_id = policy_mapping_fn(agent_id)
                pol = algo.get_policy(pol_id)

                # Deterministic (greedy) action; pass the current RNN state.
                # compute_single_action handles batching for you.
                act, new_state, _ = pol.compute_single_action(
                    agent_obs,
                    state=rnn_state[agent_id],
                    explore=False,
                )
                # Store next RNN state for this agent
                rnn_state[agent_id] = new_state

                actions[agent_id] = np.asarray(act, dtype=np.int64)
                step_log[agent_id] = np.asarray(act).tolist()

            trajectory.append(step_log)
            obs, rewards, terminated, truncated, info_dict = env.step(actions)

            for aid, r in rewards.items():
                per_agent_return[aid] += float(r)
                total_return += float(r)

            done = terminated.get("__all__", False) or truncated.get("__all__", False)
            t += 1

        return trajectory, per_agent_return, total_return, t, obs

    def print_greedy_summary(self, greedy_traj):
        # Map indices -> actual values
        em_vals = np.asarray(self.actions_config["emission_actions"], dtype=float)   # e.g., [-0.05, ...]
        pr_vals = np.asarray(self.actions_config["prevention_actions"], dtype=float) # e.g., [0.00, 0.02, ...]

        agents = [f"country_{i}" for i in range(self.env_config["N"])]
        T = len(greedy_traj)
        start_year = int(self.env_config["hist_end"]) + 1
        years = list(range(start_year, start_year + T))

        print(f"\nGreedy policy choices over {T} years ({years[0]}–{years[-1]}):")

        policy_logger = {}
        for ag in agents:
            # Collect indices chosen each year
            em_idx = [int(step[ag][0]) for step in greedy_traj]
            pr_idx = [int(step[ag][1]) for step in greedy_traj]
            # Convert to actual % values
            em_series = [em_vals[i] * 100.0 for i in em_idx]  # percent
            pr_series = [pr_vals[i] * 100.0 for i in pr_idx]  # percent

            print(f"- {ag}:")
            print("  Emission delta (%):  " + ", ".join(f"{v:.1f}" for v in em_series))
            print("  Prevention rate (%): " + ", ".join(f"{v:.1f}" for v in pr_series))
            policy_logger[ag] = {"emission_delta": em_series, "prevention_rate": pr_series}

        return policy_logger

    def register_env_rl(self):
        def env_creator(cfg):
            # cfg is the single env_config RLlib passes.
            return ClimateMARL(
                            cfg["env_config"],
                            cfg["emission_data"],
                            cfg["economics_config"],
                            cfg["actions_config"],
                            )

        register_env("climate_marl", env_creator)
    
    def run_marl_experiment(self):
        # Prepare MARL
        self.load_surrogate()
        self.make_env_step_configs()
        self.register_env_rl()

        # DO MARL
        marl_env = ClimateMARL(self.env_config, 
                               self.emission_data, 
                               self.economics_config, 
                               self.actions_config)
        obs_sp = marl_env.observation_space("country_0")
        act_sp = marl_env.action_space("country_0")
        N = self.env_config["N"]

        # One policy per country (no sharing)
        policies = {f"country_{i}": (None, obs_sp, act_sp, {}) for i in range(N)}
        policy_mapping_fn = lambda agent_id, *_, **__: agent_id

        base_config = (
        PPOConfig()
        .framework("torch")
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .environment(
            env="climate_marl",
            env_config={
                "env_config": self.env_config,
                "emission_data": self.emission_data,
                "economics_config": self.economics_config,
                "actions_config": self.actions_config,
            },
        )
        .training(     
            model={
                "use_lstm": True,
                "lstm_cell_size": 64,
                "max_seq_len": self.env_config["horizon"],
                "vf_share_layers": True,
            },
            train_batch_size=8 * self.env_config["N"] * self.env_config["horizon"],  
            minibatch_size=4 * self.env_config["N"] * self.env_config["horizon"],              
            num_epochs=5,                  
            gamma=0.999,
            lr=3e-4,
            lr_schedule=[[0, 3e-4], [50_000, 2e-4], [150_000, 1e-4], [200_000, 5e-5]],
            clip_param=0.2,     
            entropy_coeff=0.01,
            entropy_coeff_schedule=[[0, 0.01], [100_000, 0.003], [150_000, 0.001], [200_000, 0.0001]],
        )

        .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)
        .resources(num_gpus=1)     
        .env_runners(
                rollout_fragment_length='auto',
                batch_mode="complete_episodes",
                num_env_runners=1,
                num_envs_per_env_runner=32,
                num_gpus_per_env_runner=1,
                sample_timeout_s=600,
            )
        )
        
        algo = base_config.build_algo()

        ## Make folder in marl_results to save results
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        model_engine = self.env_config["engine"]
        folder_name = f"{timestamp}_{model_engine}_homogeneous"
        results_dir = os.path.join("marl_results", folder_name)
        os.makedirs(results_dir, exist_ok=False) # Make error if already exists
        
        num_iterations = 100
        num_env_steps = 0
        per_agent_reward_logger_train = {}
        per_agent_reward_logger_greedy = {}
        per_agent_policy_logger_greedy = {}
        training_time_stats = {}
        t0 = time.time()
        for i in range(num_iterations):
            result = algo.train()
            # Per-agent (per-policy) mean rewards
            per_agent_rewards = result["env_runners"]["policy_reward_mean"]
            num_env_steps += result["env_runners"]["episodes_timesteps_total"]

            per_agent_reward_logger_train[num_env_steps] = per_agent_rewards
            print(f"Per-agent eval rewards at iteration: {i} after {num_env_steps} timesteps")
            for policy_id, mean_rew in per_agent_rewards.items():
                print(f"  {policy_id}: {mean_rew:.4f}")    
            print(f"Timers:   {result['timers']}")

            if i % 2 == 0:
                traj, agent_ret, total_ret, ep_len, obs_log = self.rollout_greedy_actions(algo, policy_mapping_fn)
                per_agent_reward_logger_greedy[num_env_steps] = agent_ret
                print("[greedy] per-agent returns:", {k: round(v, 2) for k, v in agent_ret.items()})
                policy_logger = self.print_greedy_summary(traj)
                per_agent_policy_logger_greedy[num_env_steps] = policy_logger

            training_time_stats[num_env_steps] = time.time() - t0
            print("="*60)

            if i % 10 == 0:
                
                # Collect all results
                policy_reward_intermediate_dict = {
                    "train_reward": per_agent_reward_logger_train,
                    "greedy_reward": per_agent_reward_logger_greedy,
                    "greedy_policy": per_agent_policy_logger_greedy,
                    "training_time_stats": training_time_stats,
                    }
                
                # Save algo results in results_dir as json
                with open(os.path.join(results_dir, "marl_experiment_results_intermediate.json"), "w") as f:
                    json.dump(policy_reward_intermediate_dict, f, indent=4)

        # Collect all results
        policy_reward_dict = {
            "train_reward": per_agent_reward_logger_train,
            "greedy_reward": per_agent_reward_logger_greedy,
            "greedy_policy": per_agent_policy_logger_greedy,
            "training_time_stats": training_time_stats,
            }
        
        # Save algo results in results_dir as json
        with open(os.path.join(results_dir, "marl_experiment_results.json"), "w") as f:
            json.dump(policy_reward_dict, f, indent=4)



def main():
    # Load base data
    gaspam_data, conc_data, em_data, nat_ch4_data, nat_n2o_data = load_core_data()
    shifted = em_data.shift(1)
    growth_base = em_data.divide(shifted.replace(0, np.nan))
    baseline_emission_growth = growth_base.replace([np.inf, -np.inf], np.nan).fillna(1.0).loc[2015+1:]
    historical_emissions = em_data.loc[:2015]
    baseline_emissions = em_data.loc[2015:]
    emission_shares = np.tile(np.array([0.25, 0.25, 0.25, 0.25]), (40, 1)).T

    # Env config
    env_config = {"N": 4, 
                        "engine": "scm", 
                        "horizon": 35, 
                        "G": 40, 
                        "hist_end": 2015, 
                        "future_end": 2050,
                        "random_seed": 0}

    # Economic parameters per country
    economics_config = {
        "climate_disaster_sensitivity":       [1.00, 1.00, 1.00, 1.00],  # cost of climate change per temperature increase (per country)
        "emission_reduction_sensitivity":     [1.00, 1.00, 1.00, 1.00],  # cost of having lower emissions (per country)
        "climate_prevention_sensitivity":     [1.00, 1.00, 1.00, 1.00],  # cost of mitigation technologies (per country)
        }
    
    # Actions
    actions_config = {
        "emission_actions": [-0.04, 0.0, 0.04],  # Deviation from baseline emission shares
        "prevention_actions": [0.0, 0.03, 0.08]
        }

    emission_data = {
        "historical_emissions": historical_emissions,
        "baseline_emissions": baseline_emissions,
        "emission_shares": emission_shares,
        "baseline_emission_growth": baseline_emission_growth,
        "gaspam_data": gaspam_data,
        "conc_data": conc_data,
        "nat_ch4_data": nat_ch4_data,
        "nat_n2o_data": nat_n2o_data
    }

    # Register env
    marl_env = ClimateMarlExperiment(env_config,
                                    economics_config,
                                    actions_config,
                                    emission_data)
    
    marl_env.run_marl_experiment()

if __name__ == "__main__":
    main()
