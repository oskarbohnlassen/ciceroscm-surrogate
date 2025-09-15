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
from marl_env_step import CICEROSCMEngine, CICERONetEngine
from marl_env import ClimateMARL

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

conc_data_first = 1750
conc_data_last = 2100

em_data_start = 1900
em_data_policy = 2015
em_data_end = 2015

gaspam_data, conc_data, em_data, nat_ch4_data, nat_n2o_data = load_core_data()

historical_emissions = em_data.loc[:2015]
baseline_emissions = em_data.loc[2015+1:]

test_data_dir = "/home/obola/repositories/cicero-scm-surrogate/ciceroscm/tests/test-data"

baseline_scenario = {
            "gaspam_data": gaspam_data,
            "nyend": em_data_end,
            "nystart": em_data_start,
            "emstart": em_data_policy,
            "concentrations_data": conc_data,
            "nat_ch4_data": nat_ch4_data,
            "nat_n2o_data": nat_n2o_data,
            "emissions_data": historical_emissions,      
            "udir": test_data_dir,
            "idtm": 24,
            "scenname": "baseline_scenario",
        }


def load_surrogate():
    # Load model
    run_dir="/home/obola/repositories/cicero-scm-surrogate/data/20250805_152136"
    device="cuda:0"
    weights_name="model_lstm.pth"

    model = LSTMSurrogate(n_gas=40, hidden=128, num_layers=1).to(device)
    wpath = os.path.join(run_dir, weights_name)
    state = torch.load(wpath, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    cicero_net = model

    # Get standardizer
    run_path = "/home/obola/repositories/cicero-scm-surrogate/data/20250805_152136"
    data_path = os.path.join(run_path, "processed")

    X_train = np.load(os.path.join(data_path, "X_train.npy"))

    gas_mu  = X_train.reshape(-1, 40).mean(axis=0)     # per-gas mean
    gas_std = X_train.reshape(-1, 40).std(axis=0) + 1e-6

    return cicero_net, gas_mu, gas_std


def test_scm_engine():
    # Historical emissions matrix H_hist: shape (1900..2015, G)
    # Keep column order (this must match your per-step E_t order!)
    scm_engine = CICEROSCMEngine(
        historical_emissions=historical_emissions,
        gaspam_data=gaspam_data,
        conc_data=conc_data,
        nat_ch4_data=nat_ch4_data,
        nat_n2o_data=nat_n2o_data,
        pamset_udm=cfg[0]["pamset_udm"],
        pamset_emiconc=cfg[0]["pamset_emiconc"],
        em_data_start=em_data_start,
        em_data_policy=em_data_policy,
        udir=test_data_dir,
        idtm=24,
        scenname="scm_unit_test",
    )

    for i in range(5):
        print(f"Year: {i}")
        next_emission = copy.copy(baseline_emissions.iloc[i].values)
        t0 = time.time()
        T = scm_engine.step(next_emission)
        t1 = time.time()
        print(f"Time: {t1-t0}")
        print(f"Temperature: {T}")

def test_surrogate_engine(cicero_net, gas_mu, gas_std):
    net_engine = CICERONetEngine(historical_emissions = historical_emissions,
                             model = cicero_net,
                             device="cuda:0", 
                             window=50,
                             mu=gas_mu, 
                             std=gas_std,
                             autocast=False, 
                             use_half=False)
                             

    for i in range(5):
        print(f"Year: {i}")
        next_emission = copy.copy(baseline_emissions.iloc[i].values)
        t0 = time.perf_counter()
        T = net_engine.step(next_emission)
        t1 = time.perf_counter()
        print(f"Time: {t1-t0}")
        print(f"Temperature: {T}")


class ClimateMarlExperiment():
    def __init__(self, historical_emissions, baseline_emissions):
        self.env_config = {"N": 4, 
                           "engine": "scm", 
                            "horizon": 35, 
                            "G": 40, 
                            "year0_hist": 1900, 
                            "hist_end": 2015, 
                            "future_end": 2050,
                            "random_seed": 0,
                            }

        self.historical_emissions = historical_emissions
        self.baseline_emissions = baseline_emissions

        self.emission_shares = np.tile(np.array([0.05, 0.5, 0.15, 0.3]), (40, 1)).T

        # Economic parameters per country
        self.economics_config = {
            "climate_disaster_cost":       [1.00, 1.20, 0.80, 1.05],  # cost of climate change per temperature increase (per country)
            "climate_investment_cost":     [1.00, 1.10, 0.90, 0.98],  # cost of climate investment
            "economic_growth_sensitivity": [1.00, 1.00, 1.00, 1.00],  # economic growth sensitivity to investment in prevention
            }
        # Actions
        self.actions_config = {
            "m_levels": [-0.05, -0.03, -0.01, 0.0, 0.01, 0.03, 0.05]  # Deviation from baseline emission shares
            }

        self.emission_data = {
            "historical_emissions": self.historical_emissions,
            "baseline_emissions": self.baseline_emissions,
            "emission_shares": self.emission_shares}

    def load_surrogate(self):
        # Load model
        run_dir="/home/obola/repositories/cicero-scm-surrogate/data/20250805_152136"
        device="cuda:0"
        weights_name="model_lstm.pth"

        model = LSTMSurrogate(n_gas=40, hidden=128, num_layers=1).to(device)
        wpath = os.path.join(run_dir, weights_name)
        state = torch.load(wpath, map_location=device, weights_only=False)
        model.load_state_dict(state)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        self.cicero_net = model

        # Get standardizer
        run_path = "/home/obola/repositories/cicero-scm-surrogate/data/20250805_152136"
        data_path = os.path.join(run_path, "processed")

        X_train = np.load(os.path.join(data_path, "X_train.npy"))

        self.gas_mu  = X_train.reshape(-1, 40).mean(axis=0)     # per-gas mean
        self.gas_std = X_train.reshape(-1, 40).std(axis=0) + 1e-6

    
    def make_env_step_configs(self):
        '''
        Make env step configs
        '''

        if self.env_config['engine'] == "net":
            state_dict_cpu = {k: v.cpu() for k, v in self.cicero_net.state_dict().items()}
            state_dict_np = {k: v.numpy() for k, v in state_dict_cpu.items()}

            net_params = {
                "model_module": "src.model",          # module path string
                "model_class": "LSTMSurrogate",       # class name string
                "model_kwargs": {"n_gas": 40, "hidden": 128, "num_layers": 1},
                "state_dict_np": state_dict_np,       # pure numpy
                "device": "cuda:0",
                "window": 50,
                "mu": self.gas_mu.tolist(),
                "std": self.gas_std.tolist(),
                "autocast": False,
                "use_half": False,
            }
            self.env_config["net_params"] = net_params

        elif self.env_config['engine'] == "scm":
            net_params = {
                "gaspam_data": gaspam_data,
                "conc_data": conc_data,
                "nat_ch4_data": nat_ch4_data,
                "nat_n2o_data": nat_n2o_data,
                "pamset_udm": cfg[0]["pamset_udm"],
                "pamset_emiconc": cfg[0]["pamset_emiconc"],
                "em_data_start": em_data_start,
                "em_data_policy": em_data_policy,
                "udir": test_data_dir,
                "idtm": 24,
                "scenname": "scm_marl_test"
            }
            self.env_config["net_params"] = net_params
            

        
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
        marl_env = ClimateMARL(self.env_config, self.emission_data, self.economics_config, self.actions_config)
        obs_sp = marl_env.observation_space("country_0")
        act_sp = marl_env.action_space("country_0")
        N = self.env_config["N"]

        # One policy per country (no sharing)
        policies = {f"country_{i}": (None, obs_sp, act_sp, {}) for i in range(N)}
        policy_mapping_fn = lambda agent_id, *_, **__: agent_id

        algo = (
        PPOConfig()
        .environment(
            env="climate_marl",
            env_config={
                "env_config": self.env_config,
                "emission_data": self.emission_data,
                "economics_config": self.economics_config,
                "actions_config": self.actions_config,
            },
        )
        .framework("torch")
        .api_stack(enable_rl_module_and_learner=True, enable_env_runner_and_connector_v2=True)
        .env_runners(num_env_runners=0, num_envs_per_env_runner=32)
        .rl_module(
            model_config={"fcnet_hiddens": [64], "vf_share_layers": True})
        .training(         
            train_batch_size=32 * self.env_config["horizon"],  
            minibatch_size=560,              
            num_epochs=1,                  
            gamma=0.99,
            lr=2e-3,
        )
        .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)
        .resources(num_gpus=0)     
        .build_algo()
        )

        for i in range(20):
            result = algo.train()
            mean_ret = (
                result.get("episode_reward_mean")
                or result.get("episode_return_mean")
                or result.get("env_runners", {}).get("episode_return_mean")
                or result.get("sampler_results", {}).get("episode_reward_mean")
            )
            
            print(f"iter {i}: mean_return={mean_ret}")
            print(result['timers'])


def main():

    # Test scm_engine
    #test_scm_engine()

    # Load surrogate
    # cicero_net, gas_mu, gas_std = load_surrogate()
    # print(cicero_net)

    # Test surrogate engine
    # test_surrogate_engine(cicero_net, gas_mu, gas_std)

    # Register env
    marl_env = ClimateMarlExperiment(historical_emissions, baseline_emissions)
    marl_env.run_marl_experiment()

    return

if __name__ == "__main__":
    main()
