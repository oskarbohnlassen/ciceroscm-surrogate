import sys
import os
import pickle
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0,os.path.join(os.getcwd(), '../ciceroscm/', 'src')) ## Make ciceroscm importable
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURR_DIR)  # make 'train.py' and 'model.py' importable

from ciceroscm import CICEROSCM
from ciceroscm.parallel.cscmparwrapper import run_ciceroscm_parallel
import ciceroscm.input_handler as input_handler
from train import load_processed_data, format_data
from model import LSTMSurrogate

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


def load_ml_data_model():
    run_dir = "/home/obola/repositories/cicero-scm-surrogate/data/20250926_110035"
    data_dir = os.path.join(run_dir, "processed")
    device = "cuda:0"

    # Load and format data
    X_train, y_train, X_val, y_val, X_test, y_test = load_processed_data(data_dir)
    train_loader, val_loader, test_loader, G = format_data(X_train, y_train, X_val, y_val, X_test, y_test)

    model = LSTMSurrogate(n_gas=G, hidden=256, num_layers=2).to(device)

    # Load model parameters
    model_dir = os.path.join(run_dir, "model_lstm_v1.pth")
    model.load_state_dict(torch.load(model_dir, map_location=device, weights_only=False))

    return train_loader, val_loader, test_loader, model


def format_ml_data_to_rl(train_loader, val_loader, test_loader, shuffle=False, num_workers=0, pin_memory=True):
    """
    Collapse train/val/test loaders into a single DataLoader that yields
    one sample per batch (RL-style). Assumes each incoming loader yields
    (X, y) or X; ignores y entirely.

    Returns
    -------
    rl_loader : DataLoader
        DataLoader over a TensorDataset(X) with batch_size=1.
    """
    xs = []
    
    for batch in test_loader:
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        # move to CPU and detach
        xs.append(x.detach().cpu())

    X = torch.cat(xs, dim=0)[:10000]
    ds = TensorDataset(X)  # only X; targets intentionally ignored
    rl_loader = DataLoader(ds, batch_size=1, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

    return rl_loader

def run_ml_inference(model, rl_loader, device = "cuda:0"):

    # Model eval
    model.eval().to(device)

    # The data we need
    latencies = []

    ## WARM UP
    with torch.inference_mode():
        for batch in rl_loader:
            xb = batch[0]
            xb = xb.to(device, non_blocking=True)
            _ = model(xb)
            torch.cuda.synchronize()
            break

    with torch.inference_mode():
        for i, batch in enumerate(tqdm(rl_loader, desc="CICERO-NET speed-test")):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            xb = batch[0]
            xb = xb.to(device, non_blocking=True)            
            y_hat_i = model(xb)
            t1 = time.perf_counter()
            latencies.append(t1 - t0)

    latencies_numpy = np.asarray(latencies, dtype=float)
 
    return latencies_numpy

 
def load_cicero_data():
    scenario_path = "/home/obola/repositories/cicero-scm-surrogate/data/20250926_110035/raw/scenarios.pkl"
    scenarios = pickle.load(open(scenario_path, "rb"))

    return scenarios

def format_cicero_data_into_rl(scenarios):

    rl_scenarios = []
    for scenario in scenarios:
        em_data = scenario['emissions_data'].copy()
        for policy_year in range(2015,2066):
            em_data_new = em_data.loc[:policy_year].copy()
            new_scenario = scenario.copy() 
            new_scenario['emissions_data'] = em_data_new
            new_scenario['nyend'] = policy_year

            rl_scenarios.append(new_scenario)

    return rl_scenarios

def run_cicero_rl(rl_scenarios):
    
    output_variables = ["Surface Air Temperature Change"]

    latencies = []
    for i, scenario in enumerate(tqdm(rl_scenarios[:10000], desc="CICERO-SCM speed-test")):
        cscm_dir=CICEROSCM(scenario)
        t0 = time.time()
        cscm_dir._run({"results_as_dict":True}, pamset_udm=cfg[0]["pamset_udm"], pamset_emiconc=cfg[0]["pamset_emiconc"])
        t1 = time.time()
        latencies.append(t1 - t0)

    latencies_numpy = np.asarray(latencies, dtype=float)
    return latencies_numpy

def main():
    train_loader, val_loader, test_loader, model = load_ml_data_model()
    rl_loader = format_ml_data_to_rl(train_loader, val_loader, test_loader, shuffle=False)
    surrogate_latencies_cuda = run_ml_inference(model, rl_loader, device="cuda:0")

    print(f"Mean latency surrogate cuda: {np.mean(surrogate_latencies_cuda)}")
    print(f"Sum latency surrogate cuda: {np.sum(surrogate_latencies_cuda)}")
    print(f"Num experiment surrogate cuda: {len(surrogate_latencies_cuda)}")
    np.savez("surrogate_latencies_cuda.npz", surrogate_latencies_cuda)

    surrogate_latencies_cpu = run_ml_inference(model, rl_loader, device="cpu")

    print(f"Mean latency surrogate cpu: {np.mean(surrogate_latencies_cpu)}")
    print(f"Sum latency surrogate cpu: {np.sum(surrogate_latencies_cpu)}")
    print(f"Num experiment surrogate cpu: {len(surrogate_latencies_cpu)}")
    np.savez("surrogate_latencies_cpu.npz", surrogate_latencies_cpu)

    scenarios = load_cicero_data()
    rl_scenarios = format_cicero_data_into_rl(scenarios)
    cicero_latencies = run_cicero_rl(rl_scenarios)

    print(f"Mean latency cicero: {np.mean(cicero_latencies)}")
    print(f"Sum latency cicero: {np.sum(cicero_latencies)}")
    print(f"Num experiment cicero: {len(cicero_latencies)}")
    np.savez("cicero_latencies.npz", cicero_latencies)

if __name__ == "__main__":
    main()




