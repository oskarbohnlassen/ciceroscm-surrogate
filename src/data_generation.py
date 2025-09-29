import sys
import os
import pickle
import numpy as np
import pandas as pd
import time
from tqdm.auto import tqdm

sys.path.insert(0,os.path.join(os.getcwd(), '../ciceroscm/', 'src'))

from ciceroscm import CICEROSCM
from ciceroscm.parallel.cscmparwrapper import run_ciceroscm_parallel
import ciceroscm.input_handler as input_handler

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
em_data_end = 2075

tight_bounds = (0.995, 1.005)

loose_bounds = {
    "CO2_FF":   (0.925, 1.075),
    "CO2_AFOLU":(0.925, 1.075),
    "CH4":      (0.925, 1.075),
    "N2O":      (0.925, 1.075)
}

test_data_dir = "/home/obola/repositories/cicero-scm-surrogate/ciceroscm/tests/test-data"
data_output_dir = "/home/obola/repositories/cicero-scm-surrogate/data/"
run_id = time.strftime("%Y%m%d_%H%M%S")
data_output_dir_run = os.path.join(data_output_dir, run_id)

num_scenarios = 20000

def load_core_data():
    gaspam_data = input_handler.read_components(
        os.path.join(test_data_dir, "gases_v1RCMIP.txt"))

    conc_data = input_handler.read_inputfile(
        os.path.join(test_data_dir, "ssp245_conc_RCMIP.txt"), True, conc_data_first, conc_data_last)
    
    ih = input_handler.InputHandler({"nyend": em_data_end, "nystart": em_data_start, "emstart": em_data_policy})
    
    em_data = ih.read_emissions(os.path.join(test_data_dir, "ssp245_em_RCMIP.txt"))
    nat_ch4_data = input_handler.read_natural_emissions(
        os.path.join(test_data_dir, "natemis_ch4.txt"), "CH4")
    
    nat_n2o_data = input_handler.read_natural_emissions(
        os.path.join(test_data_dir, "natemis_n2o.txt"), "N2O")
    
    return gaspam_data, conc_data, em_data, nat_ch4_data, nat_n2o_data

def generate_emission_policies(em_data, nat_ch4_data, nat_n2o_data, gaspam_data, conc_data, num_scenarios):
    # 1) year-over-year baseline multipliers  E_t / E_{t-1}
    shifted      = em_data.shift(1)
    growth_base  = em_data.divide(shifted.replace(0, np.nan))
    growth_base  = growth_base.replace([np.inf, -np.inf], np.nan).fillna(1.0)

    years_future = np.arange(em_data_policy, em_data_end + 1)
    gas_cols     = em_data.columns

    scenarios = []

    scenario_iter = tqdm(
        range(num_scenarios),
        desc="Generating scenarios",
        unit="scenario",
        leave=False,
    )

    for k in scenario_iter:
        rng   = np.random.default_rng(seed=k)        # reproducible
        delta = np.empty((len(years_future), len(gas_cols)), dtype=float)
        
        # Smooth the deltas to avoid abrupt changes
        for j, gas in enumerate(gas_cols):
            lo, hi = loose_bounds.get(gas, tight_bounds)        # pick bounds
            delta[:, j] = rng.uniform(lo, hi, size=len(years_future))
            # Smooth the deltas (EMA in log-space, alpha=0.7 as an example)
            u = np.log1p(delta[:, j])                          # log(1+d)
            for t in range(1, len(u)):
                u[t] = 0.7 * u[t-1] + 0.3 * u[t]               # EMA smoothing
            delta[:, j] = np.expm1(u)                          # back to % growth


        scale_df = pd.DataFrame(delta, index=years_future, columns=gas_cols)

        em_policy = em_data.copy()
        for t in years_future:
            gfactor = growth_base.loc[t] * scale_df.loc[t]        
            em_policy.loc[t] = em_policy.loc[t-1] * gfactor       

        new_scen = {
            "gaspam_data": gaspam_data,
            "nyend": em_data_end,
            "nystart": em_data_start,
            "emstart": em_data_policy,
            "concentrations_data": conc_data,
            "nat_ch4_data": nat_ch4_data,
            "nat_n2o_data": nat_n2o_data,
            "emissions_data": em_policy,
            "udir": test_data_dir,
            "idtm": 24,
            "scenname": f"rw_growth_smooth_{k:03d}",
        }
        scenarios.append(new_scen)

    return scenarios
    
def run_scenarios(scenarios):
    output_variables = ["Surface Air Temperature Change"]
    results = run_ciceroscm_parallel(scenarios, cfg, output_variables)
    return results

def save_raw_data(scenarios, results):
    data_output_dir_run_raw = os.path.join(data_output_dir_run, "raw")
    os.makedirs(data_output_dir_run_raw, exist_ok=True)
    ## Save as pickle
    with open(os.path.join(data_output_dir_run_raw, "scenarios.pkl"), "wb") as f:
        pickle.dump(scenarios, f)
    with open(os.path.join(data_output_dir_run_raw, "results.pkl"), "wb") as f:
        pickle.dump(results, f)


def main():
    gaspam_data, conc_data, em_data, nat_ch4_data, nat_n2o_data = load_core_data()
    scenarios = generate_emission_policies(em_data, nat_ch4_data, nat_n2o_data, gaspam_data, conc_data, num_scenarios)
    results = run_scenarios(scenarios)
    save_raw_data(scenarios, results)

if __name__ == "__main__":
    main()
