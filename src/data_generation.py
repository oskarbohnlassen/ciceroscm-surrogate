import sys
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm.auto import tqdm

sys.path.insert(0, os.path.join(os.getcwd(), '../ciceroscm/', 'src'))

from ciceroscm.parallel.cscmparwrapper import run_ciceroscm_parallel
import ciceroscm.input_handler as input_handler
from utils import load_yaml_config


CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def load_configs():
    """Load CICERO and data-generation configuration files."""
    cicero_config = load_yaml_config(CONFIG_DIR / "cicero-scm.yaml")
    data_generation_config = load_yaml_config(
        CONFIG_DIR / "data_generation.yaml", "data_generation_params"
    )
    return cicero_config, data_generation_config


class DataGenerationPipeline:
    """Coordinate CICERO scenario generation from configuration values."""

    def __init__(self, cicero_config: dict, data_generation_config: dict):
        self.cicero_config = cicero_config
        self.data_generation_config = data_generation_config
        self.cfg = cicero_config["cfg"]
        self.conc_data_first = cicero_config["conc_data_first"]
        self.conc_data_last = cicero_config["conc_data_last"]
        self.em_data_start = cicero_config["em_data_start"]
        self.em_data_policy = cicero_config["em_data_policy"]
        self.em_data_end = cicero_config["em_data_end"]
        self.test_data_dir = Path(cicero_config["test_data_dir"])

        self.num_scenarios = data_generation_config["num_samples"]
        self.smoothing_alpha = data_generation_config["alpha"]
        if not 0 <= self.smoothing_alpha <= 1:
            raise ValueError("alpha must be between 0 and 1 inclusive")

        self.data_output_dir = Path(data_generation_config["data_output_dir"])
        self.specific_gas_bounds = {
            gas: tuple(bounds)
            for gas, bounds in data_generation_config.get("specific_gas_bounds", {}).items()
        }
        self.default_gas_bounds = tuple(data_generation_config["remaining_gas_bounds"])

        self.run_id = time.strftime("%Y%m%d_%H%M%S")
        self.data_output_dir_run = self.data_output_dir / self.run_id

    def load_core_data(self):
        """Read fixed CICERO inputs needed for scenario generation."""
        gaspam_data = input_handler.read_components(
            str(self.test_data_dir / "gases_v1RCMIP.txt")
        )

        conc_data = input_handler.read_inputfile(
            str(self.test_data_dir / "ssp245_conc_RCMIP.txt"),
            True,
            self.conc_data_first,
            self.conc_data_last,
        )

        ih = input_handler.InputHandler(
            {"nyend": self.em_data_end, "nystart": self.em_data_start, "emstart": self.em_data_policy}
        )

        em_data = ih.read_emissions(str(self.test_data_dir / "ssp245_em_RCMIP.txt"))
        nat_ch4_data = input_handler.read_natural_emissions(
            str(self.test_data_dir / "natemis_ch4.txt"), "CH4"
        )

        nat_n2o_data = input_handler.read_natural_emissions(
            str(self.test_data_dir / "natemis_n2o.txt"), "N2O"
        )

        return gaspam_data, conc_data, em_data, nat_ch4_data, nat_n2o_data

    def generate_scenarios(
        self,
        em_data,
        nat_ch4_data,
        nat_n2o_data,
        gaspam_data,
        conc_data,
    ):
        """Sample emissions policies within configured bounds and smooth them."""
        shifted = em_data.shift(1)
        growth_base = em_data.divide(shifted.replace(0, np.nan))
        growth_base = growth_base.replace([np.inf, -np.inf], np.nan).fillna(1.0)

        years_future = np.arange(self.em_data_policy, self.em_data_end + 1)
        gas_cols = em_data.columns

        scenarios = []

        scenario_iter = tqdm(
            range(self.num_scenarios),
            desc="Generating scenarios",
            unit="scenario",
            leave=False,
        )

        for k in scenario_iter:
            rng = np.random.default_rng(seed=k)  # reproducible
            delta = np.empty((len(years_future), len(gas_cols)), dtype=float)

            # Smooth the deltas to avoid abrupt changes
            for j, gas in enumerate(gas_cols):
                lo, hi = self.specific_gas_bounds.get(gas, self.default_gas_bounds)
                delta[:, j] = rng.uniform(lo, hi, size=len(years_future))
                # Smooth the deltas (EMA in log-space)
                u = np.log1p(delta[:, j])  # log(1+d)
                for t in range(1, len(u)):
                    u[t] = self.smoothing_alpha * u[t - 1] + (1 - self.smoothing_alpha) * u[t]
                delta[:, j] = np.expm1(u)  # back to % growth

            scale_df = pd.DataFrame(delta, index=years_future, columns=gas_cols)

            em_policy = em_data.copy()
            for t in years_future:
                gfactor = growth_base.loc[t] * scale_df.loc[t]
                em_policy.loc[t] = em_policy.loc[t - 1] * gfactor

            new_scen = {
                "gaspam_data": gaspam_data,
                "nyend": self.em_data_end,
                "nystart": self.em_data_start,
                "emstart": self.em_data_policy,
                "concentrations_data": conc_data,
                "nat_ch4_data": nat_ch4_data,
                "nat_n2o_data": nat_n2o_data,
                "emissions_data": em_policy,
                "udir": str(self.test_data_dir),
                "idtm": 24,
                "scenname": f"rw_growth_smooth_{k:03d}",
            }
            scenarios.append(new_scen)

        return scenarios

    def run_scenarios(self, scenarios):
        """Dispatch scenarios through the CICERO parallel wrapper."""
        output_variables = ["Surface Air Temperature Change"]
        return run_ciceroscm_parallel(scenarios, self.cfg, output_variables)

    def save_raw_data(self, scenarios, results):
        """Persist generated scenarios and results for later inspection."""
        data_output_dir_run_raw = self.data_output_dir_run / "raw"
        data_output_dir_run_raw.mkdir(parents=True, exist_ok=True)
        with open(data_output_dir_run_raw / "scenarios.pkl", "wb") as f:
            pickle.dump(scenarios, f)
        with open(data_output_dir_run_raw / "results.pkl", "wb") as f:
            pickle.dump(results, f)
        self._save_run_configs()

    def _save_run_configs(self):
        """Copy the configs used for this run alongside the generated data."""
        configs_dir = self.data_output_dir_run / "configs"
        configs_dir.mkdir(parents=True, exist_ok=True)
        with open(configs_dir / "cicero_scm.yaml", "w") as f:
            yaml.safe_dump(self.cicero_config, f, sort_keys=False)
        with open(configs_dir / "data_generation.yaml", "w") as f:
            yaml.safe_dump({"data_generation_params": self.data_generation_config}, f, sort_keys=False)


def main():
    cicero_config, data_generation_config = load_configs()
    pipeline = DataGenerationPipeline(cicero_config, data_generation_config)

    gaspam_data, conc_data, em_data, nat_ch4_data, nat_n2o_data = pipeline.load_core_data()
    scenarios = pipeline.generate_scenarios(
        em_data,
        nat_ch4_data,
        nat_n2o_data,
        gaspam_data,
        conc_data,
    )
    results = pipeline.run_scenarios(scenarios)
    pipeline.save_raw_data(scenarios, results)


if __name__ == "__main__":
    main()
