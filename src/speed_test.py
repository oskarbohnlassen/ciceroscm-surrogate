import pickle
import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from marl_env_step import CICERONetEngine, CICEROSCMEngine
from src.utils.config_utils import load_yaml_config
from src.utils.data_utils import determine_variable_gases
from src.utils.model_utils import instantiate_model, parse_model_config, load_state_dict

class SpeedTestPipeline:
    """Benchmark CICERO surrogate vs. CICERO SCM using configured data and model."""

    def __init__(self):
        config = load_yaml_config("speed_test.yaml", "speed_test")
        self.config = config
        self.data_run = Path(config["data_run"]).expanduser()
        if not self.data_run.exists():
            raise FileNotFoundError(f"Data run directory not found: {self.data_run}")

        training_run_path = Path(config["training_run"]).expanduser()
        if training_run_path.is_absolute():
            self.training_run = training_run_path
        else:
            self.training_run = (self.data_run / training_run_path).resolve()
        if not self.training_run.exists():
            raise FileNotFoundError(f"Training run directory not found: {self.training_run}")

        self.cuda_available = torch.cuda.is_available()

        self.use_half = bool(config["use_half"]) and self.cuda_available
        self.autocast = bool(config["autocast"]) and self.cuda_available
        self.max_samples = int(config["max_samples"])
        self.warmup_steps = int(config["warmup_steps"])

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir_name = config["output_dir_name"]
        self.output_dir = self.training_run / output_dir_name / timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.processed_dir = self.data_run / "processed"
        self.raw_dir = self.data_run / "raw"
        self.configs_dir = self.data_run / "configs"
        self.speed_test_config_path = self.output_dir / "speed_test_config.yaml"
        with self.speed_test_config_path.open("w") as f:
            yaml.safe_dump({"speed_test": self.config}, f, sort_keys=False)

        self.cicero_config = load_yaml_config(self.configs_dir / "cicero_scm.yaml")
        self.data_generation_config = load_yaml_config(
            self.configs_dir / "data_generation.yaml", "data_generation_params"
        )
        self.data_processing_config = load_yaml_config(
            self.configs_dir / "data_processing.yaml", "data_processing_params"
        )

        cfg_entry = self.cicero_config["cfg"][0]
        self.pamset_udm = cfg_entry["pamset_udm"]
        self.pamset_emiconc = cfg_entry["pamset_emiconc"]
        self.em_data_start = self.cicero_config["em_data_start"]
        self.em_data_policy = self.cicero_config["em_data_policy"]
        self.window_size = self.data_processing_config["window_size"]

        self.training_config = load_yaml_config(self.training_run / "train_config.yaml")
        X_train = self._load_array("X_train.npy", mmap=True)
        self.mu, self.std = self._compute_normalization_stats(X_train)
        del X_train
        self.scenarios = self._load_scenarios()
        if not self.scenarios:
            raise ValueError("No scenarios available in raw data")
        gas_names = self.scenarios[0]["emissions_data"].columns
        self.variable_gases = determine_variable_gases(self.data_generation_config, gas_names)

    def _load_model(self, device: str):
        model_cfg = self.training_config["model"]
        model_type, hidden_size, num_layers, kernel_size = parse_model_config(model_cfg)
        n_gas = self._infer_gas_count()
        if len(self.variable_gases) != n_gas:
            raise ValueError(
                "Mismatch between processed gas count and selected gases list: "
                f"{n_gas} vs {len(self.variable_gases)}"
            )

        model = instantiate_model(
            model_type,
            n_gas,
            hidden_size,
            num_layers,
            kernel_size=kernel_size,
            device=device,
        )
        state_path = self.training_run / "model.pth"
        if not state_path.exists():
            raise FileNotFoundError(f"Model state dict not found: {state_path}")
        state = torch.load(state_path, map_location="cpu")
        load_state_dict(model, state)
        return model

    def _infer_gas_count(self):
        return self._load_array("X_train.npy", mmap=True).shape[2]

    def _load_array(self, name: str, mmap: bool = False):
        path = self.processed_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Processed array missing: {path}")
        if mmap:
            return np.load(path, mmap_mode="r")
        return np.load(path)

    @staticmethod
    def _compute_normalization_stats(X):
        gases = X.shape[2]
        mu = X.reshape(-1, gases).mean(axis=0)
        std = X.reshape(-1, gases).std(axis=0) + 1e-6
        return mu, std

    def _load_scenarios(self):
        scenario_path = self.raw_dir / "scenarios.pkl"
        if not scenario_path.exists():
            raise FileNotFoundError(f"Scenario pickle not found: {scenario_path}")
        with scenario_path.open("rb") as f:
            return pickle.load(f)

    @staticmethod
    def _sync_device(device: str):
        if device.startswith("cuda"):
            torch.cuda.synchronize()

    def _scenario_iterator(self):
        for scenario in self.scenarios:
            em_df = scenario["emissions_data"].copy()
            if not isinstance(em_df, pd.DataFrame):
                continue
            hist_full = em_df.loc[: self.em_data_policy]
            if len(hist_full) < self.window_size:
                continue
            future_full = em_df.loc[self.em_data_policy + 1 :]
            if future_full.empty:
                continue
            hist_subset = hist_full.loc[:, self.variable_gases]
            future_subset = future_full.loc[:, self.variable_gases]
            yield scenario, hist_full, future_full, hist_subset, future_subset

    def _measure_surrogate_latencies(self, device: str, use_half: bool, autocast: bool):
        model = self._load_model(device)
        latencies = []
        recorded = 0
        warmup_remaining = self.warmup_steps

        progress = tqdm(total=self.max_samples, desc=f"CICERONet ({device})", leave=False)

        for _, _, _, hist_df, future_df in self._scenario_iterator():
            engine = CICERONetEngine(
                historical_emissions=hist_df.to_numpy(dtype=np.float32),
                model=model,
                device=device,
                mu=self.mu,
                std=self.std,
                autocast=autocast,
                use_half=use_half,
                window_size=self.window_size,
            )
            for _, row in future_df.iterrows():
                action = row.to_numpy(dtype=np.float32)
                if warmup_remaining > 0:
                    engine.step(action)
                    warmup_remaining -= 1
                    continue
                if recorded >= self.max_samples:
                    break
                self._sync_device(device)
                t0 = time.perf_counter()
                engine.step(action)
                self._sync_device(device)
                latencies.append(time.perf_counter() - t0)
                recorded += 1
                progress.update(1)
            if recorded >= self.max_samples:
                break

        progress.close()
        return np.asarray(latencies, dtype=float)

    def _measure_cicero_latencies(self):
        latencies = []
        recorded = 0
        warmup_remaining = self.warmup_steps

        progress = tqdm(total=self.max_samples, desc="CICERO-SCM", leave=False)

        for scenario, hist_full, future_full, _, _ in self._scenario_iterator():
            engine = CICEROSCMEngine(
                historical_emissions=hist_full.copy(),
                gaspam_data=scenario["gaspam_data"],
                conc_data=scenario["concentrations_data"],
                nat_ch4_data=scenario["nat_ch4_data"],
                nat_n2o_data=scenario["nat_n2o_data"],
                pamset_udm=self.pamset_udm,
                pamset_emiconc=self.pamset_emiconc,
                em_data_start=self.em_data_start,
                em_data_policy=self.em_data_policy,
                udir=scenario["udir"],
                idtm=scenario["idtm"],
                scenname=scenario["scenname"],
            )
            for _, row in future_full.iterrows():
                action = row.to_numpy(dtype=np.float32)
                if warmup_remaining > 0:
                    engine.step(action)
                    warmup_remaining -= 1
                    continue
                if recorded >= self.max_samples:
                    break
                t0 = time.perf_counter()
                engine.step(action)
                latencies.append(time.perf_counter() - t0)
                recorded += 1
                progress.update(1)
            if recorded >= self.max_samples:
                break

        progress.close()
        return np.asarray(latencies, dtype=float)

    def run(self):
        surrogate_latencies = {
            "cpu": self._measure_surrogate_latencies(
                device="cpu", use_half=False, autocast=False
            )
        }
        if self.cuda_available:
            surrogate_latencies["cuda"] = self._measure_surrogate_latencies(
                device="cuda", use_half=self.use_half, autocast=self.autocast
            )

        if self.config['include_cicero_latencies'] == True:
            cicero_latencies = self._measure_cicero_latencies()
        else:
            cicero_latencies = np.array([], dtype=float)

        surrogate_npz_payload = {
            f"{device}_latencies": arr for device, arr in surrogate_latencies.items()
        }
        np.savez(self.output_dir / "surrogate_latencies.npz", **surrogate_npz_payload)
        np.savez(self.output_dir / "cicero_latencies.npz", cicero_latencies=cicero_latencies)

        summary_path = self.output_dir / "summary.txt"
        cic_mean = float(np.mean(cicero_latencies)) if cicero_latencies.size else float("nan")
        cic_median = float(np.median(cicero_latencies)) if cicero_latencies.size else float("nan")
        with summary_path.open("w") as f:
            f.write("CICERONet latency (s)\n")
            for device, arr in surrogate_latencies.items():
                mean = float(np.mean(arr)) if arr.size else float("nan")
                median = float(np.median(arr)) if arr.size else float("nan")
                f.write(f"  [{device}] mean: {mean:.6f}\n")
                f.write(f"  [{device}] median: {median:.6f}\n")
                f.write(f"  [{device}] samples: {len(arr)}\n")
            f.write("\nCICERO-SCM latency (s)\n")
            f.write(f"  mean: {cic_mean:.6f}\n")
            f.write(f"  median: {cic_median:.6f}\n")
            f.write(f"  samples: {len(cicero_latencies)}\n")

        print("Speed test complete. Summary written to", summary_path)
        for device, arr in surrogate_latencies.items():
            mean = float(np.mean(arr)) if arr.size else float("nan")
            print(
                f"CICERONet mean latency [{device}]: {mean:.6f}s ({len(arr)} samples)"
            )
        print(f"CICERO-SCM mean latency: {cic_mean:.6f}s ({len(cicero_latencies)} samples)")


def main():
    pipeline = SpeedTestPipeline()
    pipeline.run()


if __name__ == "__main__":
    main()
