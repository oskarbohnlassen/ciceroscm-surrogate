import pickle
from pathlib import Path
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_utils import load_yaml_config
from src.utils.data_utils import determine_variable_gases, load_simulation_data

class DataProcessingPipeline:
    """Prepare machine-learning datasets from CICERO simulation outputs."""

    def __init__(self):
        config = load_yaml_config("data_processing.yaml", "data_processing_params")

        self.config = config
        self.data_dir = Path(config["data_dir"]).expanduser()
        self.data_raw_dir = self.data_dir / "raw"
        self.data_processed_dir = self.data_dir / "processed"

        self.window_size = config["window_size"]
        splits = config["split"]
        self.train_split = splits["train"]
        self.val_split = splits["val"]
        self.test_split = splits["test"]
        total_split = self.train_split + self.val_split + self.test_split
        if not np.isclose(total_split, 1.0):
            raise ValueError("train/val/test splits must sum to 1.0")
        if any(s <= 0 for s in (self.train_split, self.val_split, self.test_split)):
            raise ValueError("train/val/test splits must be positive")

        self.random_state = config.get("random_state", 0)

        self.configs_dir = self.data_dir / "configs"
        self.configs_dir.mkdir(parents=True, exist_ok=True)
        self.cicero_config = load_yaml_config(self.configs_dir / "cicero_scm.yaml")
        self.data_generation_config = load_yaml_config(self.configs_dir / "data_generation.yaml", "data_generation_params")
        self.em_data_policy = self.cicero_config["em_data_policy"]
        self._save_run_config()
        self.variable_gases = None

    def load_simulation_data(self):
        """Load raw simulation results and scenarios from the run directory."""
        results, scenarios = load_simulation_data(self.data_raw_dir)
        self.results = results
        self.scenarios = scenarios

    def format_results(self):
 
        gas_names = self.scenarios[0]["emissions_data"].columns
        self.variable_gases = determine_variable_gases(self.data_generation_config, gas_names)

        gases = self.variable_gases

        temp_tbl = (
            self.results[self.results["variable"] == "Surface Air Temperature Change"]
            .set_index("scenario")
            .filter(regex=r"^\d")
            .astype(float)
        )
        temp_tbl.columns = temp_tbl.columns.astype(int)

        scen_trainval, scen_test = train_test_split(
            self.scenarios,
            test_size=self.test_split,
            random_state=self.random_state,
            shuffle=True,
        )

        val_prop_within_trainval = self.val_split / (self.train_split + self.val_split)
        scen_train, scen_val = train_test_split(
            scen_trainval,
            test_size=val_prop_within_trainval,
            random_state=self.random_state,
            shuffle=True,
        )

        def generate_machine_learning_data(scenario_list, split_name):
            X_list, y_list = [], []
            scenario_iter = tqdm(
                scenario_list,
                desc=f"{split_name.capitalize()} scenarios",
                unit="scenario",
                leave=False,
            )

            for scen in scenario_iter:
                name = scen["scenname"]
                em_df = scen["emissions_data"].loc[:, gases]
                years = em_df.index

                T_air = temp_tbl.loc[name, years].to_numpy()

                for t_idx in range(self.window_size, len(years) - 1):
                    t_target = years[t_idx + 1]
                    if t_target < self.em_data_policy:
                        continue

                    hist = em_df.loc[t_target - self.window_size : t_target - 1, :].to_numpy()
                    next_em = em_df.loc[t_target, :].to_numpy()
                    X_sample = np.vstack([hist, next_em[None, :]]).astype("float32")
                    y_sample = np.float32(T_air[t_idx + 1])

                    X_list.append(X_sample)
                    y_list.append(y_sample)

            X = np.stack(X_list)
            y = np.stack(y_list)
            return X, y

        X_train, y_train = generate_machine_learning_data(scen_train, "train")
        X_val, y_val = generate_machine_learning_data(scen_val, "validation")
        X_test, y_test = generate_machine_learning_data(scen_test, "test")

        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.X_test, self.y_test = X_test, y_test
        self.train_scenarios = scen_train
        self.val_scenarios = scen_val
        self.test_scenarios = scen_test

        # Compute normalization stats from training data
        self.compute_mu_std(X_train)

    def compute_mu_std(self, X_data):
        """Compute mean and standard deviation for normalization."""
 
        gases = X_data.shape[2]
        mu = X_data.reshape(-1, gases).mean(axis=0)
        std = X_data.reshape(-1, gases).std(axis=0) + 1e-6

        self.mu = mu
        self.std = std

    def save_processed_data(self):
        """Persist processed datasets and scenario splits to disk."""
        self.data_processed_dir.mkdir(parents=True, exist_ok=True)

        np.save(self.data_processed_dir / "X_train.npy", self.X_train)
        np.save(self.data_processed_dir / "y_train.npy", self.y_train)
        np.save(self.data_processed_dir / "X_val.npy", self.X_val)
        np.save(self.data_processed_dir / "y_val.npy", self.y_val)
        np.save(self.data_processed_dir / "X_test.npy", self.X_test)
        np.save(self.data_processed_dir / "y_test.npy", self.y_test)
        np.save(self.data_processed_dir / "mu_train.npy", self.mu)
        np.save(self.data_processed_dir / "std_train.npy", self.std)

        with open(self.data_processed_dir / "train_scenarios.pkl", "wb") as f:
            pickle.dump(self.train_scenarios, f)
        with open(self.data_processed_dir / "val_scenarios.pkl", "wb") as f:
            pickle.dump(self.val_scenarios, f)
        with open(self.data_processed_dir / "test_scenarios.pkl", "wb") as f:
            pickle.dump(self.test_scenarios, f)

    def _save_run_config(self):
        """Persist the data-processing configuration alongside the dataset."""
        with open(self.configs_dir / "data_processing.yaml", "w") as f:
            yaml.safe_dump({"data_processing_params": self.config}, f, sort_keys=False)

    def run(self):
        self.load_simulation_data()
        self.format_results()
        self.save_processed_data()

def main():
    pipeline = DataProcessingPipeline()
    pipeline.run()

if __name__ == "__main__":
    main()
