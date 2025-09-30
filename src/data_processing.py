import pickle
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import yaml

from utils import load_yaml_config


CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def load_config():
    """Load data-processing configuration values from YAML."""
    return load_yaml_config(CONFIG_DIR / "data_processing.yaml", "data_processing_params")


class DataProcessingPipeline:
    """Prepare machine-learning datasets from CICERO simulation outputs."""

    def __init__(self, config: dict):
        self.config = config
        self.run_path = Path(config["run_id"]).expanduser()
        self.raw_dir = self.run_path / "raw"
        self.processed_dir = self.run_path / "processed"

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

        self.configs_dir = self.run_path / "configs"
        self.configs_dir.mkdir(parents=True, exist_ok=True)
        self.cicero_config = load_yaml_config(self.configs_dir / "cicero_scm.yaml")
        self.data_generation_config = load_yaml_config(
            self.configs_dir / "data_generation.yaml", "data_generation_params"
        )
        self.em_data_policy = self.cicero_config["em_data_policy"]
        self._save_run_config()

    def load_simulation_data(self):
        """Load raw simulation results and scenarios from the run directory."""
        with open(self.raw_dir / "results.pkl", "rb") as f:
            results = pickle.load(f)

        with open(self.raw_dir / "scenarios.pkl", "rb") as f:
            scenarios = pickle.load(f)

        return results, scenarios

    def format_results(self, results, scenarios):
        """Create train/val/test datasets according to configured splits."""
        gases = scenarios[0]["emissions_data"].columns.tolist()

        temp_tbl = (
            results[results["variable"] == "Surface Air Temperature Change"]
            .set_index("scenario")
            .filter(regex=r"^\d")
            .astype(float)
        )
        temp_tbl.columns = temp_tbl.columns.astype(int)

        scen_trainval, scen_test = train_test_split(
            scenarios,
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
                em_df = scen["emissions_data"]
                years = em_df.index

                T_air = temp_tbl.loc[name, years].to_numpy()

                for t_idx in range(self.window_size, len(years) - 1):
                    t_target = years[t_idx + 1]
                    if t_target < self.em_data_policy:
                        continue

                    hist = em_df.loc[t_target - self.window_size : t_target - 1, gases].to_numpy()
                    next_em = em_df.loc[t_target, gases].to_numpy()
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

        return (
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            scen_train,
            scen_val,
            scen_test,
        )

    def save_processed_data(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        train_scenarios,
        val_scenarios,
        test_scenarios,
    ):
        """Persist processed datasets and scenario splits to disk."""
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        np.save(self.processed_dir / "X_train.npy", X_train)
        np.save(self.processed_dir / "y_train.npy", y_train)
        np.save(self.processed_dir / "X_val.npy", X_val)
        np.save(self.processed_dir / "y_val.npy", y_val)
        np.save(self.processed_dir / "X_test.npy", X_test)
        np.save(self.processed_dir / "y_test.npy", y_test)

        with open(self.processed_dir / "train_scenarios.pkl", "wb") as f:
            pickle.dump(train_scenarios, f)
        with open(self.processed_dir / "val_scenarios.pkl", "wb") as f:
            pickle.dump(val_scenarios, f)
        with open(self.processed_dir / "test_scenarios.pkl", "wb") as f:
            pickle.dump(test_scenarios, f)

    def _save_run_config(self):
        """Persist the data-processing configuration alongside the dataset."""
        with open(self.configs_dir / "data_processing.yaml", "w") as f:
            yaml.safe_dump({"data_processing_params": self.config}, f, sort_keys=False)


def main():
    config = load_config()
    pipeline = DataProcessingPipeline(config)

    results, scenarios = pipeline.load_simulation_data()
    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        train_scenarios,
        val_scenarios,
        test_scenarios,
    ) = pipeline.format_results(results, scenarios)
    pipeline.save_processed_data(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        train_scenarios,
        val_scenarios,
        test_scenarios,
    )


if __name__ == "__main__":
    main()
