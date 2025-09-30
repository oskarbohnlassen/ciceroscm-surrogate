import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import yaml
from tqdm.auto import tqdm

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.model import GRUSurrogate, LSTMSurrogate, TCNForecaster
from src.utils import load_yaml_config, validation_metrics


CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def load_config():
    """Load training configuration from YAML."""
    return load_yaml_config(CONFIG_DIR / "train.yaml")


class TrainingPipeline:
    """Train surrogate models using processed CICERO datasets."""

    def __init__(self, config: dict):
        self.config = config
        self.general_cfg = config["general"]
        self.model_cfg = config["model"]
        self.scheduler_cfg = config["scheduler"]

        self.run_path = Path(self.general_cfg["data_id"]).expanduser()
        self.processed_path = self.run_path / "processed"
        if not self.processed_path.exists():
            raise FileNotFoundError(f"Processed data not found at {self.processed_path}")

        requested_device = self.general_cfg["device"]
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but not available")
        self.device = torch.device(requested_device)
        self.use_amp = self.device.type == "cuda"

        self.model_type = self.model_cfg["model_type"].lower()
        hidden_cfg = self.model_cfg["hidden_dims"]
        if isinstance(hidden_cfg, (list, tuple)):
            hidden_str = "-".join(str(h) for h in hidden_cfg)
            hidden_for_model = hidden_cfg[0]
        else:
            hidden_str = str(hidden_cfg)
            hidden_for_model = hidden_cfg
        self.hidden_size = hidden_for_model
        self.num_layers = self.model_cfg["num_layers"]
        self.model_descriptor = f"{self.model_type}_{hidden_str}_{self.num_layers}"

        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.training_run_dir = (
            self.run_path / "training_runs" / f"{self.timestamp}_{self.model_descriptor}"
        )
        self.model_save_path = self.training_run_dir / "model.pth"

        self.epochs = self.model_cfg["epochs"]
        self.batch_size = self.model_cfg["batch_size"]
        self.num_workers = self.model_cfg["num_workers"]

        self.learning_rate = self.model_cfg["lr"]
        self.weight_decay = self.model_cfg["weight_decay"]

    def load_processed_data(self):
        """Load numpy arrays produced by the data-processing pipeline."""
        def _load(name):
            path = self.processed_path / name
            if not path.exists():
                raise FileNotFoundError(f"Expected processed file missing: {path}")
            return np.load(path)

        X_train = _load("X_train.npy")
        y_train = _load("y_train.npy")
        X_val = _load("X_val.npy")
        y_val = _load("y_val.npy")
        X_test = _load("X_test.npy")
        y_test = _load("y_test.npy")
        return X_train, y_train, X_val, y_val, X_test, y_test

    def prepare_dataloaders(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """Normalize emissions, convert to tensors, and build DataLoaders."""
        gases = X_train.shape[2]
        mu = X_train.reshape(-1, gases).mean(axis=0)
        std = X_train.reshape(-1, gases).std(axis=0) + 1e-6

        def _normalize(arr):
            return (arr - mu) / std

        X_train_norm = _normalize(X_train)
        X_val_norm = _normalize(X_val)
        X_test_norm = _normalize(X_test)

        def _to_dataset(features, targets):
            return data.TensorDataset(
                torch.tensor(features, dtype=torch.float32),
                torch.tensor(targets, dtype=torch.float32),
            )

        train_ds = _to_dataset(X_train_norm, y_train)
        val_ds = _to_dataset(X_val_norm, y_val)
        test_ds = _to_dataset(X_test_norm, y_test)

        pin_memory = self.device.type == "cuda"
        persistent_workers = self.num_workers > 0

        train_loader = data.DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        val_loader = data.DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        test_loader = data.DataLoader(
            test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

        return train_loader, val_loader, test_loader, gases

    def build_model(self, n_gas: int):
        """Instantiate the configured model architecture."""
        if self.model_type == "lstm":
            model = LSTMSurrogate(
                n_gas=n_gas,
                hidden=self.hidden_size,
                num_layers=self.num_layers,
            )
        elif self.model_type == "gru":
            model = GRUSurrogate(n_gas=n_gas, hidden=self.hidden_size)
        elif self.model_type == "tcn":
            kernel_size = self.model_cfg["kernel_size"]
            model = TCNForecaster(
                n_gas=n_gas,
                n_blocks=self.num_layers,
                hidden=self.hidden_size,
                k=kernel_size,
            )
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")
        return model.to(self.device)

    def build_optimizer(self, model: torch.nn.Module):
        return torch.optim.AdamW(
            model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

    def build_scheduler(self, optimizer: torch.optim.Optimizer):
        factor = self.scheduler_cfg["factor"]
        patience = self.scheduler_cfg["patience"]
        min_lr = self.scheduler_cfg["min_lr"]
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=factor,
            patience=patience,
            min_lr=min_lr,
        )

    def train(self):
        X_train, y_train, X_val, y_val, X_test, y_test = self.load_processed_data()
        train_loader, val_loader, test_loader, n_gas = self.prepare_dataloaders(
            X_train, y_train, X_val, y_val, X_test, y_test
        )

        model = self.build_model(n_gas)
        optimizer = self.build_optimizer(model)
        scheduler = self.build_scheduler(optimizer)
        criterion = nn.MSELoss()
        scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        progress = tqdm(range(1, self.epochs + 1), desc="epochs")

        for epoch in progress:
            model.train()
            for xb, yb in train_loader:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                ctx = torch.amp.autocast("cuda") if self.use_amp else nullcontext()
                with ctx:
                    preds = model(xb)
                    loss = criterion(preds, yb)

                if self.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            model.eval()
            val_metrics = validation_metrics(val_loader, model, self.device)
            scheduler.step(val_metrics["RMSE"])

            current_lr = optimizer.param_groups[0]["lr"]
            progress.set_postfix(
                val_RMSE=f"{val_metrics['RMSE']:.4f}°C",
                val_R2=f"{val_metrics['R2']:.3f}",
                lr=f"{current_lr:.1e}",
            )

        test_metrics = validation_metrics(test_loader, model, self.device)
        tqdm.write(
            f"Test RMSE: {test_metrics['RMSE']:.4f}°C | Test R²: {test_metrics['R2']:.3f}"
        )

        self.save_run_artifacts(model)
        return model

    def save_run_artifacts(self, model: torch.nn.Module):
        """Persist the trained model and a copy of the training configuration."""
        self.training_run_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), self.model_save_path)
        with open(self.training_run_dir / "train_config.yaml", "w") as f:
            yaml.safe_dump(self.config, f, sort_keys=False)


def main():
    config = load_config()
    pipeline = TrainingPipeline(config)
    pipeline.train()


if __name__ == "__main__":
    main()
