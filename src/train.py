import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm.auto import tqdm

# Add project root to the Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


from src.utils.config_utils import load_yaml_config
from src.utils.data_utils import load_processed_data, prepare_dataloaders
from src.utils.model_utils import instantiate_model, parse_model_config
from src.utils.train_utils import validation_metrics

class TrainingPipeline:

    def __init__(self):
        config = load_yaml_config("train.yaml")
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

        hidden_cfg = self.model_cfg["hidden_dims"]
        if isinstance(hidden_cfg, (list, tuple)):
            descriptor_hidden = "-".join(str(h) for h in hidden_cfg)
        else:
            descriptor_hidden = str(hidden_cfg)

        (
            self.model_type,
            self.hidden_size,
            self.num_layers,
            self.kernel_size,
        ) = parse_model_config(self.model_cfg)
        self.model_descriptor = f"{self.model_type}_{descriptor_hidden}_{self.num_layers}"

        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.training_run_dir = (
            self.run_path / "training_runs" / f"{self.timestamp}_{self.model_descriptor}"
        )
        self.model_save_path = self.training_run_dir / "model.pth"
        self.checkpoint_dir = self.training_run_dir / "checkpoints"

        self.epochs = self.model_cfg["epochs"]
        self.batch_size = self.model_cfg["batch_size"]
        self.num_workers = self.model_cfg["num_workers"]
        self.early_stopping = self.model_cfg["early_stopping"]

        self.learning_rate = self.model_cfg["lr"]
        self.weight_decay = self.model_cfg["weight_decay"]

    def build_model(self, n_gas: int):
        """Instantiate the configured model architecture."""
        return instantiate_model(
            self.model_type,
            n_gas,
            self.hidden_size,
            self.num_layers,
            kernel_size=self.kernel_size,
            device=self.device,
            freeze=False,
        )

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
        X_train, y_train, X_val, y_val, X_test, y_test, mu_train, std_train = load_processed_data(self.processed_path)
        
        (train_loader, val_loader, test_loader, n_gas) = prepare_dataloaders(
                                                        X_train,
                                                        y_train,
                                                        X_val,
                                                        y_val,
                                                        X_test,
                                                        y_test,
                                                        mu_train,
                                                        std_train,
                                                        device=self.device,
                                                        batch_size=self.batch_size,
                                                        num_workers=self.num_workers,)

        model = self.build_model(n_gas)
        optimizer = self.build_optimizer(model)
        scheduler = self.build_scheduler(optimizer)
        criterion = nn.MSELoss()
        scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        progress = tqdm(range(1, self.epochs + 1), desc="epochs")

        best_metric = float("inf")
        best_state = None
        best_epoch = 0
        epochs_since_improvement = 0
        last_val_metrics = None

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
            last_val_metrics = val_metrics
            scheduler.step(val_metrics["RMSE"])

            current_lr = optimizer.param_groups[0]["lr"]
            progress.set_postfix(
                val_RMSE=f"{val_metrics['RMSE']:.4f}°C",
                val_R2=f"{val_metrics['R2']:.3f}",
                lr=f"{current_lr:.1e}",
            )

            improved = val_metrics["RMSE"] < best_metric - 1e-8
            if improved:
                best_metric = val_metrics["RMSE"]
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

            if improved and epoch % 5 == 0:
                self.save_checkpoint(best_state, epoch, best_metric)

            if self.early_stopping > 0 and epochs_since_improvement >= self.early_stopping:
                tqdm.write(
                    f"Early stopping triggered at epoch {epoch}. Best epoch: {best_epoch}"
                )
                break

        if best_state is None:
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_epoch = epoch
            best_metric = (
                last_val_metrics["RMSE"] if last_val_metrics is not None else float("inf")
            )

        model.load_state_dict(best_state)
        model.to(self.device)

        test_metrics = validation_metrics(test_loader, model, self.device)
        tqdm.write(
            f"Best val RMSE: {best_metric:.4f}°C at epoch {best_epoch}. "
            f"Test RMSE: {test_metrics['RMSE']:.4f}°C | Test R²: {test_metrics['R2']:.3f}"
        )

        self.save_run_artifacts(best_state, best_epoch, best_metric, test_metrics)
        return model

    def save_checkpoint(self, state_dict, epoch, metric):
        """Persist an intermediate checkpoint when validation improves."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self.checkpoint_dir / f"model_epoch_{epoch:04d}.pth"
        torch.save(state_dict, checkpoint_path)
        meta = {"epoch": int(epoch), "val_RMSE": float(metric)}
        with open(checkpoint_path.with_suffix(".yaml"), "w") as f:
            yaml.safe_dump(meta, f, sort_keys=False)

    def save_run_artifacts(self, state_dict, best_epoch, best_metric, test_metrics):
        """Persist the best model weights, config, and summary metrics."""
        self.training_run_dir.mkdir(parents=True, exist_ok=True)
        torch.save(state_dict, self.model_save_path)
        with open(self.training_run_dir / "train_config.yaml", "w") as f:
            yaml.safe_dump(self.config, f, sort_keys=False)
        metrics_payload = {
            "best_epoch": int(best_epoch),
            "best_val_RMSE": float(best_metric),
            "test_RMSE": float(test_metrics["RMSE"]),
            "test_R2": float(test_metrics["R2"]),
        }
        with open(self.training_run_dir / "metrics.yaml", "w") as f:
            yaml.safe_dump(metrics_payload, f, sort_keys=False)


def main():
    pipeline = TrainingPipeline()
    pipeline.train()


if __name__ == "__main__":
    main()

