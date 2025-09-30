import sys
from pathlib import Path

import torch
import numpy as np
import yaml
from sklearn.metrics import mean_squared_error, r2_score


def load_yaml_config(path, nested_key=None):
    """Load YAML configuration from *path* and optionally return a nested section."""
    path = Path(path)
    with path.open("r") as stream:
        data = yaml.safe_load(stream)
    if nested_key is None:
        return data
    if nested_key not in data:
        raise KeyError(f"Expected key '{nested_key}' in {path}")
    return data[nested_key]



def validation_metrics(loader, model, device):
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            y_true.append(yb.cpu().numpy())
            y_pred.append(model(xb).view(-1).cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    return {"RMSE": rmse, "R2": r2}
