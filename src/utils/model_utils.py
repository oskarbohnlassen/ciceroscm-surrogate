from typing import Dict, Tuple

import numpy as np
import torch

from src.model import GRUSurrogate, LSTMSurrogate, TCNForecaster


def parse_model_config(model_cfg: Dict) -> Tuple[str, int, int, int]:
    model_type = model_cfg["model_type"].lower()
    hidden_cfg = model_cfg["hidden_dims"]
    if isinstance(hidden_cfg, (list, tuple)):
        hidden = hidden_cfg[0]
    else:
        hidden = hidden_cfg
    num_layers = model_cfg["num_layers"]
    kernel_size = model_cfg.get("kernel_size", 3)
    return model_type, hidden, num_layers, kernel_size


def instantiate_model(
    model_type: str,
    n_gas: int,
    hidden: int,
    num_layers: int,
    *,
    kernel_size: int = 3,
    device: str = "cpu",
    freeze: bool = False,
) -> torch.nn.Module:
    device = torch.device(device)
    model_type = model_type.lower()
    if model_type == "lstm":
        model = LSTMSurrogate(n_gas=n_gas, hidden=hidden, num_layers=num_layers)
    elif model_type == "gru":
        model = GRUSurrogate(n_gas=n_gas, hidden=hidden)
    elif model_type == "tcn":
        model = TCNForecaster(
            n_gas=n_gas,
            n_blocks=num_layers,
            hidden=hidden,
            k=kernel_size,
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    model = model.to(device)
    if freeze:
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
    else:
        model.train()
    return model


def load_state_dict(
    model: torch.nn.Module,
    state_dict: Dict,
    *,
    strict: bool = True,
    freeze: bool = True,
) -> torch.nn.Module:
    model.load_state_dict(state_dict, strict=strict)
    if freeze:
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
    return model


def numpy_state_to_torch(state_dict_np: Dict[str, np.ndarray], device: str) -> Dict[str, torch.Tensor]:
    device = torch.device(device)
    return {k: torch.from_numpy(v).to(device) for k, v in state_dict_np.items()}
