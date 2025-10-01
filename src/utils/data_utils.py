from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data


def load_processed_data(data_dir):
    data_dir = Path(data_dir)
    names = [
        "X_train.npy",
        "y_train.npy",
        "X_val.npy",
        "y_val.npy",
        "X_test.npy",
        "y_test.npy",
        "mu_train.npy",
        "std_train.npy",
    ]
    arrays = []
    for name in names:
        path = data_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Expected processed file missing: {path}")
        arrays.append(np.load(path))
    return tuple(arrays)


def load_simulation_data(data_raw_dir):
    data_raw_dir = Path(data_raw_dir)
    scenario_path = data_raw_dir / "scenarios.pkl"
    results_path = data_raw_dir / "results.pkl"

    if not scenario_path.exists() or not results_path.exists():
        raise FileNotFoundError(f"Scenario or results pickle not found: {data_raw_dir}")

    else:
        import pickle
        with open(scenario_path, "rb") as f:
            scenarios = pickle.load(f)
        with open(results_path, "rb") as f:
            results = pickle.load(f)
    return results, scenarios


def prepare_dataloaders(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    mu,
    std,
    *,
    device,
    batch_size,
    num_workers,
):
    device = torch.device(device)
    gases = X_train.shape[2]

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

    pin_memory = device.type == "cuda"
    persistent_workers = num_workers > 0

    train_loader = data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    test_loader = data.DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    return train_loader, val_loader, test_loader, gases


def determine_variable_gases(data_generation_config, gas_names):
    specific_bounds = data_generation_config.get("specific_gas_bounds", {})
    remaining_bounds = tuple(data_generation_config.get("remaining_gas_bounds", (1, 1)))

    selected = []
    for gas in gas_names:
        bounds = specific_bounds.get(gas, remaining_bounds)
        if tuple(bounds) != (1, 1):
            selected.append(gas)

    return selected or list(gas_names)


def build_emission_data(em_data, baseline_shares, policy_start_year):
    shifted = em_data.shift(1)
    growth_base = em_data.divide(shifted.replace(0, np.nan))
    baseline_growth = growth_base.replace([np.inf, -np.inf], np.nan).fillna(1.0)

    historical_emissions = em_data.loc[:policy_start_year]
    baseline_emissions = em_data.loc[policy_start_year:]
    baseline_emission_growth = baseline_growth.loc[policy_start_year + 1 :]

    if baseline_shares.ndim != 1:
        raise ValueError("baseline_emission_shares must be a 1D array per agent")

    emission_shares = np.tile(baseline_shares[:, None], (1, len(em_data.columns)))

    return {
        "historical_emissions": historical_emissions,
        "baseline_emissions": baseline_emissions.to_numpy(dtype=np.float32),
        "baseline_emission_growth": baseline_emission_growth.to_numpy(dtype=np.float32),
        "emission_shares": emission_shares.astype(np.float32),
        "gas_names": list(em_data.columns),
    }

