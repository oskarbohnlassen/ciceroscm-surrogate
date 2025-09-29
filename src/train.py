import sys
import torch
import torch.nn as nn
import torch.utils.data as td
from tqdm import tqdm
import numpy as np
import os

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.utils import validation_metrics
from src.model import GRUSurrogate, LSTMSurrogate, TCNForecaster

EPOCHS = 300
device  = "cuda"
run_path = "/home/obola/repositories/cicero-scm-surrogate/data/20250926_110035"
data_path = os.path.join(run_path, "processed")

def load_processed_data(data_path):
    X_train = np.load(os.path.join(data_path, "X_train.npy"))
    y_train = np.load(os.path.join(data_path, "y_train.npy"))
    X_val = np.load(os.path.join(data_path, "X_val.npy"))
    y_val = np.load(os.path.join(data_path, "y_val.npy"))
    X_test = np.load(os.path.join(data_path, "X_test.npy"))
    y_test = np.load(os.path.join(data_path, "y_test.npy"))
    

    return X_train, y_train, X_val, y_val, X_test, y_test

def format_data(X_train, y_train, X_val, y_val, X_test, y_test):
    G = X_train.shape[2]
    mu  = X_train.reshape(-1, G).mean(axis=0)     # per-gas mean
    std = X_train.reshape(-1, G).std(axis=0) + 1e-6
    
    def norm(a): 
        return (a - mu) / std

    X_train_norm = norm(X_train)
    X_val_norm = norm(X_val)
    X_test_norm = norm(X_test)

    train_ds = td.TensorDataset(torch.tensor(X_train_norm), torch.tensor(y_train))
    val_ds   = td.TensorDataset(torch.tensor(X_val_norm),   torch.tensor(y_val))
    test_ds   = td.TensorDataset(torch.tensor(X_test_norm),   torch.tensor(y_test))

    train_loader = td.DataLoader(train_ds, batch_size = 512, shuffle=True,
                             num_workers=2, pin_memory=True, persistent_workers=True)
    
    val_loader   = td.DataLoader(val_ds,   batch_size = 512, 
                             num_workers=2, pin_memory=True, persistent_workers=True)
    
    test_loader   = td.DataLoader(test_ds,   batch_size = 512, 
                             num_workers=2, pin_memory=True, persistent_workers=True)

    return train_loader, val_loader, test_loader, G

def train_model(X_train, y_train, X_val, y_val, X_test, y_test):
    train_loader, val_loader, test_loader, G = format_data(X_train, y_train, X_val, y_val, X_test, y_test)

    #model   = GRUSurrogate(n_gas=G).to(device)
    model = LSTMSurrogate(n_gas=G, hidden=128, num_layers=1).to(device)
    #model = TCNForecaster(n_gas=G, n_blocks=8, hidden=128, k=3).to(device)
    opt         = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler   = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    opt, mode="min", factor=0.5, patience=5,
                    min_lr=1e-6)

    criterion = nn.MSELoss()

    pbar = tqdm(range(1, EPOCHS + 1), desc="epochs")

    scaler = torch.amp.GradScaler("cuda")

    for epoch in pbar:
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad()
            with torch.amp.autocast("cuda"):
                pred = model(xb)
                loss = criterion(pred, yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        model.eval()
        val = validation_metrics(val_loader, model, device)

        # scheduler step on validation RMSE
        scheduler.step(val["RMSE"])
        current_lr = opt.param_groups[0]["lr"]

        # progress-bar postfix
        pbar.set_postfix({
            "val_RMSE": f"{val['RMSE']:.4f}°C",
            "val_R²":   f"{val['R2']:.3f}",
            "lr":       f"{current_lr:.1e}"
        })

    return model

if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = load_processed_data(data_path=data_path)
    model = train_model(X_train, y_train, X_val, y_val, X_test, y_test)
    torch.save(model.state_dict(), os.path.join(run_path, "model_lstm_v2_128_1layer.pth"))