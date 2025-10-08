from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import torch

def validation_metrics(loader, model, device, return_predictions: bool = False):
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            y_true.append(yb.cpu().numpy())
            y_pred.append(model(xb).view(-1).cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    if return_predictions:
        return {"RMSE": rmse, "R2": r2}, y_true, y_pred
    return {"RMSE": rmse, "R2": r2}