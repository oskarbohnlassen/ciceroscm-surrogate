import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class GRUSurrogate(nn.Module):
    def __init__(self, n_gas, hidden=64):
        super().__init__()
        self.gru = nn.GRU(n_gas, hidden, batch_first=True)
        self.out = nn.Sequential(nn.Linear(hidden + n_gas, 64),
                                 nn.SiLU(),
                                 nn.Linear(64, 1))          # ΔT_air only
    def forward(self, x):                 # x shape (B, 51, G)
        hist, action = x[:, :-1, :], x[:, -1, :]
        h_last,_ = self.gru(hist)         # (B,50,H) → take last step
        h_last  = h_last[:, -1, :]        # (B,H)
        y_hat   = self.out(torch.cat([h_last, action], dim=-1)).squeeze(1)
        return y_hat
    
class LSTMSurrogate(nn.Module):
    def __init__(self, n_gas, hidden=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_gas,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
        )
        self.out = nn.Sequential(
            nn.Linear(hidden + n_gas, 64),
            nn.GELU(),
            nn.Linear(64, 1)  # ΔT_air only
        )

    def forward(self, x):                 # x: (B, 51, G)
        hist, action = x[:, :-1, :], x[:, -1, :]  # (B,50,G), (B,G)
        seq_out, _ = self.lstm(hist)      # seq_out: (B,50,H)
        h_last = seq_out[:, -1, :]        # last timestep hidden: (B,H)
        y_hat = self.out(torch.cat([h_last, action], dim=-1)).squeeze(1)
        return y_hat