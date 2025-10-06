import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn

class GRUSurrogate(nn.Module):
    def __init__(self, n_gas, hidden=64):
        super().__init__()
        self.gru = nn.GRU(n_gas, hidden, batch_first=True)
        self.out = nn.Sequential(nn.Linear(hidden + n_gas, 64),
                                 nn.SiLU(),
                                 nn.Linear(64, 1))              
    def forward(self, x):              
        hist, action = x[:, :-1, :], x[:, -1, :]
        h_last,_ = self.gru(hist)      
        h_last  = h_last[:, -1, :]     
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

    def forward(self, x):                 
        hist, action = x[:, :-1, :], x[:, -1, :]   
        seq_out, _ = self.lstm(hist)      
        h_last = seq_out[:, -1, :]        
        y_hat = self.out(torch.cat([h_last, action], dim=-1)).squeeze(1)
        return y_hat
    

class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, d=1):
        super().__init__()
        self.pad = nn.ConstantPad1d(( (k-1)*d, 0 ), 0)  
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, dilation=d)
    def forward(self, x):          
        return self.conv(self.pad(x))

class TCNBlock(nn.Module):
    def __init__(self, ch, k=3, d=1):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(ch, ch, k=k, d=d),
            nn.GELU(),
            nn.Conv1d(ch, ch, kernel_size=1),
        )
        self.act = nn.GELU()
    def forward(self, x):             # (B, C, T)
        return self.act(x + self.net(x))

class TCNForecaster(nn.Module):
    def __init__(self, n_gas, n_blocks=8, hidden=128, k=3):
        super().__init__()
        self.proj_in = nn.Conv1d(n_gas, hidden, kernel_size=1)
        self.blocks = nn.ModuleList([TCNBlock(hidden, k=k, d=2**b) for b in range(n_blocks)])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(hidden + n_gas, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )
    def forward(self, x):             
        hist, next_em = x[:, :-1, :], x[:, -1, :]
        h = hist.transpose(1, 2)      
        h = self.proj_in(h)
        for blk in self.blocks:
            h = blk(h)
        h = self.pool(h).squeeze(-1)  
        y = self.head(torch.cat([h, next_em], dim=-1)).squeeze(1)
        return y