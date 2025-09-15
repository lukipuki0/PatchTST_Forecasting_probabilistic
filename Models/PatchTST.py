# Models/PatchTST.py (stub temporal para desbloquear)
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.seq_len = cfg.seq_len
        self.pred_len = cfg.pred_len
        self.enc_in = cfg.enc_in  # nº de canales (1 si univariante)
        self.net = nn.Sequential(
            nn.Flatten(),                               # [B, L*C]
            nn.Linear(self.seq_len * self.enc_in, 512),
            nn.GELU(),
            nn.Linear(512, self.pred_len)              # -> [B, H]
        )

    def forward(self, x):                               # x: [B, L, C] o [B, C, L]
        if x.dim() == 3 and x.shape[1] == self.seq_len: # [B, L, C] -> [B, C, L]
            x = x.transpose(1, 2)
        # ahora x: [B, C, L] o ya [B, C, L]
        x = x.transpose(1, 2)                           # -> [B, L, C]
        return self.net(x)
