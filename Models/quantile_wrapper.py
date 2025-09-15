# Models/quantile_wrapper.py
import torch
import torch.nn as nn

class QuantileWrapper(nn.Module):
    """
    Envuelve un predictor puntual y emite K cuantiles [B,H,K].
    - Predice offsets respecto a y_hat y luego ORDENA a lo largo de K
      para garantizar monotonicidad (simple y estable).
    """
    def __init__(self, base_model: nn.Module, pred_len: int,
                 quantiles=(0.05, 0.95), hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.base = base_model
        self.pred_len = pred_len
        self.register_buffer("quantiles", torch.tensor(quantiles, dtype=torch.float32))
        self.K = len(quantiles)

        self.head = nn.Sequential(
            nn.Linear(pred_len, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, pred_len * self.K)  # offsets por horizonte y por cuantil
        )

    def forward(self, x):
        y_hat = self.base(x)                 # [B,H] o [B,H,1]
        if y_hat.dim() == 3 and y_hat.shape[-1] == 1:
            y_hat = y_hat.squeeze(-1)
        off = self.head(y_hat).view(y_hat.size(0), self.pred_len, self.K)  # [B,H,K]
        q = y_hat.unsqueeze(-1) + off                                       # [B,H,K]
        q, _ = torch.sort(q, dim=-1)                                        # garantizar orden
        return q
