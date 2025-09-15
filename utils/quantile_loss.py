# utils/quantile_loss.py
from typing import Sequence
import torch

def pinball_loss(y_true: torch.Tensor, q_pred: torch.Tensor, quantiles: Sequence[float]) -> torch.Tensor:
    """
    y_true: [B,H]
    q_pred: [B,H,Q], Q = len(quantiles) (p.ej., 2: 0.05 y 0.95)
    """
    assert q_pred.size(-1) == len(quantiles), "Dim Q != #quantiles"
    taus = torch.tensor(quantiles, device=q_pred.device, dtype=q_pred.dtype).view(1,1,-1)
    diff = y_true.unsqueeze(-1) - q_pred
    loss = torch.maximum(taus * diff, (taus - 1.0) * diff)
    return loss.mean()
