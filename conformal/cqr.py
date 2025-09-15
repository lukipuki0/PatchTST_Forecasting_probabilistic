# conformal/cqr.py
import numpy as np
import torch

@torch.no_grad()
def cqr_calibrate(y_cal: torch.Tensor, q_cal: torch.Tensor, alpha: float = 0.1, per_horizon: bool = True):
    """
    y_cal: [N,H]
    q_cal: [N,H,2] (low<=up)
    -> eps: [H] si per_horizon, o escalar si per_horizon=False
    """
    assert q_cal.size(-1) == 2, "q_cal debe ser [N,H,2]"
    low, up = q_cal[..., 0], q_cal[..., 1]
    s = torch.maximum(low - y_cal, y_cal - up)  # [N,H]

    if per_horizon:
        N = s.size(0)
        k = int(np.ceil((N + 1) * (1 - alpha)))
        k = min(max(k, 1), N)
        eps = torch.sort(s, dim=0).values[k - 1, :]  # [H]
    else:
        N = s.numel()
        k = int(np.ceil((N + 1) * (1 - alpha)))
        k = min(max(k, 1), N)
        eps = torch.sort(s.reshape(-1)).values[k - 1]  # escalar
    return eps

@torch.no_grad()
def cqr_apply(q_pred: torch.Tensor, eps: torch.Tensor):
    """
    q_pred: [B,H,2], eps: [H] o escalar
    -> lo, hi: [B,H]
    """
    low, up = q_pred[..., 0], q_pred[..., 1]
    if isinstance(eps, torch.Tensor):
        while eps.dim() < low.dim():
            eps = eps.unsqueeze(0)  # (H)->(1,H)
    return low - eps, up + eps

@torch.no_grad()
def coverage_width(y_true: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor):
    """Cobertura y ancho medio por horizonte."""
    cov = ((y_true >= lo) & (y_true <= hi)).float().mean(dim=0)
    wid = (hi - lo).mean(dim=0)
    return cov, wid
