# mainPatchTST.py — Weather dataset (10-min), K-quantiles + CQR + crps
# ---------------------------------------------------------------------

import os
import importlib
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


# ====== Importar modelo base (acepta nombres comunes) ======
_mod = importlib.import_module('Models.PatchTST')
BaseModel = None
for _name in ('Model', 'PatchTST', 'PatchTSTModel'):
    if hasattr(_mod, _name):
        BaseModel = getattr(_mod, _name)
        break
if BaseModel is None:
    raise ImportError("En Models/PatchTST.py no encontré 'Model'/'PatchTST'/'PatchTSTModel'.")

from Models.quantile_wrapper import QuantileWrapper
from utils.quantile_loss import pinball_loss
from conformal.cqr import cqr_calibrate, cqr_apply, coverage_width

# ===================== CONFIG =====================
CSV_PATH    = 'data/weather.csv'   # <-- ruta al weather.csv
TARGET_COL  = 'T (degC)'           # <-- objetivo en weather (puedes cambiarlo p.ej. 'rh (%)')

N_STEPS_IN  = 1008   # 7 días * 24h * 6 (10-min) = 1008
N_STEPS_OUT = 1
TEST_SIZE   = 0.05
CALIB_RATIO = 0.05           # porción final del train para calibración conforme
BATCH_SIZE  = 256
EPOCHS      = 200
LR          = 1e-3

# CONFIG
ALPHA = 0.05  # 95%
_q = list(np.round(np.linspace(0.01, 0.99, 99), 4))
_q += [0.025, 0.975]
QUANTILES = tuple(sorted(set(_q)))

STEP_A_GRAFICAR = 1           # horizonte a graficar (1..N_STEPS_OUT)
RESULTS_DIR = 'resultados_CQR'
INPUT_FORMAT = 'channels_last'  # 'channels_last' (B,L,C) o 'channels_first' (B,C,L)
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===================== Utilidades CRPS desde cuantiles =====================
@torch.no_grad()
def _pinball(y, q, tau):
    d = y - q
    return torch.maximum(tau * d, (tau - 1.0) * d)

def _tau_weights(taus: torch.Tensor):
    taus = taus.sort().values
    w = torch.zeros_like(taus)
    if taus.numel() == 1:
        w[:] = 1.0
        return w
    w[1:-1] = 0.5 * (taus[2:] - taus[:-2])
    w[0]    = 0.5 * (taus[1] - taus[0])
    w[-1]   = 0.5 * (taus[-1] - taus[-2])
    return w

@torch.no_grad()
def crps_from_quantiles(y_true: torch.Tensor, q_all: torch.Tensor, taus_list):
    K = q_all.size(-1)
    taus = torch.as_tensor(taus_list, dtype=q_all.dtype, device=q_all.device).view(1,1,K)
    weights = _tau_weights(taus.view(-1)).to(q_all.dtype).to(q_all.device)  # [K]
    pb = _pinball(y_true.unsqueeze(-1), q_all, taus)                        # [N,H,K]
    crps = 2.0 * torch.tensordot(pb, weights, dims=([-1],[0]))             # [N,H]
    return crps

def closest_idx(taus, target):
    arr = np.asarray(taus, dtype=float)
    return int(np.abs(arr - target).argmin())

# =============== Lectura serie (weather 10-min o genérico) ===============
df = pd.read_csv(CSV_PATH)

TIME_COL_CANDS = ['date', 'Date', 'Date Time', 'datetime', 'timestamp', 'time']
time_col = next((c for c in TIME_COL_CANDS if c in df.columns), None)

def resolve_target(df, desired):
    if desired is None: return None
    cols_lower = {c.lower(): c for c in df.columns}
    if desired.lower() in cols_lower: return cols_lower[desired.lower()]
    raise KeyError

try:
    target_col = resolve_target(df, TARGET_COL)
except KeyError:
    # heurística: usa 'OT' (ETT) o 'T (degC)' (weather) o la primera numérica
    if 'OT' in df.columns: target_col = 'OT'
    elif 'T (degC)' in df.columns: target_col = 'T (degC)'
    else:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            raise KeyError(f"No encuentro columna objetivo '{TARGET_COL}' en {CSV_PATH}")
        target_col = num_cols[0]
    print(f"[INFO] TARGET_COL no encontrado; usando '{target_col}' automáticamente.")
else:
    if target_col != TARGET_COL:
        print(f"[INFO] Normalizado TARGET_COL → '{target_col}'")

if time_col is None:
    serie = df[[target_col]].rename(columns={target_col: 'y'})
    prediction_index = pd.RangeIndex(len(serie))  # índices enteros
else:
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    df = df.dropna(subset=[time_col])
    serie = (
        df[[time_col, target_col]]
        .rename(columns={time_col: 'timestamp', target_col: 'y'})
        .set_index('timestamp')
        .sort_index()
    )
    # weather: 10-min
    try: serie = serie.asfreq('10min')
    except Exception: pass
    prediction_index = serie.index

serie['y'] = serie['y'].interpolate()
print(f"[INFO] Serie WEATHER: {len(serie)} filas. Rango: {prediction_index.min()} — {prediction_index.max()}")

# ===================== Sliding window =====================
vals = serie['y'].values
X_list, y_list = [], []
for i in range(len(vals) - N_STEPS_IN - N_STEPS_OUT + 1):
    X_list.append(vals[i:i + N_STEPS_IN])
    y_list.append(vals[i + N_STEPS_IN : i + N_STEPS_IN + N_STEPS_OUT])
X = np.array(X_list)  # [N,L]
y = np.array(y_list)  # [N,H]
prediction_start_ts = prediction_index[N_STEPS_IN : len(X) + N_STEPS_IN]

# ===================== Splits (train/test) =====================
split_idx = int(len(X) * (1 - TEST_SIZE))
X_train_all, X_test = X[:split_idx], X[split_idx:]
y_train_all, y_test = y[:split_idx], y[split_idx:]
print(f"[SPLIT] train_all={len(X_train_all)} | test={len(X_test)}")

# Calibración = cola del train
calib_size = max(1, int(len(X_train_all) * CALIB_RATIO))
X_train, X_calib = X_train_all[:-calib_size], X_train_all[-calib_size:]
y_train, y_calib = y_train_all[:-calib_size], y_train_all[-calib_size:]
print(f"[CALIB] calib={len(X_calib)} ({CALIB_RATIO:.0%} de train) | train_final={len(X_train)}")

# ===================== Escalado =====================
scaler_X = StandardScaler()
X_train_s = scaler_X.fit_transform(X_train)
X_calib_s = scaler_X.transform(X_calib)
X_test_s  = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_s = scaler_y.fit_transform(y_train.reshape(-1, N_STEPS_OUT))
y_calib_s = scaler_y.transform(y_calib.reshape(-1, N_STEPS_OUT))
y_test_s  = scaler_y.transform(y_test.reshape(-1, N_STEPS_OUT))

joblib.dump(scaler_X, os.path.join(RESULTS_DIR, 'scaler_X.pkl'))
joblib.dump(scaler_y, os.path.join(RESULTS_DIR, 'scaler_y.pkl'))

# reshape al formato que espera el modelo base
if INPUT_FORMAT == 'channels_last':   # [B,L,C]
    X_train_s = X_train_s.reshape(-1, N_STEPS_IN, 1)
    X_calib_s = X_calib_s.reshape(-1, N_STEPS_IN, 1)
    X_test_s  = X_test_s.reshape(-1, N_STEPS_IN, 1)
elif INPUT_FORMAT == 'channels_first':  # [B,C,L]
    X_train_s = X_train_s.reshape(-1, 1, N_STEPS_IN)
    X_calib_s = X_calib_s.reshape(-1, 1, N_STEPS_IN)
    X_test_s  = X_test_s.reshape(-1, 1, N_STEPS_IN)
else:
    raise ValueError("INPUT_FORMAT debe ser 'channels_last' o 'channels_first'.")

X_train_t = torch.tensor(X_train_s, dtype=torch.float32)
y_train_t = torch.tensor(y_train_s, dtype=torch.float32)
X_calib_t = torch.tensor(X_calib_s, dtype=torch.float32)
y_calib_t = torch.tensor(y_calib_s, dtype=torch.float32)
X_test_t  = torch.tensor(X_test_s,  dtype=torch.float32)
y_test_t  = torch.tensor(y_test_s,  dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
calib_loader = DataLoader(TensorDataset(X_calib_t, y_calib_t), batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(TensorDataset(X_test_t,  y_test_t),  batch_size=BATCH_SIZE, shuffle=False)

# ===================== Modelo base + Wrapper K-cuantiles =====================
class BasePointAdapter(nn.Module):
    """Si tu base devuelve (mu, sigma) o [B,H,1], nos quedamos con y_hat [B,H]."""
    def __init__(self, base): super().__init__(); self.base = base
    def forward(self, x):
        out = self.base(x)
        if isinstance(out, (tuple, list)): out = out[0]
        if out.dim() == 3 and out.shape[-1] == 1: out = out.squeeze(-1)
        return out  # [B,H]

# Configs (ajusta números si quieres)
class Configs:
    def __init__(self):
        self.enc_in = 1
        self.seq_len = N_STEPS_IN
        self.pred_len = N_STEPS_OUT
        self.e_layers = 3
        self.n_heads = 8
        self.d_model = 128
        self.d_ff = 256
        self.dropout = 0.2
        self.fc_dropout = 0.2
        self.head_dropout = 0.2
        self.individual = False
        self.patch_len = 16
        self.stride = 8
        self.padding_patch = 'end'
        self.revin = True
        self.affine = True
        self.subtract_last = False
        self.decomposition = False
        self.kernel_size = 25

configs = Configs()
base_raw = BaseModel(configs).to(device)
base = BasePointAdapter(base_raw).to(device)
model = QuantileWrapper(base_model=base, pred_len=N_STEPS_OUT, quantiles=QUANTILES).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ===================== Entrenamiento (pinball multi-quantil) =====================
print(f"[TRAIN] device={device} | K={len(QUANTILES)} quantiles | alpha={ALPHA}")
train_losses = []
for epoch in range(EPOCHS):
    model.train()
    tot = 0.0
    for Xb, Yb in train_loader:
        Xb, Yb = Xb.to(device), Yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        q = model(Xb)                            # [B,H,K]
        loss = pinball_loss(Yb, q, QUANTILES)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tot += loss.item() * Xb.size(0)
    avg = tot / len(train_loader.dataset)
    train_losses.append(avg)
    print(f"Epoch {epoch+1}/{EPOCHS}  pinball={avg:.5f}")

torch.save(model.state_dict(), os.path.join(RESULTS_DIR, 'best_kq.pt'))

# ===================== Calibración conforme (extremos α/2 y 1-α/2) =====================
i_lo = closest_idx(QUANTILES, ALPHA/2.0)
i_up = closest_idx(QUANTILES, 1.0 - ALPHA/2.0)

with torch.no_grad():
    model.eval()
    Qs_all, Ys = [], []
    for Xb, Yb in calib_loader:
        Xb = Xb.to(device)
        q_all = model(Xb).cpu()      # [B,H,K]
        Qs_all.append(q_all); Ys.append(Yb)
q_cal_all = torch.cat(Qs_all, 0)     # [N,H,K]
y_cal     = torch.cat(Ys, 0)         # [N,H]
q_cal_ext = torch.stack([q_cal_all[..., i_lo], q_cal_all[..., i_up]], dim=-1)  # [N,H,2]

eps = cqr_calibrate(y_cal, q_cal_ext, alpha=ALPHA, per_horizon=True)  # [H]
np.save(os.path.join(RESULTS_DIR, 'conformal_eps.npy'), eps.numpy())
print(f"[CALIB] eps guardado en {os.path.join(RESULTS_DIR, 'conformal_eps.npy')}")

# ===================== Test: CP sobre extremos + QCRPS con todos los cuantiles =====================
with torch.no_grad():
    model.eval()
    Q_pred, LOs, HIs, Ys = [], [], [], []
    for Xb, Yb in test_loader:
        Xb = Xb.to(device)
        q_all = model(Xb)                       # [B,H,K] (escala normalizada)
        Q_pred.append(q_all.cpu())
        lo, hi = cqr_apply(torch.stack([q_all[..., i_lo], q_all[..., i_up]], dim=-1),
                           eps.to(q_all.device))
        LOs.append(lo.cpu()); HIs.append(hi.cpu()); Ys.append(Yb)

q_test_all_s = torch.cat(Q_pred, 0).numpy()   # [N,H,K]
lo_s = torch.cat(LOs, 0).numpy()              # [N,H]
hi_s = torch.cat(HIs, 0).numpy()
y_s  = torch.cat(Ys, 0).numpy()

# -------- Inversión de escala --------
y_true = scaler_y.inverse_transform(y_s)                     # [N,H]
lo = scaler_y.inverse_transform(lo_s)                        # [N,H]
hi = scaler_y.inverse_transform(hi_s)

N, H, K = q_test_all_s.shape
q_flat = q_test_all_s.reshape(N*K, H)
q_inv_flat = scaler_y.inverse_transform(q_flat)
q_test_all = q_inv_flat.reshape(N, H, K)                     # [N,H,K]

# -------- Métricas globales --------
y_t = torch.tensor(y_true, dtype=torch.float32)
q_t = torch.tensor(q_test_all, dtype=torch.float32)
crps = crps_from_quantiles(y_t, q_t, QUANTILES)              # [N,H]

lo_t = torch.tensor(lo, dtype=torch.float32)
hi_t = torch.tensor(hi, dtype=torch.float32)
cov, width = coverage_width(y_t, lo_t, hi_t)

print(f"[TEST] crps medio={crps.mean().item():.4f} | Cobertura={cov.mean().item():.4f} | Ancho={width.mean().item():.4f}")

# ---------- Rolling sin sobrantes (ajuste automático del bloque) ----------
Ntest = y_true.shape[0]
def nearest_divisor(n: int, target: int) -> int:
    best = 1; bestdiff = abs(target - 1)
    r = int(n**0.5)
    for d in range(1, r + 1):
        if n % d == 0:
            for cand in (d, n // d):
                diff = abs(target - cand)
                if diff < bestdiff:
                    best, bestdiff = cand, diff
    return best

ROLL_BLOCK = 0  # 1 semana (10-min). Pon 144 para 1 día, 0 para desactivar.
MAX_LINES  = 12     # imprime como máx. 12 bloques (primeros 11 y el último)

if ROLL_BLOCK and ROLL_BLOCK > 0:
    block = nearest_divisor(Ntest, ROLL_BLOCK)
    print(f"[ROLL] block_len={block} (ajustado desde {ROLL_BLOCK} para evitar sobrantes; N={Ntest})")
    for s in range(0, Ntest, block):
        e = s + block
        cov_b  = ((y_t[s:e] >= lo_t[s:e]) & (y_t[s:e] <= hi_t[s:e])).float().mean().item()
        wid_b  = (hi_t[s:e] - lo_t[s:e]).mean().item()
        crps_b = crps[s:e].mean().item()
        print(f"  [{s:5d}:{e:5d}]  cov={cov_b:.3f}  width={wid_b:.3f}  crps={crps_b:.3f}")
# ====== Estadísticos extra de CRPS ======
import numpy as np

crps_np = crps.detach().cpu().numpy()  # crps es tu tensor 1D sobre todo el test

crps_mean    = float(crps_np.mean())
crps_median  = float(np.median(crps_np))
p10, p25, p75, p90, p95 = np.percentile(crps_np, [10, 25, 75, 90, 95])
crps_total   = float(crps_np.sum())    # "total" = suma sobre todas las muestras

# imprime el resumen enriquecido
print(
    "[TEST] "
    f"crps medio={crps_mean:.4f}  |  mediana={crps_median:.4f}  |  "
    f"p10={p10:.4f}  p25={p25:.4f}  p75={p75:.4f}  p90={p90:.4f}  p95={p95:.4f}  |  "
    f"Cobertura={cov.mean().item():.4f}  |  Ancho={width.mean().item():.4f}"
)

# imprime el CRPS "total" (suma)
print(f"[TEST] crps total (suma)={crps_total:.4f}")

# ====== (Opcional) CRPS "final" de los últimos N puntos ======
LAST_N = 0   # pon >0 para activar, p.ej. 500
if LAST_N and LAST_N > 0:
    n = min(LAST_N, crps_np.shape[0])
    crps_final = float(crps_np[-n:].mean())
    print(f"[TEST] crps final (últimos {n})={crps_final:.4f}")

# ===================== Visualización =====================
h = max(0, min(N_STEPS_OUT-1, STEP_A_GRAFICAR-1))
ts_plot = prediction_start_ts[split_idx:]
mid = 0.5 * (lo + hi)

plt.figure(figsize=(15,6))
plt.plot(ts_plot, y_true[:, h], label='Real', alpha=0.9)
plt.plot(ts_plot, mid[:, h],    label='Pred (centro)', linestyle='--', alpha=0.8)
plt.fill_between(ts_plot, lo[:, h], hi[:, h], alpha=0.2, label=f'CQR {int((1-ALPHA)*100)}%')
plt.title(f'Conformal CQR — horizonte {h+1} (weather, target={target_col})')
plt.xlabel('Tiempo'); plt.ylabel('Valor'); plt.grid(alpha=0.3, linestyle='--'); plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, f'cqr_h{h+1}.png'))
plt.show()

plt.figure(figsize=(8,4))
plt.plot(train_losses, label='Pinball loss (train)')
plt.title('Curva de pérdida (K-quantiles)'); plt.xlabel('Época'); plt.ylabel('Loss'); plt.grid(alpha=0.3, linestyle='--'); plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'loss_kquantiles.png'))
plt.show()

# ===== Guardar resumen en TXT (simple) =====
from pathlib import Path
from datetime import datetime
import json
import os

# aseguramos métricas agregadas
crps_mean   = float(crps_np.mean())
crps_median = float(np.median(crps_np))
p10, p25, p75, p90, p95 = map(float, np.percentile(crps_np, [10, 25, 75, 90, 95]))
crps_total  = float(crps_np.sum())
cov_mean    = float(cov.mean().item())
width_mean  = float(width.mean().item())

# nombre de corrida sin depender de "args"
stamp    = datetime.now().strftime("%Y%m%d-%H%M%S")
run_name = f"{stamp}_seq{N_STEPS_IN}_pred{N_STEPS_OUT}_alpha{ALPHA}_K{len(QUANTILES)}"

# carpeta y archivo
log_dir  = Path("resultados_CQR")
log_dir.mkdir(parents=True, exist_ok=True)
log_path = log_dir / f"{run_name}.txt"

payload = {
    "stamp": stamp,
    "target": str(target_col) if 'target_col' in globals() else None,
    "seq_len": int(N_STEPS_IN),
    "pred_len": int(N_STEPS_OUT),
    "alpha": float(ALPHA),
    "K": int(len(QUANTILES)),
    "quantiles": list(map(float, QUANTILES))[:20] + (["..."] if len(QUANTILES) > 20 else []),

    "metrics": {
        "crps_mean": crps_mean,
        "crps_median": crps_median,
        "crps_p10": p10,
        "crps_p25": p25,
        "crps_p75": p75,
        "crps_p90": p90,
        "crps_p95": p95,
        "crps_total": crps_total,
        "coverage": cov_mean,
        "width": width_mean
    },

    # info útil extra si existe el archivo de eps
    "conformal_eps_file": os.path.join("resultados_CQR", "conformal_eps.npy")
}

with open(log_path, "w", encoding="utf-8") as f:
    f.write("# Resumen de la corrida\n")
    f.write(json.dumps(payload, ensure_ascii=False, indent=2))
    f.write("\n")

print(f"[LOG] TXT guardado en: {log_path}")
