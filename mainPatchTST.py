import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from Models.PatchTST import Model

# ----------- CONFIGURACIÓN -----------
CSV_PATH = 'data/generacion_eolica_oct-nov-dic22.csv'
CENTRAL = 'PE AURORA'
N_STEPS_IN = 168
N_STEPS_OUT = 1    
TEST_SIZE = 0.1     


STEP_A_GRAFICAR = 1

EPOCHS = 20
BATCH_SIZE = 256
PRUEBA_NAME = 'experimento_final_robusto'

RESULTS_DIR = f'resultados_{PRUEBA_NAME}'
os.makedirs(RESULTS_DIR, exist_ok=True)

# ----------- LECTURA Y PREPARACIÓN DE DATOS (Sin cambios) -----------
df = pd.read_csv(CSV_PATH)
df['timestamp'] = pd.to_datetime(df['timestamp'])
serie = (
    df[['timestamp', CENTRAL]]
    .set_index('timestamp')
    .rename(columns={CENTRAL: 'generacion'})
    .sort_index()
    .asfreq('h')
)
serie['generacion'] = serie['generacion'].interpolate()
print(f"Serie de tiempo cargada. Rango de fechas: {serie.index.min()} a {serie.index.max()}")

# ----------- SLIDING WINDOW (Sin cambios) -----------
valores = serie['generacion'].values
X_list, y_list = [], []
for i in range(len(valores) - N_STEPS_IN - N_STEPS_OUT + 1):
    X_list.append(valores[i:i + N_STEPS_IN])
    y_list.append(valores[i + N_STEPS_IN : i + N_STEPS_IN + N_STEPS_OUT])
X = np.array(X_list)
y = np.array(y_list)

# ----------- SPLIT ENTRE TRAIN Y TEST (POR PORCENTAJE) (Sin cambios) -----------
split_idx = int(len(X) * (1 - TEST_SIZE))
prediction_start_timestamps = serie.index[N_STEPS_IN : len(X) + N_STEPS_IN]
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
print(f"División de datos realizada con TEST_SIZE = {TEST_SIZE}:")
print(f" - Muestras de entrenamiento: {len(X_train)}")
print(f" - Muestras de prueba (Test): {len(X_test)}")
print(f" - Primera predicción de prueba comienza en: {prediction_start_timestamps[split_idx]}")

# ----------- ESCALADO, MODELO Y ENTRENAMIENTO (Sin cambios) -----------
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
scaler_y = StandardScaler()
# Reshape 'y' por si N_STEPS_OUT es 1
y_train_reshaped = y_train.reshape(-1, N_STEPS_OUT)
y_test_reshaped = y_test.reshape(-1, N_STEPS_OUT)
y_train_scaled = scaler_y.fit_transform(y_train_reshaped)
y_test_scaled = scaler_y.transform(y_test_reshaped)
joblib.dump(scaler_X, os.path.join(RESULTS_DIR, 'scaler_X.pkl'))
joblib.dump(scaler_y, os.path.join(RESULTS_DIR, 'scaler_y.pkl'))
X_train_scaled = X_train_scaled.reshape(-1, N_STEPS_IN, 1)
X_test_scaled = X_test_scaled.reshape(-1, N_STEPS_IN, 1)
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
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
        self.forecast_type = 'probabilistic'
configs = Configs()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model(configs).to(device)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
def gaussian_nll_loss(mu, sigma, y):
    sigma = torch.clamp(sigma, min=1e-6)
    nll = torch.log(sigma) + 0.5 * ((y - mu) / sigma)**2
    return nll.mean()
criterion = gaussian_nll_loss
train_losses = []
print(f'Entrenando en {device}...')
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        mu, sigma = model(X_batch)
        loss = criterion(mu, sigma, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
    avg_loss = total_loss / len(train_loader.dataset)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")

# ----------- EVALUACIÓN Y DES-ESCALADO (Sin cambios) -----------
model.eval()
preds_mu_scaled, preds_sigma_scaled, targets_scaled = [], [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        mu, sigma = model(X_batch)
        preds_mu_scaled.append(mu.cpu().numpy())
        preds_sigma_scaled.append(sigma.cpu().numpy())
        targets_scaled.append(y_batch.cpu().numpy())
preds_mu_scaled = np.concatenate(preds_mu_scaled)
preds_sigma_scaled = np.concatenate(preds_sigma_scaled)
targets_scaled = np.concatenate(targets_scaled)
scaler_y = joblib.load(os.path.join(RESULTS_DIR, 'scaler_y.pkl'))
preds_mu_inv = scaler_y.inverse_transform(preds_mu_scaled)
targets_inv = scaler_y.inverse_transform(targets_scaled)
preds_sigma_inv = preds_sigma_scaled * scaler_y.scale_
mse_original = mean_squared_error(targets_inv, preds_mu_inv)
print(f"Test MSE (sobre la media, escala original): {mse_original:.4f}")

# ------------ SECCIÓN DE VISUALIZACIÓN UNIFICADA Y FLEXIBLE ------------

# Asegurarse que el paso a graficar sea válido
if STEP_A_GRAFICAR > N_STEPS_OUT or STEP_A_GRAFICAR < 1:
    print(f"ADVERTENCIA: STEP_A_GRAFICAR ({STEP_A_GRAFICAR}) es inválido. Se usará el paso 1.")
    STEP_A_GRAFICAR = 1

# Selecciona la columna (el paso) a graficar. El índice es `step - 1`.
step_index = STEP_A_GRAFICAR - 1

# Timestamps de los valores reales que vamos a comparar
# Se ajustan para coincidir con el paso del horizonte que estamos graficando
test_timestamps_reales = prediction_start_timestamps[split_idx:]
plot_timestamps = test_timestamps_reales + pd.to_timedelta(step_index, 'h')

# Seleccionamos los datos del paso correspondiente
plot_real = targets_inv[:, step_index]
plot_pred_mu = preds_mu_inv[:, step_index]
plot_pred_sigma = preds_sigma_inv[:, step_index]

# Intervalo de confianza
lower_bound = plot_pred_mu - 1.96 * plot_pred_sigma
upper_bound = plot_pred_mu + 1.96 * plot_pred_sigma

print(f"Graficando el paso de predicción número {STEP_A_GRAFICAR} para todo el conjunto de prueba...")
plt.figure(figsize=(15, 7))
plt.plot(plot_timestamps, plot_real, label='Real', color='blue', zorder=2)
plt.plot(plot_timestamps, plot_pred_mu, label='Predicción (Media)', color='red', linestyle='--', zorder=3, alpha=0.8)
plt.fill_between(plot_timestamps, lower_bound, upper_bound, color='red', alpha=0.2, label='Intervalo de Confianza (95%)', zorder=1)
plt.title(f'Predicción de {STEP_A_GRAFICAR} Hora(s) Adelante para {CENTRAL} (Conjunto de Prueba Completo)')
plt.xlabel('Timestamp')
plt.ylabel('Generación')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig(os.path.join(RESULTS_DIR, f'comparacion_predicciones_{N_STEPS_OUT}steps_step{STEP_A_GRAFICAR}.png'))
plt.show()

# Curva de pérdida (Sin cambios)
plt.figure(figsize=(8,4))
plt.plot(train_losses, label='Loss entrenamiento')
plt.xlabel('Época')
plt.ylabel('NLL Loss')
plt.title('Curva de pérdida (entrenamiento)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'loss_entrenamiento.png'))
plt.show()