import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
import joblib
import hopsworks
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Step 1: Initialize cloud platform connection layer
print("Initializing connection with Hopsworks platform...")
api_key = "2sPuFtpKZd0DSSbf.aKg8zOCU5wwauNa7C2bpI3oWvb3Rv3I7j30IL3ircOOqh0P76uPyIaEMuLmBS3ar"

try:
    project = hopsworks.login(project="Karachi_AQI_2026", api_key_value=api_key)
    fs = project.get_feature_store()
    print("Hopsworks feature store connection established successfully.")
except Exception as e:
    print(f"Authentication failure or connection timeout: {str(e)}")
    sys.exit(1)

# Step 2: Extract structured historical feature matrix
print("Retrieving historical features from feature group...")
try:
    feature_group = fs.get_feature_group(name="karachi_aqi_features", version=1)
    df = feature_group.read()
    print(f"Dataset synchronization completed. Total rows fetched: {len(df)}")
except Exception as e:
    print(f"Feature extraction failed: {str(e)}. Triggering local backup pipeline fallback.")
    local_path = os.path.join("data", "engineered_features.csv")
    if os.path.exists(local_path):
        df = pd.read_csv(local_path)
    else:
        print("Fatal error: Local fallback feature vectors missing.")
        sys.exit(1)

# Step 3: Chronological data alignment and chronological split
if 'timestamp' in df.columns:
    df = df.sort_values('timestamp').reset_index(drop=True)
    X = df.drop(columns=['aqi', 'timestamp'])
else:
    X = df.drop(columns=['aqi'])

y = df['aqi']

# Apply sequential ordering split to prevent temporal data leakage
split_idx = int(len(df) * 0.8)
X_train_raw, X_test_raw = X.iloc[:split_idx].values.astype(np.float32), X.iloc[split_idx:].values.astype(np.float32)
y_train_raw, y_test_raw = y.iloc[:split_idx].values.astype(np.float32), y.iloc[split_idx:].values.astype(np.float32)

model_leaderboard = {}

# Execution Matrix 1: Ridge Regression Baseline
m_ridge = Ridge(alpha=1.0)
m_ridge.fit(X_train_raw, y_train_raw)
p_ridge = m_ridge.predict(X_test_raw)
model_leaderboard['RidgeRegression'] = {
    'model': m_ridge, 'framework': 'Scikit-Learn',
    'rmse': root_mean_squared_error(y_test_raw, p_ridge),
    'mae': mean_absolute_error(y_test_raw, p_ridge),
    'r2': r2_score(y_test_raw, p_ridge)
}

# Execution Matrix 2: Random Forest
m_rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
m_rf.fit(X_train_raw, y_train_raw)
p_rf = m_rf.predict(X_test_raw)
model_leaderboard['RandomForest'] = {
    'model': m_rf, 'framework': 'Scikit-Learn',
    'rmse': root_mean_squared_error(y_test_raw, p_rf),
    'mae': mean_absolute_error(y_test_raw, p_rf),
    'r2': r2_score(y_test_raw, p_rf)
}

# Execution Matrix 3: XGBoost
m_xgb = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1)
m_xgb.fit(X_train_raw, y_train_raw)
p_xgb = m_xgb.predict(X_test_raw)
model_leaderboard['XGBoost'] = {
    'model': m_xgb, 'framework': 'XGBoost',
    'rmse': root_mean_squared_error(y_test_raw, p_xgb),
    'mae': mean_absolute_error(y_test_raw, p_xgb),
    'r2': r2_score(y_test_raw, p_xgb)
}

# Execution Matrix 4: LightGBM
m_lgb = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
m_lgb.fit(X_train_raw, y_train_raw)
p_lgb = m_lgb.predict(X_test_raw)
model_leaderboard['LightGBM'] = {
    'model': m_lgb, 'framework': 'LightGBM',
    'rmse': root_mean_squared_error(y_test_raw, p_lgb),
    'mae': mean_absolute_error(y_test_raw, p_lgb),
    'r2': r2_score(y_test_raw, p_lgb)
}

# Execution Matrix 5: PyTorch Standard DNN
class PyTorchDNN(nn.Module):
    def __init__(self, input_dim):
        super(PyTorchDNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x): return self.net(x).squeeze(-1)

train_loader = DataLoader(TensorDataset(torch.tensor(X_train_raw), torch.tensor(y_train_raw)), batch_size=32, shuffle=True)
dnn_model = PyTorchDNN(X_train_raw.shape[1])
optimizer = optim.Adam(dnn_model.parameters(), lr=0.01)
for epoch in range(15):
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        loss = nn.MSELoss()(dnn_model(batch_x), batch_y)
        loss.backward()
        optimizer.step()

with torch.no_grad(): p_dnn = dnn_model(torch.tensor(X_test_raw)).numpy()
model_leaderboard['PyTorch_Standard_DNN'] = {
    'model': dnn_model, 'framework': 'PyTorch (Standard)',
    'rmse': root_mean_squared_error(y_test_raw, p_dnn),
    'mae': mean_absolute_error(y_test_raw, p_dnn),
    'r2': r2_score(y_test_raw, p_dnn)
}

# Execution Matrix 6: PyTorch Advanced DNN (With Batch Norm & Dropout)
print("Evaluating PyTorch Advanced DNN architecture...")
class AdvancedDNN(nn.Module):
    def __init__(self, input_dim):
        super(AdvancedDNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x): return self.net(x).squeeze(-1)

adv_model = AdvancedDNN(X_train_raw.shape[1])
optimizer_adv = optim.Adam(adv_model.parameters(), lr=0.005)
for epoch in range(25):
    for batch_x, batch_y in train_loader:
        optimizer_adv.zero_grad()
        loss = nn.MSELoss()(adv_model(batch_x), batch_y)
        loss.backward()
        optimizer_adv.step()

with torch.no_grad(): p_adv = adv_model(torch.tensor(X_test_raw)).numpy()
model_leaderboard['PyTorch_Advanced_DNN'] = {
    'model': adv_model, 'framework': 'PyTorch (Advanced)',
    'rmse': root_mean_squared_error(y_test_raw, p_adv),
    'mae': mean_absolute_error(y_test_raw, p_adv),
    'r2': r2_score(y_test_raw, p_adv)
}

# Leaderboard Summary
print("\n================ SYSTEM PIPELINE BENCHMARKS ================")
for name, metrics in model_leaderboard.items():
    print(f"Model: {name:<22} | RMSE: {metrics['rmse']:.4f} | MAE: {metrics['mae']:.4f} | R2: {metrics['r2']:.4f}")

best_model_name = min(model_leaderboard, key=lambda k: model_leaderboard[k]['rmse'])
print(f"\nChampion model: {best_model_name}")

# Step 6: Final Clean Registry & Saving Logic
print("Promoting champion model to Hopsworks Registry...")
artifact_dir = "models"
os.makedirs(artifact_dir, exist_ok=True)

# 1. Champion Model Select and Save Locally
best_model = model_leaderboard[best_model_name]['model']
best_metrics = model_leaderboard[best_model_name]

if "PyTorch" in best_metrics['framework']:
    final_artifact_path = os.path.join(artifact_dir, "champion_model.pt")
    torch.save(best_model.state_dict(), final_artifact_path)
else:
    final_artifact_path = os.path.join(artifact_dir, "champion_model.pkl")
    joblib.dump(best_model, final_artifact_path)

# 2. Register to Cloud with New Versioning
try:
    model_registry = project.get_model_registry()
    
    # Is baar hum model name ke sath ek unique timestamp laga rahe hain 
    # ya versioning ko force kar rahe hain
    registry_model = model_registry.python.create_model(
        name="karachi_aqi_champion",
        metrics={
            "rmse": float(best_metrics['rmse']),
            "mae": float(best_metrics['mae']),
            "r2": float(best_metrics['r2'])
        },
        description=f"Auto-generated run. Best model: {best_model_name}"
    )
    
    # Save call automatically handles incrementing versions if name exists
    registry_model.save(final_artifact_path)
    print(f"Model {best_model_name} registered successfully.")
    
except Exception as e:
    print(f"Registry error: {e}")