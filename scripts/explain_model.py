import pandas as pd
import shap
import torch
import torch.nn as nn
import joblib
import os
import matplotlib.pyplot as plt

# 1. Class Definitions 
class PyTorchDNN(nn.Module):
    def __init__(self, input_dim):
        super(PyTorchDNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x): return self.net(x).squeeze(-1)

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

# 2. Loading Data
X_test = pd.read_csv("models/validation_features.csv")
input_dim = X_test.shape[1]

# Fallback: Agar csv me explicit target drops na huye hon to backup checks
if 'aqi' in X_test.columns: X_test = X_test.drop(columns=['aqi'])
if 'timestamp' in X_test.columns: X_test = X_test.drop(columns=['timestamp'])
if 'date' in X_test.columns: X_test = X_test.drop(columns=['date'])

# 3. Model Loading Logic
model_path = "models/champion_model.pkl" 

if model_path.endswith('.pkl'):
    model = joblib.load(model_path)
    # Tree/Linear Explainer wrapper for scikit-learn/xgboost/lightgbm
    explainer = shap.Explainer(model, X_test)
else:
    model = PyTorchDNN(input_dim) # Apni winning architecture config select karein
    model.load_state_dict(torch.load(model_path))
    model.eval()
    # Deep/Sampling explainer wrapper mapping
    explainer = shap.Explainer(model, X_test)

# 4. SHAP Calculation
shap_values = explainer(X_test)

# CRITICAL FIX: Explicitly assign feature names to shap object if lost
if hasattr(shap_values, "feature_names") and (shap_values.feature_names is None or isinstance(shap_values.feature_names[0], int)):
    shap_values.feature_names = list(X_test.columns)

# 5. Plotting (Converting to clean horizontal Bar Chart)
plt.figure(figsize=(10, 6))
# max_display=10 ya 15 krlo taake top important features clean nazar ayen
shap.plots.bar(shap_values, max_display=12, show=False)
plt.tight_layout()
plt.savefig("models/shap_summary.png", dpi=300)
plt.close()

print("SHAP analysis complete with Feature Names! Bar chart saved to models/shap_summary.png")