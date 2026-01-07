import pandas as pd
import numpy as np
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv("results/step2_clean.csv", low_memory=False)

target_col = [c for c in df.columns if "yield" in c.lower()][0]

cat_cols = [
    "A",
    "B",
    "Synthesis Method of Perovskite",
    "cocatalyst",
    "Light Type",
]

num_cols = [
    "Band Gap*",
    "Bet Surface Area (m2g-1)",
    "Calcination Time (h)",
    "cat W (g)",
    "Reaction Temperature (C )",
]

# -----------------------------
# Split phases
# -----------------------------
liq = df[df["Liq or Gas"].str.lower().str.contains("liq")].copy()
gas = df[df["Liq or Gas"].str.lower().str.contains("gas")].copy()

X_liq = liq[cat_cols + num_cols]
y_liq = np.log1p(liq[target_col])

X_gas = gas[cat_cols + num_cols]
y_gas = np.log1p(gas[target_col])

# -----------------------------
# Preprocessing
# -----------------------------
preprocess = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ],
    remainder="drop"
)

# -----------------------------
# Models
# -----------------------------
rf_params = dict(
    n_estimators=500,
    random_state=42,
    n_jobs=-1
)

pipe_liq = Pipeline([
    ("preprocess", preprocess),
    ("model", RandomForestRegressor(**rf_params))
])

pipe_gas = Pipeline([
    ("preprocess", preprocess),
    ("model", RandomForestRegressor(**rf_params))
])

# -----------------------------
# Train ONCE
# -----------------------------
pipe_liq.fit(X_liq, y_liq)
pipe_gas.fit(X_gas, y_gas)

# -----------------------------
# Save pipelines
# -----------------------------
joblib.dump(pipe_liq, "models/rf_liq_pipeline.joblib")
joblib.dump(pipe_gas, "models/rf_gas_pipeline.joblib")

print("✅ Pipelines trained and saved cleanly")
print("Liq features:", pipe_liq.named_steps["model"].n_features_in_)
print("Gas features:", pipe_gas.named_steps["model"].n_features_in_)
