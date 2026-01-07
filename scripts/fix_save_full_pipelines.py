import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestRegressor

# -------------------------
# Load cleaned dataset
# -------------------------
DATA = "results/step2_clean.csv"
df = pd.read_csv(DATA, low_memory=False)

# -------------------------
# Columns (same as training)
# -------------------------
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

target_col = [c for c in df.columns if "yield" in c.lower()][0]

# -------------------------
# Split by phase
# -------------------------
df_liq = df[df["Liq or Gas"].str.lower().str.contains("liq")]
df_gas = df[df["Liq or Gas"].str.lower().str.contains("gas")]

# -------------------------
# Preprocessing
# -------------------------
preprocess = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
)

# -------------------------
# Load EXISTING trained RF models
# -------------------------
rf_liq = joblib.load("models/rf_liq.joblib")
rf_gas = joblib.load("models/rf_gas.joblib")

# -------------------------
# Build FULL pipelines
# -------------------------
pipe_liq = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", rf_liq),
    ]
)

pipe_gas = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", rf_gas),
    ]
)

# -------------------------
# Fit preprocess ONLY (no retraining)
# -------------------------
X_liq = df_liq[cat_cols + num_cols]
X_gas = df_gas[cat_cols + num_cols]

pipe_liq.named_steps["preprocess"].fit(X_liq)
pipe_gas.named_steps["preprocess"].fit(X_gas)

# -------------------------
# Save FIXED pipelines
# -------------------------
joblib.dump(pipe_liq, "models/rf_liq_pipeline.joblib")
joblib.dump(pipe_gas, "models/rf_gas_pipeline.joblib")

print("✅ Full pipelines saved successfully")
