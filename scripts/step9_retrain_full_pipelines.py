import pandas as pd
import joblib
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

# ----------------------------
# Load cleaned dataset
# ----------------------------
DATA = "results/step2_clean.csv"
df = pd.read_csv(DATA, low_memory=False)

# ----------------------------
# Columns (MATCH STREAMLIT INPUTS)
# ----------------------------
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

# ----------------------------
# Split by phase
# ----------------------------
df_liq = df[df["Liq or Gas"].str.lower().str.contains("liq")].copy()
df_gas = df[df["Liq or Gas"].str.lower().str.contains("gas")].copy()

# Log transform target
y_liq = np.log1p(df_liq[target_col])
y_gas = np.log1p(df_gas[target_col])

X_liq = df_liq[cat_cols + num_cols]
X_gas = df_gas[cat_cols + num_cols]

# ----------------------------
# Preprocessing
# ----------------------------
preprocess = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
)

# ----------------------------
# Models
# ----------------------------
rf_params = dict(
    n_estimators=500,
    random_state=42,
    n_jobs=-1
)

pipe_liq = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", RandomForestRegressor(**rf_params))
    ]
)

pipe_gas = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", RandomForestRegressor(**rf_params))
    ]
)

# ----------------------------
# Train
# ----------------------------
pipe_liq.fit(X_liq, y_liq)
pipe_gas.fit(X_gas, y_gas)

# ----------------------------
# Save
# ----------------------------
joblib.dump(pipe_liq, "models/rf_liq_pipeline.joblib")
joblib.dump(pipe_gas, "models/rf_gas_pipeline.joblib")

print("✅ Full pipelines retrained and saved")
