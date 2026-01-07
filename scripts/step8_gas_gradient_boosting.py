import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
import joblib

# -----------------------
# Paths
# -----------------------
DATA_PATH = "results/step2_clean.csv"
MODEL_PATH = "models/gb_gas.joblib"

# -----------------------
# Load data
# -----------------------
df = pd.read_csv(DATA_PATH, low_memory=False)

# -----------------------
# Filter Gas phase
# -----------------------
df_gas = df[df["Liq or Gas"].str.lower() == "gas"].reset_index(drop=True)

print(f"Gas-phase rows: {len(df_gas)}")

# -----------------------
# Target (log1p)
# -----------------------
y = np.log1p(
    pd.to_numeric(df_gas["total yield (μmol gcat-1 h-1)"], errors="coerce")
)

# -----------------------
# Feature selection
# -----------------------
categorical_cols = [
    "A", "B", "Synthesis Method of Perovskite",
    "cocatalyst", "Light Type"
]

numerical_cols = [
    "Band Gap*", "Bet Surface Area (m2g-1)",
    "Calcination Time (h)", "Reaction Temperature (C )",
    "cat W (g)"
]

X = df_gas[categorical_cols + numerical_cols]

# -----------------------
# Preprocessing
# -----------------------
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)
numerical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", categorical_transformer, categorical_cols),
        ("num", numerical_transformer, numerical_cols),
    ]
)

# -----------------------
# Model
# -----------------------
gb_model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    random_state=42
)

pipeline = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", gb_model)
    ]
)

# -----------------------
# Cross-validation
# -----------------------
cv = KFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(
    pipeline,
    X,
    y,
    cv=cv,
    scoring="r2"
)

print(f"Gas Gradient Boosting CV R² (log-space): "
      f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# -----------------------
# Train final model
# -----------------------
pipeline.fit(X, y)

joblib.dump(pipeline, MODEL_PATH)
print(f"Saved model to: {MODEL_PATH}")
