import pandas as pd
import numpy as np
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("results/step2_clean.csv", low_memory=False)
target = [c for c in df.columns if "yield" in c.lower()][0]

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

def train_and_save(phase, out_path):
    d = df[df["Liq or Gas"].str.lower().str.contains(phase)]
    X = d[cat_cols + num_cols]
    y = np.log1p(d[target])

    preprocess = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    pipe = Pipeline(
        [
            ("preprocess", preprocess),
            ("model", RandomForestRegressor(
                n_estimators=500,
                random_state=42,
                n_jobs=-1
            )),
        ]
    )

    pipe.fit(X, y)

    print(
        f"{phase.upper()} features:",
        pipe.named_steps["model"].n_features_in_
    )

    joblib.dump(pipe, out_path)

train_and_save("liq", "models/rf_liq_pipeline.joblib")
train_and_save("gas", "models/rf_gas_pipeline.joblib")
