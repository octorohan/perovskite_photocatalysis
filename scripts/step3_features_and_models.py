"""
Step 3 — Feature encoding, Train/Test split, Random Forest (regression) and Decision Tree (classification)
Uses tertile binning for yield (Low/Medium/High).
Saves models and a summary CSV.
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, confusion_matrix, classification_report
from sklearn.preprocessing import OneHotEncoder

# ---------- CONFIG ----------
PROJECT_ROOT = r"C:\Users\HP\Desktop\winter_projects\perovskite"
INPUT_IMPUTED = os.path.join(PROJECT_ROOT, "results", "step2_imputed.csv")
OUT_MODELS = os.path.join(PROJECT_ROOT, "models")
OUT_RESULTS = os.path.join(PROJECT_ROOT, "results")
os.makedirs(OUT_MODELS, exist_ok=True)
os.makedirs(OUT_RESULTS, exist_ok=True)

# Column names (exact)
COL_PHASE = "Liq or Gas"
COL_YIELD = "total yield  (μmol  gcat-1 h-1)"
# features to include (most-important set)
NUMERIC_FEATURES = [
    "Bet Surface Area (m2g-1)",
    "Calcination Time (h)",
    "Calcination Temperature (C )",
    "cat W (g)",
    "Reaction Temperature (C )",
    "Reaction Pressure (bar)",
    "Band Gap*"
]
CATEGORICAL_FEATURES = [
    "A", "B", "Dope", "Synthesis Method of Perovskite", "cocatalyst", "Light Type"
]
# for gas include H2O:CO2 as numeric
GAS_ONLY_NUMERIC = ["H2O:CO2"]

RANDOM_STATE = 42

# RF & DT hyperparams from paper (approximate reproduction)
RF_PARAMS = {"n_estimators": 500, "max_features": 5, "random_state": RANDOM_STATE, "n_jobs": -1}
DT_PARAMS = {"min_samples_split": 5, "random_state": RANDOM_STATE}
DT_CCP_ALPHA = {"liq": 0.015, "gas": 0.010}

# ---------- HELPERS ----------
def safe_read(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False)

def make_tertile_labels(series):
    # create tertiles: [0..33%] Low, (33..66%] Medium, (66..100%] High
    q1 = series.quantile(1/3)
    q2 = series.quantile(2/3)
    def f(v):
        if v <= q1:
            return "Low"
        if v <= q2:
            return "Medium"
        return "High"
    return series.apply(f), (q1, q2)

def eval_regression(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return r2, mae, rmse

# ---------- MAIN ----------
def run_phase(df_phase, phase_name):
    print(f"\n--- Processing phase: {phase_name} ---")
    df = df_phase.copy()

    # select features
    num_feats = NUMERIC_FEATURES.copy()
    if phase_name.lower() == "gas":
        num_feats = num_feats + GAS_ONLY_NUMERIC

    # ensure columns exist; fill missing numeric with median
    for c in num_feats:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].fillna(df[c].median())

    # categorical fill
    for c in CATEGORICAL_FEATURES:
        if c not in df.columns:
            df[c] = "None"
        df[c] = df[c].fillna("None").astype(str)

    # target
    y_reg = pd.to_numeric(df[COL_YIELD], errors="coerce")
    missing_y = y_reg.isna().sum()
    if missing_y > 0:
        print(f"Warning: {missing_y} rows have missing yield and will be dropped for modeling.")
    mask = y_reg.notna()
    df = df.loc[mask].reset_index(drop=True)
    y_reg = y_reg.loc[mask].reset_index(drop=True)

    # classification labels (tertiles)
    y_cls, (q1, q2) = make_tertile_labels(y_reg)
    print(f"Tertile cutoffs for {phase_name}: q1={q1:.4g}, q2={q2:.4g}")

    # Build feature matrix: numeric + one-hot categorical
    X_num = df[num_feats].reset_index(drop=True)
    X_cat = df[CATEGORICAL_FEATURES].reset_index(drop=True)

    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_cat_enc = enc.fit_transform(X_cat)
    feature_names_cat = list(enc.get_feature_names_out(CATEGORICAL_FEATURES))
    X = np.hstack([X_num.values, X_cat_enc])
    feature_names = num_feats + feature_names_cat
    print(f"Feature matrix shape for {phase_name}: {X.shape}  (num features: {len(feature_names)})")

    # Train/test split (75/25)
    X_train, X_test, y_train_reg, y_test_reg, y_train_cls, y_test_cls = train_test_split(
        X, y_reg.values, y_cls.values, test_size=0.25, random_state=RANDOM_STATE, stratify=y_cls
    )

    # Random Forest regression
    rf = RandomForestRegressor(**RF_PARAMS)
    rf.fit(X_train, y_train_reg)
    y_pred_rf = rf.predict(X_test)
    r2, mae, rmse = eval_regression(y_test_reg, y_pred_rf)
    print(f"[RF-{phase_name}] R2={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")

    # Save RF
    joblib.dump(rf, os.path.join(OUT_MODELS, f"rf_{phase_name.lower()}.joblib"))
    print(f"Saved RF model -> rf_{phase_name.lower()}.joblib")

    # Decision Tree classifier on tertile classes
    dt_params_local = DT_PARAMS.copy()
    dt_params_local["ccp_alpha"] = DT_CCP_ALPHA["liq" if phase_name.lower()=="liq" else "gas"]
    dt = DecisionTreeClassifier(**dt_params_local)
    dt.fit(X_train, y_train_cls)
    y_pred_dt = dt.predict(X_test)

    # Eval classification
    cm = confusion_matrix(y_test_cls, y_pred_dt, labels=["Low","Medium","High"])
    creport = classification_report(y_test_cls, y_pred_dt, labels=["Low","Medium","High"], zero_division=0)
    acc = (y_pred_dt == y_test_cls).mean()
    print(f"[DT-{phase_name}] Accuracy={acc:.4f}")
    print("Confusion matrix (rows: true, cols: pred):")
    print(cm)
    print("Classification report:\n", creport)

    # Save DT
    joblib.dump(dt, os.path.join(OUT_MODELS, f"dt_{phase_name.lower()}.joblib"))
    print(f"Saved DT model -> dt_{phase_name.lower()}.joblib")

    # Return performance summary
    return {
        "phase": phase_name,
        "rf_r2": r2, "rf_mae": mae, "rf_rmse": rmse,
        "dt_acc": acc,
        "tertile_q1": q1, "tertile_q2": q2,
        "n_features": X.shape[1],
        "n_train": X_train.shape[0],
        "n_test": X_test.shape[0]
    }

def main():
    df = safe_read(INPUT_IMPUTED)
    # Standardize headers (strip whitespace)
    df.columns = [c.strip() for c in df.columns]

    # --- dynamic column detection by keyword (robust to small header differences) ---
    def find_col(df, keywords):
        cols = list(df.columns)
        lowcols = [c.lower() for c in cols]
        # direct substring match
        for kw in keywords:
            kw_low = kw.lower()
            for i, c in enumerate(lowcols):
                if kw_low in c:
                    return cols[i]
        # token match fallback
        for i, c in enumerate(lowcols):
            for token in keywords:
                parts = token.lower().split()
                if all(p in c for p in parts):
                    return cols[i]
        return None

    guessed = {}
    guessed['COL_PHASE']        = find_col(df, ["liq", "gas", "phase"])
    guessed['COL_YIELD']        = find_col(df, ["yield", "total yield", "umol"])
    guessed['COL_BAND']         = find_col(df, ["band gap", "band", "bandgap"])
    guessed['COL_BET']          = find_col(df, ["bet surface", "bet"])
    guessed['COL_CALC_TEMP']    = find_col(df, ["calcination temperature", "calcination temp", "calcination"])
    guessed['COL_CALC_TIME']    = find_col(df, ["calcination time"])
    guessed['COL_REACT_TEMP']   = find_col(df, ["reaction temperature", "reaction temp"])
    guessed['COL_REACT_PRESS']  = find_col(df, ["reaction pressure", "pressure"])
    guessed['COL_H2OCO2']       = find_col(df, ["h2o", "co2", "h2o:co2"])
    guessed['COL_CAT_W']        = find_col(df, ["cat w", "cat", "cat w (g)"])
    guessed['COL_A'] = 'A' if 'A' in df.columns else ( find_col(df, [" x of a", "x of a", "a "]) or find_col(df, ["A"]) )
    guessed['COL_B'] = 'B' if 'B' in df.columns else ( find_col(df, [" x of b", "x of b", "b "]) or find_col(df, ["B"]) )
    guessed['COL_DOPE']         = find_col(df, ["dope", "dopant", "doepe"])
    guessed['COL_SYNTH']        = find_col(df, ["synthesis", "synthesis method"])
    guessed['COL_CRYST']        = find_col(df, ["crystal", "crystal structure", "stracture"])
    guessed['COL_LIGHT']        = find_col(df, ["light", "light type", "wavelength"])

    print("\nAuto-detected columns:")
    for k, v in guessed.items():
        print(f"  {k}: {v}")

    # Write these found names back into global variables
    for k, v in guessed.items():
        if v is None:
            raise RuntimeError(f"Column not found for {k}. Cannot continue.")
        globals()[k] = v

    # Normalize phase labels
    df["phase_norm"] = df[COL_PHASE].astype(str).str.lower().str.strip()
    df["phase_label"] = df["phase_norm"].apply(
        lambda s: "Liq" if s.startswith("liq") else ("Gas" if s.startswith("gas") else s)
    )

    summaries = []
    for phase_label in ["Liq", "Gas"]:
        df_phase = df[df["phase_label"] == phase_label].copy()
        if len(df_phase) == 0:
            print(f"No rows for phase {phase_label}, skipping.")
            continue
        summary = run_phase(df_phase, phase_label)
        summaries.append(summary)

    # Save summary CSV
    summary_df = pd.DataFrame(summaries)
    out_path = os.path.join(OUT_RESULTS, "model_performance.csv")
    summary_df.to_csv(out_path, index=False)
    print("\nSaved model performance summary →", out_path)
    print(summary_df)


if __name__ == "__main__":
    main()
