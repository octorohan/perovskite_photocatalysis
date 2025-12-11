# step3_retrain_log1p.py
"""
Retrain RandomForest (regression on log1p(y)) and DecisionTree (classification on tertiles).
Robust to column-name variations and index alignment issues.
Input: results/step2_clean.csv (preferred) or results/step2_imputed_corrected.csv (fallback).
Outputs:
 - models/rf_liq.joblib, models/rf_gas.joblib
 - models/dt_liq.joblib, models/dt_gas.joblib
 - results/model_performance_log1p.csv
 - results/feature_names_liq.json, results/feature_names_gas.json
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from math import expm1
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, confusion_matrix, classification_report

# ---------- CONFIG ----------
PROJECT_ROOT = r"C:\Users\HP\Desktop\winter_projects\perovskite"
INPUT_CLEAN = os.path.join(PROJECT_ROOT, "results", "step2_clean.csv")
INPUT_CORRECTED = os.path.join(PROJECT_ROOT, "results", "step2_imputed_corrected.csv")
OUT_MODELS = os.path.join(PROJECT_ROOT, "models")
OUT_RESULTS = os.path.join(PROJECT_ROOT, "results")

os.makedirs(OUT_MODELS, exist_ok=True)
os.makedirs(OUT_RESULTS, exist_ok=True)

# features (paper's most important)
NUMERIC_BASE = [
    "Bet Surface Area (m2g-1)",
    "Calcination Time (h)",
    "Calcination Temperature (C )",
    "cat W (g)",
    "Reaction Temperature (C )",
    "Reaction Pressure (bar)",
]
CATEGORICALS = ["A", "B", "Dope", "Synthesis Method of Perovskite", "cocatalyst", "Light Type"]
GAS_ONLY_NUMERIC = ["H2O:CO2"]

# model hyperparams
RF_PARAMS = {"n_estimators": 500, "max_features": 5, "random_state": 42, "n_jobs": -1}
DT_PARAMS = {"min_samples_split": 5, "random_state": 42}
DT_CCP_ALPHA = {"Liq": 0.015, "Gas": 0.010}

RANDOM_STATE = 42
TEST_SIZE = 0.25

# ---------- helpers ----------
def try_read_input():
    path = INPUT_CLEAN if os.path.exists(INPUT_CLEAN) else INPUT_CORRECTED
    if not os.path.exists(path):
        raise FileNotFoundError(f"No input found. Expected {INPUT_CLEAN} or {INPUT_CORRECTED}")
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    return df, path

def find_col(df, keywords):
    """Return first column name containing any keyword (case-insensitive)."""
    cols = list(df.columns)
    low = [c.lower() for c in cols]
    for kw in keywords:
        for i, c in enumerate(low):
            if kw.lower() in c:
                return cols[i]
    # fallback: try token matching
    for i, c in enumerate(low):
        for kw in keywords:
            parts = kw.lower().split()
            if all(p in c for p in parts):
                return cols[i]
    return None

def safe_median_impute(series):
    med = series.median()
    return series.fillna(med)

def build_num_feats(df_phase, detected_band_col, phase_label):
    num_feats = NUMERIC_BASE.copy()
    # include band gap if detected
    if detected_band_col and detected_band_col not in num_feats:
        num_feats.append(detected_band_col)
    # include gas-only numeric
    if phase_label.lower() == "gas":
        for g in GAS_ONLY_NUMERIC:
            if g not in df_phase.columns:
                df_phase[g] = np.nan
            if g not in num_feats:
                num_feats.append(g)
    # ensure numeric and median-impute
    for c in num_feats:
        if c not in df_phase.columns:
            df_phase[c] = np.nan
        df_phase[c] = pd.to_numeric(df_phase[c], errors="coerce")
        df_phase[c] = safe_median_impute(df_phase[c])
    return num_feats

def make_tertile_labels(y_series):
    q1 = y_series.quantile(1/3)
    q2 = y_series.quantile(2/3)
    def label(v):
        if v <= q1:
            return "Low"
        if v <= q2:
            return "Medium"
        return "High"
    return y_series.apply(label), (q1, q2)

# ---------- main phase routine ----------
def run_phase(df_phase, phase_label, yield_col, detected_band_col):
    print(f"\n--- Phase: {phase_label}  rows: {len(df_phase)} ---")
    df = df_phase.copy()

    # Build numeric features (median imputed)
    num_feats = build_num_feats(df, detected_band_col, phase_label)

    # Ensure categorical columns exist and are strings
    for c in CATEGORICALS:
        if c not in df.columns:
            df[c] = "None"
        df[c] = df[c].fillna("None").astype(str)

    # Target: numeric original yields (use positional indexing to avoid alignment issues)
    y_orig = pd.to_numeric(df[yield_col], errors="coerce")
    valid_idx = np.where(y_orig.notna().values)[0]
    if len(valid_idx) == 0:
        raise RuntimeError(f"No valid yields for phase {phase_label}")
    df = df.iloc[valid_idx].reset_index(drop=True)
    y_orig = y_orig.iloc[valid_idx].reset_index(drop=True)

    # classification labels on original y (tertiles)
    y_cls, (q1, q2) = make_tertile_labels(y_orig)
    print(f"Tertile cutoffs: q1={q1:.4g}, q2={q2:.4g}")

    # Build feature matrix
    X_num = df[num_feats].reset_index(drop=True)
    X_cat = df[CATEGORICALS].reset_index(drop=True)

    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_cat_enc = enc.fit_transform(X_cat)
    cat_names = list(enc.get_feature_names_out(CATEGORICALS))
    feature_names = num_feats + cat_names
    X = np.hstack([X_num.values, X_cat_enc])

    print(f"Feature matrix: X.shape = {X.shape} (n_features={len(feature_names)})")

    # Train/test split: stratify if possible (enough samples per class)
    stratify_arg = None
    try:
        # ensure each class has >=2 samples in train+test to allow stratify
        vc = pd.Series(y_cls).value_counts()
        if (vc >= 2).all() and len(vc) == 3:
            stratify_arg = y_cls.values
        else:
            stratify_arg = None
    except Exception:
        stratify_arg = None

    X_train, X_test, y_train_orig, y_test_orig, y_train_cls, y_test_cls = train_test_split(
        X, y_orig.values, y_cls.values, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify_arg
    )

    # Train RF on log1p(y_train)
    y_train_log = np.log1p(y_train_orig)
    rf = RandomForestRegressor(**RF_PARAMS)
    rf.fit(X_train, y_train_log)

    # Predict (log space) and invert
    y_pred_log = rf.predict(X_test)
    y_pred_orig = np.expm1(y_pred_log)

    # Metrics on original scale
    r2_orig = r2_score(y_test_orig, y_pred_orig)
    mae_orig = mean_absolute_error(y_test_orig, y_pred_orig)
    rmse_orig = float(np.sqrt(mean_squared_error(y_test_orig, y_pred_orig)))

    # log-space R2
    r2_log = r2_score(np.log1p(y_test_orig), y_pred_log)

    # Save RF
    rf_path = os.path.join(OUT_MODELS, f"rf_{phase_label.lower()}.joblib")
    joblib.dump(rf, rf_path)

    # Train Decision Tree classifier on tertiles
    ccp = DT_CCP_ALPHA.get(phase_label, 0.01)
    dt = DecisionTreeClassifier(min_samples_split=DT_PARAMS["min_samples_split"], ccp_alpha=ccp, random_state=DT_PARAMS["random_state"])
    dt.fit(X_train, y_train_cls)
    y_pred_dt = dt.predict(X_test)

    # Classification metrics
    cm = confusion_matrix(y_test_cls, y_pred_dt, labels=["Low","Medium","High"])
    creport = classification_report(y_test_cls, y_pred_dt, labels=["Low","Medium","High"], zero_division=0)

    dt_path = os.path.join(OUT_MODELS, f"dt_{phase_label.lower()}.joblib")
    joblib.dump(dt, dt_path)

    # Save feature names for later analysis
    feat_json = os.path.join(OUT_RESULTS, f"feature_names_{phase_label.lower()}.json")
    with open(feat_json, "w", encoding="utf-8") as f:
        json.dump(feature_names, f, ensure_ascii=False, indent=2)

    result = {
        "phase": phase_label,
        "n_rows": X.shape[0],
        "n_features": X.shape[1],
        "rf_r2_orig": float(r2_orig),
        "rf_mae_orig": float(mae_orig),
        "rf_rmse_orig": float(rmse_orig),
        "rf_r2_log": float(r2_log),
        "dt_accuracy": float((y_pred_dt == y_test_cls).mean()),
        "dt_confusion": cm.tolist(),
        "dt_class_report": creport,
        "rf_model": rf_path,
        "dt_model": dt_path,
        "feature_names_json": feat_json
    }
    return result

# ---------- MAIN ----------
def main():
    df, used_path = try_read_input()
    print("Using input:", used_path)

    # detect important columns
    phase_col = find_col(df, ["liq", "gas", "phase"])
    yield_col = find_col(df, ["yield", "umol", "µmol", "mmol", "gcat"])
    band_col = find_col(df, ["band gap", "band_gap", "band"])

    if phase_col is None or yield_col is None:
        raise RuntimeError("Could not detect phase or yield columns. Headers: " + ", ".join(df.columns[:20]))

    print("Detected phase column:", phase_col)
    print("Detected yield column:", yield_col)
    if band_col:
        print("Detected band gap column:", band_col)

    # standardize phase labels
    df["phase_norm"] = df[phase_col].astype(str).str.lower().str.strip()
    df["phase_label"] = df["phase_norm"].map(lambda s: "Liq" if str(s).startswith("liq") else ("Gas" if str(s).startswith("gas") else s))

    summaries = []
    for phase_label in ["Liq", "Gas"]:
        df_phase = df[df["phase_label"] == phase_label].copy()
        if df_phase.shape[0] == 0:
            print(f"No rows for phase {phase_label}, skipping.")
            continue
        try:
            res = run_phase(df_phase, phase_label, yield_col, band_col)
        except Exception as e:
            print(f"Error during processing phase {phase_label}: {e}")
            raise
        summaries.append(res)
        print(f"Saved RF -> {res['rf_model']}")
        print(f"Saved DT -> {res['dt_model']}")
        print(f"RF (orig units) R2={res['rf_r2_orig']:.4f}, MAE={res['rf_mae_orig']:.4f}, RMSE={res['rf_rmse_orig']:.4f}")
        print(f"RF (log-space) R2={res['rf_r2_log']:.4f}")
        print(f"DT accuracy: {res['dt_accuracy']}")
        print("DT confusion matrix:", res['dt_confusion'])

    perf_csv = os.path.join(OUT_RESULTS, "model_performance_log1p.csv")
    pd.DataFrame(summaries).to_csv(perf_csv, index=False)
    print("\nSaved performance summary ->", perf_csv)
    print(pd.DataFrame(summaries))

if __name__ == "__main__":
    main()
