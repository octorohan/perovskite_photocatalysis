# step5_parity_plots.py
"""
Generate parity plots (log and original), residuals, and residual vs top-feature scatter.
Saves images into figures/ and prints metric summaries.
"""

import os, sys
import numpy as np
import pandas as pd
import joblib
from math import expm1
# Use non-GUI backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = r"C:\Users\HP\Desktop\winter_projects\perovskite"
RESULTS = os.path.join(PROJECT_ROOT, "results")
FIGURES = os.path.join(PROJECT_ROOT, "figures")
MODELS = os.path.join(PROJECT_ROOT, "models")

INPUT_CLEAN = os.path.join(RESULTS, "step2_clean.csv")
INPUT_CORRECTED = os.path.join(RESULTS, "step2_imputed_corrected.csv")

NUMERIC_BASE = [
    "Bet Surface Area (m2g-1)",
    "Calcination Time (h)",
    "Calcination Temperature (C )",
    "cat W (g)",
    "Reaction Temperature (C )",
    "Reaction Pressure (bar)",
]
CATEGORICALS = ["A","B","Dope","Synthesis Method of Perovskite","cocatalyst","Light Type"]
GAS_ONLY = ["H2O:CO2"]

def try_read(path):
    encs = ["utf-8","cp1252","latin1"]
    for e in encs:
        try:
            return pd.read_csv(path, encoding=e, low_memory=False)
        except Exception:
            pass
    return pd.read_csv(path, engine="python", encoding="latin1", low_memory=False)

def find_col(df, kws):
    for c in df.columns:
        lc = c.lower()
        for k in kws:
            if k.lower() in lc:
                return c
    return None

def build_X(df_phase, band_col, phase):
    df = df_phase.copy()
    num_feats = NUMERIC_BASE.copy()
    if band_col and band_col not in num_feats:
        num_feats.append(band_col)
    if phase.lower() == "gas":
        for g in GAS_ONLY:
            if g not in df.columns:
                df[g] = np.nan
            if g not in num_feats:
                num_feats.append(g)
    # numeric impute median
    for c in num_feats:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].fillna(df[c].median())
    for c in CATEGORICALS:
        if c not in df.columns:
            df[c] = "None"
        df[c] = df[c].fillna("None").astype(str)
    # One-hot encode
    from sklearn.preprocessing import OneHotEncoder
    X_num = df[num_feats].reset_index(drop=True)
    X_cat = df[CATEGORICALS].reset_index(drop=True)
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_cat_enc = enc.fit_transform(X_cat)
    feat_names = num_feats + list(enc.get_feature_names_out(CATEGORICALS))
    X = np.hstack([X_num.values, X_cat_enc])
    return X, feat_names, df

def metrics(y_true, y_pred):
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return r2, mae, rmse

def safe_save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def run_phase(df_phase, phase, col_yield, band_col):
    model_path = os.path.join(MODELS, f"rf_{phase.lower()}.joblib")
    if not os.path.exists(model_path):
        print("Model not found:", model_path)
        return
    rf = joblib.load(model_path)
    X, feat_names, df = build_X(df_phase, band_col, phase)
    y = pd.to_numeric(df[col_yield], errors="coerce")
    mask = y.notna()
    if mask.sum() == 0:
        print("No yields for phase", phase)
        return
    Xm = X[mask.values, :]
    y_orig = y[mask].values
    # predict log-space
    y_pred_log = rf.predict(Xm)
    y_pred_orig = np.expm1(y_pred_log)
    # metrics
    r2o, mae, rmse = metrics(y_orig, y_pred_orig)
    r2log, _, _ = metrics(np.log1p(y_orig), y_pred_log)
    print(f"Phase {phase} — n={len(y_orig)} — R2(orig)={r2o:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}, R2(log)={r2log:.4f}")
    # parity plot (log)
    fig = plt.figure(figsize=(6,6))
    plt.scatter(np.log1p(y_orig), y_pred_log, alpha=0.6, s=12)
    mn = min(np.log1p(y_orig).min(), y_pred_log.min()); mx = max(np.log1p(y_orig).max(), y_pred_log.max())
    plt.plot([mn,mx],[mn,mx],"r--")
    plt.xlabel("Actual log1p(y)"); plt.ylabel("Predicted log1p(y)"); plt.title(f"Parity (log) — {phase}")
    safe_save(fig, os.path.join(FIGURES, f"parity_{phase.lower()}_log.png"))
    # parity plot (orig)
    fig = plt.figure(figsize=(6,6))
    plt.scatter(y_orig, y_pred_orig, alpha=0.6, s=12)
    mn = min(y_orig.min(), y_pred_orig.min()); mx = max(y_orig.max(), y_pred_orig.max())
    plt.plot([mn,mx],[mn,mx],"r--")
    plt.xscale("symlog") ; plt.yscale("symlog")
    plt.xlabel("Actual y (µmol/g/h)"); plt.ylabel("Predicted y (µmol/g/h)"); plt.title(f"Parity (orig, symlog) — {phase}")
    safe_save(fig, os.path.join(FIGURES, f"parity_{phase.lower()}_orig.png"))
    # residuals histogram
    res = y_orig - y_pred_orig
    fig = plt.figure(figsize=(6,4))
    plt.hist(res, bins=40)
    plt.xlabel("Residual (actual - pred)"); plt.title(f"Residuals — {phase}")
    safe_save(fig, os.path.join(FIGURES, f"residuals_{phase.lower()}.png"))
    # residual vs top feature (choose top numeric feature = first numeric)
    top_feat = feat_names[0]
    # find column index for top numeric
    x_vals = Xm[:,0]  # first numeric column corresponds to first numeric feat
    fig = plt.figure(figsize=(6,4))
    plt.scatter(x_vals, res, alpha=0.6, s=12)
    plt.axhline(0, color="r", linestyle="--")
    plt.xlabel(top_feat); plt.ylabel("Residual")
    plt.title(f"Residual vs {top_feat} — {phase}")
    safe_save(fig, os.path.join(FIGURES, f"resid_vs_{phase.lower()}_{top_feat.replace(' ','_')}.png"))

def main():
    path = INPUT_CLEAN if os.path.exists(INPUT_CLEAN) else INPUT_CORRECTED
    if not os.path.exists(path):
        raise FileNotFoundError("No input found at " + INPUT_CLEAN + " or " + INPUT_CORRECTED)
    df = try_read(path)
    df.columns = [c.strip() for c in df.columns]
    col_phase = find_col(df, ["liq","gas","phase"])
    col_yield = find_col(df, ["yield","umol","mmol","gcat"])
    col_band = find_col(df, ["band gap","band"])
    print("Using input:", path)
    print("Detected:", col_phase, col_yield, col_band)
    df["phase_norm"] = df[col_phase].astype(str).str.lower().str.strip()
    df["phase_label"] = df["phase_norm"].map(lambda s: "Liq" if str(s).startswith("liq") else ("Gas" if str(s).startswith("gas") else s))
    for phase in ["Liq","Gas"]:
        df_phase = df[df["phase_label"]==phase].copy()
        if df_phase.shape[0]==0:
            continue
        run_phase(df_phase, phase, col_yield, col_band)
    print("Saved parity & residual plots to", FIGURES)

if __name__ == "__main__":
    main()
