# step4_analysis.py
"""
Step 4 — Feature importance, PDPs, Ablation study (robust)
Saves CSVs & PNGs under results/ and figures/
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.inspection import PartialDependenceDisplay
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score

# ---------- CONFIG ----------
PROJECT_ROOT = r"C:\Users\HP\Desktop\winter_projects\perovskite"
INPUT_IMPUTED = os.path.join(PROJECT_ROOT, "results", "step2_imputed.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUT_DIR = os.path.join(PROJECT_ROOT, "results")
FIG_DIR = os.path.join(PROJECT_ROOT, "figures")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# The "most important" feature lists we used earlier
NUMERIC_FEATURES_BASE = [
    "Bet Surface Area (m2g-1)",
    "Calcination Time (h)",
    "Calcination Temperature (C )",
    "cat W (g)",
    "Reaction Temperature (C )",
    "Reaction Pressure (bar)",
    # band gap will be detected and appended if present
]
CATEGORICAL_FEATURES = ["A", "B", "Dope", "Synthesis Method of Perovskite", "cocatalyst", "Light Type"]
GAS_ONLY_NUMERIC = ["H2O:CO2"]

RF_PARAMS = {"n_estimators": 500, "max_features": 5, "random_state": 42, "n_jobs": -1}
CV_FOLDS = 5
ABLATION_FEATURES = ["Synthesis Method of Perovskite", "A", "B", "Band Gap*", "cocatalyst"]

# ---------- Helpers ----------
def safe_read_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path, low_memory=False)

def find_column(df, keywords):
    """Return first column name that contains any keyword (case-insensitive)."""
    cols = list(df.columns)
    low = [c.lower() for c in cols]
    for kw in keywords:
        kwl = kw.lower()
        for i, c in enumerate(low):
            if kwl in c:
                return cols[i]
    # try token match
    for i, c in enumerate(low):
        for kw in keywords:
            parts = kw.lower().split()
            if all(p in c for p in parts):
                return cols[i]
    return None

def build_feature_matrix(df_phase, phase_label, col_names):
    """Recreate feature matrix X, feature_names, and y (numeric) for a phase (Liq/Gas)."""
    # detect numeric features list (append band gap if available)
    num_feats = NUMERIC_FEATURES_BASE.copy()
    band_col = col_names.get("band")
    if band_col:
        # include the detected band gap column with exactly that column name
        if band_col not in num_feats:
            num_feats.append(band_col)
    if phase_label.lower() == "gas":
        for g in GAS_ONLY_NUMERIC:
            if g not in df_phase.columns:
                # if not present, create NA column
                df_phase[g] = np.nan
            if g not in num_feats:
                num_feats.append(g)

    # ensure numeric exists and coerce
    for c in num_feats:
        if c not in df_phase.columns:
            df_phase[c] = np.nan
        df_phase[c] = pd.to_numeric(df_phase[c], errors="coerce")
        # median impute numeric features (same as Step2/Step3)
        med = df_phase[c].median()
        df_phase[c] = df_phase[c].fillna(med)

    # categorical features ensure present
    for c in CATEGORICAL_FEATURES:
        if c not in df_phase.columns:
            df_phase[c] = "None"
        df_phase[c] = df_phase[c].fillna("None").astype(str)

    # target
    yield_col = col_names.get("yield")
    if yield_col is None:
        raise RuntimeError("Yield column not found (unexpected).")
    y = pd.to_numeric(df_phase[yield_col], errors="coerce")
    mask = y.notna()
    df_phase = df_phase.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)

    # Build features
    X_num = df_phase[num_feats].reset_index(drop=True)
    X_cat = df_phase[CATEGORICAL_FEATURES].reset_index(drop=True)

    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_cat_enc = enc.fit_transform(X_cat)
    feat_names_cat = list(enc.get_feature_names_out(CATEGORICAL_FEATURES))

    X = np.hstack([X_num.values, X_cat_enc])
    feature_names = num_feats + feat_names_cat

    return X, feature_names, y, enc

def save_feature_importances(feature_names, importances, phase_label):
    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    df = df.sort_values("importance", ascending=False)
    out_csv = os.path.join(OUT_DIR, f"feature_importances_{phase_label.lower()}.csv")
    df.to_csv(out_csv, index=False)
    # plot top 20
    top = df.head(20).iloc[::-1]
    plt.figure(figsize=(8, max(5, len(top) * 0.25)))
    plt.barh(top["feature"], top["importance"])
    plt.title(f"Feature importances (top 20) — {phase_label}")
    plt.tight_layout()
    out_fig = os.path.join(FIG_DIR, f"fi_{phase_label.lower()}.png")
    plt.savefig(out_fig, dpi=200)
    plt.close()
    return out_csv, out_fig, df

def plot_pdp(model, X_full, feature_names, feat_idx, outpath, phase_label):
    try:
        # PartialDependenceDisplay handles numpy arrays and feature names
        PartialDependenceDisplay.from_estimator(model, X_full, [feat_idx], feature_names=feature_names)
        plt.title(f"PDP: {feature_names[feat_idx]} ({phase_label})")
        plt.tight_layout()
        plt.savefig(outpath, dpi=200)
        plt.close()
        return True
    except Exception as e:
        print("PDP error for", feature_names[feat_idx], ":", e)
        return False

def ablation_cv_r2(X_full, y_full, feature_names, abl_tokens):
    """Drop columns whose feature name contains any token in abl_tokens (case-sensitive-ish),
       return mean CV R2 (KFold)."""
    # tokens may match OHE feature names like "Synthesis Method of Perovskite_Pechini"
    keep_idx = [i for i, fn in enumerate(feature_names) if not any(tok in fn for tok in abl_tokens)]
    if len(keep_idx) == 0:
        return np.nan
    X_reduced = X_full[:, keep_idx]
    rf = RandomForestRegressor(**RF_PARAMS)
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    scores = cross_val_score(rf, X_reduced, y_full.values.ravel(), cv=cv, scoring="r2", n_jobs=-1)
    return scores.mean()

# ---------- MAIN ----------
def main():
    print("Loading imputed data:", INPUT_IMPUTED)
    df = safe_read_csv(INPUT_IMPUTED)
    df.columns = [c.strip() for c in df.columns]

    # Auto-detect important column names
    col_names = {}
    col_names["phase"] = find_column(df, ["liq", "gas", "phase"])
    col_names["yield"] = find_column(df, ["total yield", "yield", "umol", "gcat"])
    # band gap variations
    col_names["band"] = find_column(df, ["band gap", "band_gap", "band"])
    # other important columns
    col_names["bet"] = find_column(df, ["bet surface", "bet"])
    col_names["calc_temp"] = find_column(df, ["calcination temperature", "calcination temp", "calcination"])
    col_names["calc_time"] = find_column(df, ["calcination time"])
    col_names["react_temp"] = find_column(df, ["reaction temperature", "reaction temp"])
    col_names["react_press"] = find_column(df, ["reaction pressure", "pressure"])
    col_names["h2oco2"] = find_column(df, ["h2o:co2", "h2o", "h2o co2"])
    col_names["cat_w"] = find_column(df, ["cat w", "cat_w", "cat w (g)"])
    # Print mapping
    print("Auto-detected columns mapping:")
    for k, v in col_names.items():
        print(f"  {k}: {v}")

    if col_names["phase"] is None or col_names["yield"] is None:
        raise RuntimeError("Could not auto-detect required columns 'phase' or 'yield'. Aborting.")

    # normalize phase and label
    df["phase_norm"] = df[col_names["phase"]].astype(str).str.lower().str.strip()
    df["phase_label"] = df["phase_norm"].apply(lambda s: "Liq" if str(s).startswith("liq") else ("Gas" if str(s).startswith("gas") else s))

    ablation_records = []

    for phase in ["Liq", "Gas"]:
        df_phase = df[df["phase_label"] == phase].copy()
        if df_phase.shape[0] == 0:
            print(f"No rows for phase {phase}; skipping.")
            continue

        print(f"\nProcessing phase {phase} — rows: {df_phase.shape[0]}")
        # Build feature matrix
        # update NUMERIC_FEATURES_BASE to include detected band column name if present
        # build_feature_matrix expects col_names mapping with 'band' key set to actual header if any
        X_full, feat_names, y_full, encoder = build_feature_matrix(df_phase, phase, col_names)
        print(f"Phase {phase}: X shape = {X_full.shape}, n_features = {len(feat_names)}")

        # load RF model
        model_path = os.path.join(MODELS_DIR, f"rf_{phase.lower()}.joblib")
        if not os.path.exists(model_path):
            print("RF model not found for", phase, "expected at", model_path)
            continue
        rf = joblib.load(model_path)

        # feature importances
        importances = rf.feature_importances_
        csv_path, fig_path, fi_df = save_feature_importances(feat_names, importances, phase)
        print("Saved feature importances:", csv_path, fig_path)
        top3 = fi_df.head(3)["feature"].tolist()
        print(f"Top 3 features for {phase}: {top3}")

        # PDPs for top 3
        for feat in top3:
            try:
                idx = feat_names.index(feat)
            except ValueError:
                print("Top feature not found in feature list:", feat)
                continue
            outp = os.path.join(FIG_DIR, f"pdp_{phase.lower()}_{feat.replace(' ','_').replace('/', '-')}.png")
            ok = plot_pdp(rf, X_full, feat_names, idx, outp, phase)
            if ok:
                print("Saved PDP:", outp)

        # baseline CV R2
        rf_cv = RandomForestRegressor(**RF_PARAMS)
        cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
        baseline_scores = cross_val_score(rf_cv, X_full, y_full.values.ravel(), cv=cv, scoring="r2", n_jobs=-1)
        baseline_cv = baseline_scores.mean()
        print(f"Baseline CV R2 (full features) for {phase}: {baseline_cv:.4f}")

        # Ablation: for each token in ABLATION_FEATURES, drop any feature containing that token
        for abl in ABLATION_FEATURES:
            tokens = [abl]  # simple token
            try:
                abl_r2 = ablation_cv_r2(X_full, y_full, feat_names, tokens)
            except Exception as e:
                print("Ablation error for", abl, e)
                abl_r2 = np.nan
            delta = abl_r2 - baseline_cv if (not np.isnan(abl_r2) and not np.isnan(baseline_cv)) else np.nan
            print(f"Ablation {abl}: CV_R2={abl_r2:.4f}  delta_vs_baseline={delta:.4f}")
            ablation_records.append({
                "phase": phase,
                "ablated_feature": abl,
                "cv_r2": float(abl_r2) if not np.isnan(abl_r2) else None,
                "delta_vs_baseline": float(delta) if not np.isnan(delta) else None,
                "baseline_cv_r2": float(baseline_cv)
            })

    # save ablation summary
    ab_df = pd.DataFrame(ablation_records)
    ab_path = os.path.join(OUT_DIR, "ablation_summary.csv")
    ab_df.to_csv(ab_path, index=False)
    print("\nSaved ablation summary ->", ab_path)
    print(ab_df.head(30))


if __name__ == "__main__":
    main()
