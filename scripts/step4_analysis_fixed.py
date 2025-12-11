# step4_analysis_fixed.py
"""
Fixed Step 4:
- Uses cleaned dataset (step2_clean.csv) or fallback corrected file.
- Uses log1p(y) consistently for CV & ablation.
- Loads saved RF models.
- Sanitizes Windows filenames.
- Uses Agg backend to avoid Tkinter crashes.
"""

import os, re
import numpy as np
import pandas as pd
import joblib

# ***** IMPORTANT FIX: use non-GUI backend *****
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.inspection import PartialDependenceDisplay
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from math import expm1

# ---------- CONFIG ----------
PROJECT_ROOT = r"C:\Users\HP\Desktop\winter_projects\perovskite"
INPUT_CLEAN = os.path.join(PROJECT_ROOT, "results", "step2_clean.csv")
INPUT_CORRECTED = os.path.join(PROJECT_ROOT, "results", "step2_imputed_corrected.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUT_DIR = os.path.join(PROJECT_ROOT, "results")
FIG_DIR = os.path.join(PROJECT_ROOT, "figures")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

RF_PARAMS = {"n_estimators":500, "max_features":5, "random_state":42, "n_jobs":-1}
CV_FOLDS = 5
ABLATION_FEATURES = ["Synthesis Method of Perovskite","A","B","Band Gap*","cocatalyst"]

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


# ---------- helpers ----------
def try_read(path):
    encs = ["utf-8","cp1252","latin1"]
    for e in encs:
        try:
            return pd.read_csv(path, encoding=e, low_memory=False)
        except:
            pass
    return pd.read_csv(path, engine="python", encoding="latin1", low_memory=False)


def find_col(df, kws):
    for c in df.columns:
        cl = c.lower()
        for k in kws:
            if k.lower() in cl:
                return c
    return None


def sanitize(s):
    return re.sub(r'[<>:"/\\|?*]', "_", s).replace(" ", "_")


def build_features(df_phase, band_col, phase):
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

    # numeric impute
    for c in num_feats:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].fillna(df[c].median())

    # categorical
    for c in CATEGORICALS:
        if c not in df.columns:
            df[c] = "None"
        df[c] = df[c].fillna("None").astype(str)

    X_num = df[num_feats].reset_index(drop=True)
    X_cat = df[CATEGORICALS].reset_index(drop=True)

    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_cat_enc = enc.fit_transform(X_cat)
    cat_names = list(enc.get_feature_names_out(CATEGORICALS))

    X = np.hstack([X_num.values, X_cat_enc])
    feat_names = num_feats + cat_names
    return X, feat_names, enc


# ---------- MAIN ----------
def main():
    input_path = INPUT_CLEAN if os.path.exists(INPUT_CLEAN) else INPUT_CORRECTED
    print("Reading:", input_path)

    df = try_read(input_path)
    df.columns = [c.strip() for c in df.columns]

    col_phase = find_col(df, ["liq","gas","phase"])
    col_yield = find_col(df, ["yield","umol","mmol","gcat"])
    col_band = find_col(df, ["band gap","band"])

    print("Auto-detected columns mapping:")
    print(f" phase: {col_phase}")
    print(f" yield: {col_yield}")
    print(f" band: {col_band}")

    df["phase_norm"] = df[col_phase].astype(str).str.lower().str.strip()
    df["phase_label"] = df["phase_norm"].apply(
        lambda s: "Liq" if s.startswith("liq") else ("Gas" if s.startswith("gas") else s)
    )

    ab_rows = []

    for phase in ["Liq","Gas"]:
        df_phase = df[df["phase_label"] == phase].copy()
        if len(df_phase)==0:
            continue

        print(f"\nPhase {phase}: rows={len(df_phase)}")

        X, feat_names, enc = build_features(df_phase, col_band, phase)
        print(f"Phase {phase}: X shape = {X.shape}, n_features={len(feat_names)}")

        # Load RF model
        rf_path = os.path.join(MODELS_DIR, f"rf_{phase.lower()}.joblib")
        rf = joblib.load(rf_path)

        # 1) Feature importances
        imp = rf.feature_importances_
        fi_df = pd.DataFrame({"feature": feat_names, "importance": imp}).sort_values("importance", ascending=False)
        fi_csv = os.path.join(OUT_DIR, f"feature_importances_{phase.lower()}.csv")
        fi_df.to_csv(fi_csv, index=False)

        # plot
        top = fi_df.head(20).iloc[::-1]
        plt.figure(figsize=(8, max(4, len(top)*0.25)))
        plt.barh(top["feature"], top["importance"])
        plt.title(f"Feature importances (top 20) — {phase}")
        plt.tight_layout()
        fi_png = os.path.join(FIG_DIR, f"fi_{phase.lower()}.png")
        plt.savefig(fi_png, dpi=200)
        plt.close()

        print("Saved feature importances:", fi_csv, fi_png)

        # 2) PDP for top-3
        top3 = fi_df.head(3)["feature"].tolist()
        print("Top 3 features for", phase, ":", top3)

        for feat in top3:
            try:
                idx = feat_names.index(feat)
            except ValueError:
                print("Cannot find feature:", feat)
                continue
            safe = sanitize(f"pdp_{phase}_{feat}")
            outp = os.path.join(FIG_DIR, safe + ".png")
            try:
                PartialDependenceDisplay.from_estimator(rf, X, [idx], feature_names=feat_names)
                plt.title(f"PDP - {feat} ({phase})")
                plt.tight_layout()
                plt.savefig(outp, dpi=200)
                plt.close()
                print("Saved PDP:", outp)
            except Exception as e:
                print("PDP error:", feat, e)

        # 3) Baseline CV R2 (log space)
        y = pd.to_numeric(df_phase[col_yield], errors="coerce")
        mask = y.notna()
        y_log = np.log1p(y.loc[mask])
        X_cv = X[mask.values, :]

        rf_cv = RandomForestRegressor(**RF_PARAMS)
        cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
        scores = cross_val_score(rf_cv, X_cv, y_log.values, cv=cv, n_jobs=-1, scoring="r2")
        baseline = scores.mean()
        print(f"Baseline CV R2 (log-space) for {phase}: {baseline:.4f}")

        # 4) Ablation study
        for abl in ABLATION_FEATURES:
            keep_idx = [i for i,f in enumerate(feat_names) if abl not in f]
            X_red = X_cv[:, keep_idx]
            try:
                scores2 = cross_val_score(RandomForestRegressor(**RF_PARAMS),
                                          X_red, y_log.values, cv=cv, n_jobs=-1, scoring="r2")
                abl_r2 = scores2.mean()
            except Exception as e:
                abl_r2 = np.nan

            delta = abl_r2 - baseline if not np.isnan(abl_r2) else np.nan
            print(f"Ablation {abl}: CV R2 (log)={abl_r2:.4f}, delta={delta:.4f}")

            ab_rows.append({
                "phase": phase,
                "ablated": abl,
                "cv_r2_log": abl_r2,
                "delta_log": delta,
                "baseline_log": baseline
            })

    # Save ablation
    ab_df = pd.DataFrame(ab_rows)
    ab_csv = os.path.join(OUT_DIR, "ablation_summary.csv")
    ab_df.to_csv(ab_csv, index=False)
    print("\nSaved ablation summary ->", ab_csv)
    print(ab_df.head(20))


if __name__ == "__main__":
    main()
