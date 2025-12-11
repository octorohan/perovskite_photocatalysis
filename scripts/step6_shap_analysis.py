# step6_shap_analysis.py
"""
Compute SHAP explanations for RF models (liq + gas).
Generates summary bar, beeswarm, and dependence plots for top 3 features.
"""
import os, sys
import numpy as np
import pandas as pd
import joblib
# non-GUI backend
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
        except:
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
    if phase.lower()=="gas":
        for g in GAS_ONLY:
            if g not in df.columns:
                df[g]=np.nan
            if g not in num_feats:
                num_feats.append(g)
    for c in num_feats:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].fillna(df[c].median())
    for c in CATEGORICALS:
        if c not in df.columns:
            df[c] = "None"
        df[c] = df[c].fillna("None").astype(str)
    from sklearn.preprocessing import OneHotEncoder
    X_num = df[num_feats].reset_index(drop=True)
    X_cat = df[CATEGORICALS].reset_index(drop=True)
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_cat_enc = enc.fit_transform(X_cat)
    feat_names = num_feats + list(enc.get_feature_names_out(CATEGORICALS))
    X = np.hstack([X_num.values, X_cat_enc])
    return X, feat_names, df

def ensure_shap():
    try:
        import shap
        return shap
    except Exception as e:
        print("shap not installed. Install with: pip install shap")
        raise

def main():
    shap = ensure_shap()
    path = INPUT_CLEAN if os.path.exists(INPUT_CLEAN) else INPUT_CORRECTED
    df = try_read(path)
    df.columns = [c.strip() for c in df.columns]
    col_phase = find_col(df, ["liq","gas","phase"])
    col_yield = find_col(df, ["yield","umol","mmol","gcat"])
    col_band = find_col(df, ["band gap","band"])
    df["phase_norm"] = df[col_phase].astype(str).str.lower().str.strip()
    df["phase_label"] = df["phase_norm"].apply(lambda s: "Liq" if s.startswith("liq") else ("Gas" if s.startswith("gas") else s))
    for phase in ["Liq","Gas"]:
        df_phase = df[df["phase_label"]==phase].copy()
        if df_phase.shape[0]==0:
            continue
        model_path = os.path.join(MODELS, f"rf_{phase.lower()}.joblib")
        if not os.path.exists(model_path):
            print("Model missing:", model_path); continue
        rf = joblib.load(model_path)
        X, feat_names, df_used = build_X(df_phase, col_band, phase)
        # mask valid yields
        y = pd.to_numeric(df_used[col_yield], errors="coerce")
        mask = y.notna()
        Xm = X[mask.values, :]
        if Xm.shape[0] == 0:
            continue
        expl = shap.TreeExplainer(rf)
        shap_values = expl.shap_values(Xm)
        # shap_values -> for tree ensemble may be array-like (list) if multiclass; for regression it's 2D
        if isinstance(shap_values, list):
            sv = shap_values[0]
        else:
            sv = shap_values
        # summary bar
        plt.figure(figsize=(6,4))
        shap.summary_plot(sv, features=Xm, feature_names=feat_names, show=False, plot_type="bar")
        out = os.path.join(FIGURES, f"shap_bar_{phase.lower()}.png")
        plt.savefig(out, bbox_inches="tight", dpi=200)
        plt.close()
        # beeswarm
        plt.figure(figsize=(6,6))
        shap.summary_plot(sv, features=Xm, feature_names=feat_names, show=False)
        out2 = os.path.join(FIGURES, f"shap_beeswarm_{phase.lower()}.png")
        plt.savefig(out2, bbox_inches="tight", dpi=200)
        plt.close()
        # dependence for top 3 features
        importances = np.abs(sv).mean(axis=0)
        top_idx = np.argsort(importances)[-3:][::-1]
        for i in top_idx:
            feat = feat_names[i]
            plt.figure(figsize=(6,4))
            try:
                shap.dependence_plot(i, sv, Xm, feature_names=feat_names, show=False)
                out3 = os.path.join(FIGURES, f"shap_dep_{phase.lower()}_{feat.replace(' ','_')}.png")
                plt.savefig(out3, bbox_inches="tight", dpi=200)
                plt.close()
            except Exception as e:
                print("shap dependence failed for", feat, e)
        print("Saved SHAP for", phase, "->", out, out2)
    print("SHAP plots saved in", FIGURES)

if __name__ == "__main__":
    main()
