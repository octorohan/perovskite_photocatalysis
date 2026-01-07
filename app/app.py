import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ===============================
# CONFIG
# ===============================
st.set_page_config(
    page_title="Perovskite Yield Predictor",
    layout="wide"
)

ROOT = Path(__file__).resolve().parents[1]

PIPE_LIQ = ROOT / "models" / "rf_liq_pipeline.joblib"
PIPE_GAS = ROOT / "models" / "rf_gas_pipeline.joblib"
DATA_PATH = ROOT / "results" / "step2_clean.csv"

RMSE_LOG_LIQ = 0.35   # safe stored value
RMSE_LOG_GAS = 0.50   # safe stored value

# ===============================
# LOAD
# ===============================
@st.cache_resource
def load_pipeline(path):
    return joblib.load(path)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, low_memory=False)

pipe_liq = load_pipeline(PIPE_LIQ)
pipe_gas = load_pipeline(PIPE_GAS)
df = load_data()

# ===============================
# UTILS
# ===============================
def predict_with_uncertainty(pipe, X, rmse_log):
    y_log = pipe.predict(X)[0]
    y = np.expm1(y_log)
    lo = np.expm1(y_log - rmse_log)
    hi = np.expm1(y_log + rmse_log)
    return y, lo, hi

def select_box(label, col, key):
    return st.selectbox(
        label,
        sorted(df[col].dropna().unique()),
        key=key
    )

def num_input(label, col, default, key):
    return st.number_input(
        label,
        value=float(default),
        key=key
    )

# ===============================
# UI
# ===============================
st.title("Perovskite Photocatalytic Yield Predictor")
st.caption("Liquid & Gas phase | Random Forest (log1p)")

tabs = st.tabs(["Liquid Phase", "Gas Phase"])

# ============================================================
# LIQUID TAB
# ============================================================
with tabs[0]:
    st.subheader("Liquid Phase Prediction")

    c1, c2, c3 = st.columns(3)

    with c1:
        A = select_box("A-site", "A", "liq_A")
        B = select_box("B-site", "B", "liq_B")
        synth = select_box("Synthesis Method", "Synthesis Method of Perovskite", "liq_syn")

    with c2:
        coco = select_box("Cocatalyst", "cocatalyst", "liq_coco")
        light = select_box("Light Type", "Light Type", "liq_light")
        band = num_input("Band Gap (eV)", "Band Gap*", 3.2, "liq_band")

    with c3:
        bet = num_input("BET Surface Area (m²/g)", "Bet Surface Area (m2g-1)", 10.0, "liq_bet")
        calc_t = num_input("Calcination Time (h)", "Calcination Time (h)", 6.0, "liq_calc")
        cat_w = num_input("Catalyst Mass (g)", "cat W (g)", 0.05, "liq_catw")

    X_liq = pd.DataFrame([{
        "A": A,
        "B": B,
        "Synthesis Method of Perovskite": synth,
        "cocatalyst": coco,
        "Light Type": light,
        "Band Gap*": band,
        "Bet Surface Area (m2g-1)": bet,
        "Calcination Time (h)": calc_t,
        "cat W (g)": cat_w,
        "Reaction Temperature (C )": 25.0
    }])

    if st.button("Predict Liquid Yield"):
        mean, lo, hi = predict_with_uncertainty(
            pipe_liq, X_liq, RMSE_LOG_LIQ
        )

        st.success(f"**Predicted Yield:** {mean:.2f} μmol g⁻¹ h⁻¹")
        st.info(f"**Likely Range:** {lo:.2f} – {hi:.2f} μmol g⁻¹ h⁻¹")

        st.caption(
            "Uncertainty range estimated from cross-validated log-space RMSE."
        )

# ============================================================
# GAS TAB
# ============================================================
with tabs[1]:
    st.subheader("Gas Phase Prediction")

    c1, c2, c3 = st.columns(3)

    with c1:
        A = select_box("A-site", "A", "gas_A")
        B = select_box("B-site", "B", "gas_B")
        synth = select_box("Synthesis Method", "Synthesis Method of Perovskite", "gas_syn")

    with c2:
        light = select_box("Light Type", "Light Type", "gas_light")
        band = num_input("Band Gap (eV)", "Band Gap*", 3.2, "gas_band")
        bet = num_input("BET Surface Area (m²/g)", "Bet Surface Area (m2g-1)", 10.0, "gas_bet")

    with c3:
        calc_t = num_input("Calcination Time (h)", "Calcination Time (h)", 6.0, "gas_calc")
        cat_w = num_input("Catalyst Mass (g)", "cat W (g)", 0.05, "gas_catw")
        react_t = num_input("Reaction Temperature (°C)", "Reaction Temperature (C )", 25.0, "gas_rt")

    X_gas = pd.DataFrame([{
        "A": A,
        "B": B,
        "Synthesis Method of Perovskite": synth,
        "cocatalyst": "none",   # ← REQUIRED dummy column
        "Light Type": light,
        "Band Gap*": band,
        "Bet Surface Area (m2g-1)": bet,
        "Calcination Time (h)": calc_t,
        "cat W (g)": cat_w,
        "Reaction Temperature (C )": react_t
    }])

    if st.button("Predict Gas Yield"):
        mean, lo, hi = predict_with_uncertainty(
            pipe_gas, X_gas, RMSE_LOG_GAS
        )

        st.success(f"**Predicted Yield:** {mean:.2f} μmol g⁻¹ h⁻¹")
        st.info(f"**Likely Range:** {lo:.2f} – {hi:.2f} μmol g⁻¹ h⁻¹")

        st.caption(
            "Gas-phase predictions are inherently noisier; uncertainty reflects this."
        )

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.caption(
    "Models trained on experimentally reported perovskite photocatalysis data. "
    "Predictions are intended for exploratory and research use only."
)
