"""
Step 2 — Imputation & Band-gap Linear Regression
Implements the imputation rules agreed:
- Mode fill for Reaction Temperature & Reaction Pressure (grouped by Liq or Gas)
- Calcination Temperature missing -> 25 C
- BET: fill with mean for same Perovskite & Crystal stracture
- H2O:CO2 (gas): fill by average from same Light Type AND similar cat W (±10%)
- Band Gap: train LinearRegression on rows with band gap and predict missing ones
Saves:
 - results/step2_imputed.csv
 - models/bandgap_lr.joblib
 - logs/step2_log.txt
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# ----------------------------
# CONFIG (edit if needed)
# ----------------------------
PROJECT_ROOT = r"C:\Users\HP\Desktop\winter_projects\perovskite"
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "perovskite_full.csv")
OUTPUT_IMPUTED = os.path.join(PROJECT_ROOT, "results", "step2_imputed.csv")
OUTPUT_MODEL = os.path.join(PROJECT_ROOT, "models", "bandgap_lr.joblib")
LOG_PATH = os.path.join(PROJECT_ROOT, "logs", "step2_log.txt")

# Column name mapping (exact names you provided)
COL_PHASE = "Liq or Gas"
COL_YIELD = "total yield  (μmol  gcat-1 h-1)"
COL_BAND = "Band Gap*"
COL_BET = "Bet Surface Area (m2g-1)"
COL_CALC_TEMP = "Calcination Temperature (C )"
COL_CALC_TIME = "Calcination Time (h)"
COL_REACT_TEMP = "Reaction Temperature (C )"
COL_REACT_PRESS = "Reaction Pressure (bar)"
COL_H2OCO2 = "H2O:CO2"
COL_CAT_W = "cat W (g)"
COL_A = "A"
COL_B = "B"
COL_DOPE = "Dope"
COL_SYNTH = "Synthesis Method of Perovskite"
COL_CRYST = "Crystal stracture"
COL_LIGHT = "Light Type"

NUMERIC_COLUMNS_TO_USE_FOR_BANDGAP = [
    "x of A", "x of A1/B1", "x of B", "x of X",
    COL_BET, COL_CALC_TEMP, COL_CALC_TIME, COL_CAT_W,
    COL_REACT_TEMP, COL_REACT_PRESS, "wavelenght"  # wavelength numeric if present
]

CATEGORICAL_COLUMNS_TO_USE_FOR_BANDGAP = [
    COL_A, COL_B, COL_DOPE, COL_SYNTH, COL_CRYST, COL_LIGHT
]

# tolerance for "similar" cat weight (±10%)
CAT_W_TOLERANCE = 0.10

# ----------------------------
# Helpers
# ----------------------------
def safe_read_csv(path):
    """
    Robust CSV reader that:
    - tries utf-8 / cp1252 / latin1 encodings
    - attempts delimiter sniffing with csv.Sniffer
    - falls back to trying common separators [',',';','\\t']
    - uses engine='python' when C engine fails due to tokenization
    Returns a pandas DataFrame.
    """
    import csv
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    encodings = ["utf-8", "cp1252", "latin1"]
    tried = []
    last_err = None

    # Helper: attempt to read with given encoding and sep, engine options
    def try_read(enc, sep=None, engine="c", **kwargs):
        try:
            if sep is None:
                return pd.read_csv(path, encoding=enc, low_memory=False, engine=engine, **kwargs)
            else:
                return pd.read_csv(path, encoding=enc, sep=sep, low_memory=False, engine=engine, **kwargs)
        except Exception as e:
            raise e

    # First, try to sniff delimiter using a small sample and csv.Sniffer
    sniff_sep = None
    try:
        with open(path, "rb") as f:
            raw_sample = f.read(8192)
        # Try decodings for sniffing
        for enc in encodings:
            try:
                sample = raw_sample.decode(enc, errors="replace")
                sniffer = csv.Sniffer()
                dialect = sniffer.sniff(sample)
                sniff_sep = dialect.delimiter
                print(f"Sniffer detected delimiter '{sniff_sep}' using encoding {enc}")
                break
            except Exception:
                continue
    except Exception:
        sniff_sep = None

    # Try combos: encoding -> (sniff_sep, common_seps) -> engine 'c' then 'python'
    common_seps = [",", ";", "\t", "|"]
    for enc in encodings:
        # try sniffed sep first (if any)
        seps_to_try = []
        if sniff_sep:
            seps_to_try.append(sniff_sep)
        seps_to_try.extend([s for s in common_seps if s != sniff_sep])
        for sep in seps_to_try:
            tried.append((enc, sep, "c"))
            try:
                df = try_read(enc, sep=sep, engine="c")
                print(f"Read CSV using encoding={enc}, sep='{sep}', engine=c")
                return df
            except Exception as e_c:
                # try python engine which is more tolerant of irregular rows
                tried.append((enc, sep, "python"))
                try:
                    df = try_read(enc, sep=sep, engine="python")
                    print(f"Read CSV using encoding={enc}, sep='{sep}', engine=python")
                    return df
                except Exception as e_py:
                    last_err = e_py
                    # continue to next sep/encoding

    # As a last resort, try reading with latin1 and engine=python with quoting relaxed
    try:
        df = pd.read_csv(path, encoding="latin1", sep=None, engine="python", quoting=csv.QUOTE_NONE, error_bad_lines=False)
        print("Fallback read: latin1, engine=python, quoting=QUOTE_NONE (with error_bad_lines suppressed)")
        return df
    except Exception:
        pass

    # If we reach here, collect diagnostics and raise a helpful error
    diag = {
        "tried": tried,
        "last_exception": str(last_err)
    }
    raise RuntimeError(f"Failed to read CSV. Diagnostics: {diag}")

def write_log(lines):
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote log -> {LOG_PATH}")

def numeric_col(series):
    # coerce to numeric, preserving NaNs
    return pd.to_numeric(series, errors="coerce")

# ----------------------------
# Main
# ----------------------------
def main():
    os.makedirs(os.path.join(PROJECT_ROOT, "results"), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "models"), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "logs"), exist_ok=True)

    log_lines = []
    df = safe_read_csv(DATA_PATH)
    n_rows = len(df)
    print(f"Loaded {DATA_PATH} ({n_rows} rows)")
    log_lines.append(f"Loaded {DATA_PATH} ({n_rows} rows)")

    # Standardize some column names (strip)
    df.columns = [c.strip() for c in df.columns]

    # Quick missingness summary for target columns
    cols_of_interest = [COL_PHASE, COL_YIELD, COL_BAND, COL_BET, COL_CALC_TEMP, COL_CALC_TIME,
                        COL_REACT_TEMP, COL_REACT_PRESS, COL_H2OCO2, COL_CAT_W]
    present = {c: (c in df.columns) for c in cols_of_interest}
    log_lines.append("Column presence:")
    for c, exists in present.items():
        log_lines.append(f"  {c}: {'YES' if exists else 'MISSING'}")

    # Count initial missingness
    missing_before = {c: int(df[c].isna().sum()) if c in df.columns else None for c in cols_of_interest}
    log_lines.append("Missing counts before imputation:")
    for k, v in missing_before.items():
        log_lines.append(f"  {k}: {v}")

    # ----------------------------
    # Rule 1: Mode fill for Reaction Temperature & Pressure grouped by Phase
    # ----------------------------
    if COL_REACT_TEMP in df.columns and COL_REACT_PRESS in df.columns and COL_PHASE in df.columns:
        # make a copy
        df[COL_REACT_TEMP] = numeric_col(df[COL_REACT_TEMP])
        df[COL_REACT_PRESS] = numeric_col(df[COL_REACT_PRESS])
        phases = df[COL_PHASE].astype(str).unique()
        filled_rt = 0
        filled_rp = 0
        for ph in phases:
            mask_ph = df[COL_PHASE].astype(str) == ph
            # mode for Reaction Temperature
            try:
                mode_temp = df.loc[mask_ph, COL_REACT_TEMP].mode().iloc[0]
            except Exception:
                mode_temp = np.nan
            # mode for Reaction Pressure
            try:
                mode_press = df.loc[mask_ph, COL_REACT_PRESS].mode().iloc[0]
            except Exception:
                mode_press = np.nan

            # fill only where missing and phase matches
            fill_mask_temp = mask_ph & df[COL_REACT_TEMP].isna()
            fill_mask_press = mask_ph & df[COL_REACT_PRESS].isna()
            df.loc[fill_mask_temp, COL_REACT_TEMP] = mode_temp
            df.loc[fill_mask_press, COL_REACT_PRESS] = mode_press
            filled_rt += int(fill_mask_temp.sum())
            filled_rp += int(fill_mask_press.sum())

        log_lines.append(f"Filled Reaction Temperature with mode by phase: {filled_rt} values filled")
        log_lines.append(f"Filled Reaction Pressure with mode by phase: {filled_rp} values filled")
        print(f"Reaction temp filled: {filled_rt}, Reaction pressure filled: {filled_rp}")
    else:
        log_lines.append("Reaction Temperature or Pressure or Phase column missing; skipped mode fill.")

    # ----------------------------
    # Rule 2: Missing Calcination Temperature -> 25 C (no calcination)
    # ----------------------------
    if COL_CALC_TEMP in df.columns:
        df[COL_CALC_TEMP] = numeric_col(df[COL_CALC_TEMP])
        missing_calc_before = df[COL_CALC_TEMP].isna().sum()
        df[COL_CALC_TEMP] = df[COL_CALC_TEMP].fillna(25.0)
        missing_calc_after = df[COL_CALC_TEMP].isna().sum()
        log_lines.append(f"Calcination Temperature: filled missing with 25C. Before={missing_calc_before}, After={missing_calc_after}")
        print(f"Calcination temp filled: {missing_calc_before - missing_calc_after}")
    else:
        log_lines.append("Calcination Temperature column missing; skipped.")

    # ----------------------------
    # Rule 3: BET fill by mean for same Perovskite & Crystal structure
    # ----------------------------
    if COL_BET in df.columns and "Perovskite" in df.columns and COL_CRYST in df.columns:
        df[COL_BET] = numeric_col(df[COL_BET])
        missing_bet_before = df[COL_BET].isna().sum()
        # group mean fill
        df[COL_BET] = df.groupby(["Perovskite", COL_CRYST])[COL_BET].transform(lambda x: x.fillna(x.mean()))
        # if still missing (no group mean), fill with overall mean
        df[COL_BET] = df[COL_BET].fillna(df[COL_BET].mean())
        missing_bet_after = df[COL_BET].isna().sum()
        log_lines.append(f"Bet Surface Area: filled by (Perovskite, Crystal) mean. Before={missing_bet_before}, After={missing_bet_after}")
        print(f"BET filled: {missing_bet_before - missing_bet_after}")
    else:
        log_lines.append("Bet Surface Area or Perovskite or Crystal column missing; skipped BET group-fill.")

    # ----------------------------
    # Rule 4: H2O:CO2 filling for GAS rows using same Light Type AND similar cat W (±10%)
    # ----------------------------
    if COL_H2OCO2 in df.columns and COL_PHASE in df.columns and COL_LIGHT in df.columns and COL_CAT_W in df.columns:
        df[COL_H2OCO2] = numeric_col(df[COL_H2OCO2])
        df[COL_CAT_W] = numeric_col(df[COL_CAT_W])
        mask_gas = df[COL_PHASE].astype(str).str.lower() == "gas"
        missing_h2oco2_before = df.loc[mask_gas, COL_H2OCO2].isna().sum()
        filled_h2o = 0
        # For each gas row missing H2O:CO2, find candidates with same Light Type and cat W within ±10%
        gas_idx = df.loc[mask_gas].index
        for idx in gas_idx:
            if pd.isna(df.at[idx, COL_H2OCO2]):
                light = str(df.at[idx, COL_LIGHT])
                catw = df.at[idx, COL_CAT_W]
                if pd.isna(catw):
                    continue
                tol = CAT_W_TOLERANCE
                candidates = df[
                    (df[COL_PHASE].astype(str).str.lower() == "gas") &
                    (df[COL_LIGHT].astype(str) == light) &
                    (df[COL_CAT_W].notna())
                ]
                if len(candidates) == 0:
                    continue
                # find catW within tolerance
                low = catw * (1 - tol)
                high = catw * (1 + tol)
                candidates_similar = candidates[(candidates[COL_CAT_W] >= low) & (candidates[COL_CAT_W] <= high)]
                if len(candidates_similar) == 0:
                    # fallback to all candidates with same light
                    val = candidates[COL_H2OCO2].mean()
                else:
                    val = candidates_similar[COL_H2OCO2].mean()
                if pd.notna(val):
                    df.at[idx, COL_H2OCO2] = val
                    filled_h2o += 1
        missing_h2oco2_after = df.loc[mask_gas, COL_H2OCO2].isna().sum()
        log_lines.append(f"H2O:CO2 (gas): filled {filled_h2o} values using same Light Type and ±{int(CAT_W_TOLERANCE*100)}% cat W. Before={missing_h2oco2_before}, After={missing_h2oco2_after}")
        print(f"H2O:CO2 filled: {filled_h2o}")
    else:
        log_lines.append("H2O:CO2 or Phase or Light Type or cat W missing; skipped H2O:CO2 fill.")

    # ----------------------------
    # Rule 5: Band gap imputation via Linear Regression
    # ----------------------------
    # Prepare features for bandgap model
    # We'll use a mix of numeric columns and one-hot encoded categorical columns (safe fallback)
    df_work = df.copy()
    # ensure numeric columns exist and coerce
    for col in NUMERIC_COLUMNS_TO_USE_FOR_BANDGAP:
        if col in df_work.columns:
            df_work[col] = numeric_col(df_work[col])
    # ensure categorical columns exist
    for col in CATEGORICAL_COLUMNS_TO_USE_FOR_BANDGAP:
        if col not in df_work.columns:
            df_work[col] = "None"

    # Build feature matrix for rows where Band Gap exists
    mask_has_band = df_work[COL_BAND].notna()
    n_has = int(mask_has_band.sum())
    n_missing_band = int(df_work[COL_BAND].isna().sum())
    log_lines.append(f"Band gap values present: {n_has}, missing: {n_missing_band}")
    print(f"Band gap present: {n_has}, missing: {n_missing_band}")

    if n_has > 10:
        # create features
        feature_df = pd.DataFrame(index=df_work.index)
        # numeric features
        for col in NUMERIC_COLUMNS_TO_USE_FOR_BANDGAP:
            if col in df_work.columns:
                feature_df[col] = df_work[col]
        # categorical -> one-hot
        for col in CATEGORICAL_COLUMNS_TO_USE_FOR_BANDGAP:
            if col in df_work.columns:
                # replace NaN with 'None'
                feature_df[col] = df_work[col].fillna("None").astype(str)
        # get_dummies (one-hot)
        cat_cols = [c for c in CATEGORICAL_COLUMNS_TO_USE_FOR_BANDGAP if c in feature_df.columns]
        feature_df = pd.get_dummies(feature_df, columns=cat_cols, dummy_na=False)

        # Fill numeric NaNs with median
        for c in feature_df.columns:
            if feature_df[c].dtype.kind in "biufc":
                feature_df[c] = feature_df[c].fillna(feature_df[c].median())

        # split X,y for bandgap-known rows
        X_all = feature_df.loc[mask_has_band]
        y_all = numeric_col(df_work.loc[mask_has_band, COL_BAND])

        # Determine reasonable test_size:
        # If there are missing band gaps, use missing_count / available for approximate test size, capped at 0.2
        test_size = 0.1
        if n_missing_band > 0 and n_has > 0:
            approx = n_missing_band / float(n_has)
            test_size = min(0.20, max(0.05, approx))
        # ensure test_size is within (0.05, 0.2)
        test_size = float(test_size)

        X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=test_size, random_state=42)
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_train_pred = lr.predict(X_train)
        y_test_pred = lr.predict(X_test)
        r2_train = r2_score(y_train, y_train_pred)
        r2_test = r2_score(y_test, y_test_pred)
        mse_test = mean_squared_error(y_test, y_test_pred)
        rmse_test = float(np.sqrt(mse_test))


        log_lines.append(f"Band-gap LR trained on {len(X_train)} rows, tested on {len(X_test)} rows")
        log_lines.append(f"Band-gap LR R2_train={r2_train:.4f}, R2_test={r2_test:.4f}, RMSE_test={rmse_test:.4f}")
        print(f"Band-gap LR R2_train={r2_train:.4f}, R2_test={r2_test:.4f}, RMSE_test={rmse_test:.4f}")

        # Save model if requested
        joblib.dump(lr, OUTPUT_MODEL)
        log_lines.append(f"Saved band-gap LR model to {OUTPUT_MODEL}")
        print(f"Saved band-gap LR -> {OUTPUT_MODEL}")

        # Predict missing band gaps using full feature matrix
        if n_missing_band > 0:
            X_full = feature_df
            preds = lr.predict(X_full.loc[df_work[COL_BAND].isna()])
            df.loc[df_work[COL_BAND].isna(), COL_BAND] = preds
            log_lines.append(f"Predicted and filled {int(n_missing_band)} missing band-gap values using LR.")
            print(f"Predicted and filled {int(n_missing_band)} missing band-gap values.")
    else:
        log_lines.append("Not enough band-gap rows to train LR; skipped band-gap imputation.")
        print("Not enough band-gap rows to train LR; skipping.")

    # ----------------------------
    # Final: write imputed CSV and logs
    # ----------------------------
    df.to_csv(OUTPUT_IMPUTED, index=False)
    log_lines.append(f"Saved imputed dataset to: {OUTPUT_IMPUTED}")
    # summary of missing after all imputations
    missing_after = {c: int(df[c].isna().sum()) if c in df.columns else None for c in cols_of_interest}
    log_lines.append("Missing counts after imputation:")
    for k, v in missing_after.items():
        log_lines.append(f"  {k}: {v}")
    # verbose counts
    print("Missingness after imputation (sample):")
    for k, v in missing_after.items():
        print(f"  {k}: {v}")
    write_log(log_lines)
    print("STEP 2 complete. Files produced:")
    print(" -", OUTPUT_IMPUTED)
    print(" -", OUTPUT_MODEL)
    print(" -", LOG_PATH)


if __name__ == "__main__":
    main()
