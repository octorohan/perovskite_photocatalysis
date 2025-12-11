# step5_unit_correction_v2.py
"""
Robust unit correction script.

- If results/step2_imputed_corrected.csv exists -> prints summary and exits.
- Else tries to read results/step2_imputed_backup.csv (created earlier).
- Reads CSV trying utf-8, cp1252, latin1.
- Detects yield + phase columns, converts Gas yields from mmol -> μmol (x1000).
- Sets canonical yield header: "total yield (μmol gcat-1 h-1)"
- Writes results/step2_imputed_corrected.csv
- Prints before/after summaries and top-10 tables.
"""
import os
import sys
import shutil
import pandas as pd
import numpy as np

PROJECT_ROOT = r"C:\Users\HP\Desktop\winter_projects\perovskite"
BACKUP_PATH = os.path.join(PROJECT_ROOT, "results", "step2_imputed_backup.csv")
CORRECTED_PATH = os.path.join(PROJECT_ROOT, "results", "step2_imputed_corrected.csv")

def try_read(path):
    encs = ["utf-8", "cp1252", "latin1"]
    last = None
    for e in encs:
        try:
            df = pd.read_csv(path, encoding=e, low_memory=False)
            print(f"Read {os.path.basename(path)} with encoding={e}")
            return df
        except Exception as ex:
            last = ex
    # final fallback - try python engine
    try:
        df = pd.read_csv(path, engine="python", encoding="latin1")
        print(f"Read {os.path.basename(path)} with engine=python, encoding=latin1")
        return df
    except Exception:
        raise last or Exception("Failed to read CSV with tried encodings.")

def find_col(df, keywords):
    cols = list(df.columns)
    low = [c.lower() for c in cols]
    for kw in keywords:
        kwl = kw.lower()
        for i, c in enumerate(low):
            if kwl in c:
                return cols[i]
    return None

def summarize(s):
    return {
        "count": int(s.count()),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "std": float(s.std()),
        "min": float(s.min()),
        "25%": float(s.quantile(0.25)),
        "50%": float(s.quantile(0.50)),
        "75%": float(s.quantile(0.75)),
        "max": float(s.max())
    }

def main():
    # If corrected already exists, print summary and exit
    if os.path.exists(CORRECTED_PATH):
        print("Corrected file already exists:", CORRECTED_PATH)
        df = try_read(CORRECTED_PATH)
        df.columns = [c.strip() for c in df.columns]
        phase_col = find_col(df, ["liq","gas","phase"])
        yield_col = find_col(df, ["yield","total yield","umol","mmol","gcat"])
        if phase_col is None or yield_col is None:
            print("Could not detect phase/yield in corrected file.")
            return
        mask_gas = df[phase_col].astype(str).str.lower().str.startswith("gas")
        gas = pd.to_numeric(df.loc[mask_gas, yield_col], errors="coerce")
        print("Gas summary (corrected):", summarize(gas))
        print("Top 10 Gas yields (corrected):")
        disp_cols = [c for c in ["Perovskite", phase_col, yield_col] if c in df.columns]
        print(df.loc[mask_gas, disp_cols].sort_values(by=yield_col, ascending=False).head(10).to_string(index=False))
        return

    # Else, load backup
    if not os.path.exists(BACKUP_PATH):
        raise FileNotFoundError(f"Backup not found: {BACKUP_PATH}. Run the earlier correction that created backup.")
    df = try_read(BACKUP_PATH)
    df.columns = [c.strip() for c in df.columns]

    phase_col = find_col(df, ["liq","gas","phase"])
    yield_col = find_col(df, ["total yield","yield","umol","mmol","gcat"])
    if phase_col is None or yield_col is None:
        raise RuntimeError(f"Could not auto-detect phase or yield columns. phase={phase_col} yield={yield_col}")

    print("Detected phase column:", phase_col)
    print("Detected yield column:", yield_col)

    # coerce numeric
    df[yield_col] = pd.to_numeric(df[yield_col], errors="coerce")

    mask_gas = df[phase_col].astype(str).str.lower().str.startswith("gas")
    gas_before = df.loc[mask_gas, yield_col].copy()
    print("\nGas BEFORE summary:")
    print(summarize(gas_before))
    print("\nTop 10 Gas BEFORE:")
    disp_cols = [c for c in ["Perovskite", phase_col, yield_col] if c in df.columns]
    print(df.loc[mask_gas, disp_cols].sort_values(by=yield_col, ascending=False).head(10).to_string(index=False))

    # Multiply gas yields by 1000
    df_corrected = df.copy()
    df_corrected.loc[mask_gas, yield_col] = df_corrected.loc[mask_gas, yield_col] * 1000.0

    # rename yield column to canonical μmol name
    canonical = "total yield (μmol gcat-1 h-1)"
    if canonical in df_corrected.columns:
        new_yield_col = canonical
    else:
        # if original header contained 'mmol', replace it; else overwrite to canonical
        if "mmol" in yield_col.lower():
            # replace 'mmol' substring with 'µmol' or 'μmol'
            new_yield_col = yield_col.replace("mmol", "µmol")
        else:
            new_yield_col = canonical
        # if clash, append _umol
        if new_yield_col in df_corrected.columns and new_yield_col != yield_col:
            new_yield_col = yield_col + "_umol"
        df_corrected = df_corrected.rename(columns={yield_col: new_yield_col})

    # Save corrected
    df_corrected.to_csv(CORRECTED_PATH, index=False)
    print("\nSaved corrected dataset to:", CORRECTED_PATH)

    # Print after summary
    gas_after = pd.to_numeric(df_corrected.loc[mask_gas, new_yield_col], errors="coerce")
    print("\nGas AFTER summary:")
    print(summarize(gas_after))
    print("\nTop 10 Gas AFTER:")
    disp_cols_after = [c for c in ["Perovskite", phase_col, new_yield_col] if c in df_corrected.columns]
    print(df_corrected.loc[mask_gas, disp_cols_after].sort_values(by=new_yield_col, ascending=False).head(10).to_string(index=False))

if __name__ == "__main__":
    main()
