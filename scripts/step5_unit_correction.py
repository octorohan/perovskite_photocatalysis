"""
Step 5 — Unit correction: convert Gas-phase yields from mmol -> μmol by *1000.

- Detects yield and phase columns automatically.
- Backs up results/step2_imputed.csv -> results/step2_imputed_backup.csv
- Writes results/step2_imputed_corrected.csv (and does NOT overwrite original)
- Prints before/after summary for Gas yields (count, mean, median, std, min, max)
- Also prints top 10 rows (Perovskite, phase, yield) before and after for visual check.
"""

import os
import pandas as pd
import numpy as np
import shutil

PROJECT_ROOT = r"C:\Users\HP\Desktop\winter_projects\perovskite"
INPUT_PATH = os.path.join(PROJECT_ROOT, "results", "step2_imputed.csv")
BACKUP_PATH = os.path.join(PROJECT_ROOT, "results", "step2_imputed_backup.csv")
OUT_PATH = os.path.join(PROJECT_ROOT, "results", "step2_imputed_corrected.csv")

def find_col(df, keywords):
    cols = list(df.columns)
    low = [c.lower() for c in cols]
    for kw in keywords:
        kwl = kw.lower()
        for i, c in enumerate(low):
            if kwl in c:
                return cols[i]
    return None

def summarize(series):
    return {
        "count": int(series.count()),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "std": float(series.std()),
        "min": float(series.min()),
        "25%": float(series.quantile(0.25)),
        "50%": float(series.quantile(0.50)),
        "75%": float(series.quantile(0.75)),
        "max": float(series.max())
    }

def main():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError("Cannot find input: " + INPUT_PATH)
    # make a backup
    shutil.copy2(INPUT_PATH, BACKUP_PATH)
    print("Backed up original to:", BACKUP_PATH)

    df = pd.read_csv(INPUT_PATH, low_memory=False)
    # normalize headers
    df.columns = [c.strip() for c in df.columns]

    # detect columns
    phase_col = find_col(df, ["liq","gas","phase"])
    yield_col = find_col(df, ["yield","total yield","umol","mmol","gcat"])
    if phase_col is None or yield_col is None:
        raise RuntimeError(f"Could not auto-detect phase or yield columns. phase_col={phase_col}, yield_col={yield_col}")

    print("Detected phase column:", phase_col)
    print("Detected yield column:", yield_col)

    # coerce yield to numeric
    df[yield_col] = pd.to_numeric(df[yield_col], errors="coerce")

    # split gas rows
    mask_gas = df[phase_col].astype(str).str.lower().str.startswith("gas")
    gas_before = df.loc[mask_gas, yield_col].copy()

    print("\n--- Gas yields BEFORE correction ---")
    print(summarize(gas_before))
    print("\nTop 10 Gas yields BEFORE (desc):")
    display_cols = [c for c in ["Perovskite", phase_col, yield_col] if c in df.columns]
    print(df.loc[mask_gas, display_cols].sort_values(by=yield_col, ascending=False).head(10).to_string(index=False))

    # Multiply gas yields by 1000 to convert mmol -> μmol
    df_corrected = df.copy()
    df_corrected.loc[mask_gas, yield_col] = df_corrected.loc[mask_gas, yield_col] * 1000.0

    # If the yield header contains 'mmol', rename to μmol in the corrected file
    new_yield_col = yield_col
    if "mmol" in yield_col.lower():
        new_yield_col = yield_col.lower().replace("mmol", "μmol")
        # create a proper header name (keep parentheses etc.)
        new_yield_col = new_yield_col.replace("mmol", "μmol")
        # if new name collides, append '_umol'
        if new_yield_col in df_corrected.columns and new_yield_col != yield_col:
            new_yield_col = yield_col + "_umol"
        df_corrected = df_corrected.rename(columns={yield_col: new_yield_col})
        print(f"\nRenamed yield column in corrected file: '{yield_col}' -> '{new_yield_col}'")
    else:
        # optionally, standardize name to 'total yield (μmol gcat-1 h-1)' if user wants
        # We'll set a canonical name if original header didn't mention units
        canonical = "total yield (μmol gcat-1 h-1)"
        if canonical not in df_corrected.columns:
            df_corrected = df_corrected.rename(columns={yield_col: canonical})
            new_yield_col = canonical
            print(f"\nRenamed yield column in corrected file: '{yield_col}' -> '{canonical}'")
        else:
            new_yield_col = canonical

    # Summarize after correction
    gas_after = df_corrected.loc[mask_gas, new_yield_col].copy()
    print("\n--- Gas yields AFTER correction ---")
    print(summarize(gas_after))
    print("\nTop 10 Gas yields AFTER (desc):")
    print(df_corrected.loc[mask_gas, display_cols if display_cols[0] != yield_col else [display_cols[0], display_cols[1], new_yield_col]].sort_values(by=new_yield_col, ascending=False).head(10).to_string(index=False))

    # Save corrected file
    df_corrected.to_csv(OUT_PATH, index=False)
    print("\nSaved corrected dataset to:", OUT_PATH)
    print("NOTE: original file preserved as backup at:", BACKUP_PATH)
    print("Now you should re-run step3 and step4 scripts to retrain models with corrected units.")

if __name__ == "__main__":
    main()
