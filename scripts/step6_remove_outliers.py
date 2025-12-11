# step6_remove_outliers.py
"""
Step 6 — Remove Gas-phase outliers (y > mean + 3*std) after unit correction.
Produces: results/step2_clean.csv
"""

import pandas as pd
import numpy as np
import os

PROJECT_ROOT = r"C:\Users\HP\Desktop\winter_projects\perovskite"
INPUT = os.path.join(PROJECT_ROOT, "results", "step2_imputed_corrected.csv")
OUTPUT = os.path.join(PROJECT_ROOT, "results", "step2_clean.csv")

def find_col(df, keys):
    for c in df.columns:
        cl = c.lower()
        for k in keys:
            if k.lower() in cl:
                return c
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
    if not os.path.exists(INPUT):
        raise FileNotFoundError(INPUT)

    df = pd.read_csv(INPUT, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    phase_col = find_col(df, ["liq", "gas", "phase"])
    yield_col = find_col(df, ["yield", "μmol", "umol", "gcat"])
    print("Detected phase column:", phase_col)
    print("Detected yield column:", yield_col)

    df[yield_col] = pd.to_numeric(df[yield_col], errors="coerce")

    # isolate gas
    mask_gas = df[phase_col].astype(str).str.lower().str.startswith("gas")
    gas = df.loc[mask_gas, yield_col].copy()

    print("\nGas BEFORE outlier removal:")
    print(summarize(gas))

    threshold = gas.mean() + 3*gas.std()
    print(f"\nOutlier threshold = {threshold:.3f} μmol/g/h")

    mask_outlier = gas > threshold
    num_outliers = mask_outlier.sum()
    print(f"Outliers detected: {num_outliers}")

    # remove outliers
    df_clean = df.copy()
    df_clean = df_clean.loc[~(mask_gas & mask_outlier)]

    # AFTER stats
    gas_after = df_clean.loc[
        df_clean[phase_col].astype(str).str.lower().str.startswith("gas"),
        yield_col
    ]

    print("\nGas AFTER outlier removal:")
    print(summarize(gas_after))

    # Save
    df_clean.to_csv(OUTPUT, index=False)
    print("\nSaved clean dataset ->", OUTPUT)

    print("\nTop 10 Gas yields AFTER:")
    gas_sorted = df_clean.loc[
        df_clean[phase_col].astype(str).str.lower().str.startswith("gas"),
        [yield_col]
    ].sort_values(by=yield_col, ascending=False).head(10)
    print(gas_sorted.to_string())

if __name__ == "__main__":
    main()
