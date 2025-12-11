import pandas as pd
import numpy as np
import os

PROJECT_ROOT = r"C:\Users\HP\Desktop\winter_projects\perovskite"
CORRECTED = os.path.join(PROJECT_ROOT, "results", "step2_imputed_corrected.csv")

df = pd.read_csv(CORRECTED, low_memory=False)
df.columns = [c.strip() for c in df.columns]

# detect phase + yield columns
phase_col = next((c for c in df.columns if "liq" in c.lower() or "gas" in c.lower()), None)
yield_col = next((c for c in df.columns if "yield" in c.lower()), None)
print("Detected yield:", yield_col)
print("Detected phase:", phase_col)

# GAS subset
mask_gas = df[phase_col].astype(str).str.lower().str.startswith("gas")
gas = df.loc[mask_gas].copy()

# print summary
print("\n--- Gas yields AFTER (summary) ---")
print(gas[yield_col].describe().to_string())

# print top 10
disp_cols = ["Perovskite", phase_col, yield_col]
disp_cols = [c for c in disp_cols if c in df.columns]

print("\nTop 10 Gas yields AFTER (descending):")
print(gas[disp_cols].sort_values(by=yield_col, ascending=False).head(10).to_string(index=False))
