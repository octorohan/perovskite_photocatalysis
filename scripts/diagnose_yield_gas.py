# diagnose_yield_gas.py
import pandas as pd, numpy as np, os
p = r"C:\Users\HP\Desktop\winter_projects\perovskite\results\step2_imputed.csv"
df = pd.read_csv(p, low_memory=False)
# auto-detect yield + phase columns
yield_col = next((c for c in df.columns if "yield" in c.lower()), None)
phase_col = next((c for c in df.columns if "liq" in c.lower() or "gas" in c.lower() or "phase" in c.lower()), None)
print("Detected cols -> yield:", yield_col, "phase:", phase_col)
df[ yield_col ] = pd.to_numeric(df[yield_col], errors="coerce")
gas = df[df[phase_col].astype(str).str.lower().str.startswith("gas")].copy()
print("Gas rows:", len(gas))
print("\nYield summary (Gas):")
print(gas[yield_col].describe().to_string())
print("\nTop 20 yields (descending):")
top = gas.sort_values(by=yield_col, ascending=False)[[yield_col, "Perovskite", "A", "B", "Synthesis Method of Perovskite"]].head(20)
print(top.to_string(index=False))
print("\nCount of yields > mean+3*std :", ((gas[yield_col] > gas[yield_col].mean() + 3*gas[yield_col].std())).sum())
# also show number of zeros and negatives
print("Zeros:", (gas[yield_col] == 0).sum(), "Negatives:", (gas[yield_col] < 0).sum())
