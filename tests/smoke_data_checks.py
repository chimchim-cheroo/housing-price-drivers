from pathlib import Path
import sys, pandas as pd

P = Path("data/cleaned/housing_model_quarterly_panel.csv")
if not P.exists():
    print("[FAIL] panel not found:", P); sys.exit(1)

df = pd.read_csv(P)
required = [
    "quarter","hpi_sydney","hpi_melbourne","cash_rate",
    "population_nsw","population_vic",
    "approvals_units_nsw","approvals_units_vic"
]
missing = [c for c in required if c not in df.columns]
if missing:
    print("[FAIL] missing columns:", missing); sys.exit(1)

for c in ["hpi_sydney","hpi_melbourne","cash_rate"]:
    if not pd.api.types.is_numeric_dtype(df[c]):
        print(f"[FAIL] {c} not numeric"); sys.exit(1)

print("[OK] smoke check passed | rows:", len(df), "| range:", df['quarter'].min(), "→", df['quarter'].max())
