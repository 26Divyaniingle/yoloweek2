
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "runs" / "day3" / "baseline_n_50" / "results.csv"

if not RESULTS.exists():
    raise FileNotFoundError(
        f"Run results not found: {RESULTS}. Complete training first."
    )

df = pd.read_csv(RESULTS)
df.columns = df.columns.str.strip()

print("\nAvailable columns:")
print(df.columns.tolist())

print("\nLast 5 epochs:")
print(df.tail())

metric_cols = [
    c for c in df.columns
    if "metrics/mAP50(" in c or "metrics/mAP50-95(" in c
]

if metric_cols:
    print("\nBest observed values:")
    for col in metric_cols:
        idx = df[col].idxmax()
        print(
            f"{col}: {df.loc[idx, col]:.4f} "
            f"at epoch index {idx}"
        )

print("\nSaved results are from actual training; "
      "do not replace missing metrics with estimates.")
