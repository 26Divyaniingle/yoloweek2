
from pathlib import Path
import pandas as pd
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "day3/runs/day3/baseline_n_50/weights/best.pt"
DATA = ROOT / "dataset/raw/data.yaml"
OUT = ROOT / "day4/reports/threshold_comparison.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

model = YOLO(str(WEIGHTS))
rows = []

for threshold in [0.001, 0.10, 0.25, 0.40]:
    result = model.val(
        data=str(DATA),
        split="val",
        imgsz=640,
        batch=4,
        conf=threshold,
        plots=False,
        verbose=False,
        workers=0,
    )
    rows.append({
        "confidence_threshold": threshold,
        "precision": float(result.box.mp),
        "recall": float(result.box.mr),
        "mAP50": float(result.box.map50),
        "mAP50_95": float(result.box.map),
    })

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
print(df.to_string(index=False))
print(f"\nSaved: {OUT}")
