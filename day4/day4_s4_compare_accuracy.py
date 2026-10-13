
from pathlib import Path
import pandas as pd
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "dataset/raw/data.yaml"
PT = ROOT / "day3/runs/day3/baseline_n_50/weights/best.pt"
ONNX = ROOT / "day3/runs/day3/baseline_n_50/weights/best.onnx"
OUT = ROOT / "day4/reports/pt_vs_onnx_metrics.csv"

if not PT.exists():
    raise FileNotFoundError(PT)
if not ONNX.exists():
    raise FileNotFoundError(
        "ONNX not found. Replace ONNX with the exact exported path."
    )

rows = []

for label, model_path in [("PyTorch", PT), ("ONNX", ONNX)]:
    model = YOLO(str(model_path), task="detect")
    metrics = model.val(
        data=str(DATA),
        split="val",
        imgsz=640,
        batch=1,
        conf=0.001,
        plots=False,
        verbose=False,
        workers=0,
    )

    rows.append({
        "model": label,
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
    })

df = pd.DataFrame(rows)
OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)

print(df.to_string(index=False))
print(f"\nSaved: {OUT}")
