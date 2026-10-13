
from pathlib import Path
import pandas as pd
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "dataset" / "raw" / "data.yaml"
WEIGHTS = (
    ROOT / "day3" / "runs" / "day3" / "baseline_n_50"
    / "weights" / "best.pt"
)
OUT = ROOT / "day4" / "reports"
OUT.mkdir(parents=True, exist_ok=True)

if not DATA_YAML.exists():
    raise FileNotFoundError(f"Dataset YAML not found: {DATA_YAML}")
if not WEIGHTS.exists():
    raise FileNotFoundError(f"Model weights not found: {WEIGHTS}")

with open(DATA_YAML, "r", encoding="utf-8") as f:
    data_config = yaml.safe_load(f)

print("Dataset classes:", data_config.get("names"))
print("Configured test path:", data_config.get("test"))

model = YOLO(str(WEIGHTS))

# Standard validation: keep these settings consistent for comparisons.
metrics = model.val(
    data=str(DATA_YAML),
    split="val",
    imgsz=640,
    batch=4,
    conf=0.001,
    iou=0.7,
    plots=True,
    project=str(ROOT / "day4" / "runs"),
    name="baseline_val",
    exist_ok=True,
    workers=0,
)

box = metrics.box
names = model.names

rows = []
for i, class_id in enumerate(box.ap_class_index):
    class_id = int(class_id)
    rows.append({
        "class_id": class_id,
        "class_name": names[class_id],
        "precision": float(box.p[i]),
        "recall": float(box.r[i]),
        "mAP50": float(box.ap50[i]),
        "mAP50_95": float(box.ap[i]),
    })

df = pd.DataFrame(rows).sort_values("mAP50_95")
csv_path = OUT / "per_class_metrics.csv"
df.to_csv(csv_path, index=False)

print("\nPER-CLASS METRICS")
print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\nOverall precision:", round(float(box.mp), 4))
print("Overall recall:", round(float(box.mr), 4))
print("Overall mAP50:", round(float(box.map50), 4))
print("Overall mAP50-95:", round(float(box.map), 4))
print("\nSaved:", csv_path)
print("Plots and validation outputs are under day4/runs/baseline_val")
