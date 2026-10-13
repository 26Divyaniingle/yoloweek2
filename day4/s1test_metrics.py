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

test_metrics = model.val(
    data=str(DATA_YAML),
    split="test",
    imgsz=640,
    batch=4,
    conf=0.001,
    plots=True,
    project=str(ROOT / "day4" / "runs"),
    name="baseline_test",
    exist_ok=True,
    workers=0,
)