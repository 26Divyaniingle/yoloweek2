
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "day3/runs/day3/baseline_n_50/weights/best.pt"
DATA_YAML = ROOT / "dataset/raw/data.yaml"
OUT = ROOT / "day4/errors/predictions"

OUT.mkdir(parents=True, exist_ok=True)
model = YOLO(str(WEIGHTS))

# Validate and save diagnostic plots first.
model.val(
    data=str(DATA_YAML),
    split="val",
    imgsz=640,
    batch=4,
    conf=0.001,
    plots=True,
    project=str(ROOT / "day4/runs"),
    name="error_review",
    exist_ok=True,
    workers=0,
)

# Save prediction images for manual review.
# Resolve the validation directory from data.yaml before running.
import yaml

with open(DATA_YAML, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

yaml_dir = DATA_YAML.parent
val_path = Path(config["val"])
if not val_path.is_absolute():
    val_path = (yaml_dir / val_path).resolve()

if not val_path.exists():
    raise FileNotFoundError(f"Validation images not found: {val_path}")

results = model.predict(
    source=str(val_path),
    imgsz=640,
    conf=0.001,
    save=True,
    save_conf=True,
    project=str(OUT),
    name="all_val_predictions",
    exist_ok=True,
    batch=4,
    verbose=False,
)

print(f"Processed {len(results)} validation images.")
print("Prediction images saved under:", OUT / "all_val_predictions")
