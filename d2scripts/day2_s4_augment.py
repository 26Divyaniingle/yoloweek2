
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DATA_YAML = PROJECT_ROOT / "dataset" / "raw" / "data.yaml"

if not DATA_YAML.exists():
    raise FileNotFoundError(f"Dataset config not found: {DATA_YAML}")

model = YOLO(str(PROJECT_ROOT / "yolov8n.pt"))

model.train(
    data=str(DATA_YAML),
    epochs=40,
    imgsz=640,
    batch=4,
    workers=0,
    device="cpu",
    project=str(ROOT / "runs" / "day2"),
    name="augmentation_preview",
    exist_ok=True,
    plots=True,

    # Built-in augmentation settings
    mosaic=1.0,
    mixup=0.1,
    hsv_h=0.015,
    hsv_s=0.5,
    hsv_v=0.3,
    degrees=5.0,
    scale=0.3,
    fliplr=0.5,
    copy_paste=0.0,
)

print("Training and augmentation preview finished.")
print("Inspect runs/day2/augmentation_preview/")
