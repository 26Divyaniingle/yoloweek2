
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DATA_YAML = PROJECT_ROOT / "dataset" / "raw" / "data.yaml"

if not DATA_YAML.exists():
    raise FileNotFoundError(
        f"Dataset YAML not found: {DATA_YAML}"
    )

model = YOLO("yolov8n.pt")

results = model.train(
    data=str(DATA_YAML),
    epochs=50,
    imgsz=640,
    batch=4,
    workers=0,
    patience=15,
    optimizer="auto",
    lr0=0.01,
    lrf=0.01,
    warmup_epochs=3,
    pretrained=True,
    seed=42,
    deterministic=True,
    device="cpu",
    project=str(ROOT / "runs" / "day3"),
    name="baseline_n_50",
    exist_ok=True,
    plots=True,
    save=True,
    verbose=True
)

print("Baseline training completed.")
print("Results folder:", ROOT / "runs" / "day3" / "baseline_n_50")
