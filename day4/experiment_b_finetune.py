
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "dataset/raw/data.yaml"
BASE_WEIGHTS = (
    ROOT / "day3/runs/day3/baseline_n_50/weights/best.pt"
)

model = YOLO(str(BASE_WEIGHTS))

model.train(
    data=str(DATA),
    imgsz=640,
    batch=4,
    epochs=30,
    patience=10,
    workers=0,
    seed=42,
    pretrained=True,
    project=str(ROOT / "day4/experiments/runs"),
    name="experiment_b_finetune",
    exist_ok=True,
    plots=True,
)
