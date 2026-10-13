
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = (
    ROOT / "day3/runs/day3/baseline_n_50/weights/best.pt"
)

if not WEIGHTS.exists():
    raise FileNotFoundError(WEIGHTS)

model = YOLO(str(WEIGHTS))

exported_path = model.export(
    format="onnx",
    imgsz=640,
    batch=1,
    dynamic=False,
    simplify=False,
    opset=17,
)

print("Export completed.")
print("Exported ONNX model:", exported_path)
