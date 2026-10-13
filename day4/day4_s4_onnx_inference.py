
from pathlib import Path
import time
import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

# Replace this with the exact path printed by the export script.
ONNX_PATH = (
    ROOT / "day3/runs/day3/baseline_n_50/weights/best.onnx"
)

IMAGE_PATH = ROOT / "dataset/images/val"

if not ONNX_PATH.exists():
    raise FileNotFoundError(
        f"ONNX file not found: {ONNX_PATH}. "
        "Use the exact export path printed by the export script."
    )

if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        f"Image folder not found: {IMAGE_PATH}. "
        "Check your dataset's actual validation-image path."
    )

image_files = []
for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
    image_files.extend(IMAGE_PATH.glob(pattern))

if not image_files:
    raise RuntimeError(f"No images found in {IMAGE_PATH}")

model = YOLO(str(ONNX_PATH), task="detect")

# Warm-up
model.predict(
    source=str(image_files[0]),
    imgsz=640,
    conf=0.25,
    verbose=False,
)

times = []
for image_path in image_files[:20]:
    start = time.perf_counter()
    result = model.predict(
        source=str(image_path),
        imgsz=640,
        conf=0.25,
        verbose=False,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    times.append(elapsed_ms)

    output_path = (
        ROOT / "day4" / "exports" / f"{image_path.stem}_onnx.jpg"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    annotated = result[0].plot()
    cv2.imwrite(str(output_path), annotated)

print(f"Images processed: {len(times)}")
print(f"Mean end-to-end latency: {sum(times)/len(times):.2f} ms")
print("Saved annotated predictions under day4/exports")
