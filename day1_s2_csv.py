
from pathlib import Path
import csv
from ultralytics import YOLO

IMAGE_DIR = Path("images/test")
OUTPUT_DIR = Path("outputs/session2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

model = YOLO("yolov8n.pt")

image_paths = sorted(
    p for p in IMAGE_DIR.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
)

output_csv = OUTPUT_DIR / "detections.csv"

fields = [
    "image_name", "class_id", "class",
    "confidence", "x1", "y1", "x2", "y2"
]

with open(output_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()

    # Stream results instead of retaining every result in memory.
    for result in model.predict(
        source=[str(p) for p in image_paths],
        stream=True,
        conf=0.25,
        imgsz=640,
        verbose=False
    ):
        if result.boxes is None:
            continue

        for box in result.boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            writer.writerow({
                "image_name": Path(result.path).name,
                "class_id": class_id,
                "class": result.names[class_id],
                "confidence": round(confidence, 4),
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2)
            })

print(f"CSV exported: {output_csv.resolve()}")