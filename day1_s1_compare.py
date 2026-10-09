from pathlib import Path
from time import perf_counter
import csv
import cv2              
from ultralytics import YOLO

IMAGE_DIR = Path("images/test")
OUTPUT_DIR = Path("outputs/session1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

image_paths = sorted(
    p for p in IMAGE_DIR.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
)[:30]

if not image_paths:
    raise SystemExit("Add test images to the images folder.")

models = {
    "nano": YOLO("yolov8n.pt"),
    "small": YOLO("yolov8s.pt"),
    "medium": YOLO("yolov8m.pt"),
}

rows = []

for model_name, model in models.items():
    # Warm-up helps reduce first-inference overhead.
    model.predict(
        source=str(image_paths[0]),
        imgsz=640,
        verbose=False
    )

    for image_path in image_paths:
        start = perf_counter()
        result = model.predict(
            source=str(image_path),
            conf=0.25,
            iou=0.45,
            imgsz=640,
            device="cpu",
            max_det=300,
            verbose=False
        )[0]
        latency_ms = (perf_counter() - start) * 1000

        boxes = result.boxes
        confidences = boxes.conf.tolist() if boxes is not None else []
        count = len(confidences)

        rows.append({
            "model": model_name,
            "image": image_path.name,
            "latency_ms": round(latency_ms, 2),
            "detections": count,
            "mean_confidence": round(
                sum(confidences) / count, 4
            ) if count else 0.0
        })

with open(OUTPUT_DIR / "comparison.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

    print(f"Processed {len(image_paths)} images.")
    print(f"Results saved to {OUTPUT_DIR / 'comparison.csv'}")