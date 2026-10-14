
from pathlib import Path
import cv2
from ultralytics import YOLO

VIDEO_PATH = "videos/traffic.mp4"
OUTPUT_DIR = Path("outputs/session3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise SystemExit(f"Cannot open video: {VIDEO_PATH}")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 25

writer = cv2.VideoWriter(
    str(OUTPUT_DIR / "zone_count.mp4"),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

# Adjust these points to match your video's road/region.
zone = [
    (int(width * 0.2), int(height * 0.3)),
    (int(width * 0.8), int(height * 0.3)),
    (int(width * 0.95), int(height * 0.85)),
    (int(width * 0.1), int(height * 0.85))
]

# COCO IDs: person=0, car=2, motorcycle=3, bus=5, truck=7.
vehicle_classes = [2, 3, 5, 7]
cv2.namedWindow("YOLO Zone Counter", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLO Zone Counter", 540, 960)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = model.predict(
        frame,
        classes=vehicle_classes,
        conf=0.3,
        imgsz=640,
        verbose=False
    )[0]

    inside_count = 0
    cv2.polylines(
        frame, [__import__("numpy").array(zone)],
        True, (0, 255, 255), 2
    )

    if result.boxes is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            inside = cv2.pointPolygonTest(
                __import__("numpy").array(zone, dtype="int32"),
                (float(cx), float(cy)),
                False
            ) >= 0

            if inside:
                inside_count += 1
                color = (0, 255, 0)
            else:
                color = (255, 150, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (cx, cy), 4, color, -1)

    cv2.putText(
        frame, f"Vehicles in zone: {inside_count}",
        (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
        1, (0, 255, 0), 2
    )
    display_frame = cv2.resize(frame, (540, 960))
    cv2.imshow("YOLO Zone Counter", display_frame)

    writer.write(frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()
print("Saved outputs/session3/zone_count.mp4")