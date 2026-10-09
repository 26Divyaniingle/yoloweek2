
from pathlib import Path
import cv2
from ultralytics import YOLO

VIDEO_PATH = "videos/traffic.mp4"
OUTPUT_DIR = Path("outputs/session4")
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

line_y = height // 2
counted_ids = set()
previous_positions = {}
switch_observations = []

writer = cv2.VideoWriter(
    str(OUTPUT_DIR / "tracked_crossings.mp4"),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[0],  # People; use [2, 3, 5, 7] for vehicles.
        conf=0.3,
        imgsz=640,
        verbose=False
    )[0]

    cv2.line(
        frame, (0, line_y), (width, line_y),
        (0, 255, 255), 2
    )

    if result.boxes is not None and result.boxes.id is not None:
        boxes = result.boxes
        ids = boxes.id.int().cpu().tolist()
        coords = boxes.xyxy.int().cpu().tolist()

        for track_id, (x1, y1, x2, y2) in zip(ids, coords):
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            old_y = previous_positions.get(track_id)
            if old_y is not None:
                crossed = (
                    (old_y < line_y <= cy)
                    or (old_y > line_y >= cy)
                )
                if crossed and track_id not in counted_ids:
                    counted_ids.add(track_id)

            previous_positions[track_id] = cy

            cv2.rectangle(
                frame, (x1, y1), (x2, y2), (0, 255, 0), 2
            )
            cv2.putText(
                frame, f"ID {track_id}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2
            )

    cv2.putText(
        frame, f"Unique crossings: {len(counted_ids)}",
        (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
        1, (0, 255, 0), 2
    )

    writer.write(frame)
    cv2.imshow("YOLO Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()
print(f"Unique track IDs crossing line: {len(counted_ids)}")
print(f"Saved {OUTPUT_DIR / 'tracked_crossings.mp4'}")