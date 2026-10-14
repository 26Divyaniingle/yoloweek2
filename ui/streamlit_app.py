
from pathlib import Path
from datetime import datetime, timezone
from io import BytesIO
import base64
import tempfile

import cv2
import pandas as pd
import requests
import streamlit as st
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
API_URL = "http://127.0.0.1:8000"
MODEL_PATH = (
    ROOT / "day3" / "runs" / "day3" / "baseline_n_50"
    / "weights" / "best.pt"
)
LOG_PATH = ROOT / "logs" / "predictions.csv"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="YOLO Object Detection",
    page_icon="🚦",
    layout="wide",
)

st.title("YOLO Object Detection Dashboard")
st.caption("Upload an image or video to detect objects.")

mode = st.radio("Choose input type", ["Image", "Video"], horizontal=True)
confidence = st.slider(
    "Confidence threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.25,
    step=0.05,
)

def log_rows(rows):
    if not rows:
        return

    df = pd.DataFrame(rows)
    write_header = not LOG_PATH.exists()
    df.to_csv(
        LOG_PATH,
        mode="a",
        header=write_header,
        index=False,
    )

def save_csv(df, filename):
    path = OUTPUT_DIR / filename
    df.to_csv(path, index=False)
    return path

@st.cache_resource
def get_video_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    return YOLO(str(MODEL_PATH))

if mode == "Image":
    uploaded = st.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png", "webp"],
    )

    if uploaded is not None and st.button("Run detection"):
        with st.spinner("Calling FastAPI..."):
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    params={"confidence": confidence},
                    files={
                        "file": (
                            uploaded.name,
                            uploaded.getvalue(),
                            uploaded.type or "application/octet-stream",
                        )
                    },
                    timeout=120,
                )
                response.raise_for_status()
                result = response.json()

            except requests.RequestException as exc:
                st.error(
                    "Could not call the API. Check that FastAPI "
                    f"is running at {API_URL}. Details: {exc}"
                )
                st.stop()

        image_bytes = base64.b64decode(
            result["annotated_image_base64"]
        )
        st.image(image_bytes, caption="Annotated prediction")

        detections = result["detections"]
        df = pd.DataFrame(detections)

        col1, col2 = st.columns(2)
        col1.metric("Detections", result["num_detections"])
        col2.metric("API latency", f"{result['latency_ms']} ms")

        if df.empty:
            st.info("No objects detected at this confidence threshold.")
        else:
            st.dataframe(df, use_container_width=True)

        log_rows([
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "filename": uploaded.name,
                "mode": "image",
                "confidence_threshold": confidence,
                "num_detections": result["num_detections"],
                "latency_ms": result["latency_ms"],
            }
        ])

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download detections CSV",
            data=csv_bytes,
            file_name="image_detections.csv",
            mime="text/csv",
        )

elif mode == "Video":
    uploaded = st.file_uploader(
        "Upload video",
        type=["mp4", "avi", "mov"],
    )

    if uploaded is not None and st.button("Process video"):
        try:
            model = get_video_model()
        except Exception as exc:
            st.error(f"Could not load video model: {exc}")
            st.stop()

        # OpenCV needs a file path, so temporarily save the upload.
        with tempfile.NamedTemporaryFile(
            suffix=Path(uploaded.name).suffix,
            delete=False,
        ) as tmp:
            tmp.write(uploaded.getvalue())
            input_path = Path(tmp.name)

        output_path = OUTPUT_DIR / "annotated_video.mp4"
        csv_path = OUTPUT_DIR / "video_detections.csv"
        rows = []

        cap = cv2.VideoCapture(str(input_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if not cap.isOpened() or width <= 0 or height <= 0:
            cap.release()
            input_path.unlink(missing_ok=True)
            st.error("Could not read the uploaded video.")
            st.stop()

        if fps <= 0:
            fps = 25.0

        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        if not writer.isOpened():
            cap.release()
            input_path.unlink(missing_ok=True)
            st.error("Could not create output video.")
            st.stop()

        progress = st.progress(0.0)
        status = st.empty()
        frame_index = 0

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                result = model.predict(
                    source=frame,
                    imgsz=640,
                    conf=confidence,
                    verbose=False,
                )[0]

                annotated = result.plot()
                writer.write(annotated)

                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls.item())
                        x1, y1, x2, y2 = [
                            round(float(v), 2)
                            for v in box.xyxy[0].tolist()
                        ]

                        rows.append({
                            "frame": frame_index,
                            "timestamp_seconds": round(
                                frame_index / fps, 3
                            ),
                            "class_id": class_id,
                            "class_name": model.names[class_id],
                            "confidence": round(
                                float(box.conf.item()), 4
                            ),
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                        })

                frame_index += 1
                if total_frames > 0:
                    progress.progress(
                        min(frame_index / total_frames, 1.0)
                    )
                status.write(f"Processed {frame_index} frames")

        finally:
            cap.release()
            writer.release()
            input_path.unlink(missing_ok=True)

        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)

        if frame_index == 0 or not output_path.exists():
            st.error("No frames were processed.")
            st.stop()

        st.success(f"Processed {frame_index} frames.")
        st.video(str(output_path))

        st.download_button(
            "Download annotated video",
            data=output_path.read_bytes(),
            file_name="annotated_video.mp4",
            mime="video/mp4",
        )
        st.download_button(
            "Download video detections CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="video_detections.csv",
            mime="text/csv",
        )

        log_rows([
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "filename": uploaded.name,
                "mode": "video",
                "confidence_threshold": confidence,
                "num_detections": len(df),
                "latency_ms": None,
            }
        ])

        st.dataframe(df.head(100), use_container_width=True)
