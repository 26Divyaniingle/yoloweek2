
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
import base64
import time
import os
import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = Path(
    os.getenv(
        "YOLO_MODEL_PATH",
        str(
            ROOT
            / "day3"
            / "runs"
            / "day3"
            / "baseline_n_50"
            / "weights"
            / "best.pt"
        ),
    )
)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}

model = None
model_lock = Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model weights not found: {MODEL_PATH}")

    # Load once when the server starts, not on every request.
    model = YOLO(str(MODEL_PATH))
    yield
    model = None


app = FastAPI(
    title="YOLO Object Detection API",
    description="Image detection using a trained YOLO model.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def home():
    return {
        "message": "YOLO Object Detection API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {
        "status": "ok" if model is not None else "not_ready",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    confidence: float = 0.25,
):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not 0.0 <= confidence <= 1.0:
        raise HTTPException(
            status_code=400,
            detail="confidence must be between 0 and 1",
        )

    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Upload a JPG, PNG, or WEBP image",
        )

    content = await file.read(MAX_UPLOAD_BYTES + 1)

    if not content:
        raise HTTPException(status_code=400, detail="Empty upload")

    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds 10 MB limit",
        )

    image_array = np.frombuffer(content, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid image",
        )

    start = time.perf_counter()

    try:
        # Ultralytics inference is synchronised to protect the shared model.
        with model_lock:
            results = model.predict(
                source=image,
                imgsz=640,
                conf=confidence,
                verbose=False,
            )

        result = results[0]
        detections = []

        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls.item())
                x1, y1, x2, y2 = [
                    round(float(v), 2)
                    for v in box.xyxy[0].tolist()
                ]

                detections.append({
                    "class_id": class_id,
                    "class_name": model.names[class_id],
                    "confidence": round(float(box.conf.item()), 4),
                    "bbox_xyxy": [x1, y1, x2, y2],
                })

        annotated = result.plot()
        ok, encoded = cv2.imencode(".jpg", annotated)

        if not ok:
            raise HTTPException(
                status_code=500,
                detail="Could not encode annotated image",
            )

        elapsed_ms = round(
            (time.perf_counter() - start) * 1000, 2
        )

        return JSONResponse({
            "filename": file.filename,
            "image_width": int(image.shape[1]),
            "image_height": int(image.shape[0]),
            "confidence_threshold": confidence,
            "num_detections": len(detections),
            "latency_ms": elapsed_ms,
            "detections": detections,
            "annotated_image_base64": base64.b64encode(
                encoded.tobytes()
            ).decode("ascii"),
        })

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {type(exc).__name__}",
        ) from exc
    finally:
        await file.close()
