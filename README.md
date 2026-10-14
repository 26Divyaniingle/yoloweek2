# YOLO Object Detection — Cars & Pedestrians

## Overview

This project trains and serves a YOLO object detection model for a ten-class dataset containing traffic-related objects. It includes a FastAPI backend for image inference and a Streamlit interface for image and video demos.

## Features

- YOLOv8n inference
- FastAPI `/health` and `/predict` endpoints
- Image upload validation
- JSON detection results with class, confidence, and bounding-box coordinates
- Streamlit image/video demo
- Annotated outputs and downloadable detection CSV
- Basic prediction logging

## Project Structure

```text
app/       FastAPI application
ui/        Streamlit interface
day3/      Training artifacts (not necessarily committed)
dataset/   Local dataset (not necessarily committed)
day4/      Evaluation and export scripts
logs/      Prediction logs
outputs/   Generated output files
```

## Requirements

- Python 3.12
- A compatible Ultralytics/PyTorch environment
- The trained model checkpoint
- Dataset is required for evaluation, but not for inference

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Place the trained checkpoint at the path configured in `app/main.py`.

## Run the API

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` to test the API.

## Run the UI

In a second terminal:

```powershell
python -m streamlit run ui/streamlit_app.py
```

## API Endpoints

- `GET /` — API information
- `GET /health` — service and model status
- `POST /predict` — image upload and object detections

The prediction endpoint returns JSON detections and a Base64-encoded annotated image.

## Baseline Results

Record the verified validation metrics here.

| Metric | Value |
|---|---:|
| Precision | 0.789 |
| Recall | 0.320 |
| mAP50 | 0.356 |
| mAP50-95 | 0.231 |

These values came from the initial YOLOv8n training run. Replace them if a later, verified checkpoint is selected.

## Limitations

- Performance depends on dataset quality and class coverage.
- Some classes have very few validation examples.
- Low recall means that real objects can be missed.
- Video processing is frame-by-frame and does not perform unique-object tracking.
- Validation results do not guarantee real-world performance.
- This is a learning/demo project, not a safety-certified traffic monitoring system.