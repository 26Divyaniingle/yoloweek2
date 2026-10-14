# Results Summary — YOLO Traffic Object Detection

## 1. Objective

The objective of this project is to build an object detection system for traffic-related road scenes using YOLO. The system detects dataset-defined classes such as vehicles, pedestrians, bicycles, traffic lights, crosswalks, and roundabout-related signs.

The project includes model training, evaluation, error analysis, improvement experiments, and prototype deployment using FastAPI and Streamlit.

## 2. Dataset

- **Source:** [Roboflow Universe — Cars-Pedestrians, Version 6](https://universe.roboflow.com/road-pzinl/cars-pedestrians/dataset/6)
- **Version:** 6
- **Licence:** CC BY 4.0; provide attribution and comply with the licence terms.
- **Configuration:** `dataset/raw/data.yaml`
- **Training images:** 245
- **Validation images:** 70
- **Validation instances:** 524
- **Test split size:** To be verified.

### Classes

1. `40`
2. `60`
3. `bike`
4. `crosswalk`
5. `green light`
6. `pedestrian`
7. `red light`
8. `roundabout 1`
9. `roundabout 2`
10. `vehicule`

The dataset is relatively small, and some classes have very few validation examples. Per-class metrics should therefore be interpreted cautiously.

## 3. Model and Training Settings

- **Model:** YOLOv8n with pretrained weights
- **Image size:** 640 × 640
- **Batch size:** 4
- **Epochs configured:** 50
- **Workers:** 0
- **Seed:** 42
- **Early stopping patience:** 15
- **Optimizer:** AdamW selected automatically
- **Hardware:** Intel Core i5-9400F CPU, approximately 2.90 GHz
- **PyTorch:** CPU build, version 2.14.1
- **Ultralytics:** 8.4.174

Baseline checkpoint:

`day3/runs/day3/baseline_n_50/weights/best.pt`

Fine-tuned checkpoint:

`day4/experiments/runs/experiment_b_finetune/weights/best.pt`

## 4. Baseline Results

The following results were recorded for the Day 3 baseline on the validation split.

| Metric | Result |
|---|---:|
| Precision | 0.789 |
| Recall | 0.320 |
| mAP@0.50 | 0.356 |
| mAP@0.50:0.95 | 0.231 |
| Preprocessing | 2.4 ms/image |
| Inference | 45.4 ms/image |
| Postprocessing | 0.7 ms/image |

### Interpretation

The model achieved relatively high precision but low recall. This indicates that many ground-truth objects were missed at the evaluated settings. The baseline therefore needs improvement, particularly in recall and overall detection quality.

These results are validation metrics and do not establish real-world traffic safety performance.

## 5. Error Analysis

Review 20 validation predictions and classify the actual errors.

Suggested categories:

- False negatives or missed objects
- False positives or background detections
- Incorrect class predictions
- Poor bounding-box localization
- Duplicate detections
- Possible annotation errors
- Ambiguous examples

**Review results:** Pending completion.

Record the actual number of errors in each category after inspecting the images and their ground-truth annotations. Do not invent error counts or label errors.

## 6. Improvement Experiments

Compare the baseline and improvement experiments using the same validation split and consistent evaluation settings.

| Experiment | Precision | Recall | mAP@0.50 | mAP@0.50:0.95 |
|---|---:|---:|---:|---:|
| Day 3 baseline | 0.789 | 0.320 | 0.356 | 0.231 |
| Improvement A | Pending | Pending | Pending | Pending |
| Day 4 fine-tuning | Pending | Pending | Pending | Pending |

The Day 4 checkpoint exists, but improvement must be confirmed using measured evaluation results. Record the exact settings for each experiment and explain any differences in the validation protocol.

## 7. Deployment

- **FastAPI:** Prototype API with `/health` and `/predict` endpoints.
- **Streamlit:** Prototype interface for uploading images and viewing predictions; video inference is handled separately.
- **Model configuration:** `YOLO_MODEL_PATH` can select the checkpoint used by the API.
- **Docker:** The recorded image build failed during Python dependency installation because a package download timed out. A successful Docker build and container run remain unverified.
- **ONNX export:** Status to be verified.
- **Deployment latency:** Pending measurement.

Report deployment latency separately from model-validation latency. Record the hardware, image size, number of runs, and whether upload/network overhead is included.

## 8. Limitations

1. Baseline recall is low at 0.320.
2. Baseline recall was zero for `60`, `roundabout 1`, and `roundabout 2` in the recorded validation results. These classes had very few validation instances.
3. The small dataset limits confidence in rare-class performance.
4. CPU-only hardware can make training and inference slower.
5. Tracking is not integrated into the current detector-only API.
6. Docker deployment and ONNX inference have not yet been verified as successful.
7. Validation results do not prove suitability for real-world traffic control or autonomous driving.

