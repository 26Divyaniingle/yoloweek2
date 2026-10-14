# YOLO Week 2 — Day 2 Dataset Plan

## 1. Project objective

Develop an object detection model for road scenes using the Cars & Pedestrians dataset from Roboflow Universe.

Dataset: https://universe.roboflow.com/road-pzinl/cars-pedestrians/dataset/6

## 2. Target classes

| Class | Definition | Important edge cases |
|---|---|---|
| Bike | A visible bicycle or two-wheeler matching the dataset's actual class definition | Partially hidden bikes, distant bikes, overlapping vehicles |
| Pedestrian | A visible person on or near the road | Groups, occlusion, side views, partially visible people |
| Vehicule | A road vehicle matching the dataset's original class definition | Cars at different distances, partial vehicles, crowded traffic |

Note: Confirm the exact meaning of each class from the dataset examples. Preserve the class IDs and names supplied in `data.yaml`.

## 3. Data quantity targets

- Initial dataset: use all valid images supplied by the downloaded version.
- Annotation exercise: review at least 250 images if available.
- Additional data target: collect more diverse images if validation performance is weak.
- Split: target 70% training, 20% validation and 10% test.
- Keep images from the same video or near-identical frames in only one split.

## 4. Data sources

- Primary source: selected Roboflow Universe dataset version.
- Additional source: permitted road-scene images with suitable annotations.
- Optional source: original video frames, annotated manually and checked before inclusion.

## 5. Variation to capture

- Daylight, shadows, cloudy weather and low light.
- Near and distant objects.
- Front, side and rear views.
- Partial occlusion by other road users.
- Crowded intersections and sparse roads.
- Motion blur, truncation and small objects.

## 6. Quality checks

- Every image opens successfully.
- Every labelled image has a matching YOLO text file.
- Class IDs are valid.
- Normalized bounding-box coordinates are within [0, 1].
- Duplicate or near-duplicate images do not leak between splits.
- Class imbalance and empty label files are reviewed.
- All annotations follow the written annotation rules.

## 7. Evaluation

Use validation mAP50, mAP50-95, precision and recall to evaluate model performance. Keep the test split untouched until final evaluation.
