# YOLO Week 2 — Day 3 Model Evaluation Report

**Model:** YOLOv8n (`baseline_n_50/weights/best.pt`)  
**Dataset:** `dataset/raw/data.yaml` (Cars & Pedestrians)  
**Split:** Validation (`val`)  
**Hardware / Device:** Intel Core i5-9400F CPU  
**Image Size:** 640 × 640  
**Batch Size:** 4  

---

## 1. Overall Summary Metrics

| Metric | Score | Percentage |
| :--- | :--- | :--- |
| **Precision (P)** | **0.789** | 78.9% |
| **Recall (R)** | **0.320** | 32.0% |
| **mAP@50** | **0.356** | 35.6% |
| **mAP@50–95** | **0.231** | 23.1% |

- **Total Validation Images:** 70  
- **Total Ground Truth Instances:** 524  
- **Inference Latency:** ~50.3 ms / image (Total: ~53.7 ms per image including pre/post-processing on CPU)

---

## 2. Per-Class Performance Breakdown

| Class | Instances | Precision (P) | Recall (R) | mAP@50 | mAP@50–95 | Status / Observation |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **vehicule** | 325 | **0.814** | **0.752** | **0.849** | **0.633** | Strongest performer; adequate training data |
| **bike** | 15 | 0.675 | 0.733 | 0.566 | 0.377 | Good balance of precision & recall |
| **crosswalk** | 10 | 0.576 | 0.500 | 0.601 | 0.433 | Moderate detection capability |
| **pedestrian** | 117 | 0.772 | 0.479 | 0.594 | 0.292 | High precision, but misses over half of instances |
| **green light** | 29 | 0.922 | 0.207 | 0.376 | 0.232 | Detects accurately when confident, but low recall (small objects) |
| **40** (speed sign) | 3 | 0.724 | 0.333 | 0.335 | 0.235 | Very few instances; limited learning |
| **red light** | 15 | 0.407 | 0.200 | 0.236 | 0.105 | Struggles with small scale / occlusion |
| **60** (speed sign) | 3 | 1.000 | 0.000 | 0.004 | 0.001 | Severe underrepresentation (almost no detections) |
| **roundabout 1** | 3 | 1.000 | 0.000 | 0.000 | 0.000 | Zero recall (no true positives detected) |
| **roundabout 2** | 4 | 1.000 | 0.000 | 0.000 | 0.000 | Zero recall (no true positives detected) |

---

## 3. Key Diagnostic Findings

1. **High Precision vs. Low Recall Disparity (P: 0.789 vs. R: 0.320):**
   - The model is conservative: when it predicts a bounding box, it is correct nearly 79% of the time.
   - However, it misses approximately 68% of actual objects in the validation set.
2. **Severe Class Imbalance:**
   - `vehicule` (325 instances) and `pedestrian` (117 instances) dominate the dataset, accounting for >84% of all instances.
   - Infrequent classes (`roundabout 1`, `roundabout 2`, `40`, `60`) have only 3–4 samples in validation and perform poorly (mAP near 0).
3. **Small Object Challenges:**
   - Traffic lights (`red light`, `green light`) show very low recall (~20%), typical for small-scale targets at 640×640 resolution without specialized feature pyramid tuning or higher resolution.

---

## 4. Next Step Recommendations

- **Screening & Hyperparameter Tuning:** Evaluate smaller learning rates (`lr0=0.005`) or alternative optimizers as planned in `day3_s3_tuning.py`.
- **Data Augmentation:** Apply scale jitter and mosaic augmentations to improve small object recall (traffic lights, pedestrians).
- **Higher Resolution Validation:** Test with `imgsz=1280` or `imgsz=800` to evaluate whether small targets gain significant recall.
