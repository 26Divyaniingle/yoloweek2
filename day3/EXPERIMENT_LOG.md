# YOLO Week 2 — Day 3 Experiment Log

Experiment detail	Actual result
Model-	YOLOv8n
Epochs completed-	50/50
Training images-	245
Validation images-	70
Batch size-	4
Image size-	640 × 640
Training time	0.894 hours (~53.6 min)
Precision	0.789
Recall	0.320
mAP@50	0.356 (35.6%)
mAP@50–95	0.231 (23.1%)
Final box loss	0.9291
Final classification loss	0.8410
Final DFL loss	0.9330
Important observations
- Training successfully completed all 50 epochs.
- Precision recall se higher hai, isliye model kuch objects miss kar raha ho sakta hai. Per-class metrics aur confusion matrix inspect karna useful hoga.
- optimizer=auto ne AdamW select kiya aur tumhara lr0=0.01 ignore kiya. Actual learning rate 0.000714 tha.
- Dataset mein 40 aur 60 jaise class names bhi dikh rahe hain. 
