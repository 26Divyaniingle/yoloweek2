
from pathlib import Path
from ultralytics import YOLO
import csv
import time

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DATA = PROJECT_ROOT / "dataset" / "raw" / "data.yaml"
PROJECT = ROOT / "runs" / "day3_tuning"

if not DATA.exists():
    raise FileNotFoundError(f"Dataset YAML missing: {DATA}")

experiments = [
    {"name": "control_10ep", "imgsz": 640, "batch": 4, "lr0": 0.01},
    {"name": "imgsz_512", "imgsz": 512, "batch": 4, "lr0": 0.01},
    {"name": "batch_2", "imgsz": 640, "batch": 2, "lr0": 0.01},
    {"name": "lr_0005", "imgsz": 640, "batch": 4, "lr0": 0.005},
]

summary = []

for exp in experiments:
    print(f"\nStarting experiment: {exp['name']}")
    model = YOLO("yolov8n.pt")
    start = time.perf_counter()

    model.train(
        data=str(DATA),
        epochs=10,
        imgsz=exp["imgsz"],
        batch=exp["batch"],
        workers=0,
        patience=10,
        optimizer="SGD",
        lr0=exp["lr0"],
        lrf=0.01,
        warmup_epochs=2,
        pretrained=True,
        seed=42,
        deterministic=True,
        device="cpu",
        project=str(PROJECT),
        name=exp["name"],
        exist_ok=True,
        plots=True,
        save=True,
        verbose=True,
    )

    elapsed = time.perf_counter() - start
    run_dir = PROJECT / exp["name"]
    results_csv = run_dir / "results.csv"

    row = {
        **exp,
        "epochs_requested": 10,
        "duration_minutes": round(elapsed / 60, 2),
        "mAP50": "",
        "mAP50_95": "",
    }

    if results_csv.exists():
        with results_csv.open(encoding="utf-8", newline="") as f:
            records = list(csv.DictReader(f))

        if records:
            final = records[-1]
            row["mAP50"] = final.get(
                "metrics/mAP50(B)", ""
            )
            row["mAP50_95"] = final.get(
                "metrics/mAP50-95(B)", ""
            )

    summary.append(row)

    with (PROJECT / "experiment_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

print("\nScreening complete.")
print("Summary:", PROJECT / "experiment_summary.csv")
