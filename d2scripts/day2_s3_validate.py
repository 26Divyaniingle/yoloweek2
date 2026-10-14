
from pathlib import Path
from collections import Counter
import hashlib
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DATASET = PROJECT_ROOT / "dataset" / "raw"
YAML_PATH = DATASET / "data.yaml"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

if not YAML_PATH.exists():
    raise FileNotFoundError(f"Missing dataset YAML: {YAML_PATH}")

with YAML_PATH.open("r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

names = config.get("names", {})
if isinstance(names, list):
    names = dict(enumerate(names))
else:
    names = {int(k): v for k, v in names.items()}

dataset_root = Path(config.get("path", DATASET))
if not dataset_root.is_absolute():
    dataset_root = (YAML_PATH.parent / dataset_root).resolve()

print("Dataset root:", dataset_root)
print("Classes:", names)

issues = []
class_counts = Counter()
empty_labels = 0
image_count = 0
label_count = 0
hash_to_images = {}

# Resolve split paths declared in data.yaml.
for split in ("train", "val", "valid", "test"):
    if split not in config:
        continue

    image_dir = Path(config[split])
    if not image_dir.is_absolute():
        image_dir = (dataset_root / image_dir).resolve()

    if not image_dir.is_dir():
        issues.append(f"Missing {split} image directory: {image_dir}")
        continue

    # Standard YOLO structure uses a sibling labels directory.
    label_dir = image_dir.parent.parent / "labels" / image_dir.name
    # Also support layouts like train/images + train/labels.
    if not label_dir.is_dir():
        label_dir = image_dir.parent / "labels"

    print(f"\nChecking {split}: {image_dir}")
    images = sorted(
        p for p in image_dir.rglob("*")
        if p.suffix.lower() in IMAGE_EXTS
    )
    image_count += len(images)

    for image_path in images:
        relative = image_path.relative_to(image_dir)
        label_path = label_dir / relative.with_suffix(".txt")

        # Detect exact duplicate image contents.
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        hash_to_images.setdefault(digest, []).append(str(image_path))

        try:
            with Image.open(image_path) as im:
                im.verify()
        except Exception as exc:
            issues.append(f"Corrupt image: {image_path} ({exc})")
            continue

        if not label_path.exists():
            issues.append(f"Missing label: {label_path}")
            continue

        label_count += 1
        lines = label_path.read_text(encoding="utf-8").splitlines()
        if not any(line.strip() for line in lines):
            empty_labels += 1
            continue

        for line_no, line in enumerate(lines, start=1):
            if not line.strip():
                continue

            parts = line.split()
            if len(parts) != 5:
                issues.append(
                    f"Malformed label: {label_path}:{line_no}"
                )
                continue

            try:
                cls = int(parts[0])
                x, y, w, h = map(float, parts[1:])
            except ValueError:
                issues.append(
                    f"Non-numeric label: {label_path}:{line_no}"
                )
                continue

            if cls not in names:
                issues.append(
                    f"Invalid class {cls}: {label_path}:{line_no}"
                )
                continue

            class_counts[cls] += 1

            if not all(0 <= v <= 1 for v in (x, y, w, h)):
                issues.append(
                    f"Out-of-range coordinates: {label_path}:{line_no}"
                )

            if w <= 0 or h <= 0:
                issues.append(
                    f"Zero/negative box size: {label_path}:{line_no}"
                )

            if x - w / 2 < -1e-6 or x + w / 2 > 1 + 1e-6:
                issues.append(
                    f"Box exceeds horizontal image bounds: "
                    f"{label_path}:{line_no}"
                )

            if y - h / 2 < -1e-6 or y + h / 2 > 1 + 1e-6:
                issues.append(
                    f"Box exceeds vertical image bounds: "
                    f"{label_path}:{line_no}"
                )

print("\n========== SUMMARY ==========")
print("Images found:", image_count)
print("Matching label files checked:", label_count)
print("Empty label files:", empty_labels)
print("\nObject counts by class:")

for cls, name in names.items():
    print(f"{cls}: {name} = {class_counts[cls]}")

print("\nDuplicate image groups:")
duplicate_groups = [
    paths for paths in hash_to_images.values() if len(paths) > 1
]
for group in duplicate_groups:
    print(group)

print("\nIssues:", len(issues))
for issue in issues:
    print("-", issue)

report_path = ROOT / "outputs" / "session3_validation.txt"
report_path.parent.mkdir(parents=True, exist_ok=True)
report_path.write_text(
    "\n".join(issues) if issues else "No validation issues found.",
    encoding="utf-8"
)
print("\nIssue report:", report_path)
