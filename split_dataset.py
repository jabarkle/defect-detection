#!/usr/bin/env python3
"""
Dataset Split Script for 3D Print Defect Detection
Group 1 - 24-641 Project 1

Extracts images from Images_CVAT.zip and cleaned annotations from
YOLO1.1CVATFINAL.zip, matches them, and splits into train/val/test (70/15/15).
Uses stratified splitting based on primary class per image.
"""

import os
import zipfile
import shutil
import random
import yaml
from pathlib import Path
from collections import Counter, defaultdict

# ── Configuration ──────────────────────────────────────────────────────────
PROJECT_DIR = Path("/home/jesse/Desktop/AI Manufacturing Project 1")
IMAGES_ZIP = Path("/home/jesse/Desktop/Images_CVAT.zip")
ANNOTATIONS_ZIP = PROJECT_DIR / "YOLO1.1CVATFINAL.zip"
OUTPUT_DIR = PROJECT_DIR / "dataset_split"
TEMP_DIR = PROJECT_DIR / "_temp_extract"

CLASS_NAMES = ["spaghetti", "stringing", "warping"]
SPLIT_RATIOS = {"train": 0.70, "valid": 0.15, "test": 0.15}
RANDOM_SEED = 42


def extract_annotations(zip_path, dest):
    """Extract annotation .txt files from CVAT YOLO 1.1 export."""
    print(f"Extracting annotations from {zip_path.name}...")
    annotations = {}
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for info in zf.infolist():
            if info.filename.startswith("obj_train_data/") and info.filename.endswith(".txt"):
                basename = Path(info.filename).stem  # e.g. imagename
                dest_path = dest / "labels" / f"{basename}.txt"
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, open(dest_path, 'wb') as dst:
                    dst.write(src.read())
                annotations[basename] = dest_path
    print(f"  Extracted {len(annotations)} annotation files")
    return annotations


def extract_matching_images(zip_path, basenames, dest):
    """Extract only the images whose basenames match our annotations."""
    print(f"Extracting matching images from {zip_path.name}...")
    images = {}
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for info in zf.infolist():
            if info.filename.startswith("obj_train_data/") and info.filename.endswith(".jpg"):
                basename = Path(info.filename).stem
                if basename in basenames:
                    dest_path = dest / "images" / f"{basename}.jpg"
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(info) as src, open(dest_path, 'wb') as dst:
                        dst.write(src.read())
                    images[basename] = dest_path
    print(f"  Extracted {len(images)} matching images")
    return images


def get_primary_class(label_path):
    """Get the primary (most frequent) class in an annotation file.
    Returns -1 for empty annotations (background images)."""
    classes = []
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                classes.append(int(line.split()[0]))
    if not classes:
        return -1  # background / no objects
    return Counter(classes).most_common(1)[0][0]


def stratified_split(basenames, label_dir, ratios, seed=42):
    """Split basenames into train/valid/test using stratified sampling."""
    random.seed(seed)

    # Group by primary class
    class_groups = defaultdict(list)
    for bn in basenames:
        label_path = label_dir / f"{bn}.txt"
        primary = get_primary_class(label_path)
        class_groups[primary].append(bn)

    splits = {"train": [], "valid": [], "test": []}

    for cls, items in sorted(class_groups.items()):
        random.shuffle(items)
        n = len(items)
        n_train = int(n * ratios["train"])
        n_valid = int(n * ratios["valid"])
        # rest goes to test
        splits["train"].extend(items[:n_train])
        splits["valid"].extend(items[n_train:n_train + n_valid])
        splits["test"].extend(items[n_train + n_valid:])

    # Shuffle within each split
    for key in splits:
        random.shuffle(splits[key])

    return splits


def copy_split_files(splits, temp_dir, output_dir):
    """Copy files into the final dataset_split directory structure."""
    for split_name, basenames in splits.items():
        img_dst = output_dir / split_name / "images"
        lbl_dst = output_dir / split_name / "labels"
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        for bn in basenames:
            src_img = temp_dir / "images" / f"{bn}.jpg"
            src_lbl = temp_dir / "labels" / f"{bn}.txt"
            if src_img.exists():
                shutil.copy2(src_img, img_dst / f"{bn}.jpg")
            if src_lbl.exists():
                shutil.copy2(src_lbl, lbl_dst / f"{bn}.txt")


def create_data_yaml(output_dir):
    """Create the YOLO data.yaml configuration file."""
    data = {
        "path": str(output_dir),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    print(f"\nCreated {yaml_path}")
    return yaml_path


def print_split_statistics(splits, label_dir):
    """Print detailed statistics for each split."""
    print("\n" + "=" * 65)
    print("DATASET SPLIT SUMMARY")
    print("=" * 65)

    total = sum(len(v) for v in splits.values())
    print(f"Total images: {total}\n")

    for split_name in ["train", "valid", "test"]:
        basenames = splits[split_name]
        n = len(basenames)
        pct = n / total * 100

        # Count objects per class
        class_counts = Counter()
        empty = 0
        for bn in basenames:
            lbl = label_dir / f"{bn}.txt"
            with open(lbl, 'r') as f:
                lines = [l.strip() for l in f if l.strip()]
            if not lines:
                empty += 1
            for line in lines:
                cls_id = int(line.split()[0])
                class_counts[cls_id] += 1

        print(f"  {split_name:6s}: {n:5d} images ({pct:5.1f}%)")
        for i, name in enumerate(CLASS_NAMES):
            print(f"          {name:12s}: {class_counts.get(i, 0):5d} objects")
        if empty:
            print(f"          {'(empty)':12s}: {empty:5d} images")

    print("=" * 65)


def validate_pairs(output_dir):
    """Spot-check that every image has a label and vice versa."""
    issues = []
    for split in ["train", "valid", "test"]:
        imgs = {p.stem for p in (output_dir / split / "images").glob("*.jpg")}
        lbls = {p.stem for p in (output_dir / split / "labels").glob("*.txt")}
        img_only = imgs - lbls
        lbl_only = lbls - imgs
        if img_only:
            issues.append(f"  {split}: {len(img_only)} images without labels")
        if lbl_only:
            issues.append(f"  {split}: {len(lbl_only)} labels without images")

    if issues:
        print("\nWARNING - Mismatches found:")
        for i in issues:
            print(i)
    else:
        print("\nValidation PASSED: every image has a matching label in all splits.")


def main():
    # Clean up any previous output
    if OUTPUT_DIR.exists():
        print(f"Removing existing {OUTPUT_DIR}...")
        shutil.rmtree(OUTPUT_DIR)
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
    TEMP_DIR.mkdir(parents=True)

    # Step 1: Extract cleaned annotations
    annotations = extract_annotations(ANNOTATIONS_ZIP, TEMP_DIR)
    annotation_basenames = set(annotations.keys())

    # Step 2: Extract only matching images
    images = extract_matching_images(IMAGES_ZIP, annotation_basenames, TEMP_DIR)

    # Step 3: Keep only basenames that have BOTH image and annotation
    matched = annotation_basenames & set(images.keys())
    print(f"\nMatched pairs: {len(matched)}  "
          f"(annotations: {len(annotation_basenames)}, images found: {len(images)})")

    if len(matched) != len(annotation_basenames):
        missing = annotation_basenames - set(images.keys())
        print(f"WARNING: {len(missing)} annotations have no matching image!")
        for m in list(missing)[:5]:
            print(f"  - {m}")

    # Step 4: Stratified split
    label_dir = TEMP_DIR / "labels"
    splits = stratified_split(sorted(matched), label_dir, SPLIT_RATIOS, RANDOM_SEED)

    # Step 5: Copy into final structure
    OUTPUT_DIR.mkdir(parents=True)
    copy_split_files(splits, TEMP_DIR, OUTPUT_DIR)

    # Step 6: Create data.yaml
    create_data_yaml(OUTPUT_DIR)

    # Step 7: Statistics & validation
    print_split_statistics(splits, label_dir)
    validate_pairs(OUTPUT_DIR)

    # Cleanup temp
    print(f"\nCleaning up temp directory...")
    shutil.rmtree(TEMP_DIR)

    print("\nDone! Dataset ready at:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
