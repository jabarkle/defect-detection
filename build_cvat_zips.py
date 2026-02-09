#!/usr/bin/env python3
"""
Build YOLO 1.1 formatted zip files for CVAT import.

Creates:
- Images_CVAT.zip: Contains obj.data, obj.names, train.txt, and obj_train_data/ with images
- Annotations_CVAT.zip: Contains obj.names and all .txt annotation files
"""

import os
import zipfile
from pathlib import Path

# Configuration
BASE_DIR = Path("/home/jesse/Desktop/AI Manufacturing Project 1")
DATASET_DIR = BASE_DIR / "dataset_split"
OUTPUT_DIR = BASE_DIR

# Class names (from data.yaml)
CLASS_NAMES = ["spaghetti", "stringing", "warping"]

def collect_files():
    """Collect all images and labels from train, valid, and test splits."""
    images = []
    labels = []

    for split in ["train", "valid", "test"]:
        img_dir = DATASET_DIR / split / "images"
        lbl_dir = DATASET_DIR / split / "labels"

        if img_dir.exists():
            for img_file in img_dir.iterdir():
                if img_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    images.append(img_file)

                    # Find corresponding label
                    label_file = lbl_dir / (img_file.stem + ".txt")
                    if label_file.exists():
                        labels.append(label_file)
                    else:
                        # Create empty label entry for images without annotations
                        labels.append(None)

    return images, labels

def create_obj_data(num_classes):
    """Create obj.data content."""
    return f"""classes = {num_classes}
names = obj.names
train = train.txt
"""

def create_obj_names(class_names):
    """Create obj.names content."""
    return "\n".join(class_names) + "\n"

def create_train_txt(image_filenames):
    """Create train.txt content with paths like obj_train_data/image.jpg"""
    lines = [f"obj_train_data/{fname}" for fname in image_filenames]
    return "\n".join(lines) + "\n"

def build_images_zip(images, labels, output_path):
    """Build the Images zip with YOLO 1.1 structure (images + annotations together)."""
    print(f"Building {output_path}...")

    image_filenames = [img.name for img in images]

    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Add obj.data
        zf.writestr("obj.data", create_obj_data(len(CLASS_NAMES)))
        print("  Added obj.data")

        # Add obj.names
        zf.writestr("obj.names", create_obj_names(CLASS_NAMES))
        print("  Added obj.names")

        # Add train.txt
        zf.writestr("train.txt", create_train_txt(image_filenames))
        print("  Added train.txt")

        # Add images AND annotations to obj_train_data/
        for i, (img_path, lbl_path) in enumerate(zip(images, labels)):
            # Add image
            img_arcname = f"obj_train_data/{img_path.name}"
            zf.write(img_path, img_arcname)

            # Add corresponding annotation
            txt_arcname = f"obj_train_data/{img_path.stem}.txt"
            if lbl_path and lbl_path.exists():
                zf.write(lbl_path, txt_arcname)
            else:
                # Create empty annotation for images without labels
                zf.writestr(txt_arcname, "")

            if (i + 1) % 500 == 0:
                print(f"  Added {i + 1}/{len(images)} image+annotation pairs...")

        print(f"  Added {len(images)} images + annotations to obj_train_data/")

    print(f"Created {output_path}")
    print(f"  Size: {os.path.getsize(output_path) / (1024*1024):.1f} MB")

def build_annotations_zip(images, labels, output_path):
    """Build the Annotations zip with obj.names and .txt files."""
    print(f"\nBuilding {output_path}...")

    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Add obj.names
        zf.writestr("obj.names", create_obj_names(CLASS_NAMES))
        print("  Added obj.names")

        # Add annotation .txt files
        added_count = 0
        empty_count = 0
        for img_path, lbl_path in zip(images, labels):
            txt_filename = img_path.stem + ".txt"

            if lbl_path and lbl_path.exists():
                zf.write(lbl_path, txt_filename)
                added_count += 1
            else:
                # Create empty annotation file for images without labels
                zf.writestr(txt_filename, "")
                empty_count += 1

        print(f"  Added {added_count} annotation files")
        if empty_count > 0:
            print(f"  Created {empty_count} empty annotation files (images without labels)")

    print(f"Created {output_path}")
    print(f"  Size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")

def main():
    print("=" * 60)
    print("YOLO 1.1 CVAT Zip Builder")
    print("=" * 60)

    # Collect files
    print("\nCollecting files from dataset_split...")
    images, labels = collect_files()
    print(f"Found {len(images)} images")
    print(f"Found {sum(1 for l in labels if l is not None)} label files")

    # Build Images zip (with annotations included)
    images_zip_path = OUTPUT_DIR / "Images_CVAT.zip"
    build_images_zip(images, labels, images_zip_path)

    # Build Annotations zip
    annotations_zip_path = OUTPUT_DIR / "Annotations_CVAT.zip"
    build_annotations_zip(images, labels, annotations_zip_path)

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"\nCreated files:")
    print(f"  1. {images_zip_path}")
    print(f"  2. {annotations_zip_path}")
    print("\nUpload instructions:")
    print("  1. In CVAT, create task/import dataset with Images_CVAT.zip (select YOLO 1.1)")
    print("  2. Then upload Annotations_CVAT.zip as annotations (select YOLO 1.1)")

if __name__ == "__main__":
    main()
