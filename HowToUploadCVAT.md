# How to Upload Dataset to CVAT (YOLO 1.1 Format)

This guide documents the correct process for uploading our 3D print defect detection dataset to CVAT using YOLO 1.1 format.

## Prerequisites

- Dataset in Ultralytics/Roboflow format (what we have in `dataset_split/`)
- Python 3 with standard libraries

## The Problem

CVAT's YOLO 1.1 format expects a **specific structure** that differs from the Ultralytics/Roboflow format. Uploading raw images and annotations separately doesn't work - everything must be in one zip with the correct structure.

## Required YOLO 1.1 Structure

```
archive.zip/
├── obj.data              # Config file
├── obj.names             # Class names (one per line)
├── train.txt             # List of image paths
└── obj_train_data/       # Images AND annotations together
    ├── image1.jpg
    ├── image1.txt        # Annotation for image1
    ├── image2.jpg
    ├── image2.txt        # Annotation for image2
    └── ...
```

### File Contents

**obj.data:**
```
classes = 3
names = obj.names
train = train.txt
```

**obj.names:**
```
spaghetti
stringing
warping
```

**train.txt:**
```
obj_train_data/image1.jpg
obj_train_data/image2.jpg
...
```

## Step-by-Step Process

### Step 1: Run the Build Script

```bash
cd "/home/jesse/Desktop/AI Manufacturing Project 1"
python build_cvat_zips.py
```

This creates `Images_CVAT.zip` (~720 MB) containing:
- All 4,030 images
- All 4,030 annotation `.txt` files
- Config files (`obj.data`, `obj.names`, `train.txt`)

### Step 2: Upload to CVAT

1. Go to CVAT and create a new task or import dataset
2. Select **YOLO 1.1** as the format
3. Upload `Images_CVAT.zip`
4. Wait for import to complete

**That's it!** No separate annotation upload needed - everything is in one file.

## Common Errors and Solutions

### Error: "Dataset must contain a file: obj.data"

**Cause:** Your zip doesn't have the required YOLO 1.1 structure.

**Solution:** Use the `build_cvat_zips.py` script to create a properly formatted zip.

### Error: "No such file or directory: .../obj_train_data/image.txt"

**Cause:** The annotation `.txt` files are not inside `obj_train_data/` folder alongside the images.

**Solution:** The script now places both images AND their `.txt` annotations together in `obj_train_data/`.

### Error: UTF-16 encoding issues

**Cause:** Config files (train.txt, obj.data) saved with wrong encoding.

**Solution:** The Python script creates files with UTF-8 encoding by default.

## Script Location

The build script is located at:
```
/home/jesse/Desktop/AI Manufacturing Project 1/build_cvat_zips.py
```

## What the Script Does

1. Collects all images from `dataset_split/train/`, `valid/`, and `test/` folders
2. Collects corresponding label `.txt` files
3. Creates `obj.data` with class count and file references
4. Creates `obj.names` with class names (spaghetti, stringing, warping)
5. Creates `train.txt` with paths like `obj_train_data/image.jpg`
6. Packages everything into `Images_CVAT.zip` with correct structure

## Re-running After Dataset Changes

If you modify the dataset (add/remove images, update annotations):

1. Make changes in the `dataset_split/` folder
2. Re-run `python build_cvat_zips.py`
3. Upload the new `Images_CVAT.zip` to CVAT

## Notes

- The `Annotations_CVAT.zip` file is also created but **not needed** for CVAT import since annotations are included in the main zip
- Total dataset: 4,030 images across train/valid/test splits
- 3 classes: spaghetti, stringing, warping
