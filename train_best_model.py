#!/usr/bin/env python3
"""
Extended Training — Best Model for Demo Day
Group 1 - 24-641 Project 1

Our initial 15-epoch sweep found that YOLOv11s with SGD at lr=0.002 (0.2×
default) achieved the highest validation mAP50 (0.333) across all experiments.
YOLOv11 also has the fastest inference (~16 ms/image), which is ideal for
real-time demo.

This script trains that configuration for 50 epochs to push accuracy higher,
then evaluates on the test set and appends results to results_summary.json.

Usage:
    python train_best_model.py
"""

import gc
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import pandas as pd
from ultralytics import YOLO

# ── Configuration ──────────────────────────────────────────────────────────
PROJECT_DIR = Path("/home/jesse/Desktop/AI Manufacturing Project 1")
DATASET_DIR = PROJECT_DIR / "dataset_split"
DATA_YAML = DATASET_DIR / "data.yaml"
RESULTS_DIR = PROJECT_DIR / "training_results"

MODEL_PATH = PROJECT_DIR / "yolo11s.pt"        # YOLOv11 small
TAG = "yolov11_best_extended"
EPOCHS = 50
BATCH = 16
LR = 0.002          # 0.2× default — best from our LR sweep
OPTIMIZER = "SGD"    # Must use SGD so lr0 is respected
IMG_SIZE = 640
CONF_THRESHOLD = 0.5


def clear_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


# ── Training ───────────────────────────────────────────────────────────────

def train():
    output_dir = RESULTS_DIR / TAG
    csv_path = output_dir / "results.csv"

    # Skip if already complete
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if len(df) >= EPOCHS:
            print(f"Training already complete ({len(df)} epochs). Skipping.")
            return output_dir

    clear_gpu()

    print(f"\n{'='*60}")
    print(f"BEST MODEL EXTENDED TRAINING")
    print(f"  Model : YOLOv11s ({MODEL_PATH.name})")
    print(f"  Config: SGD  lr={LR}  batch={BATCH}  imgsz={IMG_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}\n")

    model = YOLO(str(MODEL_PATH))
    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        lr0=LR,
        optimizer=OPTIMIZER,
        project=str(RESULTS_DIR),
        name=TAG,
        exist_ok=True,
        plots=True,
        save=True,
        verbose=True,
        workers=8,
    )
    return output_dir


# ── Evaluation ─────────────────────────────────────────────────────────────

def evaluate(output_dir):
    best_pt = output_dir / "weights" / "best.pt"
    if not best_pt.exists():
        print("ERROR: best.pt not found — training may have failed.")
        return None

    print(f"\nEvaluating {TAG} on test set (conf={CONF_THRESHOLD})...")
    model = YOLO(str(best_pt))

    # Official metrics
    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        conf=CONF_THRESHOLD,
        save_json=True,
        plots=True,
    )

    # Inference speed (batch=1)
    test_images = list((DATASET_DIR / "test" / "images").glob("*.jpg"))[:50]
    t0 = time.perf_counter()
    for img in test_images:
        model.predict(str(img), conf=CONF_THRESHOLD, verbose=False, save=False)
    elapsed = time.perf_counter() - t0
    per_image_ms = (elapsed / len(test_images)) * 1000

    # FP / FN counting
    fp_count = fn_count = tp_count = total_gt = total_pred = 0
    test_labels_dir = DATASET_DIR / "test" / "labels"

    for img_path in (DATASET_DIR / "test" / "images").glob("*.jpg"):
        label_path = test_labels_dir / f"{img_path.stem}.txt"

        gt_boxes = []
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        gt_boxes.append(int(parts[0]))
        total_gt += len(gt_boxes)

        preds = model.predict(str(img_path), conf=CONF_THRESHOLD, verbose=False, save=False)
        pred_boxes = []
        for r in preds:
            if r.boxes is not None:
                for box in r.boxes:
                    pred_boxes.append(int(box.cls[0]))
        total_pred += len(pred_boxes)

        for cls_id in range(3):
            gt_n = gt_boxes.count(cls_id)
            pr_n = pred_boxes.count(cls_id)
            matched = min(gt_n, pr_n)
            tp_count += matched
            fp_count += max(0, pr_n - gt_n)
            fn_count += max(0, gt_n - pr_n)

    info = {
        "tag": TAG,
        "model": "YOLOv11s",
        "epochs": EPOCHS,
        "optimizer": OPTIMIZER,
        "lr0": LR,
        "batch": BATCH,
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "inference_ms_per_image": round(per_image_ms, 2),
        "conf_threshold": CONF_THRESHOLD,
        "total_gt_objects": total_gt,
        "total_pred_objects": total_pred,
        "true_positives": tp_count,
        "false_positives": fp_count,
        "false_negatives": fn_count,
    }

    print(f"\n{'='*60}")
    print(f"BEST MODEL RESULTS")
    print(f"  mAP50     : {info['mAP50']:.4f}")
    print(f"  mAP50-95  : {info['mAP50_95']:.4f}")
    print(f"  Precision : {info['precision']:.4f}")
    print(f"  Recall    : {info['recall']:.4f}")
    print(f"  Infer/img : {info['inference_ms_per_image']} ms")
    print(f"  TP={tp_count}  FP={fp_count}  FN={fn_count}")
    print(f"{'='*60}")

    return info


# ── Save to summary ───────────────────────────────────────────────────────

def update_summary(eval_info):
    """Append the extended-training results to results_summary.json."""
    summary_path = PROJECT_DIR / "results_summary.json"

    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
    else:
        summary = {"experiments": {}, "test_evaluations": []}

    # Add experiment entry
    csv_path = RESULTS_DIR / TAG / "results.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        last = df.iloc[-1]
        summary["experiments"][TAG] = {
            "epochs": int(last["epoch"]) + 1,
            "final_val_box_loss": float(last.get("val/box_loss", 0)),
            "final_val_cls_loss": float(last.get("val/cls_loss", 0)),
            "final_val_dfl_loss": float(last.get("val/dfl_loss", 0)),
            "final_mAP50": float(last.get("metrics/mAP50(B)", 0)),
            "final_mAP50_95": float(last.get("metrics/mAP50-95(B)", 0)),
            "final_precision": float(last.get("metrics/precision(B)", 0)),
            "final_recall": float(last.get("metrics/recall(B)", 0)),
        }

    # Add / replace test evaluation
    if eval_info:
        # Remove any prior entry for this tag
        summary["test_evaluations"] = [
            e for e in summary.get("test_evaluations", []) if e.get("tag") != TAG
        ]
        summary["test_evaluations"].append(eval_info)

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults appended to {summary_path}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    output_dir = train()
    eval_info = evaluate(output_dir)
    update_summary(eval_info)

    print(f"\nDone. Best weights at: {output_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
