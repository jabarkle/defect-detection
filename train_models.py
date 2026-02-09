#!/usr/bin/env python3
"""
YOLO Model Training Script for 3D Print Defect Detection
Group 1 - 24-641 Project 1

Trains YOLOv9 and YOLOv11 with various hyperparameters, evaluates on the
test set, generates all required plots, and writes results_summary.json.

Usage:
    python train_models.py            # Full training (all experiments)
    python train_models.py --quick    # Quick sanity check (3 epochs)
    python train_models.py --plots    # Re-generate plots from existing results
    python train_models.py --eval     # Re-run test evaluation only
"""

import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from ultralytics import YOLO

# Help avoid CUDA fragmentation on smaller GPUs
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ── Configuration ──────────────────────────────────────────────────────────
PROJECT_DIR = Path("/home/jesse/Desktop/AI Manufacturing Project 1")
DATASET_DIR = PROJECT_DIR / "dataset_split"
DATA_YAML = DATASET_DIR / "data.yaml"
RESULTS_DIR = PROJECT_DIR / "training_results"
PLOTS_DIR = PROJECT_DIR / "plots"

YOLOV9_MODEL = PROJECT_DIR / "yolov9s.pt"
YOLOV11_MODEL = PROJECT_DIR / "yolo11s.pt"

EPOCHS = 15
IMG_SIZE = 640
DEFAULT_BATCH = 16
DEFAULT_LR = 0.01

# Image size fallbacks for large batches that may OOM at 640
# (RTX 4060 Laptop 8 GB cannot fit YOLOv9s batch=32 at 640px)
IMGSZ_FALLBACKS = [640, 576, 512]

# Experiment matrix
LEARNING_RATES = {"default": DEFAULT_LR, "5x": DEFAULT_LR * 5, "0.2x": DEFAULT_LR * 0.2}
BATCH_SIZES = {"batch_8": 8, "batch_16": DEFAULT_BATCH, "batch_32": 32}

# ── Helpers ────────────────────────────────────────────────────────────────

def setup_directories():
    RESULTS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)


def _clear_gpu():
    """Free GPU memory between experiments."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


def _is_done(tag, min_epochs=EPOCHS):
    """Check if an experiment already completed (has full results.csv)."""
    csv = RESULTS_DIR / tag / "results.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        if len(df) >= min_epochs:
            print(f"  SKIP {tag} — already complete ({len(df)} epochs)")
            EXPERIMENT_PLAN[tag] = RESULTS_DIR / tag
            return True
    return False


def _train_once(model_path, tag, epochs, batch, lr, optimizer, imgsz, workers):
    """Run a single training attempt. Raises RuntimeError on OOM."""
    model = YOLO(str(model_path))
    results = model.train(
        data=str(DATA_YAML),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        lr0=lr,
        optimizer=optimizer,
        project=str(RESULTS_DIR),
        name=tag,
        exist_ok=True,
        plots=True,
        save=True,
        verbose=True,
        workers=workers,
    )
    return results


def train_model(model_path, model_name, experiment_name,
                epochs=EPOCHS, batch=DEFAULT_BATCH, lr=DEFAULT_LR,
                optimizer="auto"):
    """Train a YOLO model with automatic image-size fallback on OOM."""
    tag = f"{model_name}_{experiment_name}"

    # Skip if already done
    if _is_done(tag, min_epochs=epochs):
        return None, RESULTS_DIR / tag

    _clear_gpu()

    # Determine image sizes to try: only use fallbacks for large batches
    sizes_to_try = IMGSZ_FALLBACKS if batch >= 32 else [IMG_SIZE]
    workers = 0 if batch >= 32 else 8

    for imgsz in sizes_to_try:
        # Remove partial results from a prior failed attempt
        partial_dir = RESULTS_DIR / tag
        if partial_dir.exists() and not (partial_dir / "results.csv").exists():
            shutil.rmtree(partial_dir)

        _clear_gpu()
        print(f"\n{'='*60}")
        print(f"TRAINING  {tag}   epochs={epochs}  batch={batch}  lr={lr}"
              f"  optimizer={optimizer}  imgsz={imgsz}")
        print(f"{'='*60}")

        try:
            _train_once(model_path, tag, epochs, batch, lr, optimizer, imgsz, workers)
            return None, RESULTS_DIR / tag
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n  *** OOM at imgsz={imgsz} — trying smaller ***\n")
                _clear_gpu()
                continue
            raise

    # All sizes exhausted
    print(f"\n  *** FAILED {tag} — all image sizes OOM'd. ***\n")
    return None, RESULTS_DIR / tag


def load_csv(results_dir):
    """Load results.csv from a training run, return DataFrame or None."""
    csv_path = results_dir / "results.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        return df
    return None


# ── Evaluation ─────────────────────────────────────────────────────────────

def evaluate_on_test(results_dir, tag, conf=0.5):
    """Run validation on the test split and measure inference time."""
    best_pt = results_dir / "weights" / "best.pt"
    if not best_pt.exists():
        print(f"  SKIP eval for {tag}: no best.pt found")
        return None

    print(f"\n  Evaluating {tag} on test set (conf={conf})...")
    model = YOLO(str(best_pt))

    # ── Metrics on test split ──
    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        conf=conf,
        save_json=True,
        plots=True,
    )

    # ── Inference speed (batch=1) ──
    test_images = list((DATASET_DIR / "test" / "images").glob("*.jpg"))[:50]
    if test_images:
        t0 = time.perf_counter()
        for img in test_images:
            model.predict(str(img), conf=conf, verbose=False, save=False)
        elapsed = time.perf_counter() - t0
        per_image_ms = (elapsed / len(test_images)) * 1000
    else:
        per_image_ms = None

    # ── FP / FN counting ──
    fp_count = 0
    fn_count = 0
    tp_count = 0
    total_gt = 0
    total_pred = 0

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

        preds = model.predict(str(img_path), conf=conf, verbose=False, save=False)
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
        "tag": tag,
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "inference_ms_per_image": round(per_image_ms, 2) if per_image_ms else None,
        "conf_threshold": conf,
        "total_gt_objects": total_gt,
        "total_pred_objects": total_pred,
        "true_positives": tp_count,
        "false_positives": fp_count,
        "false_negatives": fn_count,
    }
    print(f"  {tag}  mAP50={info['mAP50']:.4f}  mAP50-95={info['mAP50_95']:.4f}  "
          f"FP={fp_count}  FN={fn_count}  infer={info['inference_ms_per_image']}ms")
    return info


# ── Plotting ───────────────────────────────────────────────────────────────

def _save(fig, name):
    fig.savefig(PLOTS_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot: {name}")


def plot_val_loss_vs_epochs(results_map, filename):
    """Validation loss (box+cls+dfl summed) vs epochs for each entry."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, df in results_map.items():
        if df is None:
            continue
        val_loss = df["val/box_loss"] + df["val/cls_loss"] + df["val/dfl_loss"]
        ax.plot(df["epoch"] + 1, val_loss, marker="o", markersize=4, label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total Validation Loss")
    ax.set_title("Validation Loss vs. Epochs")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, filename)


def plot_map50_95_vs_epochs(results_map, filename):
    """mAP50-95 vs epochs."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, df in results_map.items():
        if df is None:
            continue
        ax.plot(df["epoch"] + 1, df["metrics/mAP50-95(B)"], marker="o", markersize=4, label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP50-95")
    ax.set_title("mAP50-95 vs. Epochs")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, filename)


def plot_six_panel(v9_df, v11_df, filename):
    """box_loss, cls_loss, dfl_loss, Precision, Recall, mAP50 — both models."""
    metrics = [
        ("val/box_loss",          "Box Loss (val)"),
        ("val/cls_loss",          "Cls Loss (val)"),
        ("val/dfl_loss",          "DFL Loss (val)"),
        ("metrics/precision(B)",  "Precision"),
        ("metrics/recall(B)",     "Recall"),
        ("metrics/mAP50(B)",      "mAP50"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for ax, (col, title) in zip(axes.flatten(), metrics):
        for df, name, marker in [(v9_df, "YOLOv9", "o"), (v11_df, "YOLOv11", "s")]:
            if df is not None and col in df.columns:
                ax.plot(df["epoch"] + 1, df[col], marker=marker, markersize=4, label=name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.set_title(f"{title} vs. Epochs")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, filename)


def plot_lr_comparison(results_map, model_name):
    """Val loss + mAP50 side-by-side for each learning rate."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    for lr_label, df in results_map.items():
        if df is None:
            continue
        val_loss = df["val/box_loss"] + df["val/cls_loss"] + df["val/dfl_loss"]
        ax.plot(df["epoch"] + 1, val_loss, marker="o", markersize=4, label=lr_label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total Validation Loss")
    ax.set_title(f"{model_name} — Val Loss vs. Learning Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for lr_label, df in results_map.items():
        if df is None:
            continue
        ax.plot(df["epoch"] + 1, df["metrics/mAP50(B)"], marker="o", markersize=4, label=lr_label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP50")
    ax.set_title(f"{model_name} — mAP50 vs. Learning Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, f"{model_name}_lr_comparison.png")


def plot_batch_comparison(results_map, model_name):
    """Val loss + mAP50 side-by-side for each batch size."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    for bs_label, df in results_map.items():
        if df is None:
            continue
        val_loss = df["val/box_loss"] + df["val/cls_loss"] + df["val/dfl_loss"]
        ax.plot(df["epoch"] + 1, val_loss, marker="o", markersize=4, label=bs_label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total Validation Loss")
    ax.set_title(f"{model_name} — Val Loss vs. Batch Size")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for bs_label, df in results_map.items():
        if df is None:
            continue
        ax.plot(df["epoch"] + 1, df["metrics/mAP50(B)"], marker="o", markersize=4, label=bs_label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP50")
    ax.set_title(f"{model_name} — mAP50 vs. Batch Size")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, f"{model_name}_batch_comparison.png")


# ── Experiment orchestration ───────────────────────────────────────────────

EXPERIMENT_PLAN = {}  # filled at runtime: tag -> results_dir


def _tag(model, exp):
    return f"{model}_{exp}"


def run_all_training():
    """Train every required experiment, avoiding redundant runs."""
    setup_directories()

    for model_name, model_path in [("yolov9", YOLOV9_MODEL), ("yolov11", YOLOV11_MODEL)]:
        if not model_path.exists():
            print(f"WARNING: {model_path} not found — skipping {model_name}")
            continue

        # 1. Default (lr=default, batch=16) — this IS lr_default AND batch_16
        #    Uses SGD so lr0 is respected (auto optimizer overrides lr0)
        tag = _tag(model_name, "default")
        _, rdir = train_model(model_path, model_name, "default", optimizer="SGD")
        EXPERIMENT_PLAN[tag] = rdir

        # 2. LR 5x
        tag = _tag(model_name, "lr_5x")
        _, rdir = train_model(model_path, model_name, "lr_5x",
                              lr=LEARNING_RATES["5x"], optimizer="SGD")
        EXPERIMENT_PLAN[tag] = rdir

        # 3. LR 0.2x
        tag = _tag(model_name, "lr_0.2x")
        _, rdir = train_model(model_path, model_name, "lr_0.2x",
                              lr=LEARNING_RATES["0.2x"], optimizer="SGD")
        EXPERIMENT_PLAN[tag] = rdir

        # 4. Batch 8
        tag = _tag(model_name, "batch_8")
        _, rdir = train_model(model_path, model_name, "batch_8", batch=8)
        EXPERIMENT_PLAN[tag] = rdir

        # 5. Batch 32 — uses automatic imgsz fallback on OOM
        tag = _tag(model_name, "batch_32")
        _, rdir = train_model(model_path, model_name, "batch_32", batch=32)
        EXPERIMENT_PLAN[tag] = rdir

    print(f"\nAll training complete. {len(EXPERIMENT_PLAN)} experiments saved.")


def run_all_evaluations():
    """Evaluate the best default model for each architecture on the test set."""
    eval_results = []
    for model_name in ["yolov9", "yolov11"]:
        tag = _tag(model_name, "default")
        rdir = EXPERIMENT_PLAN.get(tag) or RESULTS_DIR / tag
        if rdir.exists():
            info = evaluate_on_test(rdir, tag)
            if info:
                eval_results.append(info)
    return eval_results


def discover_experiments():
    """Populate EXPERIMENT_PLAN from existing result directories on disk."""
    if RESULTS_DIR.exists():
        for d in sorted(RESULTS_DIR.iterdir()):
            if d.is_dir() and (d / "results.csv").exists():
                EXPERIMENT_PLAN[d.name] = d


def generate_all_plots():
    """Generate every required plot from existing results CSVs."""
    setup_directories()
    discover_experiments()
    print(f"\nGenerating plots from {len(EXPERIMENT_PLAN)} experiments...")

    csvs = {tag: load_csv(rdir) for tag, rdir in EXPERIMENT_PLAN.items()}

    v9_def = csvs.get("yolov9_default")
    v11_def = csvs.get("yolov11_default")

    # 1. Validation loss vs epochs (both models)
    plot_val_loss_vs_epochs({"YOLOv9": v9_def, "YOLOv11": v11_def},
                            "val_loss_vs_epochs.png")

    # 2. mAP50-95 vs epochs (both models)
    plot_map50_95_vs_epochs({"YOLOv9": v9_def, "YOLOv11": v11_def},
                            "mAP50_95_vs_epochs.png")

    # 3. Six-panel: box/cls/dfl loss, Precision, Recall, mAP50
    plot_six_panel(v9_def, v11_def, "six_panel_comparison.png")

    # 4 & 5. LR comparison for each model
    for model_name in ["yolov9", "yolov11"]:
        nice = "YOLOv9" if model_name == "yolov9" else "YOLOv11"
        lr_map = {
            "default (1x)": csvs.get(f"{model_name}_default"),
            "5x":           csvs.get(f"{model_name}_lr_5x"),
            "0.2x":         csvs.get(f"{model_name}_lr_0.2x"),
        }
        plot_lr_comparison(lr_map, nice)

    # 6 & 7. Batch comparison for each model (default = batch_16)
    for model_name in ["yolov9", "yolov11"]:
        nice = "YOLOv9" if model_name == "yolov9" else "YOLOv11"
        batch_map = {
            "batch 8":  csvs.get(f"{model_name}_batch_8"),
            "batch 16 (default)": csvs.get(f"{model_name}_default"),
            "batch 32": csvs.get(f"{model_name}_batch_32"),
        }
        plot_batch_comparison(batch_map, nice)

    print("All plots generated.")


def write_summary(eval_results):
    """Write results_summary.json with all experiment metrics."""
    discover_experiments()
    summary = {"experiments": {}, "test_evaluations": eval_results}

    for tag, rdir in EXPERIMENT_PLAN.items():
        df = load_csv(rdir)
        if df is not None:
            last = df.iloc[-1]
            summary["experiments"][tag] = {
                "epochs": int(last["epoch"]) + 1,
                "final_val_box_loss": float(last.get("val/box_loss", 0)),
                "final_val_cls_loss": float(last.get("val/cls_loss", 0)),
                "final_val_dfl_loss": float(last.get("val/dfl_loss", 0)),
                "final_mAP50": float(last.get("metrics/mAP50(B)", 0)),
                "final_mAP50_95": float(last.get("metrics/mAP50-95(B)", 0)),
                "final_precision": float(last.get("metrics/precision(B)", 0)),
                "final_recall": float(last.get("metrics/recall(B)", 0)),
            }

    out_path = PROJECT_DIR / "results_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {out_path}")


# ── CLI ────────────────────────────────────────────────────────────────────

def quick_test():
    setup_directories()
    for model_name, model_path in [("yolov9", YOLOV9_MODEL), ("yolov11", YOLOV11_MODEL)]:
        if model_path.exists():
            _, rdir = train_model(model_path, model_name, "quick_test", epochs=3)
            EXPERIMENT_PLAN[_tag(model_name, "quick_test")] = rdir
            break
    else:
        print("No model files found!")


def main():
    if "--quick" in sys.argv:
        quick_test()
        return

    if "--plots" in sys.argv:
        generate_all_plots()
        return

    if "--eval" in sys.argv:
        discover_experiments()
        evals = run_all_evaluations()
        write_summary(evals)
        return

    # Full pipeline
    run_all_training()
    evals = run_all_evaluations()
    generate_all_plots()
    write_summary(evals)


if __name__ == "__main__":
    main()
