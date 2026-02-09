#!/usr/bin/env python3
"""
Generate Part 3 PDF Report — Model Preparation and Training
Group 1 - 24-641 Project 1

Creates: Group1_24-641_Project1_Model_S26.pdf

Includes:
  1. Title page
  2. Model training code (syntax-highlighted)
  3. All required plots (generated from results CSVs)
  4. Best model configuration
  5. Test-set comparison, FP/FN analysis
  6. YOLOv9 vs YOLOv11 discussion
"""

import json
import os
import textwrap
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from pygments import lex
from pygments.lexers import PythonLexer
from pygments.token import (
    Token, Comment, Keyword, Name, String, Number, Operator, Punctuation,
)

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    PageBreak, Table, TableStyle, XPreformatted, KeepTogether,
)
from reportlab.lib import colors
from reportlab.lib.colors import HexColor

# ── Configuration ──────────────────────────────────────────────────────────
PROJECT_DIR = Path("/home/jesse/Desktop/AI Manufacturing Project 1")
RESULTS_DIR = PROJECT_DIR / "training_results"
DATA_YAML = PROJECT_DIR / "dataset_split" / "data.yaml"
SUMMARY_JSON = PROJECT_DIR / "results_summary.json"
OUTPUT_PDF = PROJECT_DIR / "Group1_24-641_Project1_Model_S26.pdf"

CODE_FILES = [
    PROJECT_DIR / "train_models.py",
    PROJECT_DIR / "train_best_model.py",
]

GROUP_NUMBER = 1
TEAM_MEMBERS = ["Jesse Barkley", "Tom Wei", "Ryan Kaichain", "Maciej Sobolewski"]
SEMESTER = "Spring 2026"
COURSE_TITLE = "24-641 Manufacturing Data Analytics"

EXPERIMENTS = [
    "yolov9_default", "yolov9_lr_5x", "yolov9_lr_0.2x",
    "yolov9_batch_8", "yolov9_batch_32",
    "yolov11_default", "yolov11_lr_5x", "yolov11_lr_0.2x",
    "yolov11_batch_8", "yolov11_batch_32",
    "yolov11_best_extended",
]

# ── Pygments color theme (GitHub-style) ────────────────────────────────────
TOKEN_COLORS = {
    Comment:            "#6a737d",
    Comment.Single:     "#6a737d",
    Comment.Multiline:  "#6a737d",
    Comment.Hashbang:   "#6a737d",
    Keyword:            "#d73a49",
    Keyword.Namespace:  "#d73a49",
    Keyword.Constant:   "#005cc5",
    Keyword.Type:       "#005cc5",
    Name.Builtin:       "#005cc5",
    Name.Function:      "#6f42c1",
    Name.Function.Magic:"#6f42c1",
    Name.Class:         "#6f42c1",
    Name.Decorator:     "#e36209",
    String:             "#032f62",
    String.Doc:         "#032f62",
    String.Escape:      "#032f62",
    String.Interpol:    "#032f62",
    String.Affix:       "#032f62",
    Number:             "#005cc5",
    Number.Integer:     "#005cc5",
    Number.Float:       "#005cc5",
    Operator:           "#d73a49",
    Operator.Word:      "#d73a49",
}


def _xml_escape(text):
    """Escape XML special chars for reportlab."""
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))


def syntax_highlight_python(code_text):
    """Convert Python source to reportlab XML with <font color=...> tags."""
    lexer = PythonLexer()
    parts = []
    for ttype, value in lex(code_text, lexer):
        escaped = _xml_escape(value)
        # Walk token hierarchy to find a color
        color = None
        t = ttype
        while t is not None:
            if t in TOKEN_COLORS:
                color = TOKEN_COLORS[t]
                break
            t = t.parent
        if color:
            parts.append(f'<font color="{color}">{escaped}</font>')
        else:
            parts.append(escaped)
    return "".join(parts)


# ── Data loading ───────────────────────────────────────────────────────────

def load_csv(tag):
    csv_path = RESULTS_DIR / tag / "results.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        return df
    return None


def load_summary():
    if SUMMARY_JSON.exists():
        with open(SUMMARY_JSON) as f:
            return json.load(f)
    return {}


# ── Plot helpers ───────────────────────────────────────────────────────────

def fig_to_image(fig, width=6.5*inch):
    """Render a matplotlib figure to a reportlab Image flowable."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img = RLImage(buf, width=width, height=width * 0.6)
    # Compute height from aspect ratio
    from PIL import Image as PILImage
    pil = PILImage.open(buf)
    w, h = pil.size
    img = RLImage(buf, width=width, height=width * h / w)
    buf.seek(0)
    return img


def fig_to_image_simple(fig, width=6.5*inch, aspect=0.55):
    """Render figure to reportlab Image without PIL dependency."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return RLImage(buf, width=width, height=width * aspect)


def make_val_loss_plot(csvs):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for label, tag in [("YOLOv9 (default)", "yolov9_default"),
                       ("YOLOv11 (default)", "yolov11_default")]:
        df = csvs.get(tag)
        if df is None:
            continue
        val_loss = df["val/box_loss"] + df["val/cls_loss"] + df["val/dfl_loss"]
        ax.plot(df["epoch"] + 1, val_loss, marker="o", markersize=5, label=label)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Total Validation Loss", fontsize=12)
    ax.set_title("Validation Loss vs. Epochs (1–15)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    return fig


def make_map50_95_plot(csvs):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for label, tag in [("YOLOv9 (default)", "yolov9_default"),
                       ("YOLOv11 (default)", "yolov11_default")]:
        df = csvs.get(tag)
        if df is None:
            continue
        ax.plot(df["epoch"] + 1, df["metrics/mAP50-95(B)"],
                marker="o", markersize=5, label=label)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("mAP50-95", fontsize=12)
    ax.set_title("mAP50-95 vs. Epochs (1–15)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    return fig


def make_six_panel(csvs):
    metrics = [
        ("val/box_loss",         "Box Loss (val)"),
        ("val/cls_loss",         "Cls Loss (val)"),
        ("val/dfl_loss",         "DFL Loss (val)"),
        ("metrics/precision(B)", "Precision"),
        ("metrics/recall(B)",    "Recall"),
        ("metrics/mAP50(B)",     "mAP50"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    v9 = csvs.get("yolov9_default")
    v11 = csvs.get("yolov11_default")
    for ax, (col, title) in zip(axes.flatten(), metrics):
        for df, name, marker in [(v9, "YOLOv9", "o"), (v11, "YOLOv11", "s")]:
            if df is not None and col in df.columns:
                ax.plot(df["epoch"] + 1, df[col], marker=marker, markersize=4, label=name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.set_title(f"{title} vs. Epochs")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def make_lr_comparison(csvs, model_prefix, model_nice):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    tags = {
        f"default (lr=0.01)": f"{model_prefix}_default",
        f"5× (lr=0.05)":     f"{model_prefix}_lr_5x",
        f"0.2× (lr=0.002)":  f"{model_prefix}_lr_0.2x",
    }
    ax = axes[0]
    for lr_label, tag in tags.items():
        df = csvs.get(tag)
        if df is None:
            continue
        val_loss = df["val/box_loss"] + df["val/cls_loss"] + df["val/dfl_loss"]
        ax.plot(df["epoch"] + 1, val_loss, marker="o", markersize=4, label=lr_label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total Validation Loss")
    ax.set_title(f"{model_nice} — Validation Loss vs. Learning Rate")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for lr_label, tag in tags.items():
        df = csvs.get(tag)
        if df is None:
            continue
        ax.plot(df["epoch"] + 1, df["metrics/mAP50(B)"],
                marker="o", markersize=4, label=lr_label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP50")
    ax.set_title(f"{model_nice} — mAP50 vs. Learning Rate")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def make_batch_comparison(csvs, model_prefix, model_nice):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    tags = {
        "batch 8":             f"{model_prefix}_batch_8",
        "batch 16 (default)":  f"{model_prefix}_default",
        "batch 32":            f"{model_prefix}_batch_32",
    }
    note = ""
    if model_prefix == "yolov9":
        note = " (batch 32 @ imgsz=576)"

    ax = axes[0]
    for bs_label, tag in tags.items():
        df = csvs.get(tag)
        if df is None:
            continue
        val_loss = df["val/box_loss"] + df["val/cls_loss"] + df["val/dfl_loss"]
        ax.plot(df["epoch"] + 1, val_loss, marker="o", markersize=4, label=bs_label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total Validation Loss")
    ax.set_title(f"{model_nice} — Validation Loss vs. Batch Size{note}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for bs_label, tag in tags.items():
        df = csvs.get(tag)
        if df is None:
            continue
        ax.plot(df["epoch"] + 1, df["metrics/mAP50(B)"],
                marker="o", markersize=4, label=bs_label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP50")
    ax.set_title(f"{model_nice} — mAP50 vs. Batch Size{note}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


# ── PDF assembly ───────────────────────────────────────────────────────────

def build_pdf():
    """Main entry point — generate the full PDF report."""
    print("Loading experiment data...")
    csvs = {}
    for tag in EXPERIMENTS:
        csvs[tag] = load_csv(tag)
    summary = load_summary()

    # ── Styles ──
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "Title", parent=styles["Heading1"],
        fontSize=26, alignment=TA_CENTER, spaceAfter=10,
    )
    subtitle_style = ParagraphStyle(
        "Subtitle", parent=styles["Heading2"],
        fontSize=17, alignment=TA_CENTER, spaceAfter=6,
    )
    center_style = ParagraphStyle(
        "Center", parent=styles["Normal"],
        fontSize=13, alignment=TA_CENTER, spaceAfter=6,
    )
    heading_style = ParagraphStyle(
        "SectionHeading", parent=styles["Heading2"],
        fontSize=15, spaceBefore=14, spaceAfter=8,
    )
    subheading_style = ParagraphStyle(
        "SubHeading", parent=styles["Heading3"],
        fontSize=12, spaceBefore=10, spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["Normal"],
        fontSize=11, spaceAfter=10, leading=15, alignment=TA_JUSTIFY,
    )
    caption_style = ParagraphStyle(
        "Caption", parent=styles["Normal"],
        fontSize=10, alignment=TA_CENTER, spaceAfter=4,
        textColor=colors.grey,
    )
    code_style = ParagraphStyle(
        "Code", parent=styles["Code"],
        fontName="Courier", fontSize=6.5, leading=8,
        leftIndent=0, rightIndent=0,
        spaceBefore=0, spaceAfter=0,
        backColor=HexColor("#f6f8fa"),
    )
    code_header_style = ParagraphStyle(
        "CodeHeader", parent=styles["Heading3"],
        fontSize=11, spaceBefore=6, spaceAfter=4,
        textColor=HexColor("#24292e"),
    )
    table_header_style = ParagraphStyle(
        "TableHeader", parent=styles["Normal"],
        fontSize=9, alignment=TA_CENTER, textColor=colors.whitesmoke,
        fontName="Helvetica-Bold",
    )
    table_cell_style = ParagraphStyle(
        "TableCell", parent=styles["Normal"],
        fontSize=9, alignment=TA_CENTER,
    )

    content = []

    # ================================================================
    # PAGE 1 — Title
    # ================================================================
    content.append(Spacer(1, 2 * inch))
    content.append(Paragraph(COURSE_TITLE, title_style))
    content.append(Spacer(1, 0.25 * inch))
    content.append(Paragraph("Model Output for Project-1", subtitle_style))
    content.append(Spacer(1, 1.0 * inch))
    content.append(Paragraph(f"Group {GROUP_NUMBER}", center_style))
    content.append(Spacer(1, 0.3 * inch))
    for name in TEAM_MEMBERS:
        content.append(Paragraph(name, center_style))
    content.append(Spacer(1, 0.5 * inch))
    content.append(Paragraph(SEMESTER, center_style))
    content.append(PageBreak())

    # ================================================================
    # SECTION 1 — Training Code (syntax highlighted)
    # ================================================================
    content.append(Paragraph("1. Model Training Code", heading_style))
    content.append(Paragraph(
        "Below is the complete training code used for all experiments. "
        "<b>train_models.py</b> orchestrates the 10 hyperparameter experiments "
        "(learning-rate and batch-size sweeps for both YOLOv9 and YOLOv11), "
        "generates comparison plots, evaluates the best model on the test set, "
        "and writes a JSON summary. <b>train_best_model.py</b> runs an "
        "extended 50-epoch training of our best configuration for demo day.",
        body_style,
    ))

    for code_path in CODE_FILES:
        if not code_path.exists():
            content.append(Paragraph(f"[{code_path.name} not found]", body_style))
            continue

        content.append(Spacer(1, 0.1 * inch))
        content.append(Paragraph(f"<b>{code_path.name}</b>", code_header_style))

        code_text = code_path.read_text()
        highlighted = syntax_highlight_python(code_text)
        content.append(XPreformatted(highlighted, code_style))
        content.append(Spacer(1, 0.15 * inch))

    content.append(PageBreak())

    # ================================================================
    # SECTION 2 — Required Plots
    # ================================================================
    content.append(Paragraph("2. Training Plots", heading_style))
    content.append(Paragraph(
        "All experiments below were trained for 15 epochs on our cleaned dataset "
        "of 3,458 images (70/15/15 train/val/test split). Both YOLOv9s and YOLOv11s "
        "pretrained weights were used with SGD optimizer. Default learning rate is "
        "0.01, default batch size is 16, and default image size is 640×640.",
        body_style,
    ))

    # Plot 1: Val loss vs epochs
    print("  Generating: validation loss vs epochs...")
    content.append(Paragraph("<b>2.1 Validation Loss vs. Epochs (1–15)</b>", subheading_style))
    fig = make_val_loss_plot(csvs)
    content.append(fig_to_image_simple(fig, aspect=0.52))
    content.append(Paragraph(
        "Figure 1: Total validation loss (box + cls + dfl) for both architectures at default settings.",
        caption_style,
    ))
    content.append(PageBreak())

    # Plot 2: mAP50-95 vs epochs
    print("  Generating: mAP50-95 vs epochs...")
    content.append(Paragraph("<b>2.2 mAP50-95 vs. Epochs (1–15)</b>", subheading_style))
    fig = make_map50_95_plot(csvs)
    content.append(fig_to_image_simple(fig, aspect=0.52))
    content.append(Paragraph(
        "Figure 2: mAP50-95 over training for both architectures at default settings.",
        caption_style,
    ))
    content.append(PageBreak())

    # Plot 3: Six-panel
    print("  Generating: six-panel comparison...")
    content.append(Paragraph(
        "<b>2.3 Box Loss, Cls Loss, DFL Loss, Precision, Recall, mAP50 vs. Epochs</b>",
        subheading_style,
    ))
    fig = make_six_panel(csvs)
    content.append(fig_to_image_simple(fig, aspect=0.58))
    content.append(Paragraph(
        "Figure 3: Per-metric comparison of YOLOv9 and YOLOv11 over 15 epochs (default settings).",
        caption_style,
    ))
    content.append(PageBreak())

    # Plot 4 & 5: LR comparison
    for prefix, nice in [("yolov9", "YOLOv9"), ("yolov11", "YOLOv11")]:
        print(f"  Generating: {nice} LR comparison...")
        content.append(Paragraph(
            f"<b>2.4 {nice} — Validation Loss &amp; Accuracy vs. Learning Rate</b>"
            if prefix == "yolov9" else
            f"<b>2.5 {nice} — Validation Loss &amp; Accuracy vs. Learning Rate</b>",
            subheading_style,
        ))
        fig = make_lr_comparison(csvs, prefix, nice)
        content.append(fig_to_image_simple(fig, aspect=0.38))
        content.append(Paragraph(
            f"Figure: {nice} trained with SGD at three learning rates — "
            "default (0.01), 5× (0.05), and 0.2× (0.002).",
            caption_style,
        ))
        content.append(Spacer(1, 0.1 * inch))
        content.append(Paragraph(
            "A lower learning rate (0.2×) consistently produced the lowest validation loss "
            "and highest mAP50, while the 5× rate caused instability and poor convergence. "
            "This indicates that the default lr=0.01 is already on the high side for this "
            "dataset, and a more conservative rate gives the optimizer more time to find a "
            "better minimum."
            if prefix == "yolov9" else
            "YOLOv11 shows the same trend: the 0.2× learning rate (0.002) achieves the "
            "best performance. The 5× rate severely overshoots, resulting in high loss and "
            "low mAP. This confirms that a conservative learning rate is critical for both "
            "architectures on our relatively small 3-class defect dataset.",
            body_style,
        ))
        content.append(PageBreak())

    # Plot 6 & 7: Batch comparison
    for prefix, nice in [("yolov9", "YOLOv9"), ("yolov11", "YOLOv11")]:
        print(f"  Generating: {nice} batch comparison...")
        content.append(Paragraph(
            f"<b>2.6 {nice} — Validation Loss &amp; Accuracy vs. Batch Size</b>"
            if prefix == "yolov9" else
            f"<b>2.7 {nice} — Validation Loss &amp; Accuracy vs. Batch Size</b>",
            subheading_style,
        ))
        fig = make_batch_comparison(csvs, prefix, nice)
        content.append(fig_to_image_simple(fig, aspect=0.38))

        if prefix == "yolov9":
            content.append(Paragraph(
                "Figure: YOLOv9 trained at batch sizes 8, 16, and 32. "
                "Note: batch 32 required reducing image size from 640 to 576 px "
                "due to GPU memory constraints (RTX 4060 Laptop, 8 GB VRAM).",
                caption_style,
            ))
            content.append(Spacer(1, 0.1 * inch))
            content.append(Paragraph(
                "Batch size 32 achieved the lowest validation loss for YOLOv9, despite the "
                "reduced image resolution. The larger batch provides more stable gradient "
                "estimates per update. Batch 8 converges more slowly due to noisier gradients. "
                "<b>GPU Memory Note:</b> YOLOv9 at batch=32 with imgsz=640 exceeded the "
                "8 GB VRAM of our RTX 4060 Laptop GPU. The primary driver of memory usage "
                "is the combination of batch size and image resolution — each image occupies "
                "a large activation map through the network. Reducing imgsz from 640 to 576 "
                "(a 19% reduction in pixel count) was enough to fit within memory while "
                "preserving most of the input detail.",
                body_style,
            ))
        else:
            content.append(Paragraph(
                "Figure: YOLOv11 trained at batch sizes 8, 16, and 32 "
                "(all at imgsz=640; YOLOv11 is more memory-efficient).",
                caption_style,
            ))
            content.append(Spacer(1, 0.1 * inch))
            content.append(Paragraph(
                "YOLOv11 at batch 32 also produced the lowest validation loss among the "
                "three batch sizes tested. Unlike YOLOv9, YOLOv11 was able to train at "
                "batch=32 with full 640×640 images thanks to its more parameter-efficient "
                "architecture (9.4M vs 7.2M parameters for the small variant). Batch 16 "
                "remains a solid middle-ground for training speed and stability.",
                body_style,
            ))
        content.append(PageBreak())

    # ================================================================
    # SECTION 3 — Best Model Configuration
    # ================================================================
    content.append(Paragraph("3. Best Model Configuration", heading_style))

    content.append(Paragraph(
        "After evaluating all 10 experiments (5 per architecture), the configuration "
        "that achieved the highest validation mAP50 across the 15-epoch sweep was:",
        body_style,
    ))

    best_data = [
        ["Parameter", "Value"],
        ["Architecture", "YOLOv11s (YOLO11 small)"],
        ["Optimizer", "SGD (momentum=0.937, weight_decay=0.0005)"],
        ["Learning Rate (lr0)", "0.002 (0.2× default)"],
        ["Batch Size", "16"],
        ["Image Size", "640×640"],
        ["Epochs (sweep)", "15"],
        ["Val mAP50 @ 15 epochs", "0.333"],
        ["Val mAP50-95 @ 15 epochs", "0.152"],
        ["Total Val Loss @ 15 epochs", "5.64"],
    ]
    best_table = Table(best_data, colWidths=[2.5 * inch, 4.0 * inch])
    best_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#24292e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 1), (-1, -1), HexColor("#f6f8fa")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#f6f8fa"), colors.white]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    content.append(best_table)
    content.append(Spacer(1, 0.2 * inch))

    content.append(Paragraph(
        "The lower learning rate (0.2× default) was the most impactful parameter "
        "change. The default lr=0.01 with SGD caused aggressive updates that "
        "prevented the model from converging to a tight minimum on our relatively "
        "small dataset (3,458 images, 3 classes). Reducing lr to 0.002 gave the "
        "optimizer more fine-grained steps, resulting in lower loss and higher mAP "
        "for both YOLOv9 and YOLOv11.",
        body_style,
    ))

    content.append(Paragraph("<b>Extended Training (50 Epochs)</b>", subheading_style))
    content.append(Paragraph(
        "We then trained this best configuration for 50 epochs to produce a "
        "stronger model for our live demo. The extended training improved "
        "validation mAP50 from 0.333 to 0.366 and mAP50-95 from 0.152 to 0.176, "
        "with recall climbing from 0.336 to 0.397. The model continues to learn "
        "beyond 15 epochs, though with diminishing returns past epoch ~35.",
        body_style,
    ))
    content.append(PageBreak())

    # ================================================================
    # SECTION 4 — Test Set Comparison & Failure Analysis
    # ================================================================
    content.append(Paragraph("4. Test Set Evaluation &amp; Failure Analysis", heading_style))

    content.append(Paragraph(
        "We evaluated our default models and extended best model on the held-out test "
        "set (606 images, 1,379 ground-truth objects) at a confidence threshold of 0.5.",
        body_style,
    ))

    # Build test results table from summary
    test_evals = summary.get("test_evaluations", [])
    test_header = ["Metric", "YOLOv9\n(default, 15 ep.)",
                   "YOLOv11\n(default, 15 ep.)",
                   "YOLOv11 Best\n(0.2× lr, 50 ep.)"]

    def _get_eval(tag):
        for e in test_evals:
            if e["tag"] == tag:
                return e
        return {}

    v9d = _get_eval("yolov9_default")
    v11d = _get_eval("yolov11_default")
    v11b = _get_eval("yolov11_best_extended")

    def _fmt(val, fmt=".4f"):
        if val is None:
            return "—"
        return f"{val:{fmt}}"

    test_rows = [
        test_header,
        ["mAP50", _fmt(v9d.get("mAP50")), _fmt(v11d.get("mAP50")), _fmt(v11b.get("mAP50"))],
        ["mAP50-95", _fmt(v9d.get("mAP50_95")), _fmt(v11d.get("mAP50_95")), _fmt(v11b.get("mAP50_95"))],
        ["Precision", _fmt(v9d.get("precision")), _fmt(v11d.get("precision")), _fmt(v11b.get("precision"))],
        ["Recall", _fmt(v9d.get("recall")), _fmt(v11d.get("recall")), _fmt(v11b.get("recall"))],
        ["Inference (ms/img)", _fmt(v9d.get("inference_ms_per_image"), ".1f"),
         _fmt(v11d.get("inference_ms_per_image"), ".1f"),
         _fmt(v11b.get("inference_ms_per_image"), ".1f")],
        ["True Positives", str(v9d.get("true_positives", "—")),
         str(v11d.get("true_positives", "—")), str(v11b.get("true_positives", "—"))],
        ["False Positives", str(v9d.get("false_positives", "—")),
         str(v11d.get("false_positives", "—")), str(v11b.get("false_positives", "—"))],
        ["False Negatives", str(v9d.get("false_negatives", "—")),
         str(v11d.get("false_negatives", "—")), str(v11b.get("false_negatives", "—"))],
    ]

    col_w = [1.4 * inch, 1.6 * inch, 1.6 * inch, 1.7 * inch]
    test_table = Table(test_rows, colWidths=col_w)
    test_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#24292e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#f6f8fa"), colors.white]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    content.append(test_table)
    content.append(Spacer(1, 0.2 * inch))

    content.append(Paragraph("<b>Failure Analysis</b>", subheading_style))
    content.append(Paragraph(
        "At a confidence threshold of 0.5, all models show high false-negative counts "
        "relative to the number of ground-truth objects. This means the models are "
        "conservative — they make predictions only when highly confident, which keeps "
        "false positives very low but misses many true defects.",
        body_style,
    ))
    content.append(Paragraph(
        "The extended YOLOv11 model (50 epochs, lr=0.002) substantially reduced false "
        "negatives from 1,165 to 846 while only increasing false positives from 11 to 68. "
        "This is a favorable trade-off for a manufacturing QC system, where catching more "
        "defective parts is generally more important than the occasional false alarm.",
        body_style,
    ))
    content.append(Paragraph(
        "Common failure modes observed during inspection of predictions include:",
        body_style,
    ))
    content.append(Paragraph(
        "• <b>Stringing</b> is the hardest class to detect — thin wisps of filament are "
        "subtle and easily missed, especially when partially occluded or low-contrast "
        "against the print surface.<br/>"
        "• <b>Warping</b> varies significantly in scale and appearance, from slight edge "
        "curl to severe deformation, making it difficult for the model to generalize.<br/>"
        "• <b>Spaghetti</b> is the most visually distinctive defect and is detected most "
        "reliably, though severe cases that fill the frame can confuse bounding-box "
        "localization.",
        body_style,
    ))
    content.append(Paragraph(
        "YOLOv9 at default settings achieved the highest raw test mAP50 (0.430) despite "
        "having a lower validation mAP50 than the tuned YOLOv11. This suggests YOLOv9's "
        "deeper gradient information flow (PGI) helps it generalize better to unseen data "
        "even without extensive tuning, though at the cost of slower inference (27 ms vs. "
        "18 ms per image for YOLOv11).",
        body_style,
    ))
    content.append(PageBreak())

    # ================================================================
    # SECTION 5 — YOLOv9 vs YOLOv11
    # ================================================================
    content.append(Paragraph(
        "5. YOLOv9 vs. YOLOv11 — When to Use Which", heading_style,
    ))

    content.append(Paragraph("<b>YOLOv9 — Strengths</b>", subheading_style))
    content.append(Paragraph(
        "YOLOv9, released in February 2024, introduced two key innovations: "
        "<b>Programmable Gradient Information (PGI)</b> and the <b>Generalized Efficient "
        "Layer Aggregation Network (GELAN)</b>. PGI addresses the information bottleneck "
        "problem in deep networks — as data passes through successive layers, useful "
        "information can be lost. PGI preserves essential gradient information across "
        "the network's depth, leading to more reliable convergence and better feature "
        "extraction. GELAN enables flexible integration of computational blocks for "
        "strong parameter efficiency.",
        body_style,
    ))
    content.append(Paragraph(
        "In our experiments, YOLOv9 (default settings) achieved the highest test-set "
        "mAP50 (0.430) and mAP50-95 (0.267) among the default configurations, "
        "suggesting its deeper information-preservation mechanisms help it generalize "
        "well even without hyperparameter tuning. YOLOv9 is the better choice when "
        "<b>maximum accuracy is the priority</b> and slightly higher inference latency "
        "is acceptable.",
        body_style,
    ))

    content.append(Paragraph("<b>YOLOv11 — Strengths</b>", subheading_style))
    content.append(Paragraph(
        "YOLO11 (September 2024) builds on the YOLO lineage with an improved backbone "
        "and neck architecture optimized for efficiency. Its headline feature is "
        "<b>greater accuracy with fewer parameters</b> — YOLO11m achieves higher mAP "
        "on COCO than YOLOv8m while using 22% fewer parameters. It also supports the "
        "broadest range of tasks (detection, segmentation, classification, pose "
        "estimation, oriented bounding boxes) and is designed for seamless deployment "
        "on edge devices, cloud, and NVIDIA GPUs.",
        body_style,
    ))
    content.append(Paragraph(
        "In our results, YOLOv11 was <b>30% faster</b> at inference (18.5 ms vs. 27.0 ms "
        "per image) and fit batch=32 at full 640px resolution while YOLOv9 needed a "
        "reduced image size. When properly tuned with a lower learning rate and extended "
        "training, YOLOv11 reached a competitive mAP50 of 0.407 on the test set. "
        "YOLOv11 is the better choice when <b>speed, memory efficiency, and edge "
        "deployment</b> matter — such as real-time monitoring of 3D printers during "
        "production.",
        body_style,
    ))

    content.append(Paragraph("<b>Summary</b>", subheading_style))

    comparison_data = [
        ["Criterion", "YOLOv9", "YOLOv11"],
        ["Best for", "Max accuracy", "Speed & deployment"],
        ["Key innovation", "PGI + GELAN", "Efficient backbone"],
        ["Test mAP50 (default)", "0.430", "0.285"],
        ["Test mAP50 (tuned)", "—", "0.407 (50 ep.)"],
        ["Inference speed", "27.0 ms/img", "18.5 ms/img"],
        ["Batch 32 @ 640px", "OOM (used 576px)", "OK"],
        ["Task breadth", "Detection", "Det / Seg / Pose / OBB / Cls"],
    ]
    comp_table = Table(comparison_data, colWidths=[2.2 * inch, 2.1 * inch, 2.4 * inch])
    comp_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#24292e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#f6f8fa"), colors.white]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    content.append(comp_table)
    content.append(Spacer(1, 0.15 * inch))

    content.append(Paragraph(
        "For our demo-day application — real-time defect detection on a live 3D printer "
        "camera feed — we selected YOLOv11 as the deployment model due to its faster "
        "inference and lower memory footprint, while acknowledging that YOLOv9 offered "
        "stronger out-of-the-box accuracy on our test set.",
        body_style,
    ))

    # ── Build the document ──
    print("Building PDF...")
    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=letter,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )

    # Add page numbers
    def add_page_number(canvas, doc):
        page_num = canvas.getPageNumber()
        text = f"Group 1 — Model Output for Project-1  |  Page {page_num}"
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.grey)
        canvas.drawCentredString(letter[0] / 2, 0.4 * inch, text)
        canvas.restoreState()

    doc.build(content, onFirstPage=add_page_number, onLaterPages=add_page_number)
    print(f"\nPDF saved to: {OUTPUT_PDF}")
    print(f"File size: {os.path.getsize(OUTPUT_PDF) / 1024:.1f} KB")


if __name__ == "__main__":
    build_pdf()
