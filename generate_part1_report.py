#!/usr/bin/env python3
"""
Generate Part 1 PDF Report for 24-641 Project 1
Group 1 - Dataset Preparation Report
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib import colors
from pathlib import Path
import os

# Configuration
PROJECT_DIR = Path("/home/jesse/Desktop/AI Manufacturing Project 1")
EXAMPLES_DIR = PROJECT_DIR / "ExamplesPart1"
OUTPUT_FILE = PROJECT_DIR / "Group1_24-641_Project1_Dataset_S26.pdf"

# Team information
GROUP_NUMBER = 1
TEAM_MEMBERS = [
    "Jesse Barkley",
    "Tom Wei",
    "Ryan Kaichain",
    "Maciej Sobolewski"
]
SEMESTER = "Spring 2026"
COURSE_TITLE = "24-641 Manufacturing Data Analytics"

# Dataset information
DATASET_URL = "https://universe.roboflow.com/purvi-rathore-5amqh/3d-print-failure-detection-efvsh"
DEFECT_TYPES = ["Spaghetti", "Stringing", "Warping"]


def create_pdf():
    """Generate the Part 1 PDF report."""
    doc = SimpleDocTemplate(
        str(OUTPUT_FILE),
        pagesize=letter,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch
    )
    
    # Styles
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        alignment=TA_CENTER,
        spaceAfter=12
    )
    
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Heading2'],
        fontSize=16,
        alignment=TA_CENTER,
        spaceAfter=6
    )
    
    center_style = ParagraphStyle(
        'Center',
        parent=styles['Normal'],
        fontSize=12,
        alignment=TA_CENTER,
        spaceAfter=6
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceBefore=12,
        spaceAfter=6
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=12,
        leading=14
    )
    
    # Build document content
    content = []
    
    # ============================================
    # PAGE 1: Title Page
    # ============================================
    content.append(Spacer(1, 2*inch))
    content.append(Paragraph(COURSE_TITLE, title_style))
    content.append(Spacer(1, 0.3*inch))
    content.append(Paragraph("Dataset Preparation for Project-1", subtitle_style))
    content.append(Spacer(1, 1*inch))
    
    content.append(Paragraph(f"Group #{GROUP_NUMBER}", center_style))
    content.append(Spacer(1, 0.3*inch))
    
    for member in TEAM_MEMBERS:
        content.append(Paragraph(member, center_style))
    
    content.append(Spacer(1, 0.5*inch))
    content.append(Paragraph(SEMESTER, center_style))
    
    content.append(PageBreak())
    
    # ============================================
    # PAGE 2: Manufacturing Process Description
    # ============================================
    content.append(Paragraph("1. Manufacturing Process and Dataset Description", heading_style))
    
    process_text = """
    <b>Manufacturing Process: Fused Deposition Modeling (FDM) 3D Printing</b><br/><br/>
    
    Fused Deposition Modeling (FDM) is an additive manufacturing process where thermoplastic 
    filament is heated and extruded through a nozzle to build objects layer by layer. This 
    process is widely used for rapid prototyping, functional parts, and custom manufacturing 
    applications.<br/><br/>
    
    During FDM printing, various defects can occur that affect part quality:<br/><br/>
    
    <b>• Spaghetti:</b> A severe print failure where filament extrudes into the air without 
    adhering properly, creating a tangled mess resembling spaghetti noodles. This typically 
    occurs when the print detaches from the bed or support structures fail.<br/><br/>
    
    <b>• Stringing:</b> Thin strings or wisps of plastic left between printed parts during 
    travel moves. Caused by oozing filament when the nozzle moves between separate areas. 
    While not catastrophic, it requires post-processing cleanup.<br/><br/>
    
    <b>• Warping:</b> Deformation of the printed part where corners or edges lift from the 
    build plate. Caused by thermal contraction as layers cool at different rates, creating 
    internal stresses that curl the part upward.
    """
    content.append(Paragraph(process_text, body_style))
    
    content.append(Spacer(1, 0.2*inch))
    content.append(Paragraph("<b>Dataset Source:</b>", body_style))
    content.append(Paragraph(f"<link href='{DATASET_URL}'>{DATASET_URL}</link>", body_style))
    
    dataset_text = """
    <br/>The dataset was obtained from Roboflow Universe, containing images of 3D print 
    failures captured during actual FDM printing operations. The original dataset includes 
    multiple defect categories, which were cleaned and annotated using CVAT (Computer Vision 
    Annotation Tool) for this project.<br/><br/>
    
    <b>Dataset Statistics:</b><br/>
    • Total annotated images: 4,030<br/>
    • Training set: 2,820 images (70%)<br/>
    • Validation set: 604 images (15%)<br/>
    • Test set: 606 images (15%)<br/>
    • Defect classes: 3 (spaghetti, stringing, warping)
    """
    content.append(Paragraph(dataset_text, body_style))
    
    content.append(PageBreak())
    
    # ============================================
    # PAGES 3+: CVAT Screenshots
    # ============================================
    content.append(Paragraph("2. CVAT Annotation Screenshots", heading_style))
    content.append(Paragraph(
        "The following screenshots show annotated examples of each defect type "
        "from CVAT (Computer Vision Annotation Tool). Each defect category has "
        "5 annotated sample images demonstrating the bounding box annotations.",
        body_style
    ))
    content.append(Spacer(1, 0.2*inch))
    
    # Add screenshots for each defect type
    for defect in DEFECT_TYPES:
        content.append(Paragraph(f"<b>2.{DEFECT_TYPES.index(defect)+1} {defect} Defect Examples</b>", heading_style))
        
        # Add 5 images per defect (2 per row + 1)
        images_added = 0
        for i in range(1, 6):
            img_path = EXAMPLES_DIR / f"{defect}{i}.png"
            if img_path.exists():
                try:
                    # Add image
                    img = Image(str(img_path), width=3.2*inch, height=2.4*inch)
                    content.append(img)
                    content.append(Paragraph(f"Figure: {defect} annotation example {i}", center_style))
                    content.append(Spacer(1, 0.1*inch))
                    images_added += 1
                except Exception as e:
                    content.append(Paragraph(f"[Image {defect}{i}.png could not be loaded: {e}]", body_style))
            else:
                content.append(Paragraph(f"[Image {defect}{i}.png not found]", body_style))
        
        if images_added > 0:
            content.append(Spacer(1, 0.2*inch))
        
        # Page break after each defect type (except last)
        if defect != DEFECT_TYPES[-1]:
            content.append(PageBreak())
    
    # Build PDF
    doc.build(content)
    print(f"PDF generated: {OUTPUT_FILE}")
    return OUTPUT_FILE


if __name__ == "__main__":
    output = create_pdf()
    print(f"\nReport saved to: {output}")
    print(f"File size: {os.path.getsize(output) / 1024:.1f} KB")
