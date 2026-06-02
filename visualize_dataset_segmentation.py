"""
Visualize Segmentation Dataset
Shows: original image + true bboxes + generated segmentation masks

Creates comparison visualizations for quality assurance
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DATASET_DIR = r"D:\Projekty\2022_01_BattPor\2025_12_Dresden\DATASET_SEG"
OUTPUT_VIZ_DIR = os.path.join(DATASET_DIR, "visualizations")

NUM_SAMPLES = 20  # Visualize 20 random samples

CLASS_NAMES = {
    0: "Agglomerate",
    1: "Pinhole-long",
    2: "Pinhole-trans",
    3: "Pinhole-round",
    4: "Crack-long",
    5: "Crack-trans",
    6: "Line-long",
    7: "Line-trans",
    8: "Line-diag",
    9: "Foreign-particle",
}

# Colors for each class (BGR for OpenCV, RGB for matplotlib)
CLASS_COLORS = {
    0: (255, 0, 0),      # Blue
    1: (0, 255, 0),      # Green
    2: (0, 0, 255),      # Red
    3: (255, 255, 0),    # Cyan
    4: (255, 0, 255),    # Magenta
    5: (0, 255, 255),    # Yellow
    6: (128, 0, 0),      # Dark Blue
    7: (0, 128, 0),      # Dark Green
    8: (0, 0, 128),      # Dark Red
    9: (128, 128, 0),    # Dark Cyan
}

# -----------------------------------------------------------------------------
# Parse Labels
# -----------------------------------------------------------------------------
def load_yolo_labels(label_file):
    """Load YOLO detection labels (bbox format)"""
    labels = []
    if not os.path.exists(label_file):
        return labels

    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            labels.append({
                'class_id': int(parts[0]),
                'x': float(parts[1]),
                'y': float(parts[2]),
                'w': float(parts[3]),
                'h': float(parts[4]),
            })
    return labels


def load_seg_labels(label_file):
    """Load segmentation labels (polygon format)"""
    polys = []
    if not os.path.exists(label_file):
        return polys

    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:  # class + at least 3 points
                continue
            class_id = int(parts[0])
            coords = [float(p) for p in parts[1:]]
            points = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
            polys.append({'class_id': class_id, 'polygon': points})
    return polys


def bbox_to_pixel(x, y, w, h, img_h, img_w):
    """Convert normalized bbox to pixel coordinates"""
    x_px = int(x * img_w)
    y_px = int(y * img_h)
    w_px = int(w * img_w)
    h_px = int(h * img_h)

    x1 = max(0, x_px - w_px // 2)
    y1 = max(0, y_px - h_px // 2)
    x2 = min(img_w, x_px + w_px // 2)
    y2 = min(img_h, y_px + h_px // 2)

    return x1, y1, x2, y2


def polygon_to_pixel(polygon, img_h, img_w):
    """Convert normalized polygon to pixel coordinates"""
    points = []
    for x_norm, y_norm in polygon:
        x_px = int(x_norm * img_w)
        y_px = int(y_norm * img_h)
        points.append((x_px, y_px))
    return np.array(points, dtype=np.int32)


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------
def visualize_image(image_path, det_label_path, seg_label_path, output_path):
    """Create visualization with bboxes and segmentation masks"""

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        return False

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    h, w = image.shape[:2]

    # Load labels
    det_labels = load_yolo_labels(det_label_path)
    seg_labels = load_seg_labels(seg_label_path)

    if not det_labels:
        return False

    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # --- Panel 1: Image + Detection BBoxes ---
    ax = axes[0]
    ax.imshow(image)
    ax.set_title("1. Original Image + Detection BBoxes (True Labels)", fontsize=12, fontweight='bold')

    for label in det_labels:
        class_id = label['class_id']
        x1, y1, x2, y2 = bbox_to_pixel(label['x'], label['y'], label['w'], label['h'], h, w)

        # Draw bbox
        color = CLASS_COLORS.get(class_id, (255, 255, 255))
        color_rgb = (color[2]/255, color[1]/255, color[0]/255)  # Convert BGR to RGB and normalize to 0-1
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor=color_rgb,
                            linewidth=2)
        ax.add_patch(rect)

        # Label
        class_name = CLASS_NAMES.get(class_id, f"C{class_id}")
        ax.text(x1, y1-5, class_name, color=color_rgb, fontsize=9,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    ax.axis('off')
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)

    # --- Panel 2: Image + Segmentation Masks ---
    ax = axes[1]
    ax.imshow(image)
    ax.set_title("2. Original Image + Segmentation Masks (Generated)", fontsize=12, fontweight='bold')

    for seg_label in seg_labels:
        class_id = seg_label['class_id']
        polygon = seg_label['polygon']

        # Convert to pixel coordinates
        points = polygon_to_pixel(polygon, h, w)

        if len(points) < 3:
            continue

        # Draw polygon
        color = CLASS_COLORS.get(class_id, (255, 255, 255))
        color_rgb = (color[2], color[1], color[0])  # BGR to RGB

        # Draw filled polygon with transparency
        poly_patch = plt.Polygon(points, fill=True, alpha=0.3, facecolor=color_rgb,
                               edgecolor=color_rgb, linewidth=2)
        ax.add_patch(poly_patch)

        # Draw points
        ax.plot(points[:, 0], points[:, 1], 'o', color=color_rgb, markersize=4)

        # Label at centroid
        cx = np.mean(points[:, 0])
        cy = np.mean(points[:, 1])
        class_name = CLASS_NAMES.get(class_id, f"C{class_id}")
        ax.text(int(cx), int(cy), class_name, color=color_rgb, fontsize=8,
               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7),
               ha='center', va='center')

    ax.axis('off')
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)

    # Save figure
    fig.tight_layout()
    fig.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)

    return True


def create_legend_figure():
    """Create legend showing class colors and names"""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')

    title_text = "Segmentation Dataset - Class Legend"
    ax.text(0.5, 0.95, title_text, ha='center', fontsize=16, fontweight='bold',
           transform=ax.transAxes)

    y_pos = 0.85
    for class_id in sorted(CLASS_NAMES.keys()):
        class_name = CLASS_NAMES[class_id]
        color = CLASS_COLORS.get(class_id, (255, 255, 255))
        color_rgb = (color[2]/255, color[1]/255, color[0]/255)  # BGR to RGB, normalize

        # Draw color box
        rect = plt.Rectangle((0.1, y_pos-0.03), 0.05, 0.04, transform=ax.transAxes,
                            facecolor=color_rgb, edgecolor='black', linewidth=1)
        ax.add_patch(rect)

        # Text
        ax.text(0.2, y_pos, f"Class {class_id}: {class_name}", fontsize=11,
               transform=ax.transAxes, va='center')

        y_pos -= 0.08

    fig.savefig(os.path.join(OUTPUT_VIZ_DIR, "00_legend.png"), dpi=100, bbox_inches='tight')
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    print("\n" + "="*70)
    print("VISUALIZE SEGMENTATION DATASET")
    print("="*70)

    # Setup output directory
    os.makedirs(OUTPUT_VIZ_DIR, exist_ok=True)
    print(f"\nOutput: {OUTPUT_VIZ_DIR}\n")

    # Create legend
    print("Creating legend...")
    create_legend_figure()
    print("  [OK] 00_legend.png\n")

    # Get all images
    train_images_dir = os.path.join(DATASET_DIR, 'images', 'train')
    train_labels_dir = os.path.join(DATASET_DIR, 'labels', 'train')

    if not os.path.exists(train_images_dir):
        print(f"[ERROR] Training images directory not found: {train_images_dir}")
        return False

    # Get image list
    image_files = [f for f in os.listdir(train_images_dir)
                   if f.lower().endswith(('.jpg', '.png'))]

    if not image_files:
        print(f"[ERROR] No images found in {train_images_dir}")
        return False

    # Sample random images
    sample_files = random.sample(image_files, min(NUM_SAMPLES, len(image_files)))
    sample_files.sort()

    print(f"Found {len(image_files)} total training images")
    print(f"Visualizing {len(sample_files)} random samples...\n")

    # Visualize samples
    success_count = 0
    for i, image_file in enumerate(sample_files):
        image_path = os.path.join(train_images_dir, image_file)
        basename = image_file.rsplit('.', 1)[0]
        det_label_path = os.path.join(train_labels_dir, basename + '.txt')
        seg_label_path = os.path.join(train_labels_dir, basename + '.txt')  # Same file
        output_path = os.path.join(OUTPUT_VIZ_DIR, f"{i+1:02d}_{basename}_comparison.png")

        try:
            success = visualize_image(image_path, det_label_path, seg_label_path, output_path)
            if success:
                success_count += 1
                status = "[OK]"
            else:
                status = "⚠"
        except Exception as e:
            status = "[ERROR]"
            print(f"  {i+1:2d}. {image_file:40s} {status} ({str(e)[:40]})")
            continue

        if (i + 1) % 5 == 0 or (i + 1) == len(sample_files):
            print(f"  {i+1:2d}. Processed {i+1} images ({success_count} successful)")

    print(f"\n[OK] Successfully created {success_count} visualizations")
    print(f"[OK] Saved to: {OUTPUT_VIZ_DIR}\n")

    # Create summary
    summary_path = os.path.join(OUTPUT_VIZ_DIR, "README.txt")
    with open(summary_path, 'w') as f:
        f.write(f"""SEGMENTATION DATASET VISUALIZATIONS
================================================================================

Generated: {OUTPUT_VIZ_DIR}
Date: 2026-04-24

Files:
  00_legend.png                    - Class color legend
  01_XXXXX_comparison.png          - Sample visualization (20 samples)
  02_XXXXX_comparison.png
  ...

Each visualization shows 2 panels:

Panel 1 (Left):
  Original image + Detection BBoxes (true labels)
  Red/blue/green rectangles = bounding boxes for each object
  Labels shown in corners

Panel 2 (Right):
  Original image + Segmentation Masks (generated from autoencoder)
  Colored polygons = precise segmentation boundaries
  Points mark polygon vertices
  Semi-transparent overlay shows mask extent

Colors:
  0: Agglomerate        (Blue)
  1: Pinhole-long       (Green)
  2: Pinhole-trans      (Red)
  3: Pinhole-round      (Cyan)
  4: Crack-long         (Magenta)
  5: Crack-trans        (Yellow)
  6: Line-long          (Dark Blue)
  7: Line-trans         (Dark Green)
  8: Line-diag          (Dark Red)
  9: Foreign-particle   (Dark Cyan)

Quality Assessment Checklist:
  [OK] Polygon follows object boundary tightly
  [OK] Polygon doesn't include background
  [OK] All objects have segmentation masks
  [OK] Polygon has sufficient points (>=3)
  [OK] Masks don't overlap between objects
  [OK] Small objects still have usable masks

Issues to Look For:
  [ERROR] Masks too small -> increase Gaussian sigma
  [ERROR] Masks too large -> decrease error threshold percentile
  [ERROR] Masks too rough -> increase Gaussian sigma
  [ERROR] Missing masks -> check detection bboxes are present
  [ERROR] Overlapping masks -> manual refinement needed

Next Step:
  Review visualizations to verify segmentation quality
  If satisfied: ready to train YOLO segmentation model
  If not: adjust parameters and regenerate

================================================================================
""")

    print(f"[OK] Created README: {summary_path}")
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)
    print(f"\nOpen visualizations in: {OUTPUT_VIZ_DIR}")
    print("Check quality before training segmentation model\n")

    return success_count > 0


if __name__ == "__main__":
    import sys
    success = main()

