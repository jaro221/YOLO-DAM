"""
Visualize Segmentation Masks from DATASET_SEG
Shows images with generated segmentation polygon masks
"""

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for file saving
import matplotlib.pyplot as plt
from pathlib import Path
import random

# Configuration
DATASET_SEG_DIR = r"D:\Projekty\2022_01_BattPor\2025_12_Dresden\DATASET_SEG"
OUTPUT_VIZ_DIR = os.path.join(DATASET_SEG_DIR, "visualizations")

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

# Colors for each class (BGR for OpenCV)
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

# Load segmentation labels
def load_seg_labels(label_file):
    """Load segmentation labels (polygon format)"""
    polys = []
    if not os.path.exists(label_file):
        return polys

    with open(label_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:  # class + at least 3 points
                continue
            class_id = int(parts[0])
            coords = [float(p) for p in parts[1:]]
            points = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
            polys.append({"class_id": class_id, "polygon": points})
    return polys


def polygon_to_pixel(polygon, img_h, img_w):
    """Convert normalized polygon to pixel coordinates"""
    points = []
    for x_norm, y_norm in polygon:
        x_px = int(x_norm * img_w)
        y_px = int(y_norm * img_h)
        points.append((x_px, y_px))
    return np.array(points, dtype=np.int32)


# Visualization
def visualize_segmentation(image_path, label_path, output_path):
    """Create visualization with segmentation masks"""

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        return False

    # Convert to RGB and normalize to 0-1
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    h, w = image.shape[:2]

    # Load labels
    seg_labels = load_seg_labels(label_path)

    if not seg_labels:
        return False

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 9))

    # Show image
    ax.imshow(image)
    ax.set_title("Segmentation Masks", fontsize=14, fontweight="bold")

    # Draw segmentation masks
    for seg_label in seg_labels:
        class_id = seg_label["class_id"]
        polygon = seg_label["polygon"]

        # Convert to pixel coordinates
        points = polygon_to_pixel(polygon, h, w)

        if len(points) < 3:
            continue

        # Get color (normalized to 0-1 for matplotlib)
        color = CLASS_COLORS.get(class_id, (255, 255, 255))
        color_rgb = (color[2] / 255.0, color[1] / 255.0, color[0] / 255.0)

        # Draw filled polygon with transparency
        poly_patch = plt.Polygon(points, fill=True, alpha=0.3,
                               facecolor=color_rgb, edgecolor=color_rgb, linewidth=2)
        ax.add_patch(poly_patch)

        # Draw points
        ax.plot(points[:, 0], points[:, 1], "o", color=color_rgb, markersize=5)

        # Label at centroid
        cx = np.mean(points[:, 0])
        cy = np.mean(points[:, 1])
        class_name = CLASS_NAMES.get(class_id, f"C{class_id}")
        ax.text(int(cx), int(cy), class_name, color=color_rgb, fontsize=10,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
               ha="center", va="center", fontweight="bold")

    ax.axis("off")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)

    # Save figure
    fig.tight_layout()
    try:
        fig.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        return True
    except Exception as e:
        print(f"    Error saving figure: {e}")
        plt.close(fig)
        return False


def create_legend_figure():
    """Create legend showing class colors and names"""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis("off")

    title_text = "Segmentation Classes - Color Legend"
    ax.text(0.5, 0.95, title_text, ha="center", fontsize=16, fontweight="bold",
           transform=ax.transAxes)

    y_pos = 0.85
    for class_id in sorted(CLASS_NAMES.keys()):
        class_name = CLASS_NAMES[class_id]
        color = CLASS_COLORS.get(class_id, (255, 255, 255))
        color_rgb = (color[2] / 255.0, color[1] / 255.0, color[0] / 255.0)

        # Draw color box
        rect = plt.Rectangle((0.1, y_pos - 0.03), 0.05, 0.04, transform=ax.transAxes,
                            facecolor=color_rgb, edgecolor="black", linewidth=1)
        ax.add_patch(rect)

        # Text
        ax.text(0.2, y_pos, f"Class {class_id}: {class_name}", fontsize=11,
               transform=ax.transAxes, va="center")

        y_pos -= 0.08

    try:
        fig.savefig(os.path.join(OUTPUT_VIZ_DIR, "00_legend.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)
        return True
    except Exception as e:
        print(f"  Error saving legend: {e}")
        plt.close(fig)
        return False


# Main
def main():
    print("\n" + "="*70)
    print("VISUALIZE SEGMENTATION MASKS")
    print("="*70)

    # Setup output directory
    os.makedirs(OUTPUT_VIZ_DIR, exist_ok=True)
    print(f"\nDataset: {DATASET_SEG_DIR}")
    print(f"Output:  {OUTPUT_VIZ_DIR}\n")

    # Create legend
    print("Creating legend...")
    if create_legend_figure():
        print("  [OK] 00_legend.png\n")
    else:
        print("  [ERROR] Failed to create legend\n")

    # Process each split
    total_success = 0

    for split in ["train", "test"]:
        images_dir = os.path.join(DATASET_SEG_DIR, "images", split)
        labels_dir = os.path.join(DATASET_SEG_DIR, "labels", split)

        if not os.path.exists(images_dir):
            print(f"[SKIP] {split.upper()} - images directory not found")
            continue

        if not os.path.exists(labels_dir):
            print(f"[SKIP] {split.upper()} - labels directory not found")
            continue

        # Get image list
        image_files = sorted([f for f in os.listdir(images_dir)
                             if f.lower().endswith((".jpg", ".png"))])

        if not image_files:
            print(f"[SKIP] {split.upper()} - no images found")
            continue

        # Sample random images
        sample_files = random.sample(image_files, min(NUM_SAMPLES, len(image_files)))
        sample_files.sort()

        print(f"{split.upper()} SPLIT")
        print(f"  Found: {len(image_files)} total images")
        print(f"  Visualizing: {len(sample_files)} random samples")

        success_count = 0
        for i, image_file in enumerate(sample_files):
            image_path = os.path.join(images_dir, image_file)
            basename = image_file.rsplit(".", 1)[0]
            label_path = os.path.join(labels_dir, basename + ".txt")
            output_path = os.path.join(OUTPUT_VIZ_DIR, f"{split}_{i+1:02d}_{basename}.png")

            try:
                success = visualize_segmentation(image_path, label_path, output_path)
                if success:
                    success_count += 1
            except Exception as e:
                print(f"    Error processing {image_file}: {str(e)[:50]}")
                continue

        total_success += success_count
        print(f"  [OK] {success_count}/{len(sample_files)} visualizations created\n")

    # Summary
    print("="*70)
    print(f"VISUALIZATION COMPLETE")
    print("="*70)
    print(f"\n[OK] Created {total_success} visualizations")
    print(f"[OK] Saved to: {OUTPUT_VIZ_DIR}\n")

    return total_success > 0


if __name__ == "__main__":
    success = main()
    import sys

