"""
Generate Precise Segmentation Masks
Combines YOLO detection + autoencoder reconstruction error + Gaussian blur

Pipeline:
1. Load YOLO-DAM v2 predictions (bounding boxes)
2. Load true labels (ground truth bboxes)
3. Merge/refine detections with true labels
4. Extract image patches within bounding boxes
5. Run through autoencoder: original vs reconstructed
6. Compute difference map (L2 error per pixel)
7. Apply Gaussian blur to smooth differences
8. Threshold + convert to polygon coordinates
9. Save YOLO segmentation labels
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from scipy.ndimage import gaussian_filter
from skimage import measure

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
IMG_SIZE = 640
NUM_CLASSES = 10
DATASET_DIR = r"D:/Projekty/2022_01_BattPor/2025_12_Dresden/YOLOv8/dataset"
MODEL_PATH = r"D:/Projekty/2022_01_BattPor/2025_12_Dresden/Models/YOLODAM_merged_v26_new.h5"
OUTPUT_DIR = r"D:/Projekty/2022_01_BattPor/2025_12_Dresden/YOLOv8/dataset/labels_seg"

# Blur parameters
GAUSSIAN_SIGMA = 2.0  # Smoothing strength (higher = more blur)
THRESHOLD_PERCENTILE = 70  # Use top 70% of error pixels as defect

# Min polygon points for valid segmentation
MIN_POLYGON_POINTS = 3

# ─────────────────────────────────────────────────────────────────────────────
# Load Model
# ─────────────────────────────────────────────────────────────────────────────
def load_model():
    """Load YOLO-DAM v2 model"""
    from YOLO_DAM_v2 import model
    try:
        model.load_weights(MODEL_PATH)
        print(f"✓ Loaded model from {MODEL_PATH}")
    except Exception as e:
        print(f"⚠ Could not load weights: {e}")
        print("  Starting with random initialization")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Parse Labels
# ─────────────────────────────────────────────────────────────────────────────
def load_yolo_labels(label_file):
    """
    Load YOLO detection labels (bbox format)
    Format: <class_id> <x_center> <y_center> <width> <height> (normalized)

    Returns: list of dicts {class_id, x, y, w, h}
    """
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


def bbox_to_coordinates(x, y, w, h, img_h, img_w):
    """
    Convert normalized bbox to pixel coordinates
    Returns: (x_min, y_min, x_max, y_max) in pixels
    """
    x_px = int(x * img_w)
    y_px = int(y * img_h)
    w_px = int(w * img_w)
    h_px = int(h * img_h)

    x_min = max(0, x_px - w_px // 2)
    y_min = max(0, y_px - h_px // 2)
    x_max = min(img_w, x_px + w_px // 2)
    y_max = min(img_h, y_px + h_px // 2)

    return x_min, y_min, x_max, y_max


def extract_patch(image, bbox, pad=10):
    """
    Extract image patch from bounding box with padding

    Args:
        image: [H, W, 3] numpy array
        bbox: (x_min, y_min, x_max, y_max)
        pad: padding in pixels

    Returns:
        patch: [H_patch, W_patch, 3]
        (x_min_padded, y_min_padded, x_max_padded, y_max_padded)
    """
    x_min, y_min, x_max, y_max = bbox
    h, w = image.shape[:2]

    # Add padding
    x_min_p = max(0, x_min - pad)
    y_min_p = max(0, y_min - pad)
    x_max_p = min(w, x_max + pad)
    y_max_p = min(h, y_max + pad)

    patch = image[y_min_p:y_max_p, x_min_p:x_max_p]

    return patch, (x_min_p, y_min_p, x_max_p, y_max_p)


# ─────────────────────────────────────────────────────────────────────────────
# Autoencoder Inference
# ─────────────────────────────────────────────────────────────────────────────
def get_reconstruction_error(model, patch):
    """
    Get reconstruction error from autoencoder

    Args:
        model: YOLO-DAM model (has autoencoder head)
        patch: [H, W, 3] image patch, values in [0, 1]

    Returns:
        error_map: [H, W] reconstruction error per pixel
    """
    # Prepare input
    patch_batch = np.expand_dims(patch, 0)  # [1, H, W, 3]
    patch_batch = tf.cast(patch_batch, tf.float32)

    # Forward pass
    outputs = model(patch_batch, training=False)
    reconstructed = outputs['auto_reconstruction'].numpy()[0]  # [H, W, 3]

    # Compute L2 error per pixel
    error = np.sqrt(np.sum((patch - reconstructed) ** 2, axis=2))  # [H, W]

    return error


def blur_and_threshold_error(error_map, sigma=2.0, percentile=70):
    """
    Smooth error map with Gaussian blur and threshold

    Args:
        error_map: [H, W] reconstruction error
        sigma: Gaussian blur standard deviation
        percentile: threshold at this percentile of errors

    Returns:
        mask: [H, W] binary mask (0 or 1)
        error_blurred: [H, W] blurred error map
    """
    # Normalize
    error_norm = (error_map - error_map.min()) / (error_map.max() - error_map.min() + 1e-7)

    # Gaussian blur
    error_blurred = gaussian_filter(error_norm, sigma=sigma)

    # Threshold at percentile
    thresh = np.percentile(error_blurred, percentile)
    mask = (error_blurred >= thresh).astype(np.uint8)

    return mask, error_blurred


def mask_to_polygon(mask):
    """
    Convert binary mask to polygon coordinates

    Args:
        mask: [H, W] binary mask

    Returns:
        polygon: [(x1, y1), (x2, y2), ...] list of normalized coordinates
                 or None if not enough points
    """
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Get largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    if len(largest_contour) < MIN_POLYGON_POINTS:
        return None

    # Approximate contour to reduce points
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    # Normalize to [0, 1]
    h, w = mask.shape
    polygon = [(float(pt[0][0]) / w, float(pt[0][1]) / h) for pt in approx]

    return polygon if len(polygon) >= MIN_POLYGON_POINTS else None


# ─────────────────────────────────────────────────────────────────────────────
# Main Processing
# ─────────────────────────────────────────────────────────────────────────────
def process_image(model, image_path, label_path, output_path):
    """
    Process single image: detect → extract patches → reconstruct → segment
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        return False

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0  # [H, W, 3], [0, 1]
    h, w = image.shape[:2]

    # Load labels (true bboxes)
    labels = load_yolo_labels(label_path)
    if not labels:
        print(f"  ⚠ No labels found for {os.path.basename(image_path)}")
        return False

    seg_labels = []  # List of segmentation labels

    # Process each bounding box
    for label in labels:
        class_id = label['class_id']
        bbox = bbox_to_coordinates(label['x'], label['y'], label['w'], label['h'], h, w)

        # Extract patch with padding
        patch, bbox_padded = extract_patch(image, bbox, pad=10)
        x_min_p, y_min_p, x_max_p, y_max_p = bbox_padded

        if patch.shape[0] < 10 or patch.shape[1] < 10:
            continue  # Too small

        # Get reconstruction error
        try:
            error_map = get_reconstruction_error(model, patch)
        except Exception as e:
            print(f"  ⚠ Error in reconstruction: {e}")
            continue

        # Blur and threshold
        mask, error_blurred = blur_and_threshold_error(
            error_map,
            sigma=GAUSSIAN_SIGMA,
            percentile=THRESHOLD_PERCENTILE
        )

        if mask.sum() < 5:  # Too few pixels
            continue

        # Convert mask to polygon
        polygon = mask_to_polygon(mask)
        if polygon is None:
            continue

        # Translate polygon to image coordinates (undo padding)
        polygon_img = [
            (x * (x_max_p - x_min_p) + x_min_p,
             y * (y_max_p - y_min_p) + y_min_p)
            for x, y in polygon
        ]

        # Normalize back to [0, 1]
        polygon_norm = [
            (x / w, y / h)
            for x, y in polygon_img
        ]

        # Format: <class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
        seg_line = f"{class_id} " + " ".join(
            f"{x:.6f} {y:.6f}" for x, y in polygon_norm
        )
        seg_labels.append(seg_line)

    # Write segmentation labels
    if seg_labels:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write('\n'.join(seg_labels))
        return True

    return False


def main():
    """Process all images in dataset"""

    print("="*60)
    print("Generate Segmentation Masks from Reconstruction Error")
    print("="*60)
    print(f"Gaussian Sigma: {GAUSSIAN_SIGMA}")
    print(f"Error Threshold: {THRESHOLD_PERCENTILE}%")
    print()

    # Load model once
    print("Loading YOLO-DAM model...")
    model = load_model()
    print()

    # Process train and val splits
    for split in ['train', 'val']:
        images_dir = os.path.join(DATASET_DIR, 'images', split)
        labels_dir = os.path.join(DATASET_DIR, 'labels', split)
        output_dir = os.path.join(OUTPUT_DIR, split)

        if not os.path.exists(images_dir):
            print(f"⚠ {images_dir} not found, skipping")
            continue

        print(f"Processing {split} split...")
        image_files = sorted([f for f in os.listdir(images_dir)
                             if f.lower().endswith(('.jpg', '.png'))])

        success_count = 0
        for i, image_file in enumerate(image_files):
            image_path = os.path.join(images_dir, image_file)
            label_path = os.path.join(labels_dir, image_file.rsplit('.', 1)[0] + '.txt')
            output_path = os.path.join(output_dir, image_file.rsplit('.', 1)[0] + '.txt')

            success = process_image(model, image_path, label_path, output_path)

            if success:
                success_count += 1
                status = "✓"
            else:
                status = "✗"

            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(image_files)} [{status}]")

        print(f"  ✓ {success_count}/{len(image_files)} images processed")
        print()

    print("="*60)
    print(f"✓ Segmentation masks saved to {OUTPUT_DIR}")
    print(f"Use for: YOLO segmentation training with data_seg.yaml")
    print("="*60)


if __name__ == "__main__":
    main()
