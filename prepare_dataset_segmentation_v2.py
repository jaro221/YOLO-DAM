"""
Advanced Segmentation Generation - Class-Specific Strategy
Uses true labels + reconstruction error + dark pixel analysis
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from scipy import ndimage

# Configuration
DATASET_DIR = r"D:\Projekty\2022_01_BattPor\DATA_DEF\YOLOv8\dataset"
TEST_DATASET_DIR = r"D:\Projekty\2022_01_BattPor\DATA_DEF\YOLOv8\test_dataset"
PREDICTIONS_DIR = r"D:\Projekty\2022_01_BattPor\2025_12_Dresden\YOLO_DAM_PREDICTIONS"
AUTOENCODER_PATH = r"D:\Projekty\2022_01_BattPor\2025_12_Dresden\Models\Autoencoder_best.h5"
OUTPUT_SEG_DIR = r"D:\Projekty\2022_01_BattPor\2025_12_Dresden\DATASET_SEG"

IMG_SIZE = 640
GAUSSIAN_SIGMA = 2.0
DARK_THRESHOLD = 30

# Class-specific strategy
CLASS_STRATEGY = {
    0: "direct_bbox",      # Agglomerate
    1: "error_dark_intersect",  # Pinhole-long
    2: "error_dark_intersect",  # Pinhole-trans
    3: "error_dark_intersect",  # Pinhole-round
    4: "error_dark_mask_average",  # Crack-long
    5: "error_dark_mask_average",  # Crack-trans
    6: "direct_bbox",      # Line-long
    7: "direct_bbox",      # Line-trans
    8: "error_dark_intersect_single",  # Line-diag (single pattern only)
    9: "direct_bbox",      # Foreign-particle
}

# ─────────────────────────────────────────────────────────────────────────────
# Load Models
# ─────────────────────────────────────────────────────────────────────────────
def load_autoencoder():
    """Load autoencoder for reconstruction error"""
    try:
        model = tf.keras.models.load_model(AUTOENCODER_PATH)
        print(f"[OK] Autoencoder loaded: {AUTOENCODER_PATH}")
        return model
    except Exception as e:
        print(f"[ERROR] Failed to load autoencoder: {e}")
        return None

# ─────────────────────────────────────────────────────────────────────────────
# Feature Extraction
# ─────────────────────────────────────────────────────────────────────────────
def get_reconstruction_error(model, patch):
    """Get reconstruction error map"""
    orig_h, orig_w = patch.shape[:2]
    patch_resized = cv2.resize(patch, (IMG_SIZE, IMG_SIZE))
    patch_batch = np.expand_dims(patch_resized, 0).astype(np.float32)
    
    try:
        outputs = model(patch_batch, training=False)
        reconstructed = outputs.numpy()[0] if hasattr(outputs, 'numpy') else outputs[0]
        reconstructed_orig = cv2.resize(reconstructed, (orig_w, orig_h))
        error = np.sqrt(np.sum((patch.astype(np.float32) / 255.0 - reconstructed_orig) ** 2, axis=2))
        return error
    except Exception as e:
        print(f"    Error computing reconstruction: {e}")
        return None

def get_dark_pixels_mask(patch, threshold=30):
    """Extract dark pixels from original patch"""
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if len(patch.shape) == 3 else patch
    mask = (gray < threshold).astype(np.float32)
    return mask

def get_smooth_dark_mask(patch, threshold=30, sigma=2.0):
    """Get dark pixels from Gaussian-smoothed patch"""
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if len(patch.shape) == 3 else patch
    smoothed = cv2.GaussianBlur(gray, (5, 5), sigma)
    mask = (smoothed < threshold).astype(np.float32)
    return mask

def normalize_and_threshold(feature_map, percentile=70):
    """Normalize feature map and threshold at percentile"""
    if np.max(feature_map) <= 0:
        return np.zeros_like(feature_map)
    feature_map = (feature_map / np.max(feature_map) * 255).astype(np.uint8)
    threshold = np.percentile(feature_map[feature_map > 0], percentile)
    binary_mask = (feature_map >= threshold).astype(np.uint8)
    return binary_mask

# ─────────────────────────────────────────────────────────────────────────────
# Polygon Generation
# ─────────────────────────────────────────────────────────────────────────────
def bbox_to_polygon(x_norm, y_norm, w_norm, h_norm, img_h, img_w):
    """Convert bbox to polygon coordinates (4 corners)"""
    x1 = int((x_norm - w_norm / 2) * img_w)
    y1 = int((y_norm - h_norm / 2) * img_h)
    x2 = int((x_norm + w_norm / 2) * img_w)
    y2 = int((y_norm + h_norm / 2) * img_h)
    
    x1, x2 = max(0, min(x1, img_w)), min(img_w, max(x2, 0))
    y1, y2 = max(0, min(y1, img_h)), min(img_h, max(y2, 0))
    
    points = [
        (x1, y1), (x2, y1), (x2, y2), (x1, y2)
    ]
    return [(p[0] / img_w, p[1] / img_h) for p in points]

def mask_to_polygon(binary_mask, x1, y1):
    """Convert binary mask to polygon (contour-based)"""
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Get largest contour
    largest = max(contours, key=cv2.contourArea)
    
    if cv2.contourArea(largest) < 5:
        return None
    
    points = []
    for pt in largest[:, 0]:
        points.append((float(pt[0]) + x1, float(pt[1]) + y1))
    
    return points

# ─────────────────────────────────────────────────────────────────────────────
# Segmentation Strategy
# ─────────────────────────────────────────────────────────────────────────────
def segment_direct_bbox(class_id, x_norm, y_norm, w_norm, h_norm, img_h, img_w):
    """Classes 0, 6, 7, 9: Direct bbox to polygon"""
    polygon = bbox_to_polygon(x_norm, y_norm, w_norm, h_norm, img_h, img_w)
    return [{"class_id": class_id, "polygon": polygon}]

def segment_error_dark_intersect(class_id, patch, bbox_coords, img_h, img_w, autoencoder):
    """Classes 1, 2, 3, 8: Reconstruction error ∩ dark pixels"""
    x1, y1, x2, y2 = bbox_coords
    
    error = get_reconstruction_error(autoencoder, patch)
    if error is None:
        return []
    
    dark_mask = get_dark_pixels_mask(patch, DARK_THRESHOLD)
    
    # Intersection: both error and dark
    error_binary = normalize_and_threshold(error, percentile=70)
    intersect = (error_binary * dark_mask).astype(np.uint8)
    
    # Smooth and extract contour
    intersect = cv2.GaussianBlur(intersect, (5, 5), GAUSSIAN_SIGMA)
    intersect = (intersect > 127).astype(np.uint8)
    
    polygon = mask_to_polygon(intersect, x1, y1)
    if polygon is None:
        return []
    
    return [{"class_id": class_id, "polygon": [(p[0] / img_w, p[1] / img_h) for p in polygon]}]

def segment_error_dark_mask_average(class_id, patch, bbox_coords, img_h, img_w, autoencoder, mask=None):
    """Classes 4, 5: Reconstruction error ∩ dark + mask average"""
    x1, y1, x2, y2 = bbox_coords
    
    error = get_reconstruction_error(autoencoder, patch)
    if error is None:
        return []
    
    dark_mask = get_dark_pixels_mask(patch, DARK_THRESHOLD)
    error_binary = normalize_and_threshold(error, percentile=70)
    
    # Combine error and dark
    combined = (error_binary.astype(np.float32) + dark_mask) / 2.0
    
    # If mask provided, include it
    if mask is not None:
        combined = (combined + mask) / 2.0
    
    # Threshold and extract
    combined = (combined > 0.5).astype(np.uint8)
    combined = cv2.GaussianBlur(combined, (5, 5), GAUSSIAN_SIGMA)
    combined = (combined > 127).astype(np.uint8)
    
    polygon = mask_to_polygon(combined, x1, y1)
    if polygon is None:
        return []
    
    return [{"class_id": class_id, "polygon": [(p[0] / img_w, p[1] / img_h) for p in polygon]}]

def segment_error_dark_single_pattern(class_id, patch, bbox_coords, img_h, img_w, autoencoder):
    """Class 8 Special: Select ONE pattern from possibly multiple defects"""
    x1, y1, x2, y2 = bbox_coords
    
    error = get_reconstruction_error(autoencoder, patch)
    if error is None:
        return []
    
    dark_mask = get_dark_pixels_mask(patch, DARK_THRESHOLD)
    error_binary = normalize_and_threshold(error, percentile=70)
    intersect = (error_binary * dark_mask).astype(np.uint8)
    intersect = cv2.GaussianBlur(intersect, (5, 5), GAUSSIAN_SIGMA)
    intersect = (intersect > 127).astype(np.uint8)
    
    # Find all contours (may be multiple for class 8)
    contours, _ = cv2.findContours(intersect, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return []
    
    # Select LARGEST contour only (main pattern matching bbox)
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 5:
        return []
    
    points = []
    for pt in largest[:, 0]:
        points.append((float(pt[0]) + x1, float(pt[1]) + y1))
    
    polygon = [(p[0] / img_w, p[1] / img_h) for p in points]
    return [{"class_id": class_id, "polygon": polygon}]

# ─────────────────────────────────────────────────────────────────────────────
# Image Processing
# ─────────────────────────────────────────────────────────────────────────────
def load_true_labels(label_file):
    """Load true YOLO labels"""
    labels = []
    if not os.path.exists(label_file):
        return labels
    
    with open(label_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            labels.append({
                "class_id": int(parts[0]),
                "x": float(parts[1]),
                "y": float(parts[2]),
                "w": float(parts[3]),
                "h": float(parts[4]),
            })
    return labels

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

def process_image(image_path, label_path, autoencoder, img_h, img_w):
    """Process single image with true labels"""
    image = cv2.imread(image_path)
    if image is None:
        return []
    
    true_labels = load_true_labels(label_path)
    if not true_labels:
        return []
    
    segmentations = []
    
    for label in true_labels:
        class_id = label["class_id"]
        strategy = CLASS_STRATEGY.get(class_id, "direct_bbox")
        
        x1, y1, x2, y2 = bbox_to_pixel(label["x"], label["y"], label["w"], label["h"], img_h, img_w)
        patch = image[y1:y2, x1:x2].copy()
        
        if patch.size == 0:
            continue
        
        try:
            if strategy == "direct_bbox":
                result = segment_direct_bbox(class_id, label["x"], label["y"], label["w"], label["h"], img_h, img_w)
            elif strategy == "error_dark_intersect":
                result = segment_error_dark_intersect(class_id, patch, (x1, y1, x2, y2), img_h, img_w, autoencoder)
            elif strategy == "error_dark_intersect_single":
                result = segment_error_dark_single_pattern(class_id, patch, (x1, y1, x2, y2), img_h, img_w, autoencoder)
            elif strategy == "error_dark_mask_average":
                result = segment_error_dark_mask_average(class_id, patch, (x1, y1, x2, y2), img_h, img_w, autoencoder)
            else:
                result = []
            
            segmentations.extend(result)
        except Exception as e:
            print(f"    Error processing class {class_id}: {str(e)[:50]}")
            continue
    
    return segmentations

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*70)
    print("ADVANCED SEGMENTATION GENERATION")
    print("Class-Specific Strategy with True Labels")
    print("="*70)
    
    # Load autoencoder
    autoencoder = load_autoencoder()
    if autoencoder is None:
        print("[ERROR] Cannot proceed without autoencoder")
        return False
    
    # Setup output directories
    for split in ["train", "test"]:
        os.makedirs(os.path.join(OUTPUT_SEG_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_SEG_DIR, "labels", split), exist_ok=True)
    
    total_segmented = 0
    
    # Process each split
    for split_name, dataset_path in [("train", DATASET_DIR)]:
        images_dir = os.path.join(dataset_path, "images", "train")
        labels_dir = os.path.join(dataset_path, "labels", "train")
        
        if not os.path.exists(images_dir):
            print(f"[SKIP] {split_name.upper()} images not found")
            continue
        
        image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".png"))])
        print(f"\n{split_name.upper()}: Processing {len(image_files)} images...")
        
        success_count = 0
        for i, image_file in enumerate(image_files):
            image_path = os.path.join(images_dir, image_file)
            basename = image_file.rsplit(".", 1)[0]
            label_path = os.path.join(labels_dir, basename + ".txt")
            
            image = cv2.imread(image_path)
            if image is None:
                continue
            
            img_h, img_w = image.shape[:2]
            segmentations = process_image(image_path, label_path, autoencoder, img_h, img_w)
            
            if segmentations:
                # Save image
                output_img = os.path.join(OUTPUT_SEG_DIR, "images", split_name, image_file)
                cv2.imwrite(output_img, image)
                
                # Save labels
                output_lbl = os.path.join(OUTPUT_SEG_DIR, "labels", split_name, basename + ".txt")
                with open(output_lbl, "w") as f:
                    for seg in segmentations:
                        polygon = seg["polygon"]
                        line = f"{seg[\"class_id\"]} " + " ".join([f"{x:.6f} {y:.6f}" for x, y in polygon]) + "\n"
                        f.write(line)
                
                success_count += 1
            
            if (i + 1) % 200 == 0 or (i + 1) == len(image_files):
                print(f"  {i+1}/{len(image_files)} - {success_count} segmented")
        
        total_segmented += success_count
        print(f"  [OK] {split_name}: {success_count}/{len(image_files)} segmented")
    
    # Process test dataset
    test_images_dir = os.path.join(TEST_DATASET_DIR, "images", "test")
    test_labels_dir = os.path.join(TEST_DATASET_DIR, "labels", "test")
    
    if os.path.exists(test_images_dir):
        image_files = sorted([f for f in os.listdir(test_images_dir) if f.lower().endswith((".jpg", ".png"))])
        print(f"\nTEST: Processing {len(image_files)} images...")
        
        success_count = 0
        for i, image_file in enumerate(image_files):
            image_path = os.path.join(test_images_dir, image_file)
            basename = image_file.rsplit(".", 1)[0]
            label_path = os.path.join(test_labels_dir, basename + ".txt")
            
            image = cv2.imread(image_path)
            if image is None:
                continue
            
            img_h, img_w = image.shape[:2]
            segmentations = process_image(image_path, label_path, autoencoder, img_h, img_w)
            
            if segmentations:
                output_img = os.path.join(OUTPUT_SEG_DIR, "images", "test", image_file)
                cv2.imwrite(output_img, image)
                
                output_lbl = os.path.join(OUTPUT_SEG_DIR, "labels", "test", basename + ".txt")
                with open(output_lbl, "w") as f:
                    for seg in segmentations:
                        polygon = seg["polygon"]
                        line = f"{seg[\"class_id\"]} " + " ".join([f"{x:.6f} {y:.6f}" for x, y in polygon]) + "\n"
                        f.write(line)
                
                success_count += 1
            
            if (i + 1) % 200 == 0 or (i + 1) == len(image_files):
                print(f"  {i+1}/{len(image_files)} - {success_count} segmented")
        
        total_segmented += success_count
        print(f"  [OK] test: {success_count}/{len(image_files)} segmented")
    else:
        print(f"[SKIP] Test dataset not found: {test_images_dir}")
    
    print("\n" + "="*70)
    print("[OK] SEGMENTATION GENERATION COMPLETE")
    print(f"[OK] Total segmented: {total_segmented}")
    print(f"[OK] Output: {OUTPUT_SEG_DIR}")
    print("="*70 + "\n")
    
    return total_segmented > 0

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
