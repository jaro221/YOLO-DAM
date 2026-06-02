"""
YOLO-DAM v3 Inference on Dataset
Generates predictions (detected bounding boxes) from trained model

Output: YOLO format predictions
  <class_id> <x_center> <y_center> <width> <height> <confidence>
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import sys

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
IMG_SIZE = 640
NUM_CLASSES = 10

# Input dataset
DATASET_DIR = r"D:\Projekty\2022_01_BattPor\DATA_DEF\YOLOv8\dataset"
MODEL_PATH = r"D:\Projekty\2022_01_BattPor\2025_12_Dresden\Models\YOLODAMv2_best_final.h5"

# Output predictions
OUTPUT_PRED_DIR = r"D:\Projekty\2022_01_BattPor\2025_12_Dresden\YOLO_DAM_PREDICTIONS"

# Confidence threshold for detection
CONF_THRESHOLD = 0.3

# ─────────────────────────────────────────────────────────────────────────────
# Load Model
# ─────────────────────────────────────────────────────────────────────────────
def load_model():
    """Load YOLO-DAM v3 model"""
    print("Loading YOLO-DAM v3 model...")
    try:
        from YOLO_DAM_v3 import model
        model.load_weights(MODEL_PATH)
        print(f"[OK] Model loaded with weights: {MODEL_PATH}")
    except Exception as e:
        print(f"[ERROR] Error loading v3: {e}")
        print("  Trying YOLO-DAM v2...")
        try:
            from YOLO_DAM_v2 import model
            model.load_weights(MODEL_PATH)
            print(f"[OK] Model loaded v2: {MODEL_PATH}")
        except Exception as e2:
            print(f"[ERROR] Error loading v2: {e2}")
            return None
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_image(image_path):
    """Load and preprocess image"""
    image = cv2.imread(image_path)
    if image is None:
        return None, None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    h, w = image.shape[:2]

    # Resize to model input size
    image_resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image_batch = np.expand_dims(image_resized, 0)

    return image_batch, (h, w)


def decode_predictions(model_outputs, scale_name, img_h, img_w, conf_thresh=0.3):
    """
    Decode YOLO-DAM predictions to bounding boxes

    Supports both M2M and O2O heads
    Returns: list of {class_id, x, y, w, h, confidence}
    """
    detections = []

    # Try M2M head (many-to-many, for recall)
    try:
        cls_output = model_outputs.get(f'{scale_name}_cls')
        reg_output = model_outputs.get(f'{scale_name}_reg')
        obj_output = model_outputs.get(f'{scale_name}_obj')

        if cls_output is None or reg_output is None or obj_output is None:
            return detections

        cls_pred = tf.sigmoid(cls_output).numpy()[0]  # [grid_h, grid_w, num_classes]
        reg_pred = reg_output.numpy()[0]  # [grid_h, grid_w, 4]
        obj_pred = tf.sigmoid(obj_output).numpy()[0]  # [grid_h, grid_w, 1]

        grid_h, grid_w = cls_pred.shape[:2]
        cell_size = IMG_SIZE / grid_h

        # Scan grid
        for gy in range(grid_h):
            for gx in range(grid_w):
                obj_score = obj_pred[gy, gx, 0]
                if obj_score < conf_thresh:
                    continue

                cls_scores = cls_pred[gy, gx] * obj_score
                class_id = np.argmax(cls_scores)
                confidence = cls_scores[class_id]

                if confidence < conf_thresh:
                    continue

                # Decode relative coordinates
                dx = reg_pred[gy, gx, 0]
                dy = reg_pred[gy, gx, 1]
                dw = reg_pred[gy, gx, 2]
                dh = reg_pred[gy, gx, 3]

                # Convert to absolute coordinates
                cell_cx = (gx + 0.5) * cell_size / IMG_SIZE
                cell_cy = (gy + 0.5) * cell_size / IMG_SIZE
                norm_cell_size = cell_size / IMG_SIZE

                x_norm = cell_cx + dx * norm_cell_size
                y_norm = cell_cy + dy * norm_cell_size
                w_norm = np.exp(dw) * norm_cell_size
                h_norm = np.exp(dh) * norm_cell_size

                # Clamp to [0, 1]
                x_norm = np.clip(x_norm, 0, 1)
                y_norm = np.clip(y_norm, 0, 1)
                w_norm = np.clip(w_norm, 1e-6, 1.0)
                h_norm = np.clip(h_norm, 1e-6, 1.0)

                detections.append({
                    'class_id': int(class_id),
                    'x': float(x_norm),
                    'y': float(y_norm),
                    'w': float(w_norm),
                    'h': float(h_norm),
                    'confidence': float(confidence),
                })

    except Exception as e:
        print(f"  Warning: Error decoding {scale_name}: {e}")

    return detections


def infer_image(model, image_path, conf_thresh=0.3):
    """
    Run inference on single image

    Returns: list of detections or None if failed
    """
    # Preprocess
    image_batch, orig_shape = preprocess_image(image_path)
    if image_batch is None:
        return None

    try:
        # Forward pass
        outputs = model(image_batch, training=False)

        # Decode all scales
        all_detections = []
        for scale in ['p2', 'p3', 'p4', 'p5']:
            scale_detections = decode_predictions(outputs, scale, orig_shape[0], orig_shape[1], conf_thresh)
            all_detections.extend(scale_detections)

        # Simple NMS (remove overlapping boxes)
        all_detections = nms(all_detections)

        return all_detections

    except Exception as e:
        print(f"  Error during inference: {e}")
        return None


def nms(detections, iou_threshold=0.5):
    """Non-maximum suppression"""
    if not detections:
        return []

    # Sort by confidence
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)

    keep = []
    while len(detections) > 0:
        keep.append(detections[0])
        if len(detections) == 1:
            break

        # Calculate IOU with first detection
        det1 = detections[0]
        ious = []
        for det2 in detections[1:]:
            iou = calc_iou(det1, det2)
            ious.append(iou)

        # Keep only low-overlap detections
        detections = [d for d, iou in zip(detections[1:], ious) if iou < iou_threshold]

    return keep


def calc_iou(det1, det2):
    """Calculate IOU between two detections"""
    x1_min = det1['x'] - det1['w'] / 2
    y1_min = det1['y'] - det1['h'] / 2
    x1_max = det1['x'] + det1['w'] / 2
    y1_max = det1['y'] + det1['h'] / 2

    x2_min = det2['x'] - det2['w'] / 2
    y2_min = det2['y'] - det2['h'] / 2
    x2_max = det2['x'] + det2['w'] / 2
    y2_max = det2['y'] + det2['h'] / 2

    inter_w = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    inter_h = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    inter_area = inter_w * inter_h

    area1 = det1['w'] * det1['h']
    area2 = det2['w'] * det2['h']
    union_area = area1 + area2 - inter_area

    iou = inter_area / (union_area + 1e-6)
    return iou


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*70)
    print("YOLO-DAM v3 INFERENCE")
    print("Generate predictions on dataset")
    print("="*70)

    # Load model
    model = load_model()
    if model is None:
        print("[ERROR] Failed to load model")
        return False

    # Setup output
    for split in ['train', 'val']:
        os.makedirs(os.path.join(OUTPUT_PRED_DIR, split), exist_ok=True)

    print(f"\nInput:  {DATASET_DIR}")
    print(f"Output: {OUTPUT_PRED_DIR}")
    print(f"Confidence threshold: {CONF_THRESHOLD}")
    print()

    # Process train dataset (main dataset)
    print("\n" + "-"*70)
    print("TRAIN DATASET")
    print("-"*70)

    for split in ['train', 'val']:
        images_dir = os.path.join(DATASET_DIR, 'images', split)
        if not os.path.exists(images_dir):
            print(f"[WARN] {split} split not found: {images_dir}")
            continue

        print(f"\nProcessing {split} split...")

        image_files = sorted([f for f in os.listdir(images_dir)
                             if f.lower().endswith(('.jpg', '.png'))])

        pred_count = 0
        total_dets = 0

        for i, image_file in enumerate(image_files):
            image_path = os.path.join(images_dir, image_file)

            # Infer
            detections = infer_image(model, image_path, CONF_THRESHOLD)

            if detections:
                # Save predictions
                output_path = os.path.join(OUTPUT_PRED_DIR, split,
                                         image_file.rsplit('.', 1)[0] + '.txt')
                with open(output_path, 'w') as f:
                    for det in detections:
                        line = f"{det['class_id']} {det['x']:.6f} {det['y']:.6f} " \
                               f"{det['w']:.6f} {det['h']:.6f} {det['confidence']:.6f}\n"
                        f.write(line)

                pred_count += 1
                total_dets += len(detections)

            # Progress
            if (i + 1) % 100 == 0 or (i + 1) == len(image_files):
                print(f"  {i+1}/{len(image_files)} images processed, {pred_count} with detections")

        print(f"  [OK] {pred_count} images with predictions")
        print(f"  [OK] {total_dets} total detections")

    # Process test dataset
    print("\n" + "-"*70)
    print("TEST DATASET")
    print("-"*70)

    test_dataset_dir = DATASET_DIR.replace('dataset', 'test_dataset')
    test_images_dir = os.path.join(test_dataset_dir, 'images', 'test')

    if os.path.exists(test_images_dir):
        print(f"\nProcessing test split...")

        image_files = sorted([f for f in os.listdir(test_images_dir)
                             if f.lower().endswith(('.jpg', '.png'))])

        pred_count = 0
        total_dets = 0

        for i, image_file in enumerate(image_files):
            image_path = os.path.join(test_images_dir, image_file)

            # Infer
            detections = infer_image(model, image_path, CONF_THRESHOLD)

            if detections:
                # Save predictions
                output_path = os.path.join(OUTPUT_PRED_DIR, 'test',
                                         image_file.rsplit('.', 1)[0] + '.txt')
                with open(output_path, 'w') as f:
                    for det in detections:
                        line = f"{det['class_id']} {det['x']:.6f} {det['y']:.6f} " \
                               f"{det['w']:.6f} {det['h']:.6f} {det['confidence']:.6f}\n"
                        f.write(line)

                pred_count += 1
                total_dets += len(detections)

            # Progress
            if (i + 1) % 100 == 0 or (i + 1) == len(image_files):
                print(f"  {i+1}/{len(image_files)} images processed, {pred_count} with detections")

        print(f"  [OK] {pred_count} images with predictions")
        print(f"  [OK] {total_dets} total detections")
    else:
        print(f"[WARN] Test dataset not found: {test_images_dir}")

    print("\n" + "="*70)
    print("✓ INFERENCE COMPLETE")
    print("="*70)
    print(f"\nPredictions saved to: {OUTPUT_PRED_DIR}")
    print("Format: <class_id> <x> <y> <w> <h> <confidence>")
    print("\nNext: Use these predictions for segmentation generation")
    print("="*70 + "\n")

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
