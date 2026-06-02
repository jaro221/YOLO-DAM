#!/usr/bin/env python3
"""
YOLO-DAM Testing Suite with RELATIVE Positioning Decoding
Modified to handle [dx, dy, dw, dh] relative regression targets
Date: 2026-04-15
"""

import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
SAVE_DIR    = os.path.join("D:/Clanky/x2026_31_Defects_Scientific_reports/DATA2")
WEIGHTS     = r"D:/Projekty/2022_01_BattPor/2025_12_Dresden/Models/YOLODAM_best_e383.h5"
TRUE_DIR    = r"D:/Projekty/2022_01_BattPor/DATA_DEF/YOLOv8/test_dataset/labels/test/"
PRED_DIR    = r"D:/Projekty/2022_01_BattPor/DATA_DEF/YOLOAUTO/run/labels_test61/"

# Derive image dir from label dir
IMG_DIR     = TRUE_DIR.replace("labels", "images")

IMG_SIZE    = 640
CONF_THRESH = 0.20
IOU_THRESH  = 0.5
NUM_CLASSES = 10
BATCH_SIZE  = 1
IMG_EXTS    = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

def detections_to_lists(detections):
    """Convert detections array to separate lists"""
    if detections is None or len(detections) == 0:
        return [], [], []
    pred_boxes   = [[d[0], d[1], d[2], d[3]] for d in detections]
    pred_classes = [int(d[5])                 for d in detections]
    pred_scores  = [float(d[4])               for d in detections]
    return pred_boxes, pred_classes, pred_scores


# ─────────────────────────────────────────────────────────────────────────────
# YOLO Model Tester with Relative Positioning
# ─────────────────────────────────────────────────────────────────────────────

class YOLOModelTester:
    """
    Complete testing suite for trained YOLO model with RELATIVE positioning.

    Handles decoding of relative regression targets [dx, dy, dw, dh]:
      dx, dy: offset from cell center / cell_size
      dw, dh: log scale (exp to recover size)
    """

    def __init__(self, model, class_names, img_size=640, conf_threshold=0.25, iou_threshold=0.45):
        self.model = model
        self.class_names = class_names
        self.img_size = img_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def load_checkpoint(self, checkpoint_path):
        """Load trained model weights"""
        print(f"Loading checkpoint from: {checkpoint_path}")
        try:
            self.model.load_weights(checkpoint_path)
            print("✅ Checkpoint loaded successfully")
        except Exception as e:
            print(f"⚠ Could not load weights: {e}")

    def preprocess_image(self, image_path):
        """Load and preprocess image for inference"""
        img = tf.io.read_file(image_path)
        img = tf.image.decode_image(img, channels=3)
        original_shape = tf.shape(img)[:2]

        img = tf.image.resize(img, [self.img_size, self.img_size])
        img = tf.cast(img, tf.float32) / 255.0
        img_batch = tf.expand_dims(img, 0)

        return img_batch, original_shape

    def decode_predictions_relative(self, preds, original_shape):
        """
        Decode RELATIVE predictions [dx, dy, dw, dh] to absolute [x, y, w, h] in [0, 1]

        For each scale:
          - Extract raw predictions [tx, ty, tw, th]
          - Apply sigmoid to tx, ty
          - Apply exp to tw, th
          - Use grid cell centers to convert to absolute coordinates
        """
        all_detections = []

        for scale in ['p2', 'p3', 'p4', 'p5']:
            if f"{scale}_cls" not in preds:
                continue

            cls_pred = preds[f"{scale}_cls"][0].numpy()   # [grid_h, grid_w, num_classes]
            reg_pred = preds[f"{scale}_reg"][0].numpy()   # [grid_h, grid_w, 4]
            obj_pred = preds[f"{scale}_obj"][0].numpy()   # [grid_h, grid_w, 1]

            # Get grid dimensions
            grid_h, grid_w = reg_pred.shape[:2]
            grid_size = grid_h  # P2=160, P3=80, P4=40, P5=20
            cell_size = self.img_size / grid_size

            # Sigmoid and exp activations
            def sig(x):
                return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))

            obj_conf = sig(obj_pred)
            cls_conf = sig(cls_pred)

            # Extract relative regression components
            tx = reg_pred[..., 0]  # raw position offset
            ty = reg_pred[..., 1]
            tw = reg_pred[..., 2]  # log scale
            th = reg_pred[..., 3]

            # Decode relative to absolute
            tx_sig = sig(tx)  # [0, 1]
            ty_sig = sig(ty)  # [0, 1]
            tw_exp = np.exp(np.clip(tw, -5, 5))  # [0, ∞)
            th_exp = np.exp(np.clip(th, -5, 5))  # [0, ∞)

            # Get objectness mask
            obj_mask = obj_conf[..., 0] > self.conf_threshold

            if not np.any(obj_mask):
                continue

            ys, xs = np.where(obj_mask)

            for y, x in zip(ys, xs):
                obj_score = obj_conf[y, x, 0]
                class_scores = cls_conf[y, x] * obj_score
                class_id = np.argmax(class_scores)
                confidence = class_scores[class_id]

                if confidence < self.conf_threshold:
                    continue

                # Decode relative to absolute normalized coordinates
                # Cell center
                cell_cx_norm = (x + 0.5) / grid_size
                cell_cy_norm = (y + 0.5) / grid_size

                # Decode position
                norm_cell_size = cell_size / self.img_size
                cx_n = cell_cx_norm + (tx_sig[y, x] - 0.5) * norm_cell_size
                cy_n = cell_cy_norm + (ty_sig[y, x] - 0.5) * norm_cell_size

                # Decode size
                w_n = tw_exp[y, x] * norm_cell_size
                h_n = th_exp[y, x] * norm_cell_size

                # Special case for class 9 (foreign particles)
                if class_id == 9:
                    w_n = 3.0 / self.img_size
                    h_n = 3.0 / self.img_size

                # Clamp to [0, 1]
                cx_n = np.clip(cx_n, 0, 1)
                cy_n = np.clip(cy_n, 0, 1)
                w_n = np.clip(w_n, 1e-6, 1.0)
                h_n = np.clip(h_n, 1e-6, 1.0)

                all_detections.append([cx_n, cy_n, w_n, h_n, confidence, float(class_id)])

        if not all_detections:
            return np.array([])

        detections = np.array(all_detections)
        detections = self.non_max_suppression(detections)
        return detections

    def non_max_suppression(self, detections):
        """Apply NMS to detections"""
        if len(detections) == 0:
            return np.array([])

        # Sort by confidence descending
        detections = detections[np.argsort(detections[:, 4])[::-1]]

        keep = []
        while len(detections) > 0:
            keep.append(detections[0])
            if len(detections) == 1:
                break
            ious = self.calculate_iou_batch(detections[0:1], detections[1:])
            detections = detections[1:][ious[0] < self.iou_threshold]

        return np.array(keep) if keep else np.array([])

    def calculate_iou_batch(self, boxes1, boxes2):
        """Calculate IoU between boxes"""
        def to_xyxy(b):
            x1 = b[:, 0] - b[:, 2] / 2
            y1 = b[:, 1] - b[:, 3] / 2
            x2 = b[:, 0] + b[:, 2] / 2
            y2 = b[:, 1] + b[:, 3] / 2
            return x1, y1, x2, y2

        x1_1, y1_1, x2_1, y2_1 = to_xyxy(boxes1)
        x1_2, y1_2, x2_2, y2_2 = to_xyxy(boxes2)

        inter_x1 = np.maximum(x1_1[:, None], x1_2)
        inter_y1 = np.maximum(y1_1[:, None], y1_2)
        inter_x2 = np.minimum(x2_1[:, None], x2_2)
        inter_y2 = np.minimum(y2_1[:, None], y2_2)

        inter_area = (np.maximum(0, inter_x2 - inter_x1) *
                      np.maximum(0, inter_y2 - inter_y1))

        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1[:, None] + area2 - inter_area

        return inter_area / (union_area + 1e-6)

    def predict_image(self, image_path):
        """Run inference on image"""
        img_batch, orig_shape = self.preprocess_image(image_path)
        preds = self.model(img_batch, training=False)
        self._last_preds = preds
        detections = self.decode_predictions_relative(preds, orig_shape)
        return detections, preds

    def visualize_predictions(self, image_path, detections, save_path=None, show_labels=True):
        """Visualize predictions vs ground truth"""
        img_np = np.array(Image.open(image_path).convert("RGB"))
        H, W = img_np.shape[:2]

        # Load ground truth
        stem = Path(image_path).stem
        gt_path = Path(TRUE_DIR) / (stem + ".txt")
        gt_boxes = []
        if gt_path.exists():
            with open(gt_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:])
                    gt_boxes.append((int((cx-bw/2)*W), int((cy-bh/2)*H),
                                    int((cx+bw/2)*W), int((cy+bh/2)*H), cls))

        # Color palette
        PALETTE = [
            (255, 56, 56), (255, 157, 51), (54, 162, 235), (255, 206, 86),
            (75, 192, 192), (153, 102, 255), (255, 159, 64), (46, 204, 113),
            (231, 76, 60), (52, 152, 219),
        ]

        def bgr(cls):
            return PALETTE[int(cls) % len(PALETTE)]

        # Draw ground truth
        canvas_gt = img_np[:, :, ::-1].copy()
        for x1, y1, x2, y2, cls in gt_boxes:
            color = bgr(cls)
            cv2.rectangle(canvas_gt, (x1, y1), (x2, y2), color, 2)
            if show_labels:
                label = self.class_names[cls]
                cv2.putText(canvas_gt, label, (x1, max(y1-4, 15)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Draw predictions
        canvas_pred = img_np[:, :, ::-1].copy()
        for det in detections:
            cx, cy, bw, bh, conf, cls = det
            cls = int(cls)
            x1, y1 = int((cx - bw/2) * W), int((cy - bh/2) * H)
            x2, y2 = int((cx + bw/2) * W), int((cy + bh/2) * H)
            color = bgr(cls)
            cv2.rectangle(canvas_pred, (x1, y1), (x2, y2), color, 2)
            if show_labels:
                label = f"{self.class_names[cls]} {conf:.2f}"
                cv2.putText(canvas_pred, label, (x1, max(y1-4, 15)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Combine panels
        total_w = W * 2 + 4
        combined = np.concatenate([canvas_gt, np.full((H, 4, 3), 50, dtype=np.uint8), canvas_pred], axis=1)

        if save_path:
            cv2.imwrite(str(save_path), combined, [cv2.IMWRITE_JPEG_QUALITY, 88])
        else:
            plt.figure(figsize=(16, 6))
            plt.imshow(combined[:, :, ::-1])
            plt.axis("off")
            plt.tight_layout()
            plt.show()
            plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main Testing Loop
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Metrics Calculation Functions
# ─────────────────────────────────────────────────────────────────────────────

def load_labels(file_path):
    """Load YOLO format labels"""
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r") as f:
        return [list(map(float, line.strip().split())) for line in f.readlines()]


def iou(box1, box2):
    """Calculate IoU between two boxes in [cx, cy, w, h] format"""
    def to_coords(box):
        x, y, w, h = box
        return [x - w / 2, y - h / 2, x + w / 2, y + h / 2]

    b1 = to_coords(box1)
    b2 = to_coords(box2)
    xi1, yi1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    xi2, yi2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def compute_ap(recalls, precisions):
    """Area under P-R curve using 11-point interpolation"""
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        p = max(precisions[recalls >= t], default=0)
        ap += p / 11
    return ap


def compute_map(true_dir, pred_dir, iou_threshold=0.5, num_classes=10):
    """Compute mAP@IOU"""
    class_preds = {c: [] for c in range(num_classes)}
    class_gt = {c: 0 for c in range(num_classes)}

    for file in os.listdir(true_dir):
        if not file.endswith(".txt"):
            continue

        gt_labels = load_labels(os.path.join(true_dir, file))
        pr_labels = load_labels(os.path.join(pred_dir, file))

        for gt in gt_labels:
            class_gt[int(gt[0])] += 1

        used_gt = set()
        pr_sorted = sorted(pr_labels, key=lambda x: x[5] if len(x) > 5 else 1.0, reverse=True)

        for pred in pr_sorted:
            pred_cls = int(pred[0])
            pred_box = pred[1:5]
            conf = pred[5] if len(pred) > 5 else 1.0

            best_iou, best_idx = 0, -1
            for gi, gt in enumerate(gt_labels):
                if gi in used_gt:
                    continue
                if int(gt[0]) != pred_cls:
                    continue
                score = iou(gt[1:5], pred_box)
                if score > best_iou:
                    best_iou, best_idx = score, gi

            is_tp = 1 if best_iou >= iou_threshold and best_idx >= 0 else 0
            if is_tp:
                used_gt.add(best_idx)
            class_preds[pred_cls].append((conf, is_tp))

    aps = []
    for c in range(num_classes):
        if class_gt[c] == 0:
            continue
        preds_c = sorted(class_preds[c], key=lambda x: x[0], reverse=True)
        tp_cum, fp_cum = 0, 0
        recalls, precisions = [], []
        for conf, is_tp in preds_c:
            if is_tp:
                tp_cum += 1
            else:
                fp_cum += 1
            recalls.append(tp_cum / (class_gt[c] + 1e-7))
            precisions.append(tp_cum / (tp_cum + fp_cum + 1e-7))
        recalls = np.array(recalls)
        precisions = np.array(precisions)
        aps.append(compute_ap(recalls, precisions))

    return np.mean(aps) if aps else 0.0


def calculate_metrics(true_dir, pred_dir, iou_threshold=0.5, num_classes=10):
    """Calculate per-class metrics: TP, FP, FN, Precision, Recall, F1"""
    stats = {
        cls: {"TP": 0, "FP": 0, "FN": 0, "WC": 0, "GT": 0}
        for cls in range(num_classes)
    }

    for file in os.listdir(true_dir):
        if not file.endswith(".txt"):
            continue

        gt_labels = load_labels(os.path.join(true_dir, file))
        pr_labels = load_labels(os.path.join(pred_dir, file))

        # Count GT
        for gt in gt_labels:
            cls = int(gt[0])
            stats[cls]["GT"] += 1

        used_gt = set()

        # Process predictions
        for pred in pr_labels:
            pred_cls, pred_box = int(pred[0]), pred[1:5]
            best_iou, best_gt_idx, best_gt_cls = 0, -1, -1

            for gt_idx, gt in enumerate(gt_labels):
                if gt_idx in used_gt:
                    continue
                gt_cls, gt_box = int(gt[0]), gt[1:5]
                score = iou(gt_box, pred_box)

                if score >= iou_threshold and score > best_iou:
                    best_iou = score
                    best_gt_idx = gt_idx
                    best_gt_cls = gt_cls

            if best_gt_idx >= 0:
                used_gt.add(best_gt_idx)
                if pred_cls == best_gt_cls:
                    stats[pred_cls]["TP"] += 1
                else:
                    stats[pred_cls]["FP"] += 1
                    stats[pred_cls]["WC"] += 1
            else:
                stats[pred_cls]["FP"] += 1

        # Count FN
        for gt_idx, gt in enumerate(gt_labels):
            if gt_idx not in used_gt:
                gt_cls = int(gt[0])
                stats[gt_cls]["FN"] += 1

    # Compute metrics
    results = {}
    pa_list = []

    for cls in range(num_classes):
        tp = stats[cls]["TP"]
        fp = stats[cls]["FP"]
        fn = stats[cls]["FN"]
        gt = stats[cls]["GT"]

        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        pa = tp / (tp + fn + 1e-6) if (tp + fn) > 0 else 0.0

        if gt > 0:
            pa_list.append(pa)

        results[cls] = {
            "GT": gt,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "WC": stats[cls]["WC"],
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1": round(f1, 4),
            "PA": round(pa, 4),
        }

    return results, stats, pa_list


def print_metrics_table(results, stats, pa_list, class_names, map_score):
    """Print metrics in formatted table"""
    all_gt = sum(s["GT"] for s in stats.values())
    all_tp = sum(s["TP"] for s in stats.values())
    all_fp = sum(s["FP"] for s in stats.values())
    all_fn = sum(s["FN"] for s in stats.values())
    all_wc = sum(s["WC"] for s in stats.values())

    all_prec = all_tp / (all_tp + all_fp + 1e-6)
    all_rec = all_tp / (all_tp + all_fn + 1e-6)
    all_f1 = 2 * all_prec * all_rec / (all_prec + all_rec + 1e-6)
    all_pa = sum(pa_list) / len(pa_list) if pa_list else 0.0

    # Print header
    header = (f"{'Class':<15} {'GT':>5} {'TP':>5} {'FP':>5} {'FN':>5} {'WC':>5} "
              f"{'Prec':>7} {'Rec':>7} {'F1':>7} {'PA':>7}")
    sep = "─" * len(header)

    print(f"\n{sep}")
    print(header)
    print(sep)

    # Print per-class metrics
    for cls, m in results.items():
        print(f"{class_names[cls]:<15} {m['GT']:>5} {m['TP']:>5} {m['FP']:>5} {m['FN']:>5} {m['WC']:>5} "
              f"{m['Precision']:>7.4f} {m['Recall']:>7.4f} {m['F1']:>7.4f} {m['PA']:>7.4f}")

    print(sep)

    # Macro average
    active = [m for m in results.values() if m["GT"] > 0]
    avg_prec = sum(m["Precision"] for m in active) / len(active) if active else 0.0
    avg_rec = sum(m["Recall"] for m in active) / len(active) if active else 0.0
    avg_f1 = sum(m["F1"] for m in active) / len(active) if active else 0.0
    avg_pa = sum(m["PA"] for m in active) / len(active) if active else 0.0

    print(f"{'MICRO (all)':<15} {all_gt:>5} {all_tp:>5} {all_fp:>5} {all_fn:>5} {all_wc:>5} "
          f"{all_prec:>7.4f} {all_rec:>7.4f} {all_f1:>7.4f} {all_pa:>7.4f}")
    print(f"{'MACRO (avg)':<15} {'':>5} {'':>5} {'':>5} {'':>5} {'':>5} "
          f"{avg_prec:>7.4f} {avg_rec:>7.4f} {avg_f1:>7.4f} {avg_pa:>7.4f}")

    print(sep)
    print(f"\nmAP@0.5 = {map_score:.4f}")
    print(f"mPA (mean Recall) = {all_pa:.4f}")
    print(f"Overall F1 = {all_f1:.4f}\n")



print("="*70)
print("YOLO-DAM Testing with RELATIVE Positioning Decoding")
print("="*70)

# Import model
print("\n[1] Importing model...")
try:
    from YOLO_DAM_v2 import model 
    print("✅ Model imported")
except ImportError as e:
    print(f"✗ Failed to import model: {e}")
    sys.exit(1)

model.load_weights("D:/Projekty/2022_01_BattPor/2025_12_Dresden/Models/YOLODAMv2_best_final.h5")

# Load weights
print(f"\n[2] Loading weights from: {WEIGHTS}")
tester = YOLOModelTester(
    model=model,
    class_names=[f"Class_{i}" for i in range(NUM_CLASSES)],
    img_size=IMG_SIZE,
    conf_threshold=CONF_THRESH,
    iou_threshold=IOU_THRESH
)
tester.load_checkpoint(WEIGHTS)

# Find images
print(f"\n[3] Finding images in: {IMG_DIR}")
img_paths = sorted([
    p for p in Path(IMG_DIR).iterdir()
    if p.suffix.lower() in IMG_EXTS
])
print(f"✅ Found {len(img_paths)} images")

# Create output directories
pred_out_dir = Path(PRED_DIR)
pred_out_dir.mkdir(parents=True, exist_ok=True)

vis_dir = pred_out_dir / "visualizations_relative"
vis_dir.mkdir(parents=True, exist_ok=True)

# Run inference
print(f"\n[4] Running inference and saving results...")
print(f"    Labels → {PRED_DIR}")
print(f"    Visualizations → {vis_dir}\n")

for idx, img_path in enumerate(img_paths):
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"  [WARN] Cannot read {img_path.name}")
        continue

    # Inference
    detections, _ = tester.predict_image(str(img_path))

    # Save YOLO labels
    out_path = pred_out_dir / (img_path.stem + ".txt")
    with open(out_path, "w") as f:
        for box in detections:
            cx, cy, bw, bh, conf, cls_id = box
            if int(cls_id) == 9:
                bw = 3 / IMG_SIZE
                bh = 3 / IMG_SIZE
            f.write(f"{int(cls_id)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {conf:.6f}\n")

    # Save visualization
    vis_path = vis_dir / (img_path.stem + "_vis.jpg")
    tester.visualize_predictions(
        image_path=str(img_path),
        detections=detections,
        save_path=str(vis_path),
        show_labels=True
    )

    print(f"  [{idx+1:>4}/{len(img_paths)}]  {img_path.name:<40}  {len(detections):>3} det")

# Calculate metrics
print(f"\n[5] Calculating metrics...")
results, stats, pa_list = calculate_metrics(
    TRUE_DIR, str(pred_out_dir),
    iou_threshold=IOU_THRESH,
    num_classes=NUM_CLASSES
)

map_score = compute_map(
    TRUE_DIR, str(pred_out_dir),
    iou_threshold=IOU_THRESH,
    num_classes=NUM_CLASSES
)

# Print results
print_metrics_table(
    results, stats, pa_list,
    [f"Class_{i}" for i in range(NUM_CLASSES)],
    map_score
)

print(f"✅ Done! Processed {len(img_paths)} images")
print(f"   Labels:          {PRED_DIR}")
print(f"   Visualizations:  {vis_dir}")
print("="*70)
