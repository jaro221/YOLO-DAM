import math
import tensorflow as tf

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
ALPHA_PER_CLASS = [
    0.25,  # 0 Agglomerate
    0.25,  # 1 Pinhole-long
    0.25,  # 2 Pinhole-trans
    0.25,  # 3 Pinhole-round
    0.50,  # 4 Crack-long         ← boost rare class
    0.25,  # 5 Crack-trans
    0.25,  # 6 Line-long
    0.25,  # 7 Line-trans
    0.25,  # 8 Line-diag
    0.75,  # 9 Foreign-particle   ← high alpha = penalise FP more
]

CLASS_WEIGHTS = tf.constant([
    1.0,   # 0 Agglomerate    — 1820 instances
    1.0,   # 1 Pinhole-long   — 1851
    1.0,   # 2 Pinhole-trans  — 2516
    1.0,   # 3 Pinhole-round  — 1530
    2.0,   # 4 Crack-long     — 1145 (fewest)
    1.0,   # 5 Crack-trans    — 2229
    1.0,   # 6 Line-long      — 2051
    1.0,   # 7 Line-trans     — 2006
    1.0,   # 8 Line-diag      — 1502
    2.0,   # 9 Foreign-particle — 1576 but hardest
], dtype=tf.float32)


# ─────────────────────────────────────────────────────────────────────────────
# CIoU Loss
# ─────────────────────────────────────────────────────────────────────────────
def ciou_loss(pred_boxes, target_boxes, eps=1e-7):
    """
    Complete IoU loss.
    pred_boxes, target_boxes: [..., 4] in [cx, cy, w, h] normalized [0,1]
    """
    pred_x1 = pred_boxes[..., 0] - pred_boxes[..., 2] / 2
    pred_y1 = pred_boxes[..., 1] - pred_boxes[..., 3] / 2
    pred_x2 = pred_boxes[..., 0] + pred_boxes[..., 2] / 2
    pred_y2 = pred_boxes[..., 1] + pred_boxes[..., 3] / 2

    target_x1 = target_boxes[..., 0] - target_boxes[..., 2] / 2
    target_y1 = target_boxes[..., 1] - target_boxes[..., 3] / 2
    target_x2 = target_boxes[..., 0] + target_boxes[..., 2] / 2
    target_y2 = target_boxes[..., 1] + target_boxes[..., 3] / 2

    inter_x1 = tf.maximum(pred_x1, target_x1)
    inter_y1 = tf.maximum(pred_y1, target_y1)
    inter_x2 = tf.minimum(pred_x2, target_x2)
    inter_y2 = tf.minimum(pred_y2, target_y2)
    inter_area = tf.maximum(inter_x2 - inter_x1, 0.0) * tf.maximum(inter_y2 - inter_y1, 0.0)

    pred_area   = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
    union_area  = pred_area + target_area - inter_area
    iou = inter_area / (union_area + eps)

    enclose_x1 = tf.minimum(pred_x1, target_x1)
    enclose_y1 = tf.minimum(pred_y1, target_y1)
    enclose_x2 = tf.maximum(pred_x2, target_x2)
    enclose_y2 = tf.maximum(pred_y2, target_y2)
    enclose_c2 = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2 + eps

    pred_cx   = (pred_x1 + pred_x2) / 2
    pred_cy   = (pred_y1 + pred_y2) / 2
    target_cx = (target_x1 + target_x2) / 2
    target_cy = (target_y1 + target_y2) / 2
    center_dist2 = (pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2

    pred_w   = pred_boxes[..., 2]
    pred_h   = pred_boxes[..., 3]
    target_w = target_boxes[..., 2]
    target_h = target_boxes[..., 3]
    v = (4 / (math.pi ** 2)) * tf.pow(
        tf.atan(target_w / (target_h + eps)) - tf.atan(pred_w / (pred_h + eps)), 2)
    alpha = tf.stop_gradient(v / (1 - iou + v + eps))

    ciou = iou - (center_dist2 / enclose_c2 + alpha * v)
    return 1 - ciou


# ─────────────────────────────────────────────────────────────────────────────
# Focal Loss
# ─────────────────────────────────────────────────────────────────────────────
def focal_loss_per_class(y_true, y_pred_logits,
                         alpha_per_class=ALPHA_PER_CLASS, gamma=2.0):
    alpha   = tf.constant(alpha_per_class, dtype=tf.float32)
    y_pred  = tf.sigmoid(y_pred_logits)
    bce     = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred_logits)
    p_t     = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    focal   = tf.pow(1.0 - p_t, gamma)
    alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
    return alpha_t * focal * bce


# ─────────────────────────────────────────────────────────────────────────────
# Detection Loss
# ─────────────────────────────────────────────────────────────────────────────
def detection_loss(preds, targets, num_classes=10,
                   label_smoothing=0.0,
                   epoch=1,
                   total_epochs=500):
    """
    Combined detection + auxiliary loss.
    Scales: P2/P3/P4/P5 — both M2M (many-to-many) and O2O (one-to-one) heads.
    Auxiliary: mask loss + autoencoder reconstruction loss (ProgLoss weighted).
    """
    total_loss = 0.0
    comps = {}
    eps = 1e-7

    # ── Progressive loss weights ───────────────────────────────────────────
    progress = tf.cast(epoch, tf.float32) / tf.cast(total_epochs, tf.float32)
    mask_w   = 0.1 + 0.4 * progress   # 0.1 → 0.5
    recon_w  = 0.1 + 0.4 * progress   # 0.1 → 0.5

    # ── Detection scales ───────────────────────────────────────────────────
    for scale in ['p2', 'p3', 'p4', 'p5']:
        if f"{scale}_cls" not in preds:
            continue

        pred_cls = tf.cast(preds[f"{scale}_cls"], tf.float32)
        pred_reg = tf.cast(preds[f"{scale}_reg"], tf.float32)
        pred_obj = tf.cast(preds[f"{scale}_obj"], tf.float32)

        t_cls = tf.cast(targets[f"{scale}_cls_t"], tf.float32)
        t_reg = tf.cast(targets[f"{scale}_reg_t"], tf.float32)
        t_obj = tf.cast(targets[f"{scale}_obj_t"], tf.float32)

        if label_smoothing > 0:
            t_cls = t_cls * (1 - label_smoothing) + label_smoothing / num_classes

        pos_mask      = tf.cast(t_obj > 0.5, tf.float32)
        pos_count     = tf.reduce_sum(pos_mask) + eps
        pospre_mask   = tf.cast(tf.sigmoid(pred_obj) > 0.5, tf.float32)
        pospre_count  = tf.reduce_sum(pospre_mask) + eps

        # Regression (CIoU)
        pred_reg_sigmoid  = tf.sigmoid(pred_reg)
        pred_boxes_flat   = tf.reshape(pred_reg_sigmoid, [-1, 4])
        target_boxes_flat = tf.reshape(t_reg, [-1, 4])
        pos_indices       = tf.where(tf.reshape(pos_mask, [-1]) > 0.5)[:, 0]
        if tf.size(pos_indices) > 0:
            reg_loss = tf.reduce_mean(ciou_loss(
                tf.gather(pred_boxes_flat, pos_indices),
                tf.gather(target_boxes_flat, pos_indices)))
        else:
            reg_loss = 0.0

        # Objectness
        pos_weight = {"p2": 2.5, "p3": 2.5, "p4": 2.0, "p5": 1.2}[scale]
        weights    = 1.0 + (pos_weight - 1.0) * t_obj
        obj_bce    = tf.nn.sigmoid_cross_entropy_with_logits(labels=t_obj, logits=pred_obj)
        obj_loss   = tf.reduce_sum(obj_bce * weights) / (tf.reduce_sum(weights) + eps)

        # Classification (focal)
        cls_bce      = focal_loss_per_class(t_cls, pred_cls)
        cls_loss     = tf.reduce_sum(cls_bce * CLASS_WEIGHTS * pos_mask) / pos_count

        m2m_loss = 2.5 * reg_loss + 1.0 * obj_loss + 3.0 * cls_loss
        total_loss += m2m_loss

        # ── One-to-one head ───────────────────────────────────────────────
        pred_cls_o2o = tf.cast(preds[f"{scale}_cls_o2o"], tf.float32)
        pred_reg_o2o = tf.cast(preds[f"{scale}_reg_o2o"], tf.float32)
        pred_obj_o2o = tf.cast(preds[f"{scale}_obj_o2o"], tf.float32)
        t_cls_o2o    = tf.cast(targets[f"{scale}_cls_t_o2o"], tf.float32)
        t_reg_o2o    = tf.cast(targets[f"{scale}_reg_t_o2o"], tf.float32)
        t_obj_o2o    = tf.cast(targets[f"{scale}_obj_t_o2o"], tf.float32)

        pos_mask_o2o  = tf.cast(t_obj_o2o > 0.5, tf.float32)
        pos_count_o2o = tf.reduce_sum(pos_mask_o2o) + eps
        pos_idx_o2o   = tf.where(tf.reshape(pos_mask_o2o, [-1]) > 0.5)[:, 0]

        if tf.size(pos_idx_o2o) > 0:
            reg_loss_o2o = tf.reduce_mean(ciou_loss(
                tf.gather(tf.reshape(tf.sigmoid(pred_reg_o2o), [-1, 4]), pos_idx_o2o),
                tf.gather(tf.reshape(t_reg_o2o, [-1, 4]), pos_idx_o2o)))
        else:
            reg_loss_o2o = 0.0

        obj_loss_o2o = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=t_obj_o2o, logits=pred_obj_o2o))
        cls_loss_o2o = tf.reduce_sum(
            focal_loss_per_class(t_cls_o2o, pred_cls_o2o) * CLASS_WEIGHTS * pos_mask_o2o
        ) / pos_count_o2o

        o2o_loss = 2.5 * reg_loss_o2o + 1.0 * obj_loss_o2o + 3.0 * cls_loss_o2o
        total_loss += o2o_loss

        comps[f"{scale}_box"]     = reg_loss
        comps[f"{scale}_obj"]     = obj_loss
        comps[f"{scale}_cls"]     = cls_loss
        comps[f"{scale}_box_o2o"] = reg_loss_o2o
        comps[f"{scale}_obj_o2o"] = obj_loss_o2o
        comps[f"{scale}_cls_o2o"] = cls_loss_o2o
        comps[f"{scale}_pos"]     = float(pos_count)
        comps[f"{scale}_pospre"]  = float(pospre_count)

    # ── Auxiliary losses ───────────────────────────────────────────────────
    if 'auto_reconstruction' in preds:
        pred_recon = tf.cast(preds['auto_reconstruction'], tf.float32)
        target_img = tf.cast(targets['AUTO'], tf.float32)

        if 'mask' in targets:
            pred_mask   = tf.cast(preds['auto_masked_recon'], tf.float32)
            gt_mask     = tf.cast(targets['mask'], tf.float32)
            mask_loss   = tf.reduce_mean(tf.square(pred_mask - gt_mask))
            total_loss += mask_w * mask_loss
            comps['mask_loss'] = mask_loss

        recon_loss  = tf.reduce_mean(tf.square(pred_recon - target_img))
        total_loss += recon_w * recon_loss
        comps['recon_loss'] = recon_loss

    return total_loss, comps


# ─────────────────────────────────────────────────────────────────────────────
# Data Augmentation (from manuscript)
# ─────────────────────────────────────────────────────────────────────────────

def augment_hsv(image):
    """HSV color space augmentation"""
    # Random hue shift
    if tf.random.uniform([]) > 0.5:
        image = tf.image.random_hue(image, 0.015)

    # Random saturation
    if tf.random.uniform([]) > 0.5:
        image = tf.image.random_saturation(image, 0.7, 1.3)

    # Random brightness
    if tf.random.uniform([]) > 0.5:
        image = tf.image.random_brightness(image, 0.4)

    # Clip to valid range
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image


def augment_flip(image, boxes):
    """Random horizontal flip with box coordinate adjustment"""
    if tf.random.uniform([]) > 0.5:
        image = tf.image.flip_left_right(image)
        # Flip box x-coordinates
        if len(boxes) > 0:
            boxes_flipped = tf.stack([
                1.0 - boxes[:, 0],  # flip center x
                boxes[:, 1],         # keep y
                boxes[:, 2],         # keep w
                boxes[:, 3]          # keep h
            ], axis=-1)
            return image, boxes_flipped
    return image, boxes


# ─────────────────────────────────────────────────────────────────────────────
# Box Size Clamping
# ─────────────────────────────────────────────────────────────────────────────

def cap_box_size(x, y, w, h, min_side, max_side):
    """Clamp box width/height to [min_side, max_side], keep center within [0,1]"""
    new_w = float(tf.clip_by_value(w, min_side, max_side))
    new_h = float(tf.clip_by_value(h, min_side, max_side))
    # Keep centre, ensure box stays within image [0,1]
    new_x = float(tf.clip_by_value(x, new_w/2, 1.0 - new_w/2))
    new_y = float(tf.clip_by_value(y, new_h/2, 1.0 - new_h/2))
    return new_x, new_y, new_w, new_h


# ─────────────────────────────────────────────────────────────────────────────
# Defect Mask Creation
# ─────────────────────────────────────────────────────────────────────────────

def create_defect_mask(boxes, classes, img_size=640):
    """
    Create binary mask:
    - 1 = good area (background)
    - 0 = defect area (inside bounding boxes)

    Args:
        boxes: numpy array [N, 4] in normalized format [x_center, y_center, w, h]
        classes: numpy array [N] class indices
        img_size: image size (640)

    Returns:
        mask: numpy array [img_size, img_size, 1] with values 0 or 1
    """
    import numpy as np

    # Start with all good (1s)
    mask = np.ones((img_size, img_size, 1), dtype=np.float32)

    if len(boxes) == 0:
        return mask

    # Convert normalized boxes to pixel coordinates
    for box in boxes:
        x_center, y_center, w, h = box

        # Convert to pixel coordinates
        x_center_px = x_center * img_size
        y_center_px = y_center * img_size
        w_px = w * img_size
        h_px = h * img_size

        # Calculate corners
        x1 = int(max(0, x_center_px - w_px / 2))
        y1 = int(max(0, y_center_px - h_px / 2))
        x2 = int(min(img_size, x_center_px + w_px / 2))
        y2 = int(min(img_size, y_center_px + h_px / 2))

        # Set defect area to 0
        mask[y1:y2, x1:x2, 0] = 0.0

    return mask


# ─────────────────────────────────────────────────────────────────────────────
# Learning Rate Schedule with Warmup
# ─────────────────────────────────────────────────────────────────────────────

class ImprovedLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Learning rate schedule with proper warmup and cosine decay"""

    def __init__(self, initial_lr=1e-3, warmup_epochs=5, total_epochs=200,
                 steps_per_epoch=800, min_lr_ratio=0.01):
        super().__init__()
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.total_steps = total_epochs * steps_per_epoch
        self.min_lr = initial_lr * min_lr_ratio

    def __call__(self, step):
        step = tf.cast(step, tf.float32)

        # Warmup: linear increase from 0 to initial_lr
        warmup_lr = self.initial_lr * (step / self.warmup_steps)

        # Cosine decay after warmup
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        progress = tf.clip_by_value(progress, 0.0, 1.0)
        cosine_decay = 0.5 * (1 + tf.cos(3.14159 * progress))
        decayed_lr = (self.initial_lr - self.min_lr) * cosine_decay + self.min_lr

        lr = tf.where(step < self.warmup_steps, warmup_lr, decayed_lr)
        return lr

    def get_config(self):
        return {
            'initial_lr': self.initial_lr,
            'warmup_steps': self.warmup_steps,
            'total_steps': self.total_steps,
            'min_lr': self.min_lr
        }


# ─────────────────────────────────────────────────────────────────────────────
# Class Distribution Analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_dataset_distribution(labels_dir):
    """Analyze class distribution in dataset"""
    import os

    class_counts = {}
    total_objects = 0

    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]

    for label_file in label_files:
        label_path = os.path.join(labels_dir, label_file)
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls = int(parts[0])
                    class_counts[cls] = class_counts.get(cls, 0) + 1
                    total_objects += 1

    print("\n" + "="*80)
    print("Dataset Class Distribution Analysis")
    print("="*80)
    print(f"{'Class':<10} {'Count':<10} {'Percentage':<15} {'Weight (inverse)':<20}")
    print("-"*80)

    class_weights = {}
    for cls in sorted(class_counts.keys()):
        count = class_counts[cls]
        percentage = (count / total_objects) * 100 if total_objects > 0 else 0
        # Calculate inverse frequency weight
        weight = total_objects / (len(class_counts) * count) if count > 0 else 1.0
        class_weights[cls] = weight
        print(f"{cls:<10} {count:<10} {percentage:<15.2f}% {weight:<20.3f}")

    print("="*80)

    return class_weights, class_counts


# ─────────────────────────────────────────────────────────────────────────────
# Duplicate Label Detection
# ─────────────────────────────────────────────────────────────────────────────

def check_duplicates(labels, file=None, verbose=True):
    """Find and remove duplicate/redundant labels"""
    seen     = []
    dupes    = []
    unique   = []

    for row in labels:
        # Round to 6 decimal places to catch float noise
        key = tuple(round(v, 6) for v in row)
        if key in seen:
            dupes.append(row)
        else:
            seen.append(key)
            unique.append(row)

    if verbose:
        name = file if file else "labels"
        print(f"\n{'─'*50}")
        print(f"  File:       {name}")
        print(f"  Total:      {len(labels)}")
        print(f"  Unique:     {len(unique)}")
        print(f"  Duplicates: {len(dupes)}")
        if dupes:
            print(f"  Duplicate rows:")
            for d in dupes:
                print(f"    cls={int(d[0])}  cx={d[1]:.6f}  cy={d[2]:.6f}  w={d[3]:.6f}  h={d[4]:.6f}")
        print(f"{'─'*50}")

    return unique, dupes
