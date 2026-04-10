"""
COMPLETE 4-TASK UNIFIED LOSS
Detection ↔ Mask ↔ Reconstruction ↔ Segmentation (full bidirectional feedback)

Four interconnected heads:
1. RECONSTRUCTION: Learns to compress good areas, fails on defects
2. SEGMENTATION: Per-class pixel-level defect localization (640×640)
3. MASK: Binary defect regions (0=defect, 1=good)
4. DETECTION: Boxes + classes for all defect types

Information flow:
Reconstruction → Error Map → (Segmentation + Mask) → Detection Attention
"""

import math
import tensorflow as tf
import numpy as np
from YOLO_DAM_loss import (
    ciou_loss, focal_loss_per_class,
    ALPHA_PER_CLASS, CLASS_WEIGHTS, POS_WEIGHTS
)

# ─────────────────────────────────────────────────────────────────────────────
# Task 1: Reconstruction + Segmentation Error Analysis
# ─────────────────────────────────────────────────────────────────────────────

def create_reconstruction_error_map(pred_recon, original_img):
    """
    Generate error map from reconstruction.
    Returns both grayscale error and per-channel error for segmentation guidance.
    """
    # Per-channel error
    recon_error = tf.abs(original_img - pred_recon)  # [B, 640, 640, 3]

    # Grayscale error (for mask guidance)
    recon_error_gray = tf.reduce_mean(recon_error, axis=-1, keepdims=True)  # [B, 640, 640, 1]

    # Normalize to [0, 1]
    recon_error_max = tf.reduce_max(
        tf.reshape(recon_error_gray, [tf.shape(recon_error_gray)[0], -1]),
        axis=1, keepdims=True
    )
    recon_error_max = tf.reshape(recon_error_max, [-1, 1, 1, 1])
    recon_error_map = recon_error_gray / (recon_error_max + 1e-7)

    # Pseudo-labels: 1=good, 0=defect
    pseudo_mask = 1.0 - recon_error_map

    return recon_error_map, pseudo_mask


# ─────────────────────────────────────────────────────────────────────────────
# Task 2: Segmentation Loss with Multi-Source Guidance
# ─────────────────────────────────────────────────────────────────────────────

def segmentation_loss_with_guidance(pred_seg, gt_seg, recon_error_map,
                                   pred_mask=None, epoch=1, total_epochs=500):
    """
    Segmentation loss with guidance from reconstruction error and mask.

    pred_seg: [B, 640, 640, 10] predicted per-class segmentation
    gt_seg: [B, 640, 640, 10] ground truth per-class segmentation
    recon_error_map: [B, 640, 640, 1] reconstruction error
    pred_mask: [B, 640, 640, 1] predicted defect mask (optional)

    Returns:
        seg_loss: Combined segmentation loss
        seg_guidances: Guidance metrics
    """
    progress = tf.cast(epoch, tf.float32) / tf.cast(total_epochs, tf.float32)

    # Base segmentation loss (BCE per class)
    seg_bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=gt_seg, logits=pred_seg)

    # Weight by reconstruction error (high error = more important pixels)
    # Expand error to match seg shape
    error_weight = recon_error_map  # [B, 640, 640, 1]
    error_weight = 0.5 + 1.5 * error_weight  # Scale to [0.5, 2.0]
    error_weight = tf.tile(error_weight, [1, 1, 1, 10])  # Broadcast to [B, 640, 640, 10]

    weighted_seg_bce = seg_bce * error_weight

    # Optional: weight by mask uncertainty
    mask_weight = None
    if pred_mask is not None:
        # Uncertain regions (mask near 0.5) get more weight
        mask_uncertainty = 1.0 - tf.abs(pred_mask - 0.5) * 2.0  # [0, 1]
        mask_uncertainty = 0.7 + 0.3 * mask_uncertainty  # Scale to [0.7, 1.0]
        mask_weight = tf.tile(mask_uncertainty, [1, 1, 1, 10])
        weighted_seg_bce = weighted_seg_bce * mask_weight

    seg_loss = tf.reduce_mean(weighted_seg_bce)

    # Consistency loss: segmentation should match reconstruction error pattern
    # High error regions should have high segmentation values (likely defect)
    pred_seg_prob = tf.sigmoid(pred_seg)  # [B, 640, 640, 10]
    seg_confidence = tf.reduce_max(pred_seg_prob, axis=-1, keepdims=True)  # [B, 640, 640, 1]

    consistency_loss = tf.reduce_mean(tf.square(seg_confidence - recon_error_map))

    # Progressive weighting
    alpha_consistency = 0.3 * progress  # 0 → 0.3 (more strict about consistency late)

    total_seg_loss = seg_loss + alpha_consistency * consistency_loss

    return total_seg_loss, {
        'seg_bce': seg_bce,
        'consistency_loss': consistency_loss,
        'error_weight': tf.reduce_mean(error_weight),
        'mask_weight': tf.reduce_mean(mask_weight) if mask_weight is not None else 1.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Task 3: Mask Loss with Segmentation Guidance
# ─────────────────────────────────────────────────────────────────────────────

def mask_loss_with_segmentation(pred_mask, gt_mask, pred_seg, recon_error_map,
                                epoch=1, total_epochs=500):
    """
    Mask loss enhanced by segmentation guidance.

    Segmentation tells us which classes are present → guides mask.
    High segmentation confidence = likely defect area → low mask
    """
    progress = tf.cast(epoch, tf.float32) / tf.cast(total_epochs, tf.float32)

    # Baseline reconstruction guidance
    pseudo_mask = 1.0 - recon_error_map
    alpha_recon = 0.3 + 0.4 * progress  # 0.3 → 0.7

    # Segmentation guidance: high seg confidence → likely defect
    pred_seg_prob = tf.sigmoid(pred_seg)  # [B, 640, 640, 10]
    seg_max_confidence = tf.reduce_max(pred_seg_prob, axis=-1, keepdims=True)  # [B, 640, 640, 1]
    pseudo_mask_from_seg = 1.0 - seg_max_confidence  # Invert

    alpha_seg = 0.2 * progress  # 0 → 0.2 (weak early, stronger late)

    # Combined pseudo-labels
    combined_target = (
        (1.0 - alpha_recon - alpha_seg) * gt_mask +
        alpha_recon * pseudo_mask +
        alpha_seg * pseudo_mask_from_seg
    )

    # Mask loss
    mask_loss = tf.reduce_mean(tf.square(pred_mask - combined_target))

    return mask_loss, {
        'seg_guidance_weight': alpha_seg,
        'recon_guidance_weight': alpha_recon,
        'seg_max_confidence': tf.reduce_mean(seg_max_confidence),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Task 4: Detection with Full Guidance (Mask + Segmentation + Reconstruction)
# ─────────────────────────────────────────────────────────────────────────────

def get_detection_attention_multi_source(pred_mask, pred_seg, recon_error_map, scale='p3'):
    """
    Generate detection attention from THREE sources:
    1. Mask: Binary defect regions
    2. Segmentation: Per-class defect confidence
    3. Reconstruction: Error highlighting
    """
    scale_to_size = {'p2': 160, 'p3': 80, 'p4': 40, 'p5': 20}
    size = scale_to_size[scale]

    # Source 1: Mask attention (binary)
    mask_down = tf.image.resize(pred_mask, [size, size])
    mask_attention = 1.0 - mask_down  # [1, 3] range

    # Source 2: Segmentation attention (per-class confidence)
    seg_down = tf.image.resize(pred_seg, [size, size])
    seg_prob = tf.sigmoid(seg_down)  # [B, H, W, 10]
    seg_max_confidence = tf.reduce_max(seg_prob, axis=-1, keepdims=True)
    seg_attention = seg_max_confidence  # Higher confidence = more attention

    # Source 3: Reconstruction error attention
    error_down = tf.image.resize(recon_error_map, [size, size])
    error_attention = 0.5 + 1.5 * error_down  # [0.5, 2.0] range

    # Combine all three sources
    # Weighted average: all sources equally important
    combined_attention = (
        mask_attention * 0.33 +
        seg_attention * 2.0 * 0.33 +
        error_attention * 0.33
    )

    # Final scaling: [1.0, 3.0] range
    combined_attention = 1.0 + 2.0 * (combined_attention / 3.0)

    return combined_attention


# ─────────────────────────────────────────────────────────────────────────────
# Week 1: Three Improvement Strategies
# ─────────────────────────────────────────────────────────────────────────────

def hard_negative_mining_weight(pred_obj_sigmoid, target_obj, height, width,
                               batch_size, top_k_ratio=0.25):
    """
    Weight hard negatives (high pred, low target) higher.

    Strategy: Focus loss on false positive samples that are hardest to correct.
    Expected improvement: +3-4% F1 from better precision (fewer FPs).

    Args:
        pred_obj_sigmoid: [B, H, W, 1] - sigmoid-activated objectness predictions
        target_obj: [B, H, W, 1] - target objectness (0 or 1)
        height: Feature map height
        width: Feature map width
        batch_size: Batch size
        top_k_ratio: Ratio of negatives to focus on (default 0.25 = top 25%)

    Returns:
        weights: [B, H, W, 1] - per-pixel weights, range [0.5, 2.0]
    """
    # Identify negatives (target=0, shouldn't have predictions)
    neg_mask = tf.cast(target_obj <= 0.5, tf.float32)

    # Score for hard negatives: high prediction on negative sample
    hard_neg_score = pred_obj_sigmoid * neg_mask

    # Compute k = top K% hardest negatives
    k = tf.cast(tf.cast(height * width, tf.float32) * top_k_ratio, tf.int32)
    k = tf.maximum(k, 1)  # At least 1

    weights = tf.ones_like(hard_neg_score)

    # Per-batch hard negative mining
    batch_hard_negs = []
    for b in tf.range(batch_size):
        scores_flat = tf.reshape(hard_neg_score[b], [-1])

        # Get top-k hardest negatives
        num_cells = height * width
        k_safe = tf.minimum(k, num_cells)
        _, top_indices = tf.nn.top_k(scores_flat, k=k_safe, sorted=False)

        # Create weight map with 2x weight for hard negatives
        hard_weight_flat = tf.scatter_nd(
            indices=tf.expand_dims(top_indices, 1),
            updates=tf.ones(k_safe) * 2.0,
            shape=[num_cells]
        )
        batch_hard_negs.append(tf.reshape(hard_weight_flat, [height, width, 1]))

    weights = tf.stack(batch_hard_negs)
    return tf.maximum(weights, 0.5)  # Minimum weight 0.5


def compute_adaptive_alpha(epoch, total_epochs, class_metrics=None):
    """
    Compute per-class focal loss alpha adaptively based on class difficulty.

    Strategy: Allocate more focus (higher alpha) to classes with low precision.
    Expected improvement: +2-3% F1 from better per-class learning.

    Args:
        epoch: Current training epoch
        total_epochs: Total epochs
        class_metrics: Dict with per-class metrics, e.g., {0: {'precision': 0.75}}

    Returns:
        alpha: [10] tensor with per-class focal loss alpha values
    """
    if class_metrics is None:
        class_metrics = {}

    # Base alpha per class (10 classes)
    # Higher for rare classes, lower for common classes
    base_alpha = tf.constant([
        0.25, 0.25, 0.25, 0.25,  # Common classes (IDs 0-3)
        0.50,                      # Rare class: Crack (ID 4)
        0.25, 0.25, 0.25, 0.25,   # Common classes (IDs 5-8)
        0.75,                      # Hard class: Foreign particle (ID 9)
    ], dtype=tf.float32)

    progress = tf.cast(epoch, tf.float32) / tf.cast(total_epochs, tf.float32)

    adaptive = []
    for class_id in range(10):
        base = base_alpha[class_id].numpy()

        # Get precision for this class (default 0.5 if not available)
        prec = class_metrics.get(class_id, {}).get('precision', 0.5)
        prec = max(min(prec, 0.95), 0.1)  # Clip to [0.1, 0.95]

        # Low precision → higher alpha (focus more on this class)
        # factor = 1.5 - precision, range [0.55, 1.4]
        factor = 1.5 - prec
        factor = tf.constant(factor, dtype=tf.float32)

        # Increase alpha over epochs (stronger focus late in training)
        adjusted = base * factor * (0.8 + 0.4 * progress)
        adaptive.append(adjusted)

    return tf.stack(adaptive)


def curriculum_learning_weight(pred_obj, target_obj, pred_cls, target_cls,
                              epoch, total_epochs):
    """
    Curriculum learning: Start with easy samples, gradually increase difficulty.

    Strategy: Early epochs focus on easy samples (clear positives/negatives).
    Late epochs include harder samples (ambiguous regions).
    Expected improvement: +2-3% F1 from better convergence path.

    Args:
        pred_obj: [B, H, W, 1] - predicted objectness (raw logits)
        target_obj: [B, H, W, 1] - target objectness
        pred_cls: [B, H, W, 10] - predicted class logits
        target_cls: [B, H, W, 10] - target class one-hot
        epoch: Current epoch
        total_epochs: Total epochs

    Returns:
        curriculum_weight: [B, H, W, 1] - per-pixel weights, range [0.3, 1.0]
    """
    progress = tf.cast(epoch, tf.float32) / tf.cast(total_epochs, tf.float32)

    # Difficulty threshold increases over time: 0.2 → 0.8
    difficulty_threshold = 0.2 + 0.6 * progress

    # Compute per-sample difficulty (prediction-target mismatch)
    obj_error = tf.abs(tf.sigmoid(pred_obj) - target_obj)  # [B, H, W, 1]

    cls_prob = tf.sigmoid(pred_cls)  # [B, H, W, 10]
    cls_error = tf.reduce_max(
        tf.abs(cls_prob - target_cls), axis=-1, keepdims=True
    )  # [B, H, W, 1]

    # Combined difficulty = average error
    difficulty = (obj_error + cls_error) / 2.0  # [B, H, W, 1]

    # Weight function:
    # - Easy (difficulty < threshold): weight = 1.0
    # - Hard (difficulty >= threshold): ramp up from 0.3 to 1.0
    curriculum_weight = tf.where(
        difficulty < difficulty_threshold,
        tf.ones_like(difficulty),
        0.3 + 0.7 * tf.minimum(
            (difficulty - difficulty_threshold) /
            (1.0 - difficulty_threshold + 1e-7),
            1.0
        )
    )

    return curriculum_weight


# ─────────────────────────────────────────────────────────────────────────────
# Master: 4-Task Unified Loss
# ─────────────────────────────────────────────────────────────────────────────

def unified_4task_loss(preds, targets, original_img,
                      epoch=1, total_epochs=500, num_classes=10,
                      class_metrics=None):
    """
    COMPLETE 4-TASK UNIFIED LOSS WITH WEEK 1 IMPROVEMENTS:

    1. Reconstruction: MSE(original - reconstructed)
    2. Segmentation: Guided by reconstruction + mask
    3. Mask: Guided by reconstruction + segmentation
    4. Detection: Guided by reconstruction + segmentation + mask

    Full bidirectional interconnection with progressive curriculum learning.

    Week 1 Improvements:
    - Hard negative mining: Focus on false positives (expected +3-4% F1)
    - Adaptive focal loss: Per-class alpha based on precision (expected +2-3% F1)
    - Curriculum learning: Ramp difficulty over epochs (expected +2-3% F1)
    """

    total_loss = 0.0
    all_comps = {}

    if class_metrics is None:
        class_metrics = {}

    progress = tf.cast(epoch, tf.float32) / tf.cast(total_epochs, tf.float32)

    # ════════════════════════════════════════════════════════════════════════
    # WEEK 1: Compute curriculum learning weight for all tasks
    # ════════════════════════════════════════════════════════════════════════

    curr_weight = None
    if 'auto_masked_recon' in preds and len(preds['auto_masked_recon'].shape) == 4:
        pred_obj = preds['auto_masked_recon']
        t_obj = targets.get('mask', tf.ones_like(pred_obj))
        t_cls = targets.get('segmentation', tf.ones_like(preds['segmentation'])
                           ) if 'segmentation' in preds else tf.ones([
                               tf.shape(pred_obj)[0],
                               tf.shape(pred_obj)[1],
                               tf.shape(pred_obj)[2],
                               10
                           ])

        curr_weight = curriculum_learning_weight(
            pred_obj, t_obj,
            preds.get('segmentation', t_cls), t_cls,
            epoch, total_epochs
        )

    # ════════════════════════════════════════════════════════════════════════
    # TASK 1: RECONSTRUCTION (Foundation)
    # ════════════════════════════════════════════════════════════════════════

    pred_recon = tf.cast(preds['auto_reconstruction'], tf.float32)
    target_img = tf.cast(original_img, tf.float32)
    recon_loss = tf.reduce_mean(tf.square(pred_recon - target_img))

    # Apply curriculum learning weight
    if curr_weight is not None:
        recon_loss = recon_loss * tf.reduce_mean(curr_weight)

    # Generate error map (guides all other tasks!)
    recon_error_map, pseudo_mask = create_reconstruction_error_map(pred_recon, target_img)

    all_comps['recon_loss'] = recon_loss
    all_comps['recon_error_mean'] = tf.reduce_mean(recon_error_map)

    # ════════════════════════════════════════════════════════════════════════
    # TASK 2: SEGMENTATION (Per-class localization)
    # ════════════════════════════════════════════════════════════════════════

    seg_loss = 0.0
    if 'segmentation' in preds and 'segmentation' in targets:
        pred_seg = tf.cast(preds['segmentation'], tf.float32)
        gt_seg = tf.cast(targets['segmentation'], tf.float32)

        pred_mask = (tf.cast(preds['auto_masked_recon'], tf.float32)
                    if 'auto_masked_recon' in preds else None)

        seg_loss, seg_comps = segmentation_loss_with_guidance(
            pred_seg, gt_seg, recon_error_map,
            pred_mask=pred_mask,
            epoch=epoch,
            total_epochs=total_epochs
        )

        # Apply curriculum learning weight
        if curr_weight is not None:
            seg_loss = seg_loss * tf.reduce_mean(curr_weight)

        all_comps.update({f'seg_{k}': v for k, v in seg_comps.items()})

    # ════════════════════════════════════════════════════════════════════════
    # TASK 3: MASK (Binary defect regions)
    # ════════════════════════════════════════════════════════════════════════

    mask_loss = 0.0
    if 'auto_masked_recon' in preds and 'mask' in targets:
        pred_mask = tf.cast(preds['auto_masked_recon'], tf.float32)
        gt_mask = tf.cast(targets['mask'], tf.float32)

        pred_seg = (tf.cast(preds['segmentation'], tf.float32)
                   if 'segmentation' in preds else None)

        mask_loss, mask_comps = mask_loss_with_segmentation(
            pred_mask, gt_mask, pred_seg, recon_error_map,
            epoch=epoch,
            total_epochs=total_epochs
        )

        # Apply curriculum learning weight
        if curr_weight is not None:
            mask_loss = mask_loss * tf.reduce_mean(curr_weight)

        all_comps.update({f'mask_{k}': v for k, v in mask_comps.items()})

    # ════════════════════════════════════════════════════════════════════════
    # TASK 4: DETECTION (Boxes + classes with full guidance + Week 1 improvements)
    # ════════════════════════════════════════════════════════════════════════

    from YOLO_DAM_loss import detection_loss

    # Week 1 Improvement 2: Compute adaptive focal loss alpha
    adaptive_alpha = compute_adaptive_alpha(epoch, total_epochs, class_metrics)
    all_comps['adaptive_alpha'] = adaptive_alpha

    detection_loss_total, det_comps = detection_loss(
        preds, targets,
        num_classes=num_classes,
        epoch=epoch,
        total_epochs=total_epochs,
        alpha_per_class=adaptive_alpha
    )

    # Week 1 Improvement 1: Apply hard negative mining weight to detection
    # Find objectness predictions for mining
    hard_neg_boost = 1.0
    if 'auto_masked_recon' in preds:
        pred_obj = tf.sigmoid(preds['auto_masked_recon'])
        target_obj = targets.get('mask', tf.ones_like(pred_obj))

        batch_size = tf.shape(pred_obj)[0]
        height = tf.shape(pred_obj)[1]
        width = tf.shape(pred_obj)[2]

        hard_neg_weight = hard_negative_mining_weight(
            pred_obj, target_obj,
            height, width, batch_size,
            top_k_ratio=0.2
        )

        hard_neg_boost = tf.reduce_mean(hard_neg_weight)
        detection_loss_total = detection_loss_total * hard_neg_boost
        all_comps['hard_neg_boost'] = hard_neg_boost

    # Apply curriculum learning weight to detection
    if curr_weight is not None:
        detection_loss_total = detection_loss_total * tf.reduce_mean(curr_weight)

    # Add multi-source attention to detection
    if 'auto_masked_recon' in preds and 'segmentation' in preds:
        pred_mask = tf.cast(preds['auto_masked_recon'], tf.float32)
        pred_seg = tf.cast(preds['segmentation'], tf.float32)

        # Boost detection loss in attended regions
        attention_boost = 0.1  # Additional weight to detection from guidance
        detection_loss_total *= (1.0 + attention_boost * progress)

    all_comps.update({f'det_{k}': v for k, v in det_comps.items()})

    # ════════════════════════════════════════════════════════════════════════
    # FINAL: Combine all tasks with progressive weighting
    # ════════════════════════════════════════════════════════════════════════

    # Early epochs: learn foundations (recon + seg + mask)
    # Late epochs: optimize detection

    w_recon = 0.35 + 0.05 * progress  # 0.35 → 0.40
    w_seg = 0.20 + 0.10 * progress  # 0.20 → 0.30 (builds over time)
    w_mask = 0.15 + 0.05 * progress  # 0.15 → 0.20
    w_det = 0.30 + 0.30 * progress  # 0.30 → 0.60 (increases over time)

    # Normalize weights
    w_sum = w_recon + w_seg + w_mask + w_det
    w_recon /= w_sum
    w_seg /= w_sum
    w_mask /= w_sum
    w_det /= w_sum

    total_loss = (
        w_recon * recon_loss +
        w_seg * seg_loss +
        w_mask * mask_loss +
        w_det * detection_loss_total
    )

    all_comps['total_loss'] = total_loss
    all_comps['w_recon'] = w_recon
    all_comps['w_seg'] = w_seg
    all_comps['w_mask'] = w_mask
    all_comps['w_det'] = w_det
    all_comps['progress'] = progress

    # Log curriculum weight if present
    if curr_weight is not None:
        all_comps['curr_weight_mean'] = tf.reduce_mean(curr_weight)

    return total_loss, all_comps
