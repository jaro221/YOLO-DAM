"""
UNIFIED INTERCONNECTED LOSS
Detection ↔ Mask ↔ Reconstruction (bidirectional feedback)

Three heads inform each other:
1. Reconstruction → Mask (error map guides mask)
2. Mask → Detection (defect regions guide detection)
3. Reconstruction → Detection (error highlights important areas)
"""

import math
import tensorflow as tf
from YOLO_DAM_loss import (
    ciou_loss, focal_loss_per_class,
    ALPHA_PER_CLASS, CLASS_WEIGHTS, POS_WEIGHTS
)

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Reconstruction → Mask (Pseudo-label generation)
# ─────────────────────────────────────────────────────────────────────────────

def create_reconstruction_error_map(pred_recon, original_img):
    """
    Generate pseudo-labels for mask from reconstruction error.

    High reconstruction error → likely defect (mask should be ~0)
    Low reconstruction error → likely good (mask should be ~1)

    Args:
        pred_recon: [B, 640, 640, 3] reconstructed image
        original_img: [B, 640, 640, 3] original image

    Returns:
        recon_error_map: [B, 640, 640, 1] normalized error
        pseudo_mask: [B, 640, 640, 1] defect likelihood (0=defect, 1=good)
    """
    # Compute per-channel error
    recon_error = tf.abs(original_img - pred_recon)  # [B, 640, 640, 3]

    # Average across channels → grayscale error map
    recon_error_gray = tf.reduce_mean(recon_error, axis=-1, keepdims=True)  # [B, 640, 640, 1]

    # Normalize to [0, 1] per batch
    recon_error_max = tf.reduce_max(
        tf.reshape(recon_error_gray, [tf.shape(recon_error_gray)[0], -1]),
        axis=1, keepdims=True
    )
    recon_error_max = tf.reshape(recon_error_max, [-1, 1, 1, 1])
    recon_error_map = recon_error_gray / (recon_error_max + 1e-7)  # [B, 640, 640, 1]

    # Invert: high error → low mask (defect), low error → high mask (good)
    pseudo_mask = 1.0 - recon_error_map

    return recon_error_map, pseudo_mask


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Mask ↔ Detection (Mutual guidance)
# ─────────────────────────────────────────────────────────────────────────────

def get_detection_attention_from_mask(pred_mask, scale='p3'):
    """
    Create detection attention weights from mask.

    Low mask value (0) = defect region → HIGH detection weight
    High mask value (1) = good region → LOW detection weight

    Args:
        pred_mask: [B, 640, 640, 1]
        scale: 'p2', 'p3', 'p4', or 'p5'

    Returns:
        attention: [B, H, W, 1] attention weights for detection
    """
    # Map scale to spatial size
    scale_to_size = {
        'p2': 160,
        'p3': 80,
        'p4': 40,
        'p5': 20
    }

    size = scale_to_size[scale]

    # Downsample mask to detection scale
    mask_downsampled = tf.image.resize(pred_mask, [size, size])  # [B, H, W, 1]

    # Invert for attention: 0 (defect) → high weight, 1 (good) → low weight
    defect_attention = 1.0 - mask_downsampled  # [B, H, W, 1]

    # Amplify difference: multiply by scaling factor
    attention = 1.0 + 2.0 * defect_attention  # Range [1.0, 3.0]
    # 1.0 in good regions, 3.0 in defect regions

    return attention


def get_reconstruction_attention_map(recon_error_map, scale='p3'):
    """
    Create detection attention weights from reconstruction error.

    High reconstruction error → HIGH detection weight (focus here!)
    Low reconstruction error → LOW detection weight

    Args:
        recon_error_map: [B, 640, 640, 1] from create_reconstruction_error_map()
        scale: 'p2', 'p3', 'p4', or 'p5'

    Returns:
        attention: [B, H, W, 1] attention weights
    """
    # Map scale to spatial size
    scale_to_size = {
        'p2': 160,
        'p3': 80,
        'p4': 40,
        'p5': 20
    }

    size = scale_to_size[scale]

    # Downsample error map to detection scale
    recon_attention = tf.image.resize(recon_error_map, [size, size])  # [B, H, W, 1]

    # Scale: 1.0 where no error, 3.0 where high error
    attention = 1.0 + 2.0 * recon_attention  # Range [1.0, 3.0]

    return attention


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Unified Multi-Task Loss
# ─────────────────────────────────────────────────────────────────────────────

def unified_detection_loss(preds, targets, scale,
                          pred_mask=None, recon_error_map=None,
                          num_classes=10, epoch=1, total_epochs=500):
    """
    Detection loss with FEEDBACK from mask and reconstruction.

    Incorporates:
    1. Mask attention: Focus detection on defect regions
    2. Reconstruction attention: Focus on reconstruction error regions

    Args:
        preds: Detection predictions (M2M and O2O)
        targets: Ground truth targets
        scale: 'p2', 'p3', 'p4', or 'p5'
        pred_mask: [B, 640, 640, 1] predicted mask (optional)
        recon_error_map: [B, 640, 640, 1] reconstruction error (optional)
        epoch, total_epochs: For progressive weighting

    Returns:
        total_loss: Combined loss
        comps: Loss components dict
    """

    total_loss = 0.0
    comps = {}
    eps = 1e-7

    # ── Progressive weights ──────────────────────────────────────────────────
    progress = tf.cast(epoch, tf.float32) / tf.cast(total_epochs, tf.float32)

    # Early: trust mask/reconstruction more
    # Late: trust GT labels more
    mask_guidance_weight = 0.6 * (1.0 - progress)  # 0.6 → 0
    recon_guidance_weight = 0.4 * (1.0 - progress)  # 0.4 → 0

    # ── Compute attention maps from mask and reconstruction ──────────────────
    mask_attention = None
    recon_attention = None
    combined_attention = None

    if pred_mask is not None:
        mask_attention = get_detection_attention_from_mask(pred_mask, scale)

    if recon_error_map is not None:
        recon_attention = get_reconstruction_attention_map(recon_error_map, scale)

    # Combine attention maps
    if mask_attention is not None and recon_attention is not None:
        combined_attention = mask_attention * 0.5 + recon_attention * 0.5
    elif mask_attention is not None:
        combined_attention = mask_attention
    elif recon_attention is not None:
        combined_attention = recon_attention
    else:
        combined_attention = tf.ones_like(preds[f"{scale}_obj"])

    # ── M2M Head Loss ────────────────────────────────────────────────────────
    pred_cls = tf.cast(preds[f"{scale}_cls"], tf.float32)
    pred_reg = tf.cast(preds[f"{scale}_reg"], tf.float32)
    pred_obj = tf.cast(preds[f"{scale}_obj"], tf.float32)

    t_cls = tf.cast(targets[f"{scale}_cls_t"], tf.float32)
    t_reg = tf.cast(targets[f"{scale}_reg_t"], tf.float32)
    t_obj = tf.cast(targets[f"{scale}_obj_t"], tf.float32)

    pos_mask = tf.cast(t_obj > 0.5, tf.float32)
    pos_count = tf.reduce_sum(pos_mask) + eps

    # Regression (CIoU) - weighted by attention
    pred_reg_sigmoid = tf.sigmoid(pred_reg)
    pred_boxes_flat = tf.reshape(pred_reg_sigmoid, [-1, 4])
    target_boxes_flat = tf.reshape(t_reg, [-1, 4])
    pos_indices = tf.where(tf.reshape(pos_mask, [-1]) > 0.5)[:, 0]

    if tf.size(pos_indices) > 0:
        reg_loss = tf.reduce_mean(ciou_loss(
            tf.gather(pred_boxes_flat, pos_indices),
            tf.gather(target_boxes_flat, pos_indices)))
    else:
        reg_loss = 0.0

    # Objectness - WITH ATTENTION ✓
    pos_weight = POS_WEIGHTS[scale]
    weights = 1.0 + (pos_weight - 1.0) * t_obj

    # Apply mask/reconstruction attention to objectness weights
    attention_flat = tf.reshape(combined_attention, [-1])
    weights_flat = tf.reshape(weights, [-1])
    weights_attended = weights_flat * attention_flat
    weights = tf.reshape(weights_attended, tf.shape(weights))

    obj_bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=t_obj, logits=pred_obj)
    obj_loss = tf.reduce_sum(obj_bce * weights) / (tf.reduce_sum(weights) + eps)

    # Classification - WITH ATTENTION ✓
    cls_bce = focal_loss_per_class(t_cls, pred_cls)

    attention_flat = tf.reshape(combined_attention, [-1])
    pos_mask_flat = tf.reshape(pos_mask, [-1])
    pos_attention = attention_flat * pos_mask_flat
    pos_attention = tf.reshape(pos_attention, tf.shape(pos_mask))

    cls_loss = tf.reduce_sum(cls_bce * CLASS_WEIGHTS * pos_attention) / (
        tf.reduce_sum(pos_attention) + eps)

    m2m_loss = 2.0 * reg_loss + 1.2 * obj_loss + 2.5 * cls_loss
    total_loss += m2m_loss

    comps[f"{scale}_box"] = reg_loss
    comps[f"{scale}_obj"] = obj_loss
    comps[f"{scale}_cls"] = cls_loss
    comps[f"{scale}_attention"] = tf.reduce_mean(combined_attention)

    # ── O2O Head Loss ────────────────────────────────────────────────────────
    pred_cls_o2o = tf.cast(preds[f"{scale}_cls_o2o"], tf.float32)
    pred_reg_o2o = tf.cast(preds[f"{scale}_reg_o2o"], tf.float32)
    pred_obj_o2o = tf.cast(preds[f"{scale}_obj_o2o"], tf.float32)

    t_cls_o2o = tf.cast(targets[f"{scale}_cls_t_o2o"], tf.float32)
    t_reg_o2o = tf.cast(targets[f"{scale}_reg_t_o2o"], tf.float32)
    t_obj_o2o = tf.cast(targets[f"{scale}_obj_t_o2o"], tf.float32)

    pos_mask_o2o = tf.cast(t_obj_o2o > 0.5, tf.float32)
    pos_count_o2o = tf.reduce_sum(pos_mask_o2o) + eps
    pos_idx_o2o = tf.where(tf.reshape(pos_mask_o2o, [-1]) > 0.5)[:, 0]

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

    o2o_loss = 3.0 * reg_loss_o2o + 0.8 * obj_loss_o2o + 3.5 * cls_loss_o2o
    total_loss += o2o_loss

    comps[f"{scale}_box_o2o"] = reg_loss_o2o
    comps[f"{scale}_obj_o2o"] = obj_loss_o2o
    comps[f"{scale}_cls_o2o"] = cls_loss_o2o

    return total_loss, comps


# ─────────────────────────────────────────────────────────────────────────────
# Master Unified Loss Function (Detection + Mask + Reconstruction)
# ─────────────────────────────────────────────────────────────────────────────

def unified_multi_task_loss(preds, targets, original_img,
                           epoch=1, total_epochs=500, num_classes=10):
    """
    UNIFIED LOSS with full interconnection:

    Detection ← (Mask + Reconstruction)
    Mask ← Reconstruction
    Reconstruction (standalone)

    Flow:
    1. Reconstruction error highlights defect regions
    2. Error map creates pseudo-labels for mask
    3. Mask + error attention guide detection

    Args:
        preds: All predictions (detection, mask, reconstruction)
        targets: All targets (detection, mask)
        original_img: [B, 640, 640, 3] original image
        epoch, total_epochs: For progressive weighting
        num_classes: Number of defect classes

    Returns:
        total_loss: Combined loss
        all_comps: All loss components for logging
    """

    total_loss = 0.0
    all_comps = {}
    eps = 1e-7

    progress = tf.cast(epoch, tf.float32) / tf.cast(total_epochs, tf.float32)

    # ════════════════════════════════════════════════════════════════════════
    # PHASE 1: Reconstruction Error (baseline)
    # ════════════════════════════════════════════════════════════════════════

    pred_recon = tf.cast(preds['auto_reconstruction'], tf.float32)
    target_img = tf.cast(original_img, tf.float32)
    recon_loss = tf.reduce_mean(tf.square(pred_recon - target_img))

    # Generate error map and pseudo-mask
    recon_error_map, pseudo_mask = create_reconstruction_error_map(
        pred_recon, target_img)

    all_comps['recon_loss'] = recon_loss
    all_comps['recon_error_mean'] = tf.reduce_mean(recon_error_map)

    # ════════════════════════════════════════════════════════════════════════
    # PHASE 2: Mask Loss (with reconstruction guidance)
    # ════════════════════════════════════════════════════════════════════════

    if 'auto_masked_recon' in preds and 'mask' in targets:
        pred_mask = tf.cast(preds['auto_masked_recon'], tf.float32)
        gt_mask = tf.cast(targets['mask'], tf.float32)

        # Combine GT mask with pseudo-mask from reconstruction
        alpha_guidance = 0.3 + 0.4 * progress  # Increase over time: 0.3 → 0.7
        combined_mask_target = (
            (1.0 - alpha_guidance) * gt_mask +
            alpha_guidance * pseudo_mask
        )

        # Mask loss
        mask_loss = tf.reduce_mean(tf.square(pred_mask - combined_mask_target))

        # Bonus: reconstruction error consistency
        # Mask should match reconstruction error pattern
        recon_consistency_loss = tf.reduce_mean(
            tf.square(pred_mask - pseudo_mask)
        )

        total_mask_loss = mask_loss + 0.3 * recon_consistency_loss

        total_loss += 0.4 * total_mask_loss  # Weight in total loss
        all_comps['mask_loss'] = total_mask_loss
        all_comps['mask_guidance_weight'] = alpha_guidance
    else:
        pred_mask = None
        total_mask_loss = 0.0

    # ════════════════════════════════════════════════════════════════════════
    # PHASE 3: Detection Loss (with mask & reconstruction attention)
    # ════════════════════════════════════════════════════════════════════════

    detection_loss_total = 0.0

    for scale in ['p2', 'p3', 'p4', 'p5']:
        if f"{scale}_cls" not in preds:
            continue

        # Detection loss WITH attention from mask and reconstruction
        scale_loss, scale_comps = unified_detection_loss(
            preds, targets, scale,
            pred_mask=pred_mask,
            recon_error_map=recon_error_map,
            num_classes=num_classes,
            epoch=epoch,
            total_epochs=total_epochs
        )

        detection_loss_total += scale_loss
        all_comps.update({f"{scale}_{k}": v for k, v in scale_comps.items()})

    total_loss += detection_loss_total

    # ════════════════════════════════════════════════════════════════════════
    # PHASE 4: Final Loss with Progressive Weighting
    # ════════════════════════════════════════════════════════════════════════

    # Early epochs: focus on learning from reconstruction
    # Late epochs: focus on perfect detection

    w_detection = 1.0
    w_mask = 0.3 + 0.2 * progress  # 0.3 → 0.5
    w_recon = 0.4 + 0.1 * progress  # 0.4 → 0.5

    final_loss = (
        w_detection * detection_loss_total +
        w_mask * total_mask_loss +
        w_recon * recon_loss
    )

    all_comps['total_loss'] = final_loss
    all_comps['w_detection'] = w_detection
    all_comps['w_mask'] = w_mask
    all_comps['w_recon'] = w_recon

    return final_loss, all_comps
