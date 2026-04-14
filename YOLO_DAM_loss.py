

import tensorflow as tf
import math

def ciou_loss(pred_boxes, target_boxes, eps=1e-7):
    """
    Complete IoU loss for better box regression
    pred_boxes, target_boxes: [..., 4] in format [cx, cy, w, h] normalized [0,1]
    Returns: CIoU loss values
    """
    # Convert to corners
    pred_x1 = pred_boxes[..., 0] - pred_boxes[..., 2] / 2
    pred_y1 = pred_boxes[..., 1] - pred_boxes[..., 3] / 2
    pred_x2 = pred_boxes[..., 0] + pred_boxes[..., 2] / 2
    pred_y2 = pred_boxes[..., 1] + pred_boxes[..., 3] / 2
    
    target_x1 = target_boxes[..., 0] - target_boxes[..., 2] / 2
    target_y1 = target_boxes[..., 1] - target_boxes[..., 3] / 2
    target_x2 = target_boxes[..., 0] + target_boxes[..., 2] / 2
    target_y2 = target_boxes[..., 1] + target_boxes[..., 3] / 2
    
    # Intersection
    inter_x1 = tf.maximum(pred_x1, target_x1)
    inter_y1 = tf.maximum(pred_y1, target_y1)
    inter_x2 = tf.minimum(pred_x2, target_x2)
    inter_y2 = tf.minimum(pred_y2, target_y2)
    
    inter_area = tf.maximum(inter_x2 - inter_x1, 0.0) * tf.maximum(inter_y2 - inter_y1, 0.0)
    
    # Union
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
    union_area = pred_area + target_area - inter_area
    
    # IoU
    iou = inter_area / (union_area + eps)
    
    # Enclosing box
    enclose_x1 = tf.minimum(pred_x1, target_x1)
    enclose_y1 = tf.minimum(pred_y1, target_y1)
    enclose_x2 = tf.maximum(pred_x2, target_x2)
    enclose_y2 = tf.maximum(pred_y2, target_y2)
    
    enclose_w = enclose_x2 - enclose_x1
    enclose_h = enclose_y2 - enclose_y1
    enclose_c2 = enclose_w ** 2 + enclose_h ** 2 + eps
    
    # Center distance
    pred_cx = (pred_x1 + pred_x2) / 2
    pred_cy = (pred_y1 + pred_y2) / 2
    target_cx = (target_x1 + target_x2) / 2
    target_cy = (target_y1 + target_y2) / 2
    
    center_dist2 = (pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2
    
    # Aspect ratio consistency
    pred_w = pred_boxes[..., 2]
    pred_h = pred_boxes[..., 3]
    target_w = target_boxes[..., 2]
    target_h = target_boxes[..., 3]
    
    v = (4 / (math.pi ** 2)) * tf.pow(
        tf.atan(target_w / (target_h + eps)) - tf.atan(pred_w / (pred_h + eps)), 2
    )
    
    alpha = v / (1 - iou + v + eps)
    alpha = tf.stop_gradient(alpha)  # Don't backprop through alpha
    
    # CIoU
    ciou = iou - (center_dist2 / enclose_c2 + alpha * v)
    loss = 1 - ciou
    
    return loss

def detection_loss(preds, targets, num_classes=10, 
                        class_weights=None, label_smoothing=0.0):
    """
    FIXED loss function with proper weighting and focal loss
    """

    total_loss = 0.0
    comps = {}
    eps = 1e-7
    auto_mask_weight = 0.5 
    auto_recon_weight = 0.5

    for scale in ['p3', 'p4', 'p5']:
        pred_cls = tf.cast(preds[f"{scale}_cls"], tf.float32)
        pred_reg = tf.cast(preds[f"{scale}_reg"], tf.float32)
        pred_obj = tf.cast(preds[f"{scale}_obj"], tf.float32)

        t_cls = tf.cast(targets[f"{scale}_cls_t"], tf.float32)
        t_reg = tf.cast(targets[f"{scale}_reg_t"], tf.float32)
        t_obj = tf.cast(targets[f"{scale}_obj_t"], tf.float32)
        
        if label_smoothing > 0:
            t_cls = t_cls * (1 - label_smoothing) + label_smoothing / num_classes

        pos_mask = tf.cast(t_obj > 0.5, tf.float32)
        pos_count = tf.reduce_sum(pos_mask) + eps

        # ---- 1. Regression (CIoU) ----
        pred_reg_sigmoid = tf.sigmoid(pred_reg)
        pred_boxes_flat = tf.reshape(pred_reg_sigmoid, [-1, 4])
        target_boxes_flat = tf.reshape(t_reg, [-1, 4])
        pos_mask_flat = tf.reshape(pos_mask, [-1])
        pos_indices = tf.where(pos_mask_flat > 0.5)[:, 0]
        
        if tf.size(pos_indices) > 0:
            pred_boxes_pos = tf.gather(pred_boxes_flat, pos_indices)
            target_boxes_pos = tf.gather(target_boxes_flat, pos_indices)
            ciou_losses = ciou_loss(pred_boxes_pos, target_boxes_pos)
            reg_loss = tf.reduce_mean(ciou_losses)
        else:
            reg_loss = 0.0

        # ---- 2. Objectness ----
        obj_bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=t_obj, logits=pred_obj)
        pos_weight = 2.0
        weights = 1.0 + (pos_weight - 1.0) * t_obj
        obj_loss = tf.reduce_sum(obj_bce * weights) / (tf.reduce_sum(weights) + eps)

        # ---- 3. Classification (Standard BCE - no focal loss needed!) ----
        cls_bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=t_cls, logits=pred_cls)
        cls_loss = tf.reduce_sum(cls_bce * pos_mask) / pos_count


        scale_loss = 2.5 * reg_loss + 1.0 * obj_loss + 3.0 * cls_loss

        
        total_loss += scale_loss
        comps[f"{scale}_box"] = reg_loss
        comps[f"{scale}_obj"] = obj_loss
        comps[f"{scale}_cls"] = cls_loss

    # ===== AUTOENCODER LOSSES =====
    if 'auto_reconstruction' in preds:
        pred_recon = tf.cast(preds['auto_reconstruction'], tf.float32)
        target_img = tf.cast(targets['AUTO'], tf.float32)
        
        
        if  'mask' in targets:
            pred_mask = tf.cast(preds['auto_masked_recon'], tf.float32)
            gt_defect_mask = tf.cast(targets['mask'], tf.float32)  # 1=good, 0=defect
            gt_attention = gt_defect_mask  # 1=defect, 0=good
            
            # 2. Mask loss
            mask_loss = tf.reduce_mean(tf.square(pred_mask - gt_attention))
            total_loss += mask_loss 

            
            per_pixel_error = tf.square(pred_recon - target_img)
            total_loss += auto_recon_weight*tf.reduce_mean(per_pixel_error)
    
    return total_loss, comps
