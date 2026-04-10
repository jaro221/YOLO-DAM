# Quick Implementation Guide: F1 0.9+ (4-Week Sprint)

## 🏃 **Fast Track to 0.9 F1**

### **Current**: 0.815 F1
### **Goal**: 0.90+ F1 in 4 weeks
### **Method**: Implement top 3 strategies + ensemble

---

## 📋 **Week-by-Week Plan**

### **Week 1: Core Improvements**

#### Task 1: Hard Negative Mining (1 day)
Add to `YOLO_DAM_loss_4tasks.py`:

```python
# Add this function
def hard_negative_mining_weight(pred_obj_sigmoid, target_obj, 
                                height, width, batch_size, top_k_ratio=0.25):
    """Weight hard negatives (high pred, low target) higher."""
    
    # Identify negatives
    neg_mask = tf.cast(target_obj <= 0.5, tf.float32)
    
    # Score for hard negatives: high prediction on negative sample
    hard_neg_score = pred_obj_sigmoid * neg_mask
    
    # Get top K% hardest
    k = tf.cast(tf.cast(height * width, tf.float32) * top_k_ratio, tf.int32)
    
    weights = tf.ones_like(hard_neg_score)
    
    # Find top-k indices per batch
    batch_hard_negs = []
    for b in tf.range(batch_size):
        scores_flat = tf.reshape(hard_neg_score[b, :, :, :], [-1])
        _, top_indices = tf.nn.top_k(scores_flat, k=k, sorted=False)
        
        hard_weight_flat = tf.scatter_nd(
            indices=tf.expand_dims(top_indices, 1),
            updates=tf.ones(k) * 2.0,  # Weight hard negatives 2x
            shape=[height * width]
        )
        batch_hard_negs.append(tf.reshape(hard_weight_flat, [height, width, 1]))
    
    weights = tf.stack(batch_hard_negs)
    return tf.maximum(weights, 0.5)  # Min weight 0.5


# In unified_4task_loss function, modify detection loss:
def unified_4task_loss(...):
    # ... existing code ...
    
    for scale in ['p2', 'p3', 'p4', 'p5']:
        # ... existing code ...
        
        # ADD THIS:
        batch_size = tf.shape(pred_obj)[0]
        height = tf.shape(pred_obj)[1]
        width = tf.shape(pred_obj)[2]
        
        hard_neg_weight = hard_negative_mining_weight(
            tf.sigmoid(pred_obj), t_obj,
            height, width, batch_size, top_k_ratio=0.2
        )
        
        # Multiply existing weights
        weights = weights * hard_neg_weight
        obj_loss = tf.reduce_sum(obj_bce * weights) / (tf.reduce_sum(weights) + eps)
        
        # ... rest of code ...
```

**Expected gain**: +3-4% F1

---

#### Task 2: Adaptive Focal Loss (1 day)
Add to `YOLO_DAM_loss_4tasks.py`:

```python
# Track class-level metrics during training
class_metrics = {i: {'precision': 0.5} for i in range(10)}

def compute_adaptive_alpha(epoch, total_epochs, class_metrics):
    """Adjust alpha based on class precision."""
    
    base_alpha = tf.constant([
        0.25, 0.25, 0.25, 0.25,  # Common classes
        0.50,                      # Rare (Crack)
        0.25, 0.25, 0.25, 0.25,
        0.75,                      # Hard (Foreign particle)
    ], dtype=tf.float32)
    
    progress = tf.cast(epoch, tf.float32) / tf.cast(total_epochs, tf.float32)
    
    adaptive = []
    for class_id, base in enumerate(base_alpha.numpy()):
        prec = class_metrics.get(class_id, {}).get('precision', 0.5)
        
        # Low precision → higher alpha (focus more)
        factor = 1.5 - min(max(prec, 0.3), 0.8)  # Range [0.7, 1.2]
        
        # Increase over epochs
        adjusted = base * factor * (0.8 + 0.4 * progress)
        adaptive.append(adjusted)
    
    return tf.constant(adaptive, dtype=tf.float32)


# In training loop:
# Track metrics every 50 batches and update class_metrics
# Use in focal_loss_per_class call:
alpha = compute_adaptive_alpha(epoch, EPOCHS, class_metrics)
cls_bce = focal_loss_per_class(t_cls, pred_cls, alpha_per_class=alpha)
```

**Expected gain**: +2-3% F1

---

#### Task 3: Curriculum Learning (1 day)
Add to `YOLO_DAM_loss_4tasks.py`:

```python
def curriculum_learning_weight(pred_obj, target_obj, pred_cls, target_cls,
                               epoch, total_epochs):
    """Start with easy samples, gradually increase difficulty."""
    
    progress = tf.cast(epoch, tf.float32) / tf.cast(total_epochs, tf.float32)
    
    # Difficulty threshold increases over time
    difficulty_threshold = 0.2 + 0.6 * progress  # 0.2 → 0.8
    
    # Compute per-sample difficulty
    obj_error = tf.abs(tf.sigmoid(pred_obj) - target_obj)
    cls_error = tf.reduce_max(
        tf.abs(tf.sigmoid(pred_cls) - target_cls), axis=-1, keepdims=True
    )
    difficulty = (obj_error + cls_error) / 2.0
    
    # Easy samples: weight=1.0, Hard samples: ramp up
    curriculum_weight = tf.where(
        difficulty < difficulty_threshold,
        1.0,
        0.3 + 0.7 * (difficulty - difficulty_threshold) / (1.0 - difficulty_threshold + 1e-7)
    )
    
    return curriculum_weight


# In unified_4task_loss:
def unified_4task_loss(...):
    # ... existing code ...
    
    # Get curriculum weight for this epoch
    curr_weight = curriculum_learning_weight(
        pred_obj, t_obj, pred_cls, t_cls,
        epoch, total_epochs
    )
    
    # Apply to all losses
    obj_loss = tf.reduce_mean(obj_bce * curr_weight)
    cls_loss = tf.reduce_mean(cls_bce * curr_weight[..., None])
```

**Expected gain**: +2-3% F1

**Week 1 Total**: 0.815 → 0.878 F1 (+6-7%)

---

### **Week 2: Augmentation & Ensemble Prep**

#### Task 4: Advanced Augmentation (2 days)
Add augmentation function to data pipeline:

```python
def advanced_augmentation(image, targets, mask):
    """Enhanced augmentation for defect detection."""
    
    # 1. Random brightness/contrast/saturation
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.image.random_saturation(image, 0.8, 1.2)
    image = tf.image.random_hue(image, 0.1)
    
    # 2. Geometric transforms
    if tf.random.uniform([]) < 0.5:
        # Random rotation (-15 to +15 degrees)
        angle = tf.random.uniform([], -15, 15) * 3.14159 / 180
        image = tfa.image.rotate(image, angle)
    
    # 3. Random scale
    scale = tf.random.uniform([], 0.9, 1.1)
    h, w = 640, 640
    new_size = tf.cast(tf.cast([h, w], tf.float32) * scale, tf.int32)
    image = tf.image.resize(image, new_size)
    image = tf.image.resize_with_pad(image, h, w)
    
    # 4. Mixup (blend with another image)
    if tf.random.uniform([]) < 0.2:
        other_image = tf.random.shuffle(image)[0:1]
        alpha = tf.random.uniform([], 0.5, 1.0)
        image = alpha * image + (1 - alpha) * other_image
    
    # 5. Clip and ensure valid range
    image = tf.clip_by_value(image, 0, 1)
    
    return image, targets, mask


# In YOLO_DAM_train.py data loading, use this augmentation
```

**Expected gain**: +1.5-2% F1

---

#### Task 5: Prepare 3 Model Configs (1 day)
Create training scripts for ensemble:

```python
# train_ensemble_a.py
LEARNING_RATE = 5e-5
WIDTH = 1.0
DEPTH = 1.0
EPOCHS = 400
MODEL_NAME = "ensemble_config_a"

# train_ensemble_b.py
LEARNING_RATE = 3e-5
WIDTH = 1.2
DEPTH = 1.1
EPOCHS = 400
MODEL_NAME = "ensemble_config_b"

# train_ensemble_c.py
LEARNING_RATE = 7e-5
WIDTH = 0.9
DEPTH = 0.95
EPOCHS = 400
MODEL_NAME = "ensemble_config_c"

# Run all three in parallel if GPUs available
# Or sequentially on single GPU
```

**Week 2 Total**: 0.878 → 0.910 F1 (+3.2%)

---

### **Week 3: Training All Configs**

Run parallel training:
```bash
# Terminal 1
python train_ensemble_a.py

# Terminal 2 (if GPU available)
python train_ensemble_b.py

# Terminal 3 (if GPU available)
python train_ensemble_c.py
```

**Expected**: Each model ~0.90-0.92 F1

---

### **Week 4: Ensemble + Post-Processing**

#### Task 6: Ensemble Inference (1 day)

```python
def ensemble_inference(image, model_a, model_b, model_c):
    """Average predictions from 3 models."""
    
    preds_a = model_a(image, training=False)
    preds_b = model_b(image, training=False)
    preds_c = model_c(image, training=False)
    
    # Average at each scale
    ensemble_preds = {}
    
    for scale in ['p2', 'p3', 'p4', 'p5']:
        # Get predictions
        obj_a = tf.sigmoid(preds_a[f'{scale}_obj'])
        obj_b = tf.sigmoid(preds_b[f'{scale}_obj'])
        obj_c = tf.sigmoid(preds_c[f'{scale}_obj'])
        
        # Average with confidence weighting
        avg_conf = (obj_a + obj_b + obj_c) / 3.0
        
        cls_a = tf.nn.softmax(preds_a[f'{scale}_cls'])
        cls_b = tf.nn.softmax(preds_b[f'{scale}_cls'])
        cls_c = tf.nn.softmax(preds_c[f'{scale}_cls'])
        
        avg_cls = (cls_a + cls_b + cls_c) / 3.0
        
        reg_a = preds_a[f'{scale}_reg']
        reg_b = preds_b[f'{scale}_reg']
        reg_c = preds_c[f'{scale}_reg']
        
        avg_reg = (reg_a + reg_b + reg_c) / 3.0
        
        # Store ensemble predictions
        ensemble_preds[f'{scale}_obj'] = avg_conf
        ensemble_preds[f'{scale}_cls'] = avg_cls
        ensemble_preds[f'{scale}_reg'] = avg_reg
    
    return ensemble_preds
```

**Expected gain**: +1-1.5% F1

---

#### Task 7: Threshold Optimization (1 day)

```python
def find_best_thresholds(val_predictions, val_targets):
    """Grid search for optimal confidence threshold per class."""
    
    best_f1_overall = 0
    best_config = {}
    
    for conf_thresh in np.arange(0.4, 0.75, 0.05):
        for iou_thresh in [0.5, 0.55, 0.6]:
            
            tp = sum([1 for pred in val_predictions 
                     if pred['conf'] > conf_thresh and 
                     compute_iou(pred['box'], pred['gt_box']) > iou_thresh])
            fp = sum([1 for pred in val_predictions 
                     if pred['conf'] > conf_thresh and 
                     compute_iou(pred['box'], pred['gt_box']) <= iou_thresh])
            fn = sum([1 for gt in val_targets 
                     if not any(compute_iou(gt['box'], pred['box']) > iou_thresh 
                               for pred in val_predictions)])
            
            prec = tp / (tp + fp + 1e-7)
            rec = tp / (tp + fn + 1e-7)
            f1 = 2 * prec * rec / (prec + rec + 1e-7)
            
            if f1 > best_f1_overall:
                best_f1_overall = f1
                best_config = {
                    'conf_threshold': conf_thresh,
                    'iou_threshold': iou_thresh,
                    'f1': f1
                }
    
    return best_config
```

**Expected gain**: +0.5-1% F1

---

## 📊 **Expected Final Results**

```
Week 1: 0.815 → 0.878 F1 (+6-7%)
Week 2: 0.878 → 0.910 F1 (+3.2%)
Week 3: Training (individual models ~0.90-0.92)
Week 4: Ensemble + Optimization → 0.92-0.95 F1

FINAL: F1 = 0.92+ ✓
```

---

## ✅ **Implementation Checklist**

**Week 1**:
- [ ] Add hard_negative_mining_weight function
- [ ] Add compute_adaptive_alpha function
- [ ] Add curriculum_learning_weight function
- [ ] Test integrated loss on 10 epochs
- [ ] Measure improvement

**Week 2**:
- [ ] Implement advanced_augmentation
- [ ] Create 3 training configs
- [ ] Start parallel training (or sequential)

**Week 3**:
- [ ] Monitor training (loss curves)
- [ ] Ensure all 3 models converging

**Week 4**:
- [ ] Implement ensemble_inference
- [ ] Run threshold optimization
- [ ] Evaluate final F1 score
- [ ] Document final configuration

---

## 🎯 **Key Success Metrics**

Track these numbers:
```
Epoch 50:   F1 should be ~0.85+
Epoch 100:  F1 should be ~0.88+
Epoch 200:  F1 should be ~0.90+
Epoch 400:  F1 should be ~0.92+
Ensemble:   F1 should be ~0.93+
```

If numbers lower, debug immediately (check loss curves, validation metrics).

---

**Target**: F1 = 0.92-0.95 in 4 weeks
**Key**: Measure after each change, iterate quickly
