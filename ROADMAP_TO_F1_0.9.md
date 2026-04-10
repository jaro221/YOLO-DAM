# Roadmap to F1 Score 0.9+ 

## Current State
```
F1: 0.815
Precision: 76-78%
Recall: 84-86%
Gap to 0.9: -8.5% (significant but achievable)
```

## Target
```
F1: 0.90+
Precision: 88-90%
Recall: 90-92%
```

---

## 🎯 **Strategy Overview (Ranked by Impact)**

```
Tier 1: MAXIMUM IMPACT (+4-5% F1)
├─ 1. Hard Negative Mining
├─ 2. Focal Loss Tuning (per-class)
└─ 3. Progressive Training with Curriculum Learning

Tier 2: HIGH IMPACT (+2-3% F1)
├─ 4. Advanced Augmentation Pipeline
├─ 5. Ensemble of Configurations
└─ 6. IoU & NMS Optimization

Tier 3: MEDIUM IMPACT (+1-2% F1)
├─ 7. Longer Training (more epochs)
├─ 8. Knowledge Distillation
└─ 9. Test-Time Augmentation

Tier 4: FINE-TUNING (+0.5-1% F1)
├─ 10. Confidence Calibration
├─ 11. Class-specific thresholds
└─ 12. Post-processing optimization
```

---

# TIER 1: MAXIMUM IMPACT STRATEGIES

## 1️⃣ **Hard Negative Mining** (+3-4% F1)

**What it is**: Identify "hard" false positives (high confidence but wrong) and weight them heavily in training.

**Why it works**: 
- Reduces false positives (improves precision)
- Forces model to learn discriminative features
- Targets the weakest predictions

### Implementation

```python
# In YOLO_DAM_loss_4tasks.py, add:

def hard_negative_mining(pred_boxes, pred_cls, pred_obj, 
                         target_boxes, target_cls, target_obj,
                         iou_threshold=0.5, top_k=0.25):
    """
    Find hard negatives: high confidence but low IoU.
    Weight these heavily in loss.
    """
    batch_size = tf.shape(pred_obj)[0]
    height = tf.shape(pred_obj)[1]
    width = tf.shape(pred_obj)[2]
    
    # Get positive positions
    pos_mask = target_obj > 0.5  # [B, H, W, 1]
    
    # Get negative positions (target_obj == 0)
    neg_mask = target_obj <= 0.5
    
    # Get predicted objectness for negatives
    pred_obj_sigmoid = tf.sigmoid(pred_obj)
    
    # Hard negatives: high predicted score but target is 0
    hard_neg_score = pred_obj_sigmoid * neg_mask
    
    # Get top K% hardest negatives per batch
    num_hard = tf.cast(tf.cast(height * width, tf.float32) * top_k, tf.int32)
    
    # For each batch, find hardest negatives
    hard_neg_weight = tf.zeros_like(pred_obj)
    
    for b in range(batch_size):
        flat_scores = tf.reshape(hard_neg_score[b], [-1])
        _, top_indices = tf.nn.top_k(flat_scores, k=num_hard, sorted=False)
        
        # Mark hard negatives with higher weight
        hard_neg_weight_flat = tf.scatter_nd(
            tf.expand_dims(top_indices, 1),
            tf.ones(num_hard),
            [height * width]
        )
        hard_neg_weight = tf.reshape(hard_neg_weight_flat, [height, width, 1])
    
    # Apply hard negative weighting
    hard_neg_weight = tf.maximum(hard_neg_weight, 0.1)  # Min weight 0.1
    
    return hard_neg_weight  # Use to weight objectness loss
```

**In unified loss**:
```python
def unified_4task_loss(...):
    # ... existing code ...
    
    # Add hard negative mining
    hard_neg_weight = hard_negative_mining(
        pred_obj_sigmoid, pred_cls, pred_obj,
        target_obj, target_cls, target_obj,
        top_k=0.25  # Top 25% hardest negatives
    )
    
    # Apply to detection loss
    obj_bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=t_obj, logits=pred_obj)
    obj_loss = tf.reduce_mean(obj_bce * hard_neg_weight * combined_attention)
```

**Expected impact**: +3-4% F1 (reduces false positives significantly)

---

## 2️⃣ **Adaptive Focal Loss Tuning** (+2-3% F1)

**What it is**: Dynamically adjust focal loss alpha per class based on performance.

**Why it works**:
- Different classes have different difficulty
- Focal loss concentrates on hard examples
- Adapting alpha helps balance learning

### Implementation

```python
# Replace static ALPHA_PER_CLASS with adaptive version

def compute_adaptive_alpha(epoch, total_epochs, class_performance):
    """
    Dynamically adjust alpha based on class-level precision.
    
    Low precision class → higher alpha (focus more)
    High precision class → lower alpha (relax)
    """
    progress = epoch / total_epochs
    
    # Base alpha values
    base_alpha = [
        0.25, 0.25, 0.25, 0.25,  # Common classes
        0.50,                       # Rare class (crack)
        0.25, 0.25, 0.25, 0.25,
        0.75,                       # Hard class (foreign particle)
    ]
    
    # Adjust by class performance
    adaptive_alpha = []
    for class_id, base in enumerate(base_alpha):
        # Get precision for this class
        class_prec = class_performance.get(class_id, 0.5)
        
        # High precision → lower alpha (easier to learn)
        # Low precision → higher alpha (focus more)
        factor = 1.5 - class_prec  # Range [0.5, 1.5]
        
        # Alpha increases over time (harder focus late)
        adjusted = base * factor * (0.8 + 0.4 * progress)
        adaptive_alpha.append(adjusted)
    
    return tf.constant(adaptive_alpha, dtype=tf.float32)
```

**In training loop**:
```python
# Track per-class metrics
class_precision = {
    0: 0.82,  # Agglomerate: high precision
    4: 0.58,  # Crack: low precision → boost alpha
    9: 0.64,  # Foreign-particle: medium → keep high alpha
}

alpha = compute_adaptive_alpha(epoch, EPOCHS, class_precision)

# Use in focal loss
focal_loss = focal_loss_per_class(y_true, y_pred_logits, 
                                 alpha_per_class=alpha, 
                                 gamma=2.0)
```

**Expected impact**: +2-3% F1 (balances per-class learning)

---

## 3️⃣ **Curriculum Learning with Warmup** (+2-3% F1)

**What it is**: Start with easy samples, gradually increase difficulty.

**Why it works**:
- Model learns good representations first
- Avoids early overfitting to hard examples
- Better convergence

### Implementation

```python
def get_curriculum_weight(epoch, total_epochs, difficulty_score):
    """
    Curriculum learning: start easy, get hard.
    
    difficulty_score: 0=easy, 1=hard
    """
    progress = epoch / total_epochs
    
    # Difficulty threshold increases over time
    threshold = 0.2 + 0.6 * progress  # 0.2 → 0.8
    
    # Weight samples by difficulty
    # Easy samples (difficulty < threshold): weight = 1.0
    # Hard samples: weight ramps up over time
    curriculum_weight = tf.where(
        difficulty_score < threshold,
        1.0,  # Easy samples: full weight
        0.5 + 1.5 * (difficulty_score - threshold) / (1.0 - threshold)  # Hard: ramp up
    )
    
    return curriculum_weight

def compute_sample_difficulty(pred_obj, target_obj, pred_cls, target_cls):
    """
    Estimate difficulty: high prediction error = hard sample.
    """
    obj_error = tf.abs(tf.sigmoid(pred_obj) - target_obj)
    cls_error = tf.reduce_max(tf.abs(tf.sigmoid(pred_cls) - target_cls), axis=-1)
    
    difficulty = (obj_error + cls_error) / 2.0
    return tf.squeeze(difficulty)
```

**In unified loss**:
```python
def unified_4task_loss(...):
    # Compute sample difficulty
    difficulty = compute_sample_difficulty(pred_obj, t_obj, pred_cls, t_cls)
    
    # Get curriculum weight
    curriculum_weight = get_curriculum_weight(epoch, total_epochs, difficulty)
    
    # Apply to all losses
    obj_loss = tf.reduce_mean(obj_bce * curriculum_weight[..., None])
    cls_loss = tf.reduce_mean(cls_bce * curriculum_weight[..., None])
```

**Expected impact**: +2-3% F1 (better convergence to good local minimum)

---

# TIER 2: HIGH IMPACT STRATEGIES

## 4️⃣ **Advanced Augmentation Pipeline** (+1.5-2.5% F1)

**What it is**: Sophisticated data augmentation targeting defect detection challenges.

### Key Augmentations

```python
def advanced_augmentation_pipeline(image, targets, mask):
    """
    Advanced augmentation:
    1. Mixup for small objects
    2. CutMix for defect regions
    3. AutoAugment policy
    4. Mosaic augmentation
    """
    
    # 1. Mixup (blend with another image)
    if tf.random.uniform([]) < 0.2:
        image, targets = apply_mixup(image, targets)
    
    # 2. CutMix (swap defect regions with other images)
    if tf.random.uniform([]) < 0.3:
        image, mask = apply_cutmix_defects(image, mask)
    
    # 3. Mosaic (combine 4 images)
    if tf.random.uniform([]) < 0.15:
        image, targets, mask = apply_mosaic(image, targets, mask)
    
    # 4. AutoAugment (learned augmentation policy)
    image = autoaugment(image)
    
    # 5. Standard augmentations
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.image.random_saturation(image, 0.8, 1.2)
    image = tf.image.random_hue(image, 0.1)
    
    # 6. Geometric augmentations
    image, targets, mask = apply_geometric_transform(
        image, targets, mask,
        rotation=15,
        scale=(0.9, 1.1),
        shear=0.1
    )
    
    return image, targets, mask
```

**Expected impact**: +1.5-2.5% F1 (better robustness)

---

## 5️⃣ **Ensemble of Configurations** (+1.5-2% F1)

**What it is**: Train multiple models with different configurations, ensemble predictions.

```python
# Train 3 different model variants

Config A: width=1.0, depth=1.0, lr=5e-5
Config B: width=1.2, depth=1.1, lr=3e-5
Config C: width=0.9, depth=0.95, lr=7e-5

# Inference: average predictions from all 3
def ensemble_inference(image, model_a, model_b, model_c):
    pred_a = model_a(image)
    pred_b = model_b(image)
    pred_c = model_c(image)
    
    # Average boxes (weighted by confidence)
    conf_a = tf.sigmoid(pred_a['p3_obj'])
    conf_b = tf.sigmoid(pred_b['p3_obj'])
    conf_c = tf.sigmoid(pred_c['p3_obj'])
    
    total_conf = conf_a + conf_b + conf_c
    
    ensemble_pred = {
        'boxes': (pred_a['p3_reg'] * conf_a + 
                 pred_b['p3_reg'] * conf_b + 
                 pred_c['p3_reg'] * conf_c) / total_conf,
        'classes': (tf.argmax(pred_a['p3_cls'], -1) +
                   tf.argmax(pred_b['p3_cls'], -1) +
                   tf.argmax(pred_c['p3_cls'], -1)) // 3,
        'confidence': total_conf / 3.0
    }
    
    return ensemble_pred
```

**Expected impact**: +1.5-2% F1 (reduces individual model biases)

---

## 6️⃣ **IoU Threshold & NMS Optimization** (+0.5-1.5% F1)

**What it is**: Optimize post-processing hyperparameters for YOUR data.

```python
def optimize_inference_parameters():
    """
    Grid search for best IoU threshold and NMS parameters.
    """
    best_f1 = 0
    best_params = {}
    
    for iou_thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
        for nms_thresh in [0.4, 0.5, 0.6, 0.7]:
            for conf_thresh in [0.4, 0.5, 0.6, 0.7]:
                # Test on validation set
                f1 = evaluate_on_val_set(
                    iou_threshold=iou_thresh,
                    nms_threshold=nms_thresh,
                    conf_threshold=conf_thresh
                )
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_params = {
                        'iou': iou_thresh,
                        'nms': nms_thresh,
                        'conf': conf_thresh
                    }
    
    return best_params
```

**Expected impact**: +0.5-1.5% F1 (often underutilized!)

---

# TIER 3: MEDIUM IMPACT STRATEGIES

## 7️⃣ **Longer Training (400-500 epochs)** (+0.5-1% F1)

```python
# Instead of 300 epochs:
EPOCHS = 400  # or 500

# With learning rate annealing
def get_lr_schedule(epoch, total_epochs):
    # Cosine annealing with warm restarts
    base_lr = 5e-5
    
    cycle = epoch % 100  # Restart every 100 epochs
    progress = cycle / 100
    
    # Cosine annealing within cycle
    lr = base_lr * (1 + tf.cos(math.pi * progress)) / 2
    
    # Exponential decay overall
    overall_decay = (epoch / total_epochs) ** 0.5
    
    return lr * overall_decay
```

**Expected impact**: +0.5-1% F1 (more epochs = better convergence)

---

## 8️⃣ **Knowledge Distillation** (+1-1.5% F1)

```python
def knowledge_distillation_loss(student_logits, teacher_logits, 
                               targets, temperature=4.0):
    """
    Train student from teacher + ground truth.
    Smoother targets help generalization.
    """
    # Soft targets from teacher
    teacher_probs = tf.nn.softmax(teacher_logits / temperature)
    student_probs = tf.nn.softmax(student_logits / temperature)
    
    # KL divergence for soft targets
    kl_loss = tf.keras.losses.KLDivergence()(teacher_probs, student_probs)
    
    # Cross-entropy for hard targets (ground truth)
    hard_loss = tf.keras.losses.binary_crossentropy(targets, student_logits)
    
    # Weighted combination
    alpha = 0.7
    total = alpha * kl_loss + (1 - alpha) * hard_loss
    
    return total
```

**Expected impact**: +1-1.5% F1 (learns from larger teacher model)

---

## 9️⃣ **Test-Time Augmentation (TTA)** (+0.5-1% F1)

```python
def test_time_augmentation(image, model, num_augments=4):
    """
    Augment at test time, average predictions.
    """
    predictions = []
    
    # Original
    pred = model(image)
    predictions.append(pred)
    
    # Augmented versions
    for _ in range(num_augments):
        aug_image = random_augment(image)  # Light augmentation
        pred = model(aug_image)
        predictions.append(pred)
    
    # Average predictions
    avg_boxes = tf.reduce_mean([p['boxes'] for p in predictions], axis=0)
    avg_conf = tf.reduce_mean([p['confidence'] for p in predictions], axis=0)
    avg_cls = tf.reduce_mean([p['classes'] for p in predictions], axis=0)
    
    return {'boxes': avg_boxes, 'confidence': avg_conf, 'classes': avg_cls}
```

**Expected impact**: +0.5-1% F1 (better predictions through averaging)

---

# TIER 4: FINE-TUNING

## 🔟 **Confidence Calibration** (+0.3-0.5% F1)

```python
def calibrate_confidence(pred_conf, val_set_preds, val_set_targets):
    """
    Adjust confidence values to match true positives.
    """
    # Find confidence threshold for each class
    calibration_curve = []
    
    for class_id in range(10):
        class_preds = val_set_preds[val_set_preds['class'] == class_id]
        
        # Compute accuracy for different confidence levels
        best_scaling = 1.0
        best_calibration = float('inf')
        
        for scale in [0.8, 0.9, 1.0, 1.1, 1.2]:
            scaled_conf = class_preds['confidence'] * scale
            calibration_error = compute_brier_score(scaled_conf, class_preds['correct'])
            
            if calibration_error < best_calibration:
                best_calibration = calibration_error
                best_scaling = scale
        
        calibration_curve.append(best_scaling)
    
    return calibration_curve
```

---

## 1️⃣1️⃣ **Per-Class Thresholds** (+0.2-0.3% F1)

```python
def find_optimal_thresholds(val_predictions, val_targets):
    """
    Different classes may need different confidence thresholds.
    """
    optimal_thresholds = {}
    
    for class_id in range(10):
        class_preds = val_predictions[val_predictions['class'] == class_id]
        
        best_f1 = 0
        best_thresh = 0.5
        
        for thresh in np.arange(0.3, 0.8, 0.01):
            tp = sum((class_preds['confidence'] > thresh) & 
                    (class_preds['is_correct'] == 1))
            fp = sum((class_preds['confidence'] > thresh) & 
                    (class_preds['is_correct'] == 0))
            fn = sum((class_preds['confidence'] <= thresh) & 
                    (class_preds['is_correct'] == 1))
            
            prec = tp / (tp + fp + 1e-7)
            rec = tp / (tp + fn + 1e-7)
            f1 = 2 * prec * rec / (prec + rec + 1e-7)
            
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        
        optimal_thresholds[class_id] = best_thresh
    
    return optimal_thresholds
```

---

# 🚀 **COMBINED STRATEGY TO REACH 0.9 F1**

## Phase 1: Implementation (Weeks 1-2)
```
Week 1:
  ✓ Hard negative mining (+3-4% F1)
  ✓ Adaptive focal loss (+2-3% F1)
  ✓ Curriculum learning (+2-3% F1)

Week 2:
  ✓ Advanced augmentation (+1.5-2.5% F1)
  ✓ Train ensemble (3 configs) (+1.5-2% F1)
  ✓ Optimize thresholds (+0.5-1.5% F1)
```

## Phase 2: Training
```
Config A: width=1.0, depth=1.0, lr=5e-5 → 400 epochs
Config B: width=1.2, depth=1.1, lr=3e-5 → 400 epochs
Config C: width=0.9, depth=0.95, lr=7e-5 → 400 epochs

Total time: ~12 weeks (running in parallel on available GPUs)
```

## Phase 3: Validation & Tuning
```
✓ Evaluate ensemble on validation set
✓ Fine-tune IoU thresholds
✓ Apply test-time augmentation
✓ Calibrate confidence scores
```

---

## 📊 **Expected Cumulative Improvement**

```
Baseline (current): F1 = 0.815

+ Hard negative mining: 0.815 + 0.035 = 0.850
+ Adaptive focal loss: 0.850 + 0.025 = 0.875
+ Curriculum learning: 0.875 + 0.025 = 0.900
+ Ensemble: 0.900 + 0.015 = 0.915
+ TTA + Calibration: 0.915 + 0.010 = 0.925

TARGET ACHIEVED: F1 = 0.92+
```

---

## ✅ **Implementation Checklist**

**TIER 1 (Critical)**:
- [ ] Implement hard negative mining
- [ ] Adaptive focal loss per class
- [ ] Curriculum learning warmup

**TIER 2 (Important)**:
- [ ] Advanced augmentation pipeline
- [ ] Train 3 model variants
- [ ] Grid search IoU/NMS thresholds

**TIER 3 (Nice to have)**:
- [ ] Extend to 400+ epochs
- [ ] Knowledge distillation from larger model
- [ ] Test-time augmentation

**TIER 4 (Polish)**:
- [ ] Confidence calibration
- [ ] Per-class thresholds

---

## 📈 **Validation Strategy**

```python
def validate_f1_improvement():
    """
    Measure improvement at each stage.
    """
    stages = [
        ("Baseline", 0.815),
        ("+ Hard negatives", 0.850),
        ("+ Adaptive focal", 0.875),
        ("+ Curriculum", 0.900),
        ("+ Ensemble", 0.915),
        ("+ TTA + Calibration", 0.925),
    ]
    
    for stage, expected_f1 in stages:
        actual_f1 = evaluate_model()
        improvement = actual_f1 - expected_f1
        print(f"{stage}: {actual_f1:.4f} (expected {expected_f1:.4f}, delta {improvement:+.4f})")
```

---

## 🎯 **Key Success Factors**

1. **Hard Negative Mining** - Eliminates false positives
2. **Adaptive Focal Loss** - Balances per-class learning
3. **Curriculum Learning** - Better convergence
4. **Ensemble** - Combines different strategies
5. **Post-processing** - Often overlooked but impactful
6. **Validation Discipline** - Measure every change

---

**Target**: F1 = 0.92-0.95 (achievable with disciplined execution)
**Timeline**: 3-4 months with parallel training
**Key Metric**: Measure F1 after each tier implementation
