# YOLO-DAM Performance Improvement Plan
**Date: 2026-04-15**  
**Current State: mAP=0.5036, mPA=0.6389, F1=0.6388**  
**Target: mAP ≥ 0.80, F1 ≥ 0.85**

---

## 🔴 Critical Issues

### Issue 1: Class_4 (Crack-long) - 9.5% Recall
**Symptoms:**
- Only 30 TP out of 342 GT (missing 285 objects!)
- FP: 31 (acceptable)
- Precision: 49.2% (moderate)

**Root Causes:**
1. Severe class imbalance (Class_4 is rare)
2. Insufficient loss weight during training
3. Possible size/appearance confusion with Class_5

**Solutions:**

#### Solution 1A: Increase Loss Weight (Immediate)
```python
# File: YOLO_DAM_loss_v2.py, line 37-48
CLASS_WEIGHTS = tf.constant([
    1.0,   # 0 Agglomerate
    1.0,   # 1 Pinhole-long
    1.0,   # 2 Pinhole-trans
    1.0,   # 3 Pinhole-round
    6.0,   # 4 Crack-long ← INCREASE from 2.0 to 6.0 (3x boost)
    1.5,   # 5 Crack-trans ← INCREASE from 1.0 to 1.5
    1.0,   # 6 Line-long
    1.0,   # 7 Line-trans
    1.0,   # 8 Line-diag
    0.8,   # 9 Foreign-particle ← DECREASE from 2.0 to 0.8
], dtype=tf.float32)
```

**Expected Impact:** +10-15% recall on Class_4

#### Solution 1B: Use Focal Loss Alpha Adjustment
```python
# File: YOLO_DAM_loss_v2.py, line 24-35
ALPHA_PER_CLASS = [
    0.25,  # 0 Agglomerate
    0.25,  # 1 Pinhole-long
    0.25,  # 2 Pinhole-trans
    0.25,  # 3 Pinhole-round
    0.50,  # 4 Crack-long ← INCREASE from 0.25 to 0.50 (harder penalty)
    0.30,  # 5 Crack-trans ← INCREASE from 0.25 to 0.30
    0.25,  # 6 Line-long
    0.25,  # 7 Line-trans
    0.25,  # 8 Line-diag
    0.60,  # 9 Foreign-particle ← INCREASE to 0.60 (penalize FP more)
]
```

**Expected Impact:** +5-8% recall on rare classes

#### Solution 1C: Hard Negative Mining
```python
# Add to YOLO_DAM_train_v2.py in training loop
# Keep predictions that are hardest to classify

def hard_negative_mining(detections, gt_labels, ratio=3.0):
    """Keep hard negatives (misclassified) proportional to positives"""
    # For Class_4: ratio=3.0 means keep 3x false positives per true positive
    # Helps model learn what NOT to detect
    pass
```

---

### Issue 2: Class_9 (Foreign-particle) - 861 False Positives!
**Symptoms:**
- TP: 201, FP: 862 (4.3:1 FP:TP ratio!)
- Precision: 18.9% (terrible)
- Recall: 34.3% (low)
- 1 correct detection, then 4-5 false ones nearby

**Root Causes:**
1. Confidence threshold too low (0.25)
2. Model confuses background noise with particles
3. NMS threshold too high (0.4) - not suppressing adjacent boxes
4. Class weight too high (2.0) - pushing model to over-detect

**Solutions:**

#### Solution 2A: Increase Confidence Threshold (Immediate - No Retraining)
```python
# File: test_relative_positioning.py, line 28-29
# Current:
CONF_THRESH = 0.25

# Change to:
CONF_THRESH = 0.35  # Moderate increase

# Better: Use class-specific thresholds
CLASS_CONF_THRESH = {
    0: 0.25,
    1: 0.25,
    2: 0.25,
    3: 0.25,
    4: 0.15,  # LOWER - rare class, encourage detection
    5: 0.25,
    6: 0.25,
    7: 0.25,
    8: 0.25,
    9: 0.55,  # MUCH HIGHER - filter noise for Class_9
}

# Update decode_predictions_relative() to use:
if confidence < CLASS_CONF_THRESH.get(class_id, CONF_THRESH):
    continue
```

**Expected Impact:** -300 to -500 FP on Class_9 (immediate, no retraining)

#### Solution 2B: Tighten NMS Threshold
```python
# File: test_relative_positioning.py, line 30
# Current:
IOU_THRESH = 0.4

# Change to:
IOU_THRESH = 0.3  # Stricter - suppress more overlapping boxes

# Or use class-specific NMS:
CLASS_NMS_THRESH = {
    9: 0.25,  # Very strict for Foreign-particle
    # Others: 0.4 (default)
}
```

**Expected Impact:** -200 FP by removing redundant detections

#### Solution 2C: Decrease Class Weight
```python
# File: YOLO_DAM_loss_v2.py
CLASS_WEIGHTS = tf.constant([
    ...
    0.5,   # 9 Foreign-particle ← DECREASE from 2.0 to 0.5
], dtype=tf.float32)
```

**Expected Impact:** Model will be less aggressive on Class_9

#### Solution 2D: Add Box Size Filtering
```python
# File: test_relative_positioning.py, in decode_predictions_relative()
# Add after decoding:

# Filter by expected size range for each class
CLASS_SIZE_RANGE = {
    0: (0.01, 0.15),  # Agglomerate: 6-96 px in 640 img
    9: (0.003, 0.012),  # Foreign-particle: 2-8 px (very small!)
    # ... others
}

if class_id in CLASS_SIZE_RANGE:
    min_size, max_size = CLASS_SIZE_RANGE[class_id]
    size = (w_n + h_n) / 2
    if not (min_size <= size <= max_size):
        continue  # Skip out-of-range predictions
```

**Expected Impact:** -100 to -200 FP on Class_9

---

### Issue 3: Class_5 (Crack-trans) - 47.4% Precision
**Symptoms:**
- TP: 321, FP: 356 (1:1 ratio)
- Recall: 42.6% (low)
- Precision: 47.4% (too many false positives)

**Root Causes:**
1. Confused with Class_4 (both are cracks)
2. Low confidence prediction scores
3. Insufficient training data

**Solutions:**

#### Solution 3A: Increase Loss Weight
```python
# File: YOLO_DAM_loss_v2.py
CLASS_WEIGHTS = tf.constant([
    ...
    2.0,   # 5 Crack-trans ← INCREASE from 1.0 to 2.0
    ...
], dtype=tf.float32)
```

#### Solution 3B: Data Augmentation for Cracks
```python
# File: YOLO_DAM_loss_v2.py, in augmentation section
# Add rotation augmentation (cracks appear at different angles)

def augment_rotate(img, boxes, angles=[-15, -10, -5, 5, 10, 15]):
    """Add random rotation to help model learn crack orientations"""
    angle = random.choice(angles)
    # Rotate image and boxes
    # This helps distinguish Class_4 vs Class_5 by orientation
    pass
```

#### Solution 3C: Use Focal Loss with Higher Gamma
```python
# File: YOLO_DAM_loss_v2.py, line 115-123
def focal_loss_per_class(y_true, y_pred_logits,
                         alpha_per_class=ALPHA_PER_CLASS, gamma=2.5):  # ← increase from 2.0
    # Higher gamma = penalize easier examples more, focus on hard ones
    pass
```

---

## 🟢 Quick Fixes (Apply Immediately, No Retraining)

### Fix 1: Class-Specific Confidence Thresholds
**File:** `test_relative_positioning.py`

```python
# Add after line 90 in decode_predictions_relative():

CLASS_CONF_THRESH = {
    0: 0.25,
    1: 0.25,
    2: 0.25,
    3: 0.25,
    4: 0.10,  # Lower for rare class
    5: 0.25,
    6: 0.25,
    7: 0.25,
    8: 0.25,
    9: 0.60,  # Much higher for noise-prone class
}

# Modify condition at line 148:
if confidence < CLASS_CONF_THRESH.get(class_id, CONF_THRESH):
    continue
```

**Expected Improvement:** 
- Class_4 recall: +5%
- Class_9 precision: +30% (FP -400)
- Overall F1: +0.05-0.08

### Fix 2: Tighter NMS for Class_9
```python
# Modify non_max_suppression() to use class-specific NMS:

def non_max_suppression(self, detections):
    """Apply NMS with class-specific thresholds"""
    if len(detections) == 0:
        return np.array([])

    detections = detections[np.argsort(detections[:, 4])[::-1]]
    keep = []
    
    while len(detections) > 0:
        keep.append(detections[0])
        if len(detections) == 1:
            break
        
        # Use class-specific NMS threshold
        class_id = int(detections[0, 5])
        nms_thresh = 0.25 if class_id == 9 else self.iou_threshold
        
        ious = self.calculate_iou_batch(detections[0:1], detections[1:])
        detections = detections[1:][ious[0] < nms_thresh]

    return np.array(keep) if keep else np.array([])
```

**Expected Improvement:**
- Class_9 precision: +20% (FP -200)

### Fix 3: Size Range Filtering
```python
# Add to decode_predictions_relative(), after line 176:

# Expected size ranges for each class (normalized to [0,1])
CLASS_SIZE_RANGE = {
    0: (0.01, 0.20),    # Agglomerate
    1: (0.01, 0.15),    # Pinhole-long
    2: (0.01, 0.15),    # Pinhole-trans
    3: (0.008, 0.12),   # Pinhole-round
    4: (0.015, 0.08),   # Crack-long
    5: (0.01, 0.08),    # Crack-trans
    6: (0.01, 0.20),    # Line-long
    7: (0.01, 0.20),    # Line-trans
    8: (0.01, 0.20),    # Line-diag
    9: (0.003, 0.015),  # Foreign-particle (very small)
}

# Filter out-of-range predictions
if class_id in CLASS_SIZE_RANGE:
    min_size, max_size = CLASS_SIZE_RANGE[class_id]
    avg_size = (w_n + h_n) / 2
    if not (min_size <= avg_size <= max_size):
        continue
```

**Expected Improvement:**
- Class_9 precision: +15%
- Overall FP: -100 to -200

---

## 🔵 Medium-Term Improvements (Requires Retraining)

### Improvement 1: Adjust Loss Weights
**Effort:** 5 minutes  
**Retraining:** Yes (full 400 epochs ~ 3-4 weeks)

```python
# File: YOLO_DAM_loss_v2.py

CLASS_WEIGHTS = tf.constant([
    1.0,   # 0 Agglomerate
    1.0,   # 1 Pinhole-long
    1.0,   # 2 Pinhole-trans
    1.0,   # 3 Pinhole-round
    5.0,   # 4 Crack-long (was 2.0) ← 2.5x boost
    1.5,   # 5 Crack-trans (was 1.0) ← 1.5x boost
    1.0,   # 6 Line-long
    1.0,   # 7 Line-trans
    1.0,   # 8 Line-diag
    0.6,   # 9 Foreign-particle (was 2.0) ← reduce by 3x
], dtype=tf.float32)

ALPHA_PER_CLASS = [
    0.25,  # 0
    0.25,  # 1
    0.25,  # 2
    0.25,  # 3
    0.50,  # 4 (was 0.25) ← harder penalty for rare class
    0.30,  # 5 (was 0.25)
    0.25,  # 6
    0.25,  # 7
    0.25,  # 8
    0.60,  # 9 (was 0.75) ← penalize FP heavily
]
```

**Expected Results After Retraining:**
- Class_4 recall: 9.5% → 25-30%
- Class_5 precision: 47.4% → 55-60%
- Class_9 FP: 861 → 400-500
- Overall mAP: 0.50 → 0.60-0.65

### Improvement 2: Enhanced Data Augmentation
```python
# File: YOLO_DAM_loss_v2.py, add rotation augmentation

def augment_rotate(img, boxes, angle_range=(-20, 20)):
    """Rotate image and boxes for crack orientation robustness"""
    angle = np.random.uniform(*angle_range)
    h, w = img.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Rotate image
    img_rot = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    # Transform box centers
    for i, box in enumerate(boxes):
        cx, cy = box[0] * w, box[1] * h
        pt = np.array([[[cx, cy]]], dtype=np.float32)
        pt_rot = cv2.transform(pt, M)[0][0]
        boxes[i][0] = pt_rot[0] / w
        boxes[i][1] = pt_rot[1] / h
    
    return img_rot, boxes
```

**Apply in training:**
```python
# In YOLO_DAM_train_v2.py training loop
if augment and len(boxes) > 0:
    img = augment_hsv(img)
    img, boxes = augment_rotate(img, boxes)  # Add rotation
    boxes_tf = tf.constant(boxes, dtype=tf.float32)
    img, boxes_tf = augment_flip(img, boxes_tf)
    boxes = boxes_tf.numpy()
```

**Expected Impact:** Better crack orientation detection

### Improvement 3: Focal Loss Gamma Tuning
```python
# File: YOLO_DAM_loss_v2.py
def focal_loss_per_class(y_true, y_pred_logits,
                         alpha_per_class=ALPHA_PER_CLASS, gamma=2.5):  # ← increase
    # gamma=2.5 focuses more on hard examples
    # gamma=2.0 (original) is standard
    pass
```

**Expected Impact:** +2-3% on hard classes

---

## 🟣 Long-Term Improvements (Major Impact)

### Strategy 1: Two-Stage Training
**Stage 1 (Epochs 1-200):** Train on all classes equally
**Stage 2 (Epochs 200-400):** Focus on problem classes (4, 5, 9) with boosted weights

```python
# In YOLO_DAM_train_v2.py

if epoch < 200:
    # Stage 1: Normal training
    weights = CLASS_WEIGHTS
else:
    # Stage 2: Boost problem classes
    weights = CLASS_WEIGHTS * tf.constant([
        1.0, 1.0, 1.0, 1.0,  # 0-3: normal
        3.0,  # 4: Crack-long (3x boost)
        2.0,  # 5: Crack-trans (2x boost)
        1.0, 1.0, 1.0,  # 6-8: normal
        0.3,  # 9: Foreign-particle (reduce)
    ])
```

**Expected Results:** Class_4 and Class_5 improvements in final epochs

### Strategy 2: Data Balancing
**Analysis:**
- Class_4: 342 samples (smallest)
- Class_9: 587 samples
- Class_3: 930 samples (largest)
- **Imbalance ratio: 2.7:1**

**Solution - Weighted Sampling:**
```python
# In YOLO_DAM_dataset_v2_RELATIVE.py

def get_class_weights(image_files, labels_dir):
    """Compute sampling weights to balance classes"""
    class_counts = {i: 0 for i in range(10)}
    
    for img_name in image_files:
        label_path = Path(labels_dir) / (Path(img_name).stem + ".txt")
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    cls = int(line.split()[0])
                    class_counts[cls] += 1
    
    # Inverse weighting: rarer classes get higher weight
    max_count = max(class_counts.values())
    weights = {cls: max_count / (count + 1) for cls, count in class_counts.items()}
    
    return weights

# Use in dataset sampling
```

**Expected Impact:** Better representation of rare classes during training

### Strategy 3: Hard Example Mining
```python
# In YOLO_DAM_train_v2.py, track hard examples

hard_examples = {cls: [] for cls in range(10)}

def track_hard_examples(loss, pred_cls, gt_cls):
    """Save examples where prediction was wrong"""
    if int(pred_cls) != int(gt_cls) and loss > loss_threshold:
        hard_examples[int(gt_cls)].append({
            'loss': loss,
            'pred': pred_cls,
            'actual': gt_cls
        })

# In epoch 200+, oversample hard examples
```

**Expected Impact:** +5-10% on problem classes

---

## 📋 Action Plan (Recommended)

### Phase 1: Immediate (Today, No Retraining)
**Time: 1-2 hours**  
**Expected Gain: +0.05 F1 score**

1. ✅ Apply class-specific confidence thresholds
2. ✅ Tighten NMS threshold for Class_9
3. ✅ Add size range filtering
4. ✅ Re-run test_relative_positioning.py
5. ✅ Compare metrics

### Phase 2: Medium-Term (This Week)
**Time: 30 minutes setup + 3-4 weeks training**  
**Expected Gain: +0.10 F1 score**

1. ✅ Update CLASS_WEIGHTS in loss function
2. ✅ Update ALPHA_PER_CLASS
3. ✅ Commit changes
4. ✅ Start retraining: `python YOLO_DAM_train_v2.py`
5. ✅ Monitor metrics weekly

### Phase 3: Advanced (After Phase 2 Results)
**Time: Depends on Phase 2 results**  
**Expected Gain: +0.05-0.10 F1 score**

1. Add rotation augmentation
2. Implement two-stage training
3. Add hard example mining
4. Monitor convergence

---

## 🎯 Expected Results Timeline

```
Current:      mAP=0.50, F1=0.64
Phase 1:      mAP=0.55, F1=0.69  (immediate)
Phase 2:      mAP=0.65, F1=0.75  (after 3-4 weeks)
Phase 3:      mAP=0.80+, F1=0.85+ (target achieved!)
```

---

## 📝 Testing Commands

### After Phase 1 (Quick Fix):
```bash
python test_relative_positioning.py
# Compare: Class_4 recall, Class_9 precision
```

### After Phase 2 (Retraining):
```bash
# Every 100 epochs during training, save checkpoint and test:
python test_relative_positioning.py --weights Models/YOLODAM_best_e100.h5
python test_relative_positioning.py --weights Models/YOLODAM_best_e200.h5
python test_relative_positioning.py --weights Models/YOLODAM_best_e300.h5
```

---

## ✅ Success Criteria

- [ ] Class_4 recall ≥ 70% (from 9.5%)
- [ ] Class_9 precision ≥ 70% (from 18.9%)
- [ ] Class_5 F1 ≥ 0.75 (from 0.45)
- [ ] Overall mAP ≥ 0.75
- [ ] Overall F1 ≥ 0.80
- [ ] All classes F1 ≥ 0.70

---

## 📞 Questions to Answer

1. **Which phase to start with?** → Start Phase 1 immediately (no retraining needed)
2. **How long does retraining take?** → 3-4 weeks (400 epochs on RTX 3090)
3. **Can I parallelize?** → Train Phase 2 while monitoring Phase 1 results
4. **How to prevent overfitting?** → Use validation metrics, early stopping at epoch 350-380
5. **What if Class_9 still has FP?** → Increase confidence threshold further (0.70) as fallback

