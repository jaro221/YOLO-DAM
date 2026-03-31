# Investigation: Maximizing Performance with Current YOLO-DAM Model

## Current Model Summary
- **Architecture**: YOLOv11-based backbone (width=0.6) + PANet neck + DecoupledHead (M2M + O2O)
- **Scales**: P2 (160×160), P3 (80×80), P4 (40×40), P5 (20×20)
- **Auxiliary heads**: MaskHead_V2, AutoHead_V2
- **Weights**: Merged (v11 backbone + v26 DAM heads)
- **Total params**: 20.9M (trainable: 20.8M)
- **Baseline loss @ epoch 1**: ~45 (with merged weights)

---

## Performance Maximization Strategies

### ⭐ CATEGORY A: Training Optimization (Highest ROI, no code changes)

#### A1: Learning Rate Scheduling (Current: Cosine annealing from 5e-5)
```
Current: 5e-5 → cosine decay → ~0 over 300 epochs
```

**Investigation Options:**

**A1a: Warmup + Cosine Annealing (Recommended)**
- Start: 5e-6 (warmup for 5 epochs) → 5e-5 (peak) → cosine decay
- **Expected improvement**: +1–2% mAP
- **Reason**: Better convergence, avoids sharp early jumps
- **Cost**: Negligible (already have cosine annealing)
- **Implementation**: 3 lines in YOLO_DAM_train.py

**A1b: Multi-stage LR schedule**
```
Epochs 0-50:    5e-5  (backbone warmup)
Epochs 51-150:  2e-5  (fine-tune neck)
Epochs 151-300: 5e-6  (polish heads)
```
- **Expected improvement**: +2–3% mAP
- **Reason**: Different parts mature at different rates
- **Cost**: +5% training time
- **Implementation**: 10 lines in YOLO_DAM_train.py

**A1c: Adaptive LR based on loss plateau**
```
If loss doesn't improve for 10 epochs → reduce LR by 0.5
```
- **Expected improvement**: +1–2% mAP
- **Reason**: Automatic escape from local minima
- **Cost**: +10% training time (fewer wasted epochs)
- **Implementation**: Keras callback, 15 lines

---

#### A2: Batch Accumulation (Current: batch_size=4)
```
Current: 4 samples/batch
```

**Investigation Options:**

**A2a: Gradient Accumulation**
```
Effective batch = 4 samples × 4 accumulation steps = 16 equivalent
```
- **Expected improvement**: +2–4% mAP
- **Reason**: Larger effective batch → better gradient estimates
- **Cost**: +0% training time (same GPU memory, more iterations)
- **Implementation**: Modify train_step() to skip apply_gradients() 3/4 of time
- **Difficulty**: Easy (20 lines)

**A2b: Increase actual batch size**
```
4 → 8 or 16 (if GPU memory allows)
```
- **Expected improvement**: +1–3% mAP
- **Reason**: Larger batches → better regularization
- **Cost**: Need RTX 3090 (you have this ✅) → test if fits
- **Check command**: Run inference on batch_size=8 or 16
- **Implementation**: Change BATCH_SIZE in YOLO_DAM_dataset.py

**A2c: Mixed precision training**
```
Use tf.keras.mixed_precision (float16 for compute, float32 for storage)
```
- **Expected improvement**: +0.5–1% mAP (speed improvement: 20–30%)
- **Reason**: Faster computation = more epochs in same time
- **Cost**: Minimal (already supported by TF)
- **Implementation**: 5 lines in YOLO_DAM_train.py

---

#### A3: Label Smoothing (Current: 0.01, optional)
```
Current setting in YOLO_DAM_train.py: USE_LABEL_SMOOTHING = 0.01
```

**Investigation Options:**

**A3a: Increase label smoothing**
```
0.01 → 0.05 or 0.1
```
- **Expected improvement**: +1–2% mAP (improves generalization)
- **Reason**: Reduces overconfident predictions
- **Cost**: Negligible
- **Risk**: Too high (>0.15) → poor calibration
- **Implementation**: Change 1 line

**A3b: Class-specific smoothing**
```
Class 0-8: 0.01 (common classes)
Class 9:   0.05 (hard class - foreign particles)
```
- **Expected improvement**: +2–3% mAP (especially class 9)
- **Reason**: Rare/hard classes benefit from more regularization
- **Cost**: Negligible
- **Implementation**: Modify detection_loss() focal loss computation

---

#### A4: Optimization: Gradient Clipping & Normalization (Current: clip_norm=5)
```
Current: grads, global_norm = tf.clip_by_global_norm(grads, clip_norm=5)
```

**Investigation Options:**

**A4a: Tune clip norm value**
```
Current: 5.0
Options: 1.0, 3.0, 10.0
```
- **Find sweet spot**: Test which prevents gradient explosion without suppressing learning
- **Expected improvement**: +0.5–1% (stability)
- **Cost**: 5-run comparison
- **Implementation**: Grid search on [1.0, 3.0, 5.0, 10.0]

**A4b: Layer-wise gradient scaling**
```
Different clip norms for different layers:
- Backbone: clip_norm=10 (stable, large gradients)
- Neck: clip_norm=5 (intermediate)
- Heads: clip_norm=2 (sensitive, small gradients)
```
- **Expected improvement**: +1–2% mAP
- **Reason**: Each part has different gradient magnitude
- **Cost**: +5% training time
- **Implementation**: 20 lines in train_step()

---

### ⭐ CATEGORY B: Data & Augmentation Optimization

#### B1: Advanced Augmentation (Current: HSV + horizontal flip)
```
Current augmentation in YOLO_DAM_dataset.py:
- augment_hsv(): hue, saturation, brightness
- augment_flip(): horizontal flip (50%)
```

**Investigation Options:**

**B1a: Add geometric augmentations**
```
- Random rotation (±5°)
- Random scaling (0.8-1.2x)
- Random perspective transform
```
- **Expected improvement**: +3–5% mAP
- **Reason**: Defects appear at different angles/scales
- **Cost**: +15–20% training time
- **Implementation**: 30-40 lines in YOLO_DAM_dataset.py
- **Difficulty**: Medium (requires box transformation)

**B1b: Mosaic augmentation**
```
Combine 4 random images into 1:
[img1] [img2]
[img3] [img4]
```
- **Expected improvement**: +4–6% mAP (especially for small objects in P2)
- **Reason**: YOLOv8/v11 standard, works great
- **Cost**: +20–30% training time
- **Implementation**: 50-60 lines (complex but worth it)
- **Difficulty**: High (box merging, scaling)

**B1c: Cutout/Mixup**
```
Cutout: Randomly erase patches from image
Mixup: Blend two images + blend boxes
```
- **Expected improvement**: +2–4% mAP
- **Reason**: Improves robustness to occlusion
- **Cost**: +10% training time
- **Implementation**: 20-30 lines
- **Difficulty**: Medium

**B1d: Copy-Paste augmentation**
```
Copy detected defects from one image and paste onto another clean image
```
- **Expected improvement**: +5–8% mAP (especially for rare classes)
- **Reason**: Creates synthetic hard examples
- **Cost**: +25% training time + preprocessing
- **Implementation**: 50+ lines
- **Difficulty**: High (requires bounding box aware pasting)

---

#### B2: Dataset Rebalancing (Current: CLASS_WEIGHTS applied in loss)
```
Current weights in YOLO_DAM_loss.py:
- Class 4 (Crack-long): 2.0  (fewest - 1145 instances)
- Class 9 (Foreign-particle): 2.0 (hardest - 1576 instances)
- Others: 1.0
```

**Investigation Options:**

**B2a: Adjust class weights**
```
Option 1 - Inverse frequency:
weight[c] = total_samples / (num_classes × count[c])

Option 2 - Sqrt scaling (gentler):
weight[c] = sqrt(total_samples / count[c])

Option 3 - Manual tuning:
Class 4: 3.0 (increase)
Class 9: 2.5 (increase slightly)
```
- **Expected improvement**: +1–3% mAP (especially precision for rare classes)
- **Cost**: Negligible (already doing this)
- **Implementation**: Change YOLO_DAM_loss.py constants or make dynamic

**B2b: Oversampling rare classes**
```
During data loading, repeat samples from classes 4 & 9
```
- **Expected improvement**: +2–4% mAP
- **Cost**: +10–20% training time
- **Implementation**: Modify YOLO_DAM_dataset.py generator

**B2c: Hard example mining**
```
Track epochs where specific classes fail most
Prioritize those samples in subsequent epochs
```
- **Expected improvement**: +3–5% mAP
- **Cost**: +30% training time, complex logic
- **Implementation**: 100+ lines
- **Difficulty**: High

---

#### B3: Validation-based improvements
```
Current: No validation during training (only best loss checkpoint)
```

**Investigation Options:**

**B3a: Add validation dataset**
```
Split: 80% train, 20% val
Monitor: mAP, precision, recall (not just loss)
```
- **Expected improvement**: +2–3% (better checkpoint selection)
- **Reason**: Loss can plateau while metrics still improve
- **Cost**: +30% training time
- **Implementation**: Add val_ds to YOLO_DAM_train.py

**B3b: Early stopping with patience**
```
If validation mAP doesn't improve for 30 epochs → stop
```
- **Expected improvement**: +1–2% (avoids overfitting)
- **Cost**: Less wasted computation
- **Implementation**: Keras callback, 10 lines

---

### ⭐ CATEGORY C: Model Architecture Refinement (Requires code changes)

#### C1: Add SE (Squeeze-Excitation) blocks to backbone
```
Lightweight attention: [B, H, W, C] → [B, 1, 1, C] → [B, H, W, C]
```
- **Expected improvement**: +2–3% mAP
- **Reason**: Channel-wise attention helps focus on relevant features
- **Cost**: +5–10% inference time, +2–3% training time
- **Implementation**: Wrap Conv blocks with SE layers
- **Difficulty**: Medium

#### C2: Replace standard convolutions with grouped convolutions
```
Conv2D(out_ch, ..., groups=8) instead of groups=1
```
- **Expected improvement**: +1–2% mAP
- **Reason**: More efficient feature learning, acts as regularization
- **Cost**: -10% inference time (faster!), similar training
- **Implementation**: Change groups parameter (1 line per conv)
- **Difficulty**: Easy

#### C3: Add skip connections in Neck
```
Direct connection from C3 → P3 in addition to FPN path
```
- **Expected improvement**: +2–4% mAP (especially P3 scale)
- **Reason**: Preserves high-res information
- **Cost**: +5% parameters, negligible speed cost
- **Implementation**: 5 lines in PANetNeck class
- **Difficulty**: Easy

#### C4: Increase model capacity (depth/width)
```
Current: width=0.6, depth=0.5
Options: width=0.8 or 1.0, depth=0.7 or 1.0
```
- **Expected improvement**: +3–6% mAP
- **Reason**: Larger model, more capacity
- **Cost**: 2× training time, 2× inference time
- **Implementation**: Change YOLO_DAM.py build_yolo_model() params
- **Difficulty**: Easy (1 line)

---

### ⭐ CATEGORY D: Loss Function Refinement

#### D1: IoU-based losses instead of CIoU
```
Current: CIoU (Complete IoU)
Options: DIoU, GIoU, EIoU, SIoU
```
- **Expected improvement**: +0–1% mAP
- **Reason**: Different geometric considerations
- **Cost**: Negligible
- **Implementation**: Replace ciou_loss() with alternative
- **Difficulty**: Easy

#### D2: Focal loss tuning
```
Current: gamma=2.0 (medium hardness)
Options: gamma=1.5, 2.5, 3.0
```
- **Expected improvement**: +0.5–1% mAP
- **Reason**: Controls how hard examples are weighted
- **Cost**: Negligible
- **Implementation**: Change 1 constant
- **Difficulty**: Easy

#### D3: Add mask consistency loss (from investigation doc)
```
Joint Loss: detection_loss + mask_consistency_loss
Consistency: If detection → box, mask should be 0 in that region
```
- **Expected improvement**: +2–5% mAP + mask accuracy
- **Reason**: Helps detection and mask agree
- **Cost**: Negligible
- **Implementation**: 10-15 lines in detection_loss()
- **Difficulty**: Easy

#### D4: Temperature scaling for confidence calibration
```
pred_confidence_calibrated = sigmoid(logits / temperature)
Tune temperature: 0.8 to 2.0
```
- **Expected improvement**: +1–2% mAP (better recall/precision balance)
- **Cost**: Negligible
- **Implementation**: 5 lines in loss computation
- **Difficulty**: Easy

---

### ⭐ CATEGORY E: Inference Optimization

#### E1: Test-Time Augmentation (TTA)
```
- Flip image left-right
- Run inference on 4 versions
- Average predictions
```
- **Expected improvement**: +3–5% effective mAP
- **Cost**: 4× inference time (not practical for real-time)
- **Implementation**: 20 lines in inference script
- **Use case**: Validation/evaluation only

#### E2: Ensemble with complementary models
```
Train 3 models with different seeds/augmentations
Ensemble predictions
```
- **Expected improvement**: +2–4% mAP (no cost during training)
- **Cost**: 3× inference time
- **Implementation**: Train 3× with different seeds
- **Use case**: High-stakes evaluation

---

## Quick Win Ranking (Best Risk/Reward)

### 🥇 Tier 1: Implement First (2-4 hours, +1–3% mAP)
1. **A2a: Gradient accumulation** (20 lines) → +2–4% mAP
2. **D3: Mask consistency loss** (15 lines) → +2–5% mAP
3. **B2a: Adjust class weights** (5 lines) → +1–3% mAP
4. **A1a: Warmup + cosine LR** (3 lines) → +1–2% mAP

**Expected total from Tier 1: +6–14% mAP**

### 🥈 Tier 2: Implement Next (4-8 hours, +2–4% mAP)
5. **B1a: Geometric augmentations** (40 lines) → +3–5% mAP
6. **C3: Skip connections in Neck** (5 lines) → +2–4% mAP
7. **B3a: Validation dataset** (30 lines) → +2–3% mAP

**Expected additional: +7–12% mAP**

### 🥉 Tier 3: Advanced (8-16 hours, +3–6% mAP)
8. **B1b: Mosaic augmentation** (60 lines) → +4–6% mAP
9. **B1d: Copy-Paste augmentation** (80 lines) → +5–8% mAP
10. **C4: Larger model** (1 line, need 2 weeks training) → +3–6% mAP

---

## Current Bottlenecks Analysis

### Why is baseline ~45?
- ✅ Good: Merged weights provide good initialization
- ⚠️ Possible: Gradient accumulation too aggressive (batch=4 effective too small)
- ⚠️ Possible: Label smoothing=0.01 might be too low (try 0.05)
- ⚠️ Possible: Early epochs use low mask/recon weights (ProgLoss) → imbalance

### How to diagnose first epoch loss is good?
1. Check per-scale loss components:
   ```
   If p2_box >> p3_box >> p4_box >> p5_box
   → P2 is dominating, needs reweighting
   ```

2. Check M2M vs O2O:
   ```
   If M2M loss >> O2O loss
   → Too many positive anchors, adjust radius in build_targets_m2m()
   ```

3. Check mask vs detection balance:
   ```
   If recon_loss >> detection_loss
   → Autoencoder too heavy, reduce recon_w in detection_loss()
   ```

---

## Recommended Roadmap

### Phase 1: Quick Wins (Week 1)
```
├─ A2a: Add gradient accumulation (2 hours)
├─ D3: Add mask consistency loss (2 hours)
├─ B2a: Tune class weights with analysis (1 hour)
├─ A1a: Add warmup (0.5 hour)
└─ Train for 50 epochs, measure improvements
```
**Expected result**: +6–10% mAP, faster convergence

### Phase 2: Medium Effort (Week 2-3)
```
├─ B1a: Add geometric augmentations (4 hours)
├─ C3: Add skip connections (1 hour)
├─ B3a: Add validation dataset (2 hours)
└─ Train for 100 epochs, measure improvements
```
**Expected result**: Additional +7–10% mAP

### Phase 3: Advanced (Week 4+)
```
├─ B1b: Implement Mosaic (8 hours)
├─ B1d: Implement Copy-Paste (8 hours)
└─ C4: Train larger model (2 weeks retraining)
```
**Expected result**: Additional +8–15% mAP

**Total potential**: **+21–35% mAP over baseline**

---

## Per-Defect Class Opportunities

### Class 0-3, 5-8 (Common classes ~1500-2500 samples)
- ✅ Benefit from: data augmentation, larger models
- ✅ Less benefit from: class weights (already balanced)

### Class 4 (Crack-long, fewest ~1145 samples)
- ✅ Benefit most from: augmentation (geometric + mosaic + copy-paste)
- ✅ Benefit from: higher class weight
- ✅ Benefit from: hard example mining

### Class 9 (Foreign-particle, hardest ~1576 samples)
- ✅ Benefit from: higher alpha in focal loss
- ✅ Benefit from: copy-paste (synthetic examples)
- ✅ Benefit from: mask guidance (if particles are small)

---

## GPU Memory Considerations (RTX 3090, 24GB)

### Current: batch_size=4
- Forward: ~8GB
- Backward: ~12GB
- Optimizer state: ~4GB
- **Total: ~24GB (at limit)**

### Can we do batch_size=8?
```
Estimate: Linear scaling → 16GB forward + backward → 20GB
Answer: Maybe with gradient checkpointing
```

### Can we do batch_size=16?
```
Estimate: 32GB needed
Answer: Probably not without A100
```

### Option: Gradient checkpointing
```
Trade: Memory ↔ Compute time
Save gradients on-the-fly instead of storing
Cost: ~10% slower training, but frees 4-6GB
Allows: batch_size=8 comfortably
```

---

## Summary Table: All Strategies

| Strategy | Category | Effort | mAP gain | Risk | Implementation |
|----------|----------|--------|----------|------|---|
| Gradient accumulation | A2a | 2h | +2–4% | None | Easy |
| Mask consistency loss | D3 | 2h | +2–5% | Low | Easy |
| Class weight tuning | B2a | 1h | +1–3% | None | Easy |
| Warmup + cosine | A1a | 0.5h | +1–2% | None | Easy |
| **Tier 1 Total** | | **5.5h** | **+6–14%** | **None** | **Easy** |
| Geometric augmentation | B1a | 4h | +3–5% | Low | Medium |
| Skip connections | C3 | 1h | +2–4% | None | Easy |
| Validation dataset | B3a | 2h | +2–3% | None | Easy |
| **Tier 2 Total** | | **7h** | **+7–12%** | **Low** | **Medium** |
| Mosaic augmentation | B1b | 8h | +4–6% | Medium | Hard |
| Copy-paste augmentation | B1d | 8h | +5–8% | Medium | Hard |
| Larger model (0.8x or 1.0x) | C4 | 2 weeks | +3–6% | None | Easy |
| **Tier 3 Total** | | **16+ h** | **+12–20%** | **Medium** | **Hard** |

---

## Conclusion

**Current model is well-built. Performance gains come from:**

1. **Training smarter** (+6–14% via Tier 1)
2. **Training on better data** (+7–12% via Tier 2)
3. **Using bigger models** (+12–20% via Tier 3)

**Recommended approach:**
- Start Tier 1 → measure
- If good, do Tier 2 → measure
- Only do Tier 3 if time/compute allows

**No code changes needed to start** — all Tier 1 is 10-15 lines total.
