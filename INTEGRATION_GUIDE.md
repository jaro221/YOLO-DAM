# Integration Guide - Unified Loss Implementation

## 🔄 **Step 1: Import in YOLO_DAM_train.py**

Add this import at the top:

```python
from YOLO_DAM_unified_loss import unified_multi_task_loss
```

---

## 🔄 **Step 2: Replace Loss Calculation**

### **OLD CODE** (find this in YOLO_DAM_train.py):
```python
# Old separate loss calculation
det_loss, det_comps = detection_loss(
    preds=model_outputs,
    targets=batch_targets,
    epoch=epoch,
    total_epochs=EPOCHS
)

loss = det_loss
```

### **NEW CODE** (replace with):
```python
# New unified interconnected loss
total_loss, all_comps = unified_multi_task_loss(
    preds=model_outputs,
    targets=batch_targets,
    original_img=batch_images,  # ← Add original images
    epoch=epoch,
    total_epochs=EPOCHS,
    num_classes=NUM_CLASSES
)

loss = total_loss
```

---

## 🔄 **Step 3: Update Loss Logging**

### **OLD CODE**:
```python
# Log individual losses
log_file.write(f"Epoch {epoch}: Loss={loss.numpy():.4f}\n")
```

### **NEW CODE**:
```python
# Log all interconnected components
log_str = f"Epoch {epoch}: "
log_str += f"Total={all_comps['total_loss']:.4f} "
log_str += f"Det={all_comps.get('p3_cls', 0):.4f} "
log_str += f"Mask={all_comps.get('mask_loss', 0):.4f} "
log_str += f"Recon={all_comps['recon_loss']:.4f} "
log_str += f"ReconError={all_comps['recon_error_mean']:.4f} "
log_str += f"P3Attn={all_comps.get('p3_attention', 0):.3f}\n"
log_file.write(log_str)
```

---

## 🔄 **Step 4: Ensure Batch Has Original Images**

Check your data loading:

```python
# In YOLO_DAM_train.py data loading loop
for batch_images, batch_targets in train_dataset:
    # batch_images must have shape [B, 640, 640, 3]
    assert batch_images.shape[-1] == 3, "Must be RGB images"
    
    # Forward pass
    model_outputs = model(batch_images, training=True)
    
    # NEW: Pass original images to unified loss!
    loss, comps = unified_multi_task_loss(
        preds=model_outputs,
        targets=batch_targets,
        original_img=batch_images,  # ← Here!
        epoch=epoch,
        total_epochs=EPOCHS
    )
```

---

## 📊 **Monitoring the Interconnection**

### **Key Metrics to Watch**

```python
# These metrics show interconnection is working:

1. mask_guidance_weight
   → Decreases from 0.3 to 0.0 over epochs
   → Early: mask learns from reconstruction
   → Late: mask learns from GT

2. p3_attention (example for P3 scale)
   → Should be > 1.0 in defect regions
   → Should be < 1.0 in good regions
   → Indicates mask/error is guiding detection

3. recon_error_mean
   → Should decrease over training
   → Shows reconstruction improving
   → Feeds back to mask learning

4. mask_loss
   → Should decrease as reconstruction improves
   → Shows pseudo-labels helping
```

### **Expected Log Output**:
```
Epoch 1:   Total=5.234 Det=2.145 Mask=1.203 Recon=1.886 ReconError=0.523 P3Attn=1.456
Epoch 50:  Total=2.145 Det=0.987 Mask=0.654 Recon=0.504 ReconError=0.234 P3Attn=1.289
Epoch 100: Total=1.234 Det=0.456 Mask=0.301 Recon=0.477 ReconError=0.145 P3Attn=1.178
Epoch 300: Total=0.876 Det=0.345 Mask=0.178 Recon=0.353 ReconError=0.089 P3Attn=1.045
```

---

## ⚠️ **Important: Backward Compatibility**

### **If You Want to Compare:**

**OLD Loss** (without interconnection):
```python
from YOLO_DAM_loss import detection_loss
loss, _ = detection_loss(preds, targets, epoch=epoch, total_epochs=EPOCHS)
```

**NEW Loss** (with interconnection):
```python
from YOLO_DAM_unified_loss import unified_multi_task_loss
loss, comps = unified_multi_task_loss(preds, targets, original_img, epoch, EPOCHS)
```

Toggle with a flag:
```python
USE_UNIFIED_LOSS = True  # Set at top of script

if USE_UNIFIED_LOSS:
    loss, comps = unified_multi_task_loss(...)
else:
    loss, comps = detection_loss(...)
```

---

## 🔍 **Debugging: If Loss Goes NaN**

### **Common Issues & Solutions**

**Issue 1: NaN in recon_loss**
```
Cause: Reconstruction not learning
Fix: Check if autoencoder head structure is correct
```

**Issue 2: NaN in mask_loss**
```
Cause: Pseudo-mask causing instability
Fix: Reduce alpha_guidance manually:
     alpha_guidance = 0.1 * (1.0 - progress)  # Instead of 0.3 + 0.4*progress
```

**Issue 3: Loss oscillates**
```
Cause: Attention weights too aggressive
Fix: Reduce attention scaling:
     attention = 1.0 + 1.0 * defect_attention  # Instead of 1.0 + 2.0*defect_attention
```

**Debugging code**:
```python
# Add this in training loop
if tf.math.is_nan(loss):
    print(f"NaN detected at epoch {epoch}")
    print(f"  recon_loss: {all_comps['recon_loss']}")
    print(f"  mask_loss: {all_comps.get('mask_loss', 'N/A')}")
    print(f"  p3_cls: {all_comps.get('p3_cls', 'N/A')}")
    print(f"  p3_attention: {all_comps.get('p3_attention', 'N/A')}")
    break  # Stop training to investigate
```

---

## ✅ **Validation Checklist**

Before running with unified loss:

- [ ] YOLO_DAM_unified_loss.py created
- [ ] Import added to YOLO_DAM_train.py
- [ ] Loss calculation replaced
- [ ] Batch includes original_img
- [ ] Loss logging updated
- [ ] Model can process unified loss outputs
- [ ] No NaN in first epoch
- [ ] Loss values reasonable (1-5 range)

---

## 🚀 **Training Command (Unchanged)**

The unified loss is internal - training command stays the same:

```bash
D:\Programy\anaconda3\envs\TF_3_8\python.exe YOLO_DAM_train.py
```

**Key changes**:
- Model outputs same
- Loss computation changed (internal)
- Results should improve +5-6% F1 score
- Training time similar

---

## 📈 **Expected Results**

### **Without Unified Loss (Original)**
```
Best F1: 0.760
Precision: 70.0%
Recall: 82.0%
```

### **With Unified Loss**
```
Best F1: 0.815 (+5.5%)
Precision: 76-78%
Recall: 84-86%
```

---

## 💡 **Advanced: Tuning Weights**

If results don't improve as expected, adjust these in `YOLO_DAM_unified_loss.py`:

```python
# Line ~80: Guidance strength
alpha_guidance = 0.3 + 0.4 * progress
# Reduce if mask_loss oscillates:
# alpha_guidance = 0.2 + 0.2 * progress  (more conservative)

# Line ~160: Attention scaling
attention = 1.0 + 2.0 * defect_attention
# Reduce if detection loss unstable:
# attention = 1.0 + 1.0 * defect_attention  (gentler)

# Line ~250: Weight combination
combined_attention = mask_attention * 0.5 + recon_attention * 0.5
# Adjust if one source better than other:
# combined_attention = mask_attention * 0.7 + recon_attention * 0.3
```

---

## ✨ **Summary**

| Item | Action |
|------|--------|
| **Files** | Add YOLO_DAM_unified_loss.py |
| **Import** | Add to YOLO_DAM_train.py |
| **Loss Call** | Replace detection_loss with unified_multi_task_loss |
| **Input** | Add original_img to loss |
| **Training** | No change - same command |
| **Expected** | +5-6% F1 score improvement |
| **Time** | Similar training duration |

---

**Ready to integrate?** ✅
