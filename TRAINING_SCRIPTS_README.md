# Training Scripts Guide

## Overview

This guide explains how to train all YOLO models from scratch and compare their performance on your defect detection dataset.

**Total Models**: 6 configurations
- 3 standard Ultralytics models (YOLOv11m, YOLOv26m, YOLOv26x)
- 3 custom YOLO-DAM ablation study configurations (A, B, C)

**Total Training Time**: ~20-24 weeks on RTX3090 GPU (300 epochs × 6 models ÷ parallel capability)

---

## Training Scripts Available

### 1. `TRAIN_ALL_MODELS.py` (Master Orchestrator)
**Purpose**: Train all 6 models sequentially in one command

**What it does**:
- Trains YOLOv11m, YOLOv26m, YOLOv26x (standard Ultralytics)
- Trains YOLO-DAM Config A (random init)
- Trains YOLO-DAM Config B (v26 pre-trained)
- Trains YOLO-DAM Config C (old model baseline)
- Automatically manages configuration changes
- Saves results to organized directories

**Run**:
```bash
cd d:\Projekty\2022_01_BattPor\2025_12_Dresden\VSCODE
D:\Programy\anaconda3\envs\TF_3_8\python.exe TRAIN_ALL_MODELS.py
```

**Time**: 20-24 weeks total

---

### 2. `TRAIN_STANDARD_MODELS.py`
**Purpose**: Train only standard Ultralytics YOLO models

**What it trains**:
- YOLOv11m
- YOLOv26m
- YOLOv26x

**Run**:
```bash
D:\Programy\anaconda3\envs\TF_3_8\python.exe TRAIN_STANDARD_MODELS.py
```

**Time**: ~3-4 weeks per model (9-12 weeks total)

**Output**: Models saved to
```
D:/Projekty/2022_01_BattPor/2025_12_Dresden/Training_Results/
├── yolov11m_trained/weights/best.pt
├── yolov26m_trained/weights/best.pt
└── yolov26x_trained/weights/best.pt
```

---

### 3. `TRAIN_YOLO_DAM_ABLATION.py`
**Purpose**: Train YOLO-DAM configurations to isolate component benefits

**What it trains**:

#### Phase 1: Config A (Random Initialization)
- Model: width=1.0, depth=1.0
- Backbone: **Random initialization** (no pre-training)
- M2M radius: 0 (fixed)
- Purpose: Measure pure architecture benefit

Expected: Precision 45-55%, Recall 78-82%

#### Phase 2: Config B (v26 Pre-trained)
- Model: width=1.0, depth=1.0
- Backbone: **v26 COCO pre-trained**
- M2M radius: 0 (fixed)
- Purpose: Measure combined benefit (architecture + pre-training)

Expected: Precision 70-75%, Recall 82-85%

#### Phase 3: Config C (Old Model Baseline)
- Model: width=0.6, depth=0.5
- Backbone: v26 COCO pre-trained
- M2M radius: 1 (adaptive - old behavior)
- Purpose: Baseline reference to show improvement from all fixes

Expected: Precision ~40%, Recall ~73%

**Run**:
```bash
D:\Programy\anaconda3\envs\TF_3_8\python.exe TRAIN_YOLO_DAM_ABLATION.py
```

**Time**: ~3-4 weeks per configuration (9-12 weeks total)

**Output**: Models saved to
```
D:/Projekty/2022_01_BattPor/2025_12_Dresden/Models/
├── YOLODAM_CONFIG_A_random.h5
├── YOLODAM_CONFIG_B_v26_pretrained.h5
└── YOLODAM_CONFIG_C_old_baseline.h5
```

---

## Data Preparation

### Dataset Location
```
D:/Projekty/2022_01_BattPor/2025_12_Dresden/YOLOv8/dataset/
├── images/
│   ├── train/       (3,104 images)
│   ├── val/         (validation images)
│   └── test/        (test images)
└── labels/
    ├── train/
    ├── val/
    └── test/
```

### Data Configuration (data.yaml)
```yaml
path: D:/Projekty/2022_01_BattPor/DATA_DEF/YOLOv11/

train: images/train
val: images/val
test: images/test

nc: 10  # Number of classes
names:
  0: dirt
  1: dent
  2: crack
  3: corrosion
  4: rust
  5: stain
  6: scratch
  7: burn
  8: weld
  9: missing
```

---

## Training Configuration

All models use:
- **Image Size**: 640×640
- **Epochs**: 300
- **Batch Size**: 4 (RTX3090 24GB VRAM)
- **Device**: GPU device 0 (change if using different GPU)
- **Optimizer**: SGD with cosine annealing learning rate
- **Early Stopping**: 50 epochs patience

### GPU Requirements
- **RTX3090** (24GB): All configurations supported
- **RTX4090** (24GB): All configurations supported
- **RTX3080** (10GB): Reduce batch size to 2

### Memory Estimate
```
Model           Params      VRAM (batch=4)
─────────────────────────────────────────
YOLOv11m        20.1M       ~3GB
YOLOv26m        20.1M       ~3GB
YOLOv26x        56.9M       ~8GB
YOLO-DAM        67.1M       ~10GB
```

---

## Training Progress Monitoring

### Expected Loss Curves

**Standard YOLO Models**:
```
Epoch 1:    Loss: 20-25
Epoch 50:   Loss: 5-8
Epoch 100:  Loss: 3-5
Epoch 300:  Loss: 1-3
```

**YOLO-DAM Config A (Random Init)**:
```
Epoch 1:    Loss: 38-40   (high, random weights)
Epoch 50:   Loss: 18-20
Epoch 100:  Loss: 12-14
Epoch 300:  Loss: 5-6
```

**YOLO-DAM Config B (v26 Pre-trained)**:
```
Epoch 1:    Loss: 28-30   (low, pre-trained backbone!)
Epoch 50:   Loss: 12-14
Epoch 100:  Loss: 8-10
Epoch 300:  Loss: 4-6
```

**YOLO-DAM Config C (Old Model)**:
```
Epoch 1:    Loss: 45-48
Epoch 50:   Loss: 20-22
Epoch 100:  Loss: 14-16
Epoch 300:  Loss: 6-8
```

### Log Files
```
D:/Projekty/2022_01_BattPor/2025_12_Dresden/Models/
├── train_log_dam_CONFIG_A.txt     (Config A training log)
├── train_log_dam_CONFIG_B.txt     (Config B training log)
└── train_log_dam_CONFIG_C.txt     (Config C training log)
```

View logs:
```bash
type d:\Projekty\2022_01_BattPor\2025_12_Dresden\Models\train_log_dam_CONFIG_B.txt | tail -50
```

---

## Comparing All Models

### Step 1: After All Training Completes

Run comprehensive evaluation:
```bash
D:\Programy\anaconda3\envs\TF_3_8\python.exe COMPREHENSIVE_TEST_AND_COMPARE.py
```

This script will:
1. Auto-discover all trained models
2. Test each on the test dataset
3. Calculate metrics (precision, recall, F1, mAP)
4. Generate Excel report with detailed comparisons

### Step 2: Review Results

Results saved to:
```
D:/Projekty/2022_01_BattPor/2025_12_Dresden/TEST_RESULTS/
└── test_run_YYYYMMDD_HHMMSS/
    ├── metrics/
    │   ├── yolov11m_metrics.json
    │   ├── yolov26m_metrics.json
    │   ├── yolov26x_metrics.json
    │   ├── YOLO-DAM-CONFIG_A_metrics.json
    │   ├── YOLO-DAM-CONFIG_B_metrics.json
    │   └── YOLO-DAM-CONFIG_C_metrics.json
    ├── predictions/
    ├── logs/
    └── comparison_report.xlsx
```

### Excel Report Contains
- **Sheet 1**: Overall metrics comparison (sorted by F1)
- **Sheet 2**: Class-level metrics (all 10 defect classes)
- **Sheet 3**: Comparison to baseline models
- **Sheet 4**: Summary and key findings

---

## Expected Results Summary

### Standard Models
| Model | Params | Precision | Recall | F1 | Training Time |
|-------|--------|-----------|--------|----|----|
| YOLOv11m | 20.1M | 72-75% | 83-85% | 0.77-0.80 | 3-4 weeks |
| YOLOv26m | 20.1M | 80-82% | 85-87% | 0.83-0.85 | 3-4 weeks |
| YOLOv26x | 56.9M | 84-86% | 87-89% | 0.86-0.87 | 3-4 weeks |

### Custom YOLO-DAM Configurations
| Config | Architecture | Pre-training | Precision | Recall | F1 |
|--------|---|---|-----------|--------|---|
| A | 1.0×1.0 | ✗ Random | 45-55% | 78-82% | 0.61-0.68 |
| B | 1.0×1.0 | ✓ v26 | 70-75% | 82-85% | 0.76-0.80 |
| C | 0.6×0.5 | ✓ v26 | 38-42% | 72-75% | 0.48-0.55 |

### Key Insights
```
Pre-training Benefit (B - A):   +20-25% precision, +4-6% recall
Architecture Benefit (A - C):   +5-15% precision, +6-10% recall
M2M Fix Benefit (A - C):        +20-25% precision (no duplicates)
Total Improvement (B - C):      +30-35% precision, +9-12% recall
```

---

## Troubleshooting

### Issue: CUDA Out of Memory
```
Solution: Reduce batch size in training script
  BATCH_SIZE = 4 → 2  (training slower but fits)
  BATCH_SIZE = 4 → 1  (if still OOM)
```

### Issue: Loss Not Decreasing (Config A or B)
```
Possible causes:
  1. Dataset not accessible (check path)
  2. GPU not detected (check nvidia-smi)
  3. TensorFlow not using GPU

Solutions:
  1. Verify dataset path in training script
  2. Check: nvidia-smi → GPU 0 available?
  3. Run: python -c "import tensorflow as tf; print(len(tf.config.list_physical_devices('GPU')))"
```

### Issue: Training Hangs
```
Solution:
  1. Press Ctrl+C to stop
  2. Check if GPU process is stuck: tasklist | findstr nvidia
  3. Verify TF_3_8 environment is active
  4. Restart training
```

### Issue: Models Not Found in Comparison Script
```
Problem: COMPREHENSIVE_TEST_AND_COMPARE.py says no models found

Solutions:
  1. Verify training actually completed
  2. Check models saved in correct location:
     - Standard: D:/Projekty/2022_01_BattPor/2025_12_Dresden/Training_Results/*/weights/best.pt
     - YOLO-DAM: D:/Projekty/2022_01_BattPor/2025_12_Dresden/Models/YOLODAM_*.h5
  3. Check file permissions (should be readable)
```

---

## Quick Start Guide

### Option 1: Train Everything (Recommended for Full Analysis)
```bash
cd d:\Projekty\2022_01_BattPor\2025_12_Dresden\VSCODE
D:\Programy\anaconda3\envs\TF_3_8\python.exe TRAIN_ALL_MODELS.py
```
- Duration: 20-24 weeks
- Result: Complete comparison of all approaches

### Option 2: Train Only Standard Models
```bash
D:\Programy\anaconda3\envs\TF_3_8\python.exe TRAIN_STANDARD_MODELS.py
```
- Duration: 9-12 weeks
- Result: Baseline comparison for standard YOLO models

### Option 3: Train Only YOLO-DAM Ablation
```bash
D:\Programy\anaconda3\envs\TF_3_8\python.exe TRAIN_YOLO_DAM_ABLATION.py
```
- Duration: 9-12 weeks
- Result: Understand benefit of each component (pre-training, architecture, M2M fix)

### Step 4: After Training, Evaluate
```bash
D:\Programy\anaconda3\envs\TF_3_8\python.exe COMPREHENSIVE_TEST_AND_COMPARE.py
```
- Duration: 1-2 hours (depends on test set size)
- Output: Excel report with detailed comparisons

---

## Files Modified During Training

The following files are temporarily modified during ablation study training:

**Modified by `TRAIN_YOLO_DAM_ABLATION.py`**:
- `YOLO_DAM.py` (restored after Config C)
- `YOLO_DAM_dataset.py` (restored after Config C)
- `YOLO_DAM_train.py` (restored after Config B)

All files are **automatically restored** to Config B state (final desired configuration) after training completes.

---

## Environment

### Conda Environment
```
Name: TF_3_8
Python: 3.8.x
TensorFlow: 2.13.x
CUDA: 11.x
cuDNN: 8.x
```

### Verify Environment
```bash
D:\Programy\anaconda3\envs\TF_3_8\python.exe -c "
import tensorflow as tf
print('TensorFlow:', tf.__version__)
print('GPUs:', len(tf.config.list_physical_devices('GPU')))
"
```

Expected output:
```
TensorFlow: 2.13.x
GPUs: 1
```

---

## Support

### Helpful Commands

List GPU status:
```bash
nvidia-smi
```

Check TensorFlow GPU usage:
```bash
D:\Programy\anaconda3\envs\TF_3_8\python.exe -c "
import tensorflow as tf
print('GPU Devices:', tf.config.list_physical_devices('GPU'))
"
```

View training logs in real-time:
```bash
type d:\Projekty\2022_01_BattPor\2025_12_Dresden\Models\train_log_dam_CONFIG_B.txt
```

Check disk space:
```bash
dir d:\Projekty\2022_01_BattPor\2025_12_Dresden\Models\
```

---

## Next Steps

1. **Start Training**: Choose one of the training scripts above
2. **Monitor Progress**: Check logs periodically for loss curves
3. **Wait for Completion**: Training takes 3-4 weeks per model
4. **Evaluate Models**: Run COMPREHENSIVE_TEST_AND_COMPARE.py
5. **Analyze Results**: Review Excel report for detailed comparisons
6. **Deploy Best Model**: Use highest-performing model in production

---

**Last Updated**: 2026-03-30
**Status**: Ready to train
