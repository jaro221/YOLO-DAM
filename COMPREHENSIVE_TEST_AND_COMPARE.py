
"""
Comprehensive Model Testing & Comparison Script
Tests multiple YOLO models on defect dataset and compares results
Outputs: Performance metrics, class-level analysis, Excel comparison

Usage:
    python COMPREHENSIVE_TEST_AND_COMPARE.py
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
from collections import defaultdict
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils.dataframe import dataframe_to_rows

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Paths
TEST_LABELS_DIR = r"D:/Projekty/2022_01_BattPor/DATA_DEF/YOLOv8/test_dataset/labels/test/"
TEST_IMAGES_DIR = TEST_LABELS_DIR.replace("labels", "images")
RESULTS_DIR     = r"D:/Projekty/2022_01_BattPor/2025_12_Dresden/TEST_RESULTS"

# Class names (10 defect classes)
CLASS_NAMES = {
    0: "Agglomerate",
    1: "Pinhole-long",
    2: "Pinhole-trans",
    3: "Pinhole-round",
    4: "Crack-long",
    5: "Crack-trans",
    6: "Line-long",
    7: "Line-trans",
    8: "Line-diag",
    9: "Foreign-particle",
}

# Models to test: (model_name, weights_path, description)
# Will be populated dynamically from training results
MODELS_TO_TEST = []

def discover_trained_models():
    """Discover trained models from results directory"""
    training_results = r"D:/Projekty/2022_01_BattPor/2025_12_Dresden/Training_Results"
    models_dir = r"D:/Projekty/2022_01_BattPor/2025_12_Dresden/Models"

    discovered = []

    # Standard Ultralytics models
    for model_name in ["yolov11m", "yolov26m", "yolov26x"]:
        model_path = os.path.join(
            training_results,
            f"{model_name}_trained",
            "weights",
            "best.pt"
        )
        if os.path.exists(model_path):
            discovered.append((model_name, model_path, f"Standard {model_name.upper()}"))

    # YOLO-DAM configs
    configs = {
        "CONFIG_A_random": "YOLO-DAM Config A (width=1.0, random init)",
        "CONFIG_B_v26_pretrained": "YOLO-DAM Config B (width=1.0, v26 pre-trained)",
        "CONFIG_C_old_baseline": "YOLO-DAM Config C (width=0.6, old baseline)",
    }

    for config_id, description in configs.items():
        model_path = os.path.join(models_dir, f"YOLODAM_{config_id}.h5")
        if os.path.exists(model_path):
            discovered.append((f"YOLO-DAM-{config_id}", model_path, description))

    return discovered

# Baseline models from literature (for comparison)
BASELINE_MODELS = {
    "YOLOv8n": {"params": 3.2, "ap": 0.5825, "f1": 0.6415, "recall": 0.7633},
    "YOLOv8m": {"params": 25.9, "ap": 0.6043, "f1": 0.6505, "recall": 0.7424},
    "YOLOv8x": {"params": 68.2, "ap": 0.6184, "f1": 0.6624, "recall": 0.7449},
    "YOLOv9t": {"params": 2.0, "ap": 0.6105, "f1": 0.6597, "recall": 0.7550},
    "YOLOv9m": {"params": 20.1, "ap": 0.6858, "f1": 0.7267, "recall": 0.7992},
    "YOLOv10n": {"params": 2.3, "ap": 0.6830, "f1": 0.7080, "recall": 0.7679},
    "YOLOv10m": {"params": 15.4, "ap": 0.5797, "f1": 0.5987, "recall": 0.6654},
    "YOLOv10l": {"params": 24.4, "ap": 0.5890, "f1": 0.5602, "recall": 0.6147},
    "YOLOv10x": {"params": 29.5, "ap": 0.6291, "f1": 0.2011, "recall": 0.1431},
    "YOLOv11n": {"params": 2.6, "ap": 0.6427, "f1": 0.6846, "recall": 0.7743},
    "YOLOv11m": {"params": 20.1, "ap": 0.7173, "f1": 0.7587, "recall": 0.8339},
    "YOLOv11x": {"params": 56.9, "ap": 0.6527, "f1": 0.6957, "recall": 0.7887},
    "YOLO26n": {"params": 2.7, "ap": 0.7080, "f1": 0.7455, "recall": 0.8067},
    "YOLO26m": {"params": 20.1, "ap": 0.8126, "f1": 0.8297, "recall": 0.8549},
    "YOLO26x": {"params": 56.9, "ap": 0.8527, "f1": 0.8620, "recall": 0.8765},
    "YOLO-DAM-old": {"params": 17.5, "ap": 0.3800, "f1": 0.4840, "recall": 0.7279},
}

# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

def create_results_directory():
    """Create organized results directory structure"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(RESULTS_DIR, f"test_run_{timestamp}")

    os.makedirs(results_path, exist_ok=True)
    os.makedirs(os.path.join(results_path, "predictions"), exist_ok=True)
    os.makedirs(os.path.join(results_path, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(results_path, "logs"), exist_ok=True)

    print(f"[OK] Results directory created: {results_path}")
    return results_path

def load_ground_truth(labels_dir):
    """Load ground truth labels from YOLO format .txt files"""
    ground_truth = {}

    for label_file in os.listdir(labels_dir):
        if not label_file.endswith(".txt"):
            continue

        img_name = label_file.replace(".txt", "")
        label_path = os.path.join(labels_dir, label_file)

        boxes = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls, x, y, w, h = map(float, parts[:5])
                    boxes.append({
                        'class': int(cls),
                        'x': x, 'y': y, 'w': w, 'h': h
                    })

        ground_truth[img_name] = boxes

    print(f"[OK] Loaded {len(ground_truth)} ground truth labels")
    return ground_truth

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes (normalized YOLO format)"""
    x1_min = box1['x'] - box1['w']/2
    y1_min = box1['y'] - box1['h']/2
    x1_max = box1['x'] + box1['w']/2
    y1_max = box1['y'] + box1['h']/2

    x2_min = box2['x'] - box2['w']/2
    y2_min = box2['y'] - box2['h']/2
    x2_max = box2['x'] + box2['w']/2
    y2_max = box2['y'] + box2['h']/2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    box1_area = box1['w'] * box1['h']
    box2_area = box2['w'] * box2['h']
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0

def test_model(model_path, test_images_dir, conf_threshold=0.25):
    """Test model on test set and return predictions"""
    print(f"\n[TEST] Loading model: {model_path}")

    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return None

    print(f"[TEST] Running inference on {test_images_dir}")
    results = model.predict(
        source=test_images_dir,
        save=False,
        conf=conf_threshold,
        imgsz=640,
        verbose=False,
    )

    predictions = {}
    for result in results:
        img_name = Path(result.path).stem

        detections = []
        if result.boxes is not None:
            for box in result.boxes:
                x, y, w, h = box.xywhn[0].cpu().numpy()
                cls = int(box.cls[0].item())
                conf = float(box.conf[0].item())

                detections.append({
                    'class': cls,
                    'x': x, 'y': y, 'w': w, 'h': h,
                    'confidence': conf
                })

        predictions[img_name] = detections

    print(f"[OK] Generated {len(predictions)} predictions")
    return predictions

def calculate_metrics(ground_truth, predictions, iou_threshold=0.5):
    """Calculate precision, recall, F1, mAP by class"""
    metrics_by_class = defaultdict(lambda: {
        'tp': 0, 'fp': 0, 'fn': 0, 'confidence': []
    })

    # Count TP, FP, FN for each class
    for img_name, gt_boxes in ground_truth.items():
        pred_boxes = predictions.get(img_name, [])

        matched_pred = [False] * len(pred_boxes)

        for gt_box in gt_boxes:
            gt_class = gt_box['class']
            best_iou = 0
            best_idx = -1

            for pred_idx, pred_box in enumerate(pred_boxes):
                if matched_pred[pred_idx] or pred_box['class'] != gt_class:
                    continue

                iou = calculate_iou(gt_box, pred_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = pred_idx

            if best_iou >= iou_threshold and best_idx >= 0:
                metrics_by_class[gt_class]['tp'] += 1
                matched_pred[best_idx] = True
                metrics_by_class[gt_class]['confidence'].append(
                    pred_boxes[best_idx]['confidence']
                )
            else:
                metrics_by_class[gt_class]['fn'] += 1

        # Count FP
        for pred_idx, matched in enumerate(matched_pred):
            if not matched:
                pred_class = pred_boxes[pred_idx]['class']
                metrics_by_class[pred_class]['fp'] += 1

    # Calculate precision, recall, F1
    results = {
        'by_class': {},
        'overall': {}
    }

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for cls_id in range(10):
        m = metrics_by_class[cls_id]
        tp, fp, fn = m['tp'], m['fp'], m['fn']

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        results['by_class'][CLASS_NAMES[cls_id]] = {
            'tp': tp, 'fp': fp, 'fn': fn,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }

        total_tp += tp
        total_fp += fp
        total_fn += fn

    # Overall metrics
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) \
        if (overall_precision + overall_recall) > 0 else 0

    results['overall'] = {
        'precision': overall_precision,
        'recall': overall_recall,
        'f1': overall_f1,
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn,
    }

    return results

def create_excel_report(all_results, results_dir):
    """Create comprehensive Excel report with all results"""
    excel_path = os.path.join(results_dir, "MODEL_COMPARISON_RESULTS.xlsx")

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:

        # Sheet 1: Overall Metrics Comparison
        overall_data = []
        for model_name, metrics in all_results.items():
            if 'overall' in metrics:
                m = metrics['overall']
                overall_data.append({
                    'Model': model_name,
                    'Precision': m['precision'],
                    'Recall': m['recall'],
                    'F1': m['f1'],
                    'TP': m['tp'],
                    'FP': m['fp'],
                    'FN': m['fn'],
                })

        df_overall = pd.DataFrame(overall_data)
        df_overall = df_overall.sort_values('F1', ascending=False)
        df_overall.to_excel(writer, sheet_name='Overall Metrics', index=False)

        # Sheet 2: Class-level Metrics
        class_data = []
        for model_name, metrics in all_results.items():
            if 'by_class' in metrics:
                for class_name, class_metrics in metrics['by_class'].items():
                    class_data.append({
                        'Model': model_name,
                        'Class': class_name,
                        'Precision': class_metrics['precision'],
                        'Recall': class_metrics['recall'],
                        'F1': class_metrics['f1'],
                        'TP': class_metrics['tp'],
                        'FP': class_metrics['fp'],
                        'FN': class_metrics['fn'],
                    })

        df_class = pd.DataFrame(class_data)
        df_class.to_excel(writer, sheet_name='Class Metrics', index=False)

        # Sheet 3: Baseline Comparison
        baseline_data = []
        for model_name, baseline_metrics in BASELINE_MODELS.items():
            baseline_data.append({
                'Model': model_name,
                'Params (M)': baseline_metrics['params'],
                'Avg Precision': baseline_metrics['ap'],
                'Avg F1': baseline_metrics['f1'],
                'Final Recall': baseline_metrics['recall'],
                'Type': 'Baseline (Literature)',
            })

        for model_name, metrics in all_results.items():
            if 'overall' in metrics:
                m = metrics['overall']
                baseline_data.append({
                    'Model': model_name,
                    'Params (M)': 'N/A',
                    'Avg Precision': m['precision'],
                    'Avg F1': m['f1'],
                    'Final Recall': m['recall'],
                    'Type': 'Our Training',
                })

        df_baseline = pd.DataFrame(baseline_data)
        df_baseline.to_excel(writer, sheet_name='All Models Comparison', index=False)

        # Sheet 4: Training Summary
        summary_data = [
            ['Test Date', datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            ['Test Images Dir', TEST_IMAGES_DIR],
            ['Test Labels Dir', TEST_LABELS_DIR],
            ['', ''],
            ['Models Tested', len(all_results)],
            ['', ''],
            ['Key Findings:', ''],
            ['Best Precision', df_overall.iloc[0]['Model'] if len(df_overall) > 0 else 'N/A'],
            ['Best Recall', df_overall.loc[df_overall['Recall'].idxmax()]['Model'] if len(df_overall) > 0 else 'N/A'],
            ['Best F1', df_overall.iloc[0]['Model'] if len(df_overall) > 0 else 'N/A'],
        ]

        df_summary = pd.DataFrame(summary_data)
        df_summary.to_excel(writer, sheet_name='Summary', index=False, header=False)

    print(f"\n[OK] Excel report saved: {excel_path}")
    return excel_path

# ─────────────────────────────────────────────────────────────────────────────
# Main Testing Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL TESTING & COMPARISON")
    print("="*80)

    # Setup
    results_dir = create_results_directory()

    # Discover trained models
    print("\n[DISCOVER] Scanning for trained models...")
    models_to_test = discover_trained_models()

    if not models_to_test:
        print("[ERROR] No trained models found!")
        print("  Expected locations:")
        print("    - Training results: D:/Projekty/2022_01_BattPor/2025_12_Dresden/Training_Results/")
        print("    - YOLO-DAM models: D:/Projekty/2022_01_BattPor/2025_12_Dresden/Models/")
        return

    print(f"[OK] Found {len(models_to_test)} models to test:")
    for model_name, _, description in models_to_test:
        print(f"  - {model_name:30s} {description}")

    # Load ground truth
    print("\n[LOAD] Loading ground truth labels...")
    ground_truth = load_ground_truth(TEST_LABELS_DIR)

    # Test each model
    all_results = {}
    for model_name, model_path, description in models_to_test:
        print(f"\n{'='*80}")
        print(f"Testing: {model_name} - {description}")
        print(f"{'='*80}")

        # Check if model exists
        if not os.path.exists(model_path):
            print(f"[WARNING] Model not found: {model_path}")
            continue

        # Run predictions
        predictions = test_model(model_path, TEST_IMAGES_DIR)
        if predictions is None:
            print(f"[ERROR] Failed to test model: {model_name}")
            continue

        # Calculate metrics
        print(f"[METRICS] Calculating performance metrics...")
        metrics = calculate_metrics(ground_truth, predictions)
        all_results[model_name] = metrics

        # Print results
        m = metrics['overall']
        print(f"\n[RESULTS] {model_name}")
        print(f"  Precision: {m['precision']:.4f}")
        print(f"  Recall: {m['recall']:.4f}")
        print(f"  F1: {m['f1']:.4f}")
        print(f"  TP: {m['tp']}, FP: {m['fp']}, FN: {m['fn']}")

        # Save detailed metrics
        metrics_file = os.path.join(results_dir, "metrics", f"{model_name}_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        print(f"  [SAVED] {metrics_file}")

    # Create Excel report
    print(f"\n{'='*80}")
    print("GENERATING EXCEL REPORT")
    print(f"{'='*80}")
    excel_file = create_excel_report(all_results, results_dir)

    # Summary
    print(f"\n{'='*80}")
    print("TESTING COMPLETE")
    print(f"{'='*80}")
    print(f"\nResults Summary:")
    print(f"  Results Directory: {results_dir}")
    print(f"  Excel Report: {excel_file}")
    print(f"  Models Tested: {len(all_results)}")

    if all_results:
        print(f"\nPerformance Summary:")
        for model_name, metrics in all_results.items():
            m = metrics['overall']
            print(f"  {model_name:30} | Precision: {m['precision']:.4f} | Recall: {m['recall']:.4f} | F1: {m['f1']:.4f}")

    print(f"\n[COMPLETE] All results saved to: {results_dir}\n")

if __name__ == "__main__":
    main()
