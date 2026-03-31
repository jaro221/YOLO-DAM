"""
Train ALL Baseline YOLO Models from Scratch
Comprehensive benchmark training all standard YOLO architectures

Models to train:
- YOLOv8: nano, medium, extra (3 sizes)
- YOLOv9: tiny, medium (2 sizes)
- YOLOv10: nano, medium, large, extra (4 sizes)
- YOLOv11: nano, medium, extra (3 sizes)
- YOLO26: nano, medium, extra (3 sizes)

Total: 15 models, ~45-60 weeks on single GPU
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Paths
DATA_YAML = r"D:/Projekty/2022_01_BattPor/DATA_DEF/YOLOv11/data.yaml"
RESULTS_DIR = r"D:/Projekty/2022_01_BattPor/2025_12_Dresden/Training_Results"
LOG_DIR = r"D:/Projekty/2022_01_BattPor/2025_12_Dresden/Models"

# Training config
IMG_SIZE = 640
EPOCHS = 300
BATCH_SIZE = 4
DEVICE = 0
PATIENCE = 50

# Models to train: (model_id, description)
# Format: model_id is what ultralytics expects (e.g., 'yolov8n.pt')
BASELINE_MODELS = [
    # YOLOv8 Family
    ("yolov8n", "YOLOv8 Nano - Fast baseline"),
    ("yolov8m", "YOLOv8 Medium - Standard medium"),
    ("yolov8x", "YOLOv8 Extra - Large model"),

    # YOLOv9 Family
    ("yolov9t", "YOLOv9 Tiny - Very fast"),
    ("yolov9m", "YOLOv9 Medium - Improved YOLOv8"),

    # YOLOv10 Family
    ("yolov10n", "YOLOv10 Nano - Fast with anchor-free"),
    ("yolov10m", "YOLOv10 Medium - Improved detection"),
    ("yolov10l", "YOLOv10 Large - Better accuracy"),
    ("yolov10x", "YOLOv10 Extra - Best accuracy"),

    # YOLOv11 Family
    ("yolov11n", "YOLOv11 Nano - Latest nano"),
    ("yolov11m", "YOLOv11 Medium - Latest medium"),
    ("yolov11x", "YOLOv11 Extra - Latest extra"),

    # YOLO26 Family (Latest!)
    ("yolov26n", "YOLO26 Nano - Next gen nano"),
    ("yolov26m", "YOLO26 Medium - Next gen medium"),
    ("yolov26x", "YOLO26 Extra - Next gen extra (EXPECTED BEST)"),
]

# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

def log_msg(msg, level="INFO"):
    """Log with timestamp"""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{level:8s}] {msg}")

def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 80)
    print(text.center(80))
    print("=" * 80 + "\n")

def print_summary_table(results):
    """Print results table"""
    print("\n" + "=" * 80)
    print("TRAINING RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Model':<20} {'Status':<15} {'Best mAP':<12} {'Time Taken':<15}")
    print("-" * 80)

    for model_name, result in results.items():
        status = "[OK]" if result["success"] else "[FAILED]"
        map_score = f"{result['best_fitness']:.4f}" if result["best_fitness"] else "N/A"
        duration = result["duration"] if result["duration"] else "Unknown"
        print(f"{model_name:<20} {status:<15} {map_score:<12} {duration:<15}")

    print("=" * 80)

# ─────────────────────────────────────────────────────────────────────────────
# Training Function
# ─────────────────────────────────────────────────────────────────────────────

def train_model(model_id, description):
    """Train a single YOLO model from scratch (no pretrained weights)"""
    log_msg(f"Loading: {model_id} (from scratch - architecture only)", "LOAD")

    start_time = datetime.now()

    try:
        # Load model ARCHITECTURE ONLY (not pretrained weights)
        # Using .yaml instead of .pt initializes with random weights
        model = YOLO(f"{model_id}.yaml")

        log_msg(f"Training: {description}", "TRAIN")

        # Train
        results = model.train(
            data=DATA_YAML,
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            device=DEVICE,
            project=RESULTS_DIR,
            name=f"{model_id}_trained",
            patience=PATIENCE,
            save=True,
            plots=True,
            verbose=False,  # Less verbose for cleaner output
        )

        duration = datetime.now() - start_time
        hours = duration.total_seconds() / 3600
        log_msg(f"COMPLETE: {model_id} | mAP: {results.best_fitness:.4f} | Time: {hours:.1f}h", "SUCCESS")

        return {
            "success": True,
            "best_fitness": results.best_fitness,
            "duration": f"{hours:.1f}h",
        }

    except Exception as e:
        log_msg(f"ERROR: {model_id}: {str(e)}", "ERROR")
        duration = datetime.now() - start_time
        hours = duration.total_seconds() / 3600
        return {
            "success": False,
            "best_fitness": None,
            "duration": f"{hours:.1f}h",
        }

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print_header("COMPREHENSIVE BASELINE MODEL TRAINING")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    log_msg(f"Dataset: {DATA_YAML}", "CONFIG")
    log_msg(f"Results: {RESULTS_DIR}", "CONFIG")
    log_msg(f"Config: IMG={IMG_SIZE}, BATCH={BATCH_SIZE}, EPOCHS={EPOCHS}", "CONFIG")
    log_msg(f"Total models to train: {len(BASELINE_MODELS)}", "CONFIG")
    log_msg(f"Estimated time: {len(BASELINE_MODELS) * 3.5:.0f}-{len(BASELINE_MODELS) * 4.5:.0f} weeks on single GPU", "CONFIG")

    print("\nModels to train:")
    for model_id, description in BASELINE_MODELS:
        print(f"  • {model_id:<15} {description}")

    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)

    results = {}
    total_start = datetime.now()

    for idx, (model_id, description) in enumerate(BASELINE_MODELS, 1):
        print("\n" + "-" * 80)
        print(f"[{idx}/{len(BASELINE_MODELS)}] {model_id}")
        print("-" * 80)

        result = train_model(model_id, description)
        results[model_id] = result

    # Save results to JSON
    total_duration = datetime.now() - total_start
    total_hours = total_duration.total_seconds() / 3600

    training_log = {
        "start_time": total_start.isoformat(),
        "end_time": datetime.now().isoformat(),
        "total_hours": total_hours,
        "total_models": len(BASELINE_MODELS),
        "results": results,
    }

    log_file = os.path.join(LOG_DIR, "baseline_training_log.json")
    with open(log_file, 'w') as f:
        json.dump(training_log, f, indent=2)

    log_msg(f"Training log saved: {log_file}", "SAVE")

    # Print summary
    print_summary_table(results)

    # Summary statistics
    successful = sum(1 for r in results.values() if r["success"])
    failed = len(results) - successful

    print(f"\nSummary:")
    print(f"  ✓ Successful: {successful}/{len(results)}")
    print(f"  ✗ Failed: {failed}/{len(results)}")
    print(f"  Total time: {total_hours:.1f} hours ({total_hours / 24:.1f} days)")

    # Ranking
    successful_results = [(name, r) for name, r in results.items() if r["success"]]
    if successful_results:
        successful_results.sort(key=lambda x: x[1]["best_fitness"], reverse=True)

        print(f"\nRanking by mAP:")
        for idx, (name, result) in enumerate(successful_results, 1):
            print(f"  {idx}. {name:<15} mAP: {result['best_fitness']:.4f}")

        print(f"\nBest Model: {successful_results[0][0]} (mAP: {successful_results[0][1]['best_fitness']:.4f})")

    print("\n" + "=" * 80)
    print("NEXT STEP: Evaluate all models")
    print("=" * 80)
    print(f"\nD:\\Programy\\anaconda3\\envs\\TF_3_8\\python.exe COMPREHENSIVE_TEST_AND_COMPARE.py\n")

if __name__ == "__main__":
    main()
