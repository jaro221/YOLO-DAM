"""
Train Standard Ultralytics YOLO Models
Trains YOLOv11m, YOLOv26m, YOLOv26x from scratch on defect detection dataset
"""

import os
from datetime import datetime
from ultralytics import YOLO

# Paths
DATA_YAML = r"D:/Projekty/2022_01_BattPor/DATA_DEF/YOLOv11/data.yaml"
RESULTS_DIR = r"D:/Projekty/2022_01_BattPor/2025_12_Dresden/Training_Results"

# Config
IMG_SIZE = 640
EPOCHS = 300
BATCH_SIZE = 4
DEVICE = 0
PATIENCE = 50  # Early stopping

# Models to train
MODELS = ["yolov11m", "yolov26m", "yolov26x"]

def log_msg(msg, level="INFO"):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{level:8s}] {msg}")

def train_model(model_name):
    """Train single YOLO model from scratch (no pretrained weights)"""
    log_msg(f"Loading {model_name} (from scratch - random init)...", "LOAD")

    try:
        # Use .yaml to load architecture only (random weights)
        # NOT .pt which would load pretrained COCO weights
        model = YOLO(f"{model_name}.yaml")

        log_msg(f"Starting training: {model_name}", "TRAIN")

        results = model.train(
            data=DATA_YAML,
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            device=DEVICE,
            project=RESULTS_DIR,
            name=f"{model_name}_trained",
            patience=PATIENCE,
            save=True,
            plots=True,
            verbose=True,
        )

        # Log results
        log_msg(f"Training complete: {model_name}", "SUCCESS")
        log_msg(f"  Best mAP: {results.best_fitness:.4f}", "RESULT")
        log_msg(f"  Results: {RESULTS_DIR}/{model_name}_trained/", "RESULT")

        return True

    except Exception as e:
        log_msg(f"Error: {e}", "ERROR")
        return False

def main():
    print("\n" + "=" * 80)
    print("TRAINING STANDARD ULTRALYTICS MODELS")
    print("=" * 80)
    print(f"Dataset: {DATA_YAML}")
    print(f"Results: {RESULTS_DIR}")
    print(f"Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | Device: GPU{DEVICE}")
    print("=" * 80 + "\n")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    results = {}

    for model_name in MODELS:
        log_msg(f"", "")
        log_msg("=" * 60, "")
        log_msg(f"Model: {model_name}", "")
        log_msg("=" * 60, "")

        success = train_model(model_name)
        results[model_name] = success

        print()

    # Summary
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)

    for model_name, success in results.items():
        status = "[OK]" if success else "[FAILED]"
        print(f"{status} {model_name}")

    print("\n" + "=" * 80)
    print("All models saved in:")
    print(f"  {RESULTS_DIR}/")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
