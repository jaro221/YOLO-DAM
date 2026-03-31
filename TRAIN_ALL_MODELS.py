"""
Master Training Orchestrator
Trains all YOLO models (standard + custom YOLO-DAM) from scratch
Compares performance across 6 configurations

Models:
1. Standard Ultralytics YOLOv11m
2. Standard Ultralytics YOLOv26m
3. Standard Ultralytics YOLOv26x
4. Custom YOLO-DAM Config A (width=1.0, random init)
5. Custom YOLO-DAM Config B (width=1.0, v26 pretrained)
6. Custom YOLO-DAM Config C (width=0.6, v26 pretrained, old M2M)
"""

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

# Paths
PROJECT_ROOT = r"D:/Projekty/2022_01_BattPor/2025_12_Dresden/VSCODE"
DATA_DIR = r"D:/Projekty/2022_01_BattPor/2025_12_Dresden/YOLOv8"
MODELS_DIR = r"D:/Projekty/2022_01_BattPor/2025_12_Dresden/Models"
RESULTS_DIR = r"D:/Projekty/2022_01_BattPor/2025_12_Dresden/Training_Results"
PYTHON_EXE = r"D:\Programy\anaconda3\envs\TF_3_8\python.exe"

# Create data.yaml for standard YOLO training
DATA_YAML = r"D:/Projekty/2022_01_BattPor/DATA_DEF/YOLOv11/data.yaml"

# Training config
IMG_SIZE = 640
EPOCHS = 300
BATCH_SIZE = 4
DEVICE = 0  # GPU device

# Defect classes
CLASSES = [
    "dirt", "dent", "crack", "corrosion",
    "rust", "stain", "scratch", "burn",
    "weld", "missing"
]

def ensure_dirs():
    """Create necessary directories"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"[OK] Result directories ready: {RESULTS_DIR}")

def log_message(msg, status="INFO"):
    """Log with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{status}] {msg}")

def train_ultralytics_model(model_name, epochs=EPOCHS):
    """Train standard Ultralytics YOLO model"""
    log_message(f"Starting training: {model_name}", "TRAIN")

    try:
        from ultralytics import YOLO

        # Load pretrained model
        model = YOLO(f"{model_name}.pt")

        # Train on custom data
        results = model.train(
            data=DATA_YAML,
            epochs=epochs,
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            device=DEVICE,
            project=RESULTS_DIR,
            name=f"{model_name}_trained",
            patience=50,  # Early stopping
            save=True,
            plots=True,
        )

        log_message(f"Completed: {model_name} | Best mAP: {results.best_fitness:.4f}", "SUCCESS")
        return True

    except Exception as e:
        log_message(f"Error training {model_name}: {e}", "ERROR")
        return False

def train_yolo_dam_config_a():
    """
    Config A: Random initialization (width=1.0, depth=1.0)
    Measures pure architecture benefit without pre-training
    """
    log_message("Config A: Training YOLO-DAM with RANDOM initialization", "TRAIN")

    # Temporarily modify YOLO_DAM_train.py to disable weight loading
    train_file = os.path.join(PROJECT_ROOT, "YOLO_DAM_train.py")

    try:
        with open(train_file, 'r') as f:
            content = f.read()

        # Comment out weight loading
        modified = content.replace(
            "print(f\"Loading merged weights: {WEIGHTS_PATH}\")",
            "# DISABLED for Config A\n    # print(f\"Loading merged weights: {WEIGHTS_PATH}\")"
        ).replace(
            "model_dam.load_weights(WEIGHTS_PATH)",
            "# model_dam.load_weights(WEIGHTS_PATH)  # CONFIG A: Random init"
        )

        # Update log path
        modified = modified.replace(
            'LOG_PATH = r"D:/Projekty/2022_01_BattPor/2025_12_Dresden/Models/train_log_dam.txt"',
            'LOG_PATH = r"D:/Projekty/2022_01_BattPor/2025_12_Dresden/Models/train_log_dam_CONFIG_A.txt"'
        )

        with open(train_file, 'w') as f:
            f.write(modified)

        # Run training
        result = subprocess.run(
            [PYTHON_EXE, "YOLO_DAM_train.py"],
            cwd=PROJECT_ROOT,
            capture_output=False
        )

        # Save model with config name
        best_model = os.path.join(MODELS_DIR, "YOLODAM_best_e300.h5")
        config_a_model = os.path.join(MODELS_DIR, "YOLODAM_CONFIG_A_random.h5")
        if os.path.exists(best_model):
            shutil.copy(best_model, config_a_model)
            log_message(f"Saved Config A model: {config_a_model}", "SUCCESS")

        return result.returncode == 0

    except Exception as e:
        log_message(f"Error in Config A training: {e}", "ERROR")
        return False

def train_yolo_dam_config_b():
    """
    Config B: v26 Pre-trained (width=1.0, depth=1.0)
    Measures combined benefit of pre-training + larger architecture
    """
    log_message("Config B: Training YOLO-DAM with v26 PRE-TRAINED weights", "TRAIN")

    train_file = os.path.join(PROJECT_ROOT, "YOLO_DAM_train.py")

    try:
        with open(train_file, 'r') as f:
            content = f.read()

        # Restore weight loading
        modified = content.replace(
            "# DISABLED for Config A\n    # print(f\"Loading merged weights: {WEIGHTS_PATH}\")",
            "print(f\"Loading merged weights: {WEIGHTS_PATH}\")"
        ).replace(
            "# model_dam.load_weights(WEIGHTS_PATH)  # CONFIG A: Random init",
            "model_dam.load_weights(WEIGHTS_PATH)"
        )

        # Update log path
        modified = modified.replace(
            'LOG_PATH = r"D:/Projekty/2022_01_BattPor/2025_12_Dresden/Models/train_log_dam_CONFIG_A.txt"',
            'LOG_PATH = r"D:/Projekty/2022_01_BattPor/2025_12_Dresden/Models/train_log_dam_CONFIG_B.txt"'
        )

        with open(train_file, 'w') as f:
            f.write(modified)

        # Run training
        result = subprocess.run(
            [PYTHON_EXE, "YOLO_DAM_train.py"],
            cwd=PROJECT_ROOT,
            capture_output=False
        )

        # Save model with config name
        best_model = os.path.join(MODELS_DIR, "YOLODAM_best_e300.h5")
        config_b_model = os.path.join(MODELS_DIR, "YOLODAM_CONFIG_B_v26.h5")
        if os.path.exists(best_model):
            shutil.copy(best_model, config_b_model)
            log_message(f"Saved Config B model: {config_b_model}", "SUCCESS")

        return result.returncode == 0

    except Exception as e:
        log_message(f"Error in Config B training: {e}", "ERROR")
        return False

def train_yolo_dam_config_c():
    """
    Config C: Old Model (width=0.6, depth=0.5, M2M radius=1)
    Baseline reference to show improvement from all fixes
    """
    log_message("Config C: Training OLD YOLO-DAM model (baseline)", "TRAIN")

    yolo_dam_file = os.path.join(PROJECT_ROOT, "YOLO_DAM.py")
    dataset_file = os.path.join(PROJECT_ROOT, "YOLO_DAM_dataset.py")
    train_file = os.path.join(PROJECT_ROOT, "YOLO_DAM_train.py")

    try:
        # Revert YOLO_DAM.py to old config
        with open(yolo_dam_file, 'r') as f:
            content = f.read()

        modified = content.replace(
            "model = build_yolo_model(width=1.0, depth=1.0)",
            "model = build_yolo_model(width=0.6, depth=0.5)"
        )

        with open(yolo_dam_file, 'w') as f:
            f.write(modified)

        # Revert YOLO_DAM_dataset.py to old M2M radius
        with open(dataset_file, 'r') as f:
            content = f.read()

        modified = content.replace(
            "radius = 0  # All objects assigned to single cell",
            "radius = 1 if max_span > 3.0 else 0  # OLD behavior"
        )

        with open(dataset_file, 'w') as f:
            f.write(modified)

        # Update train log path
        with open(train_file, 'r') as f:
            content = f.read()

        modified = content.replace(
            'LOG_PATH = r"D:/Projekty/2022_01_BattPor/2025_12_Dresden/Models/train_log_dam_CONFIG_B.txt"',
            'LOG_PATH = r"D:/Projekty/2022_01_BattPor/2025_12_Dresden/Models/train_log_dam_CONFIG_C.txt"'
        )

        with open(train_file, 'w') as f:
            f.write(modified)

        # Run training
        result = subprocess.run(
            [PYTHON_EXE, "YOLO_DAM_train.py"],
            cwd=PROJECT_ROOT,
            capture_output=False
        )

        # Save model with config name
        best_model = os.path.join(MODELS_DIR, "YOLODAM_best_e300.h5")
        config_c_model = os.path.join(MODELS_DIR, "YOLODAM_CONFIG_C_old.h5")
        if os.path.exists(best_model):
            shutil.copy(best_model, config_c_model)
            log_message(f"Saved Config C model: {config_c_model}", "SUCCESS")

        return result.returncode == 0

    except Exception as e:
        log_message(f"Error in Config C training: {e}", "ERROR")
        return False

def restore_yolo_dam_config_b():
    """Restore YOLO_DAM files to Config B (final state)"""
    log_message("Restoring YOLO-DAM to Config B (final)", "INFO")

    yolo_dam_file = os.path.join(PROJECT_ROOT, "YOLO_DAM.py")
    dataset_file = os.path.join(PROJECT_ROOT, "YOLO_DAM_dataset.py")

    try:
        # Restore YOLO_DAM.py to new config
        with open(yolo_dam_file, 'r') as f:
            content = f.read()

        modified = content.replace(
            "model = build_yolo_model(width=0.6, depth=0.5)",
            "model = build_yolo_model(width=1.0, depth=1.0)"
        )

        with open(yolo_dam_file, 'w') as f:
            f.write(modified)

        # Restore YOLO_DAM_dataset.py to fixed M2M radius
        with open(dataset_file, 'r') as f:
            content = f.read()

        modified = content.replace(
            "radius = 1 if max_span > 3.0 else 0  # OLD behavior",
            "radius = 0  # All objects assigned to single cell"
        )

        with open(dataset_file, 'w') as f:
            f.write(modified)

        log_message("Config B restored", "OK")

    except Exception as e:
        log_message(f"Error restoring Config B: {e}", "ERROR")

def print_training_plan():
    """Display training plan"""
    print("""
================================================================================
COMPREHENSIVE TRAINING PLAN - All Models from Scratch
================================================================================

STANDARD ULTRALYTICS MODELS (3 models, ~3-4 weeks each):
  1. YOLOv11m  - Fast, lightweight baseline
  2. YOLOv26m  - Medium, standard comparison
  3. YOLOv26x  - Extra-large, best performance expected

CUSTOM YOLO-DAM MODELS (3 configurations, ~3-4 weeks each):
  4. Config A (Random)       - width=1.0, random init, M2M radius=0
  5. Config B (v26 Pre-trained) - width=1.0, v26 backbone, M2M radius=0
  6. Config C (Old)          - width=0.6, v26 backbone, M2M radius=1

TOTAL TRAINING TIME: ~20-24 weeks on RTX3090 (6 models × 3-4 weeks each)

COMPARISON METRICS:
  - Precision, Recall, F1 by class (10 defect types)
  - Overall mAP@0.5 and mAP@0.5:0.95
  - Training convergence speed
  - Model size vs performance trade-off

RESULTS STORED:
  Models: {MODELS_DIR}
  Logs: {RESULTS_DIR}
  Final Report: COMPREHENSIVE_TEST_AND_COMPARE.py output (Excel)

================================================================================
EXECUTION ORDER:
================================================================================

Option A: Sequential (One model at a time)
  - Choose 1 or 6 models
  - Run independently
  - Lower GPU utilization but simpler management

Option B: Parallel (Multiple GPU cards)
  - Distribute models across GPUs
  - Faster overall completion time
  - Requires multiple GPUs

START TRAINING (Sequential - select models):
  [ ] 1. Standard YOLOv11m
  [ ] 2. Standard YOLOv26m
  [ ] 3. Standard YOLOv26x
  [ ] 4. YOLO-DAM Config A (random)
  [ ] 5. YOLO-DAM Config B (v26 pre-trained) - RECOMMENDED
  [ ] 6. YOLO-DAM Config C (old) - OPTIONAL

================================================================================
    """.format(MODELS_DIR=MODELS_DIR, RESULTS_DIR=RESULTS_DIR))

def main():
    """Main orchestrator"""
    print("\n")
    print("=" * 80)
    print("MASTER TRAINING ORCHESTRATOR")
    print("=" * 80)

    ensure_dirs()
    print_training_plan()

    log_message("Training environment ready", "OK")
    log_message(f"Python: {PYTHON_EXE}", "INFO")
    log_message(f"Data: {DATA_YAML}", "INFO")
    log_message(f"Results: {RESULTS_DIR}", "INFO")

    training_log = {
        "start_time": datetime.now().isoformat(),
        "models": {},
        "end_time": None
    }

    # Option: Train all models in sequence
    # Comment out sections to skip specific models

    print("\n" + "=" * 80)
    print("STANDARD ULTRALYTICS MODELS")
    print("=" * 80 + "\n")

    # Train standard models
    standard_models = ["yolov11m", "yolov26m", "yolov26x"]
    for model_name in standard_models:
        success = train_ultralytics_model(model_name, epochs=EPOCHS)
        training_log["models"][model_name] = {"success": success}
        print()

    print("\n" + "=" * 80)
    print("CUSTOM YOLO-DAM MODELS - ABLATION STUDY")
    print("=" * 80 + "\n")

    # Train YOLO-DAM configs
    print("PHASE 1: Config A (Random Initialization)")
    print("-" * 80)
    success_a = train_yolo_dam_config_a()
    training_log["models"]["YOLO_DAM_ConfigA"] = {"success": success_a}
    print()

    print("\nPHASE 2: Config B (v26 Pre-trained)")
    print("-" * 80)
    success_b = train_yolo_dam_config_b()
    training_log["models"]["YOLO_DAM_ConfigB"] = {"success": success_b}
    print()

    print("\nPHASE 3: Config C (Old Model Baseline)")
    print("-" * 80)
    success_c = train_yolo_dam_config_c()
    training_log["models"]["YOLO_DAM_ConfigC"] = {"success": success_c}
    print()

    # Restore Config B as final state
    restore_yolo_dam_config_b()

    # Summary
    training_log["end_time"] = datetime.now().isoformat()

    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)

    for model, result in training_log["models"].items():
        status = "[OK]" if result["success"] else "[FAILED]"
        print(f"{status} {model}")

    # Save training log
    log_file = os.path.join(RESULTS_DIR, "training_log.json")
    with open(log_file, 'w') as f:
        json.dump(training_log, f, indent=2)

    log_message(f"Training log saved: {log_file}", "OK")

    print("\n" + "=" * 80)
    print("NEXT STEP: Run Comprehensive Evaluation")
    print("=" * 80)
    print(f"\nD:\\Programy\\anaconda3\\envs\\TF_3_8\\python.exe COMPREHENSIVE_TEST_AND_COMPARE.py\n")

if __name__ == "__main__":
    main()
