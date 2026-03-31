"""
Train YOLO-DAM Ablation Study Configurations
Trains 3 configurations to isolate benefit of each component:
  - Config A: Random init (shows architecture benefit)
  - Config B: v26 pre-trained (shows combined benefit)
  - Config C: Old model (baseline reference)
"""

import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

# Paths
PROJECT_ROOT = r"D:/Projekty/2022_01_BattPor/2025_12_Dresden/VSCODE"
MODELS_DIR = r"D:/Projekty/2022_01_BattPor/2025_12_Dresden/Models"
PYTHON_EXE = r"D:\Programy\anaconda3\envs\TF_3_8\python.exe"

def log_msg(msg, level="INFO"):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{level:8s}] {msg}")

def backup_file(filepath, suffix):
    """Backup file with suffix"""
    path = Path(filepath)
    backup = path.parent / f"{path.stem}_backup_{suffix}{path.suffix}"
    shutil.copy(filepath, backup)
    log_msg(f"Backed up: {filepath}", "BACKUP")
    return backup

def restore_file(filepath, backup_path):
    """Restore from backup"""
    shutil.copy(backup_path, filepath)
    log_msg(f"Restored: {filepath}", "RESTORE")

def modify_for_config_a():
    """Modify training script for Config A (random init)"""
    train_file = os.path.join(PROJECT_ROOT, "YOLO_DAM_train.py")

    with open(train_file, 'r') as f:
        content = f.read()

    # Comment out weight loading
    modified = content.replace(
        "print(f\"Loading merged weights: {WEIGHTS_PATH}\")",
        "# [CONFIG A - DISABLED] print(f\"Loading merged weights: {WEIGHTS_PATH}\")"
    ).replace(
        "    model_dam.load_weights(WEIGHTS_PATH)",
        "    # [CONFIG A] model_dam.load_weights(WEIGHTS_PATH)  # Random init"
    )

    # Update log path
    modified = modified.replace(
        'LOG_PATH = r"D:/Projekty/2022_01_BattPor/2025_12_Dresden/Models/train_log_dam.txt"',
        'LOG_PATH = r"D:/Projekty/2022_01_BattPor/2025_12_Dresden/Models/train_log_dam_CONFIG_A.txt"'
    )

    with open(train_file, 'w') as f:
        f.write(modified)

    log_msg("Modified for Config A (weights DISABLED)", "CONFIG")

def modify_for_config_b():
    """Restore for Config B (v26 pre-trained)"""
    train_file = os.path.join(PROJECT_ROOT, "YOLO_DAM_train.py")

    with open(train_file, 'r') as f:
        content = f.read()

    # Restore weight loading
    modified = content.replace(
        "# [CONFIG A - DISABLED] print(f\"Loading merged weights: {WEIGHTS_PATH}\")",
        "print(f\"Loading merged weights: {WEIGHTS_PATH}\")"
    ).replace(
        "    # [CONFIG A] model_dam.load_weights(WEIGHTS_PATH)  # Random init",
        "    model_dam.load_weights(WEIGHTS_PATH)"
    )

    # Update log path
    modified = modified.replace(
        'LOG_PATH = r"D:/Projekty/2022_01_BattPor/2025_12_Dresden/Models/train_log_dam_CONFIG_A.txt"',
        'LOG_PATH = r"D:/Projekty/2022_01_BattPor/2025_12_Dresden/Models/train_log_dam_CONFIG_B.txt"'
    )

    with open(train_file, 'w') as f:
        f.write(modified)

    log_msg("Modified for Config B (weights ENABLED)", "CONFIG")

def modify_for_config_c():
    """Modify for Config C (old model)"""
    yolo_dam_file = os.path.join(PROJECT_ROOT, "YOLO_DAM.py")
    dataset_file = os.path.join(PROJECT_ROOT, "YOLO_DAM_dataset.py")
    train_file = os.path.join(PROJECT_ROOT, "YOLO_DAM_train.py")

    # Revert model to old config
    with open(yolo_dam_file, 'r') as f:
        content = f.read()

    modified = content.replace(
        "model = build_yolo_model(width=1.0, depth=1.0)",
        "model = build_yolo_model(width=0.6, depth=0.5)  # [CONFIG C] Old"
    )

    with open(yolo_dam_file, 'w') as f:
        f.write(modified)

    log_msg("Reverted YOLO_DAM.py to old config (width=0.6, depth=0.5)", "CONFIG")

    # Revert dataset M2M radius
    with open(dataset_file, 'r') as f:
        content = f.read()

    modified = content.replace(
        "radius = 0  # All objects assigned to single cell",
        "radius = 1 if max_span > 3.0 else 0  # [CONFIG C] Old M2M"
    )

    with open(dataset_file, 'w') as f:
        f.write(modified)

    log_msg("Reverted M2M radius to old behavior (adaptive 0-1)", "CONFIG")

    # Update log path
    with open(train_file, 'r') as f:
        content = f.read()

    modified = content.replace(
        'LOG_PATH = r"D:/Projekty/2022_01_BattPor/2025_12_Dresden/Models/train_log_dam_CONFIG_B.txt"',
        'LOG_PATH = r"D:/Projekty/2022_01_BattPor/2025_12_Dresden/Models/train_log_dam_CONFIG_C.txt"'
    )

    with open(train_file, 'w') as f:
        f.write(modified)

def restore_config_b():
    """Restore to Config B (final state)"""
    yolo_dam_file = os.path.join(PROJECT_ROOT, "YOLO_DAM.py")
    dataset_file = os.path.join(PROJECT_ROOT, "YOLO_DAM_dataset.py")

    # Restore model
    with open(yolo_dam_file, 'r') as f:
        content = f.read()

    modified = content.replace(
        "model = build_yolo_model(width=0.6, depth=0.5)  # [CONFIG C] Old",
        "model = build_yolo_model(width=1.0, depth=1.0)"
    )

    with open(yolo_dam_file, 'w') as f:
        f.write(modified)

    log_msg("Restored YOLO_DAM.py to new config (width=1.0, depth=1.0)", "RESTORE")

    # Restore dataset
    with open(dataset_file, 'r') as f:
        content = f.read()

    modified = content.replace(
        "radius = 1 if max_span > 3.0 else 0  # [CONFIG C] Old M2M",
        "radius = 0  # All objects assigned to single cell"
    )

    with open(dataset_file, 'w') as f:
        f.write(modified)

    log_msg("Restored M2M radius to fixed behavior (0 always)", "RESTORE")

def run_training():
    """Execute training"""
    log_msg(f"Working directory: {PROJECT_ROOT}", "INFO")

    result = subprocess.run(
        [PYTHON_EXE, "YOLO_DAM_train.py"],
        cwd=PROJECT_ROOT,
        capture_output=False
    )

    return result.returncode == 0

def save_model(config_name):
    """Save trained model with config name"""
    best_model = os.path.join(MODELS_DIR, "YOLODAM_best_e300.h5")
    config_model = os.path.join(MODELS_DIR, f"YOLODAM_{config_name}.h5")

    if os.path.exists(best_model):
        shutil.copy(best_model, config_model)
        size_mb = os.path.getsize(config_model) / (1024 ** 2)
        log_msg(f"Saved model: {config_model} ({size_mb:.1f} MB)", "SAVE")
        return True

    log_msg(f"Best model not found: {best_model}", "ERROR")
    return False

def main():
    print("\n" + "=" * 80)
    print("YOLO-DAM ABLATION STUDY")
    print("Training 3 configurations to isolate component benefits")
    print("=" * 80 + "\n")

    os.makedirs(MODELS_DIR, exist_ok=True)

    results = {}

    # =========================================================================
    # PHASE 1: CONFIG A (Random Initialization)
    # =========================================================================
    print("\n" + "-" * 80)
    print("PHASE 1: CONFIG A - Random Initialization (width=1.0, depth=1.0)")
    print("-" * 80)
    print("Purpose: Measure pure architecture benefit without pre-training")
    print("Expected: Precision 45-55%, Recall 78-82%")
    print()

    modify_for_config_a()

    log_msg("Starting Config A training...", "TRAIN")
    success_a = run_training()

    if success_a:
        save_model("CONFIG_A_random")
        log_msg("Config A training completed", "SUCCESS")
    else:
        log_msg("Config A training failed", "ERROR")

    results["Config_A"] = success_a

    # =========================================================================
    # PHASE 2: CONFIG B (v26 Pre-trained)
    # =========================================================================
    print("\n" + "-" * 80)
    print("PHASE 2: CONFIG B - v26 Pre-trained (width=1.0, depth=1.0)")
    print("-" * 80)
    print("Purpose: Measure combined benefit of pre-training + larger architecture")
    print("Expected: Precision 70-75%, Recall 82-85%")
    print()

    modify_for_config_b()

    log_msg("Starting Config B training...", "TRAIN")
    success_b = run_training()

    if success_b:
        save_model("CONFIG_B_v26_pretrained")
        log_msg("Config B training completed", "SUCCESS")
    else:
        log_msg("Config B training failed", "ERROR")

    results["Config_B"] = success_b

    # =========================================================================
    # PHASE 3: CONFIG C (Old Model Baseline)
    # =========================================================================
    print("\n" + "-" * 80)
    print("PHASE 3: CONFIG C - Old Model Baseline (width=0.6, depth=0.5)")
    print("-" * 80)
    print("Purpose: Baseline reference to show improvement from all fixes")
    print("Expected: Precision ~40%, Recall ~73%")
    print()

    modify_for_config_c()

    log_msg("Starting Config C training...", "TRAIN")
    success_c = run_training()

    if success_c:
        save_model("CONFIG_C_old_baseline")
        log_msg("Config C training completed", "SUCCESS")
    else:
        log_msg("Config C training failed", "ERROR")

    results["Config_C"] = success_c

    # =========================================================================
    # RESTORE CONFIG B (Final State)
    # =========================================================================
    print("\n" + "-" * 80)
    print("RESTORATION")
    print("-" * 80)

    restore_config_b()

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("ABLATION STUDY SUMMARY")
    print("=" * 80)

    for config, success in results.items():
        status = "[OK]" if success else "[FAILED]"
        print(f"{status} {config}")

    print("\n" + "=" * 80)
    print("SAVED MODELS:")
    print("=" * 80)

    models_dir = Path(MODELS_DIR)
    for model_file in sorted(models_dir.glob("YOLODAM_CONFIG_*.h5")):
        size_mb = model_file.stat().st_size / (1024 ** 2)
        print(f"  {model_file.name:45s} ({size_mb:6.1f} MB)")

    print("\n" + "=" * 80)
    print("NEXT STEP: Compare all trained models")
    print("=" * 80)
    print(f"\nD:\\Programy\\anaconda3\\envs\\TF_3_8\\python.exe COMPREHENSIVE_TEST_AND_COMPARE.py\n")

if __name__ == "__main__":
    main()
