"""
Parallel Training Orchestrator
Run multiple trainings simultaneously on different GPUs

Requirements:
  - 2+ GPUs for parallel training
  - Each GPU trains one model independently
  - Speeds up total training time significantly
"""

import os
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

PYTHON_EXE = r"D:\Programy\anaconda3\envs\TF_3_8\python.exe"
PROJECT_ROOT = r"D:/Projekty/2022_01_BattPor/2025_12_Dresden/VSCODE"

# GPU configuration
# Change these based on your setup
GPU_CONFIG = {
    "baseline_models": 0,      # Use GPU 0 for baseline models
    "yolo_dam_ablation": 1,    # Use GPU 1 for YOLO-DAM ablation
}

# Available GPU setup presets
PARALLEL_SETUPS = {
    "single_gpu": {
        "description": "Single GPU (RTX3090) - Sequential training only",
        "config": {
            "baseline_models": 0,
            "yolo_dam_ablation": 0,  # Same GPU - trains sequentially
        },
        "note": "Run baseline_models first, then yolo_dam_ablation"
    },
    "dual_gpu": {
        "description": "2 GPUs (e.g., 2x RTX3090 or RTX3090 + RTX4090)",
        "config": {
            "baseline_models": 0,      # GPU 0
            "yolo_dam_ablation": 1,    # GPU 1
        },
        "note": "Both run in parallel - total time ~45-60 weeks"
    },
    "quad_gpu": {
        "description": "4 GPUs - Train 4 models simultaneously",
        "config": {
            "yolov8_family": 0,
            "yolov9_family": 1,
            "yolov10_family": 2,
            "yolo_dam_ablation": 3,
        },
        "note": "Advanced - requires custom script"
    },
    "6gpu": {
        "description": "6 GPUs - Train all baseline + YOLO-DAM in parallel",
        "config": {
            "all_baseline": [0, 1, 2, 3, 4],  # 15 models on 5 GPUs
            "yolo_dam": 5,
        },
        "note": "Total time: ~9-12 weeks instead of 55-70"
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

def log_msg(msg, level="INFO"):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{level:8s}] {msg}")

def print_header(text):
    print("\n" + "=" * 80)
    print(text.center(80))
    print("=" * 80 + "\n")

def check_gpu_availability():
    """Check available GPUs"""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        num_gpus = len(gpus)

        log_msg(f"GPUs detected: {num_gpus}", "GPU")

        if num_gpus == 0:
            log_msg("WARNING: No GPUs detected! Training will be very slow.", "WARNING")
            return 0

        for i, gpu in enumerate(gpus):
            log_msg(f"  GPU {i}: {gpu}", "GPU")

        return num_gpus

    except Exception as e:
        log_msg(f"Error checking GPUs: {e}", "ERROR")
        return 0

def run_training(name, script, gpu_id, log_file):
    """Run training in separate thread"""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    log_msg(f"Starting {name} on GPU {gpu_id}...", "START")

    try:
        with open(log_file, 'w') as f:
            result = subprocess.run(
                [PYTHON_EXE, script],
                cwd=PROJECT_ROOT,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True
            )

        if result.returncode == 0:
            log_msg(f"Completed {name}", "SUCCESS")
        else:
            log_msg(f"Failed {name} (exit code {result.returncode})", "ERROR")

        return result.returncode == 0

    except Exception as e:
        log_msg(f"Error in {name}: {e}", "ERROR")
        return False

# ─────────────────────────────────────────────────────────────────────────────
# Training Scenarios
# ─────────────────────────────────────────────────────────────────────────────

def scenario_single_gpu():
    """Single GPU - Sequential training"""
    print_header("SCENARIO: Single GPU (Sequential)")

    print("""
You have 1 GPU (RTX3090, RTX4090, etc.)

Training plan:
  Phase 1: YOLO-DAM Ablation (9-12 weeks) - FASTER TO START
  Phase 2: Baseline Models (45-60 weeks)
  Total: 54-72 weeks

Recommended order:
  1. Start with YOLO-DAM (smaller, faster results)
  2. Then start baseline models
  3. Or do just YOLO-DAM for initial insights

Commands:
  Terminal 1 (GPU 0):
    D:\\Programy\\anaconda3\\envs\\TF_3_8\\python.exe TRAIN_YOLO_DAM_ABLATION.py
    (let run 9-12 weeks)

  After first completes, Terminal 1:
    D:\\Programy\\anaconda3\\envs\\TF_3_8\\python.exe TRAIN_BASELINE_MODELS.py
    (let run 45-60 weeks)

Result:
  All models trained sequentially on GPU 0
  Takes ~55-72 weeks total
  """)

def scenario_dual_gpu():
    """Dual GPU - Parallel training"""
    print_header("SCENARIO: Dual GPU (Parallel)")

    print("""
You have 2 GPUs (e.g., 2x RTX3090 or RTX3090 + RTX4090)

Training plan:
  GPU 0: YOLO-DAM Ablation (9-12 weeks)
  GPU 1: Baseline Models (45-60 weeks)
  Both run in parallel
  Total elapsed: 45-60 weeks (instead of 55-72!)

Time savings: 15-27% faster!

Commands:
  Terminal 1 (GPU 0):
    D:\\Programy\\anaconda3\\envs\\TF_3_8\\python.exe TRAIN_YOLO_DAM_ABLATION.py

  Terminal 2 (GPU 1):
    set CUDA_VISIBLE_DEVICES=1
    D:\\Programy\\anaconda3\\envs\\TF_3_8\\python.exe TRAIN_BASELINE_MODELS.py

Result:
  GPU 0: YOLO-DAM done first (~9-12 weeks)
  GPU 1: Baseline models run longer (45-60 weeks)
  When baseline done, you have everything tested!
  """)

def scenario_quad_gpu():
    """Quad GPU - Faster parallel training"""
    print_header("SCENARIO: Quad GPU (Faster Parallel)")

    print("""
You have 4 GPUs

Training plan:
  GPU 0: YOLOv8 family (3 models: n, m, x)
  GPU 1: YOLOv9 family (2 models: t, m)
  GPU 2: YOLOv10 family (4 models: n, m, l, x)
  GPU 3: YOLO-DAM (3 configs: A, B, C)

But note: Ultralytics doesn't easily partition 15 models across GPUs
Would need custom script with threading

Alternative:
  GPU 0: TRAIN_BASELINE_MODELS.py (uses all 4 GPUs internally)
  GPU 3: TRAIN_YOLO_DAM_ABLATION.py (uses GPU 3)

Still gets parallelism, less custom code needed
  """)

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print_header("PARALLEL TRAINING OPTIONS")

    # Check GPU availability
    num_gpus = check_gpu_availability()

    print("\n" + "=" * 80)
    print("AVAILABLE TRAINING SCENARIOS")
    print("=" * 80)

    if num_gpus == 0:
        log_msg("No GPUs found! Using CPU (very slow)", "WARNING")
        scenario_single_gpu()

    elif num_gpus == 1:
        log_msg("1 GPU detected - Sequential training", "INFO")
        scenario_single_gpu()

    elif num_gpus >= 2:
        log_msg(f"{num_gpus} GPUs detected - Parallel training available!", "INFO")
        scenario_dual_gpu()

        if num_gpus >= 4:
            log_msg("4+ GPUs - Even faster parallel options available", "INFO")
            scenario_quad_gpu()

    # Show all scenarios
    print("\n" + "=" * 80)
    print("ALL AVAILABLE SCENARIOS")
    print("=" * 80)

    for scenario_name, scenario_info in PARALLEL_SETUPS.items():
        print(f"\n{scenario_name.upper()}:")
        print(f"  {scenario_info['description']}")
        print(f"  Note: {scenario_info['note']}")

    # Manual parallel example
    print("\n" + "=" * 80)
    print("MANUAL PARALLEL TRAINING (2+ GPUs)")
    print("=" * 80)

    print("""
Open TWO terminals:

TERMINAL 1 (GPU 0 - YOLO-DAM):
  cd d:\\Projekty\\2022_01_BattPor\\2025_12_Dresden\\VSCODE
  set CUDA_VISIBLE_DEVICES=0
  D:\\Programy\\anaconda3\\envs\\TF_3_8\\python.exe TRAIN_YOLO_DAM_ABLATION.py

TERMINAL 2 (GPU 1 - Baseline):
  cd d:\\Projekty\\2022_01_BattPor\\2025_12_Dresden\\VSCODE
  set CUDA_VISIBLE_DEVICES=1
  D:\\Programy\\anaconda3\\envs\\TF_3_8\\python.exe TRAIN_BASELINE_MODELS.py

Both run simultaneously!
  GPU 0: YOLO-DAM (9-12 weeks)
  GPU 1: Baseline models (45-60 weeks)
  Actual elapsed time: 45-60 weeks (both finish when slower one is done)

Time Saved: 10-15 weeks vs sequential training!
    """)

    # Single GPU best practice
    print("\n" + "=" * 80)
    print("SINGLE GPU BEST PRACTICE")
    print("=" * 80)

    print("""
If you only have 1 GPU, recommended order:

1. Start with YOLO-DAM Ablation (faster, get results first)
   D:\\Programy\\anaconda3\\envs\\TF_3_8\\python.exe TRAIN_YOLO_DAM_ABLATION.py
   Time: 9-12 weeks
   Result: See if improvements matter

2. Then start Baseline Models (let run in background)
   D:\\Programy\\anaconda3\\envs\\TF_3_8\\python.exe TRAIN_BASELINE_MODELS.py
   Time: 45-60 weeks
   Result: Which YOLO is best?

3. After both complete (~55 weeks), evaluate
   D:\\Programy\\anaconda3\\envs\\TF_3_8\\python.exe COMPREHENSIVE_TEST_AND_COMPARE.py
   Time: 1-2 hours
   Result: Definitive comparison

Why this order?
  - Get first results in 9-12 weeks (YOLO-DAM)
  - Don't wait 45 weeks for any results
  - Baseline can run while you use computer for other work
    """)

if __name__ == "__main__":
    main()
