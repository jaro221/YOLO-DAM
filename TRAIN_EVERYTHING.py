"""
ULTIMATE Training Orchestrator
Trains EVERYTHING from scratch for complete analysis

Phase 1: All Baseline Models (15 models)
  - YOLOv8: nano, medium, extra (3)
  - YOLOv9: tiny, medium (2)
  - YOLOv10: nano, medium, large, extra (4)
  - YOLOv11: nano, medium, extra (3)
  - YOLO26: nano, medium, extra (3)

Phase 2: All YOLO-DAM Configs (3 models)
  - Config A: Random init
  - Config B: v26 Pre-trained
  - Config C: Old baseline

Total: 18 models
Estimated Time: 55-70 weeks on single GPU (or 9-12 weeks with 6 GPUs in parallel)
"""

import os
import sys
import json
import subprocess
from datetime import datetime
from pathlib import Path

PYTHON_EXE = r"D:\Programy\anaconda3\envs\TF_3_8\python.exe"
PROJECT_ROOT = r"D:/Projekty/2022_01_BattPor/2025_12_Dresden/VSCODE"
LOG_DIR = r"D:/Projekty/2022_01_BattPor/2025_12_Dresden/Models"

def log_msg(msg, level="INFO"):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{level:8s}] {msg}")

def print_header(text):
    print("\n" + "=" * 80)
    print(text.center(80))
    print("=" * 80 + "\n")

def main():
    print_header("ULTIMATE TRAINING ORCHESTRATOR - Everything from Scratch")

    log_msg("Phase 1: Baseline Models (15 total)", "PHASE")
    log_msg("Phase 2: YOLO-DAM Ablation (3 total)", "PHASE")
    log_msg("Total Models: 18", "INFO")
    log_msg("Estimated Time: 55-70 weeks single GPU | 9-12 weeks with parallelization", "TIME")

    print("\n" + "=" * 80)
    print("BASELINE MODELS (YOLOv8, v9, v10, v11, v26)")
    print("=" * 80)
    print("""
    ✓ YOLOv8:  nano, medium, extra (3)
    ✓ YOLOv9:  tiny, medium (2)
    ✓ YOLOv10: nano, medium, large, extra (4)
    ✓ YOLOv11: nano, medium, extra (3)
    ✓ YOLO26:  nano, medium, extra (3) - LATEST & BEST EXPECTED

    Total: 15 baseline models
    Expected best: YOLO26x
    Time: 45-60 weeks
    """)

    print("=" * 80)
    print("YOLO-DAM ABLATION (3 configs)")
    print("=" * 80)
    print("""
    ✓ Config A: width=1.0, random init  (expected: 45-55%)
    ✓ Config B: width=1.0, v26 pretrain (expected: 70-75%) <- BEST YOLO-DAM
    ✓ Config C: width=0.6, old model    (expected: 38-42%)

    Total: 3 YOLO-DAM configurations
    Time: 9-12 weeks
    """)

    training_log = {
        "start_time": datetime.now().isoformat(),
        "phases": {},
    }

    # =========================================================================
    # PHASE 1: Baseline Models
    # =========================================================================
    print_header("PHASE 1: BASELINE MODELS (15 total)")

    phase_start = datetime.now()

    log_msg("Starting baseline model training...", "START")
    log_msg(f"Command: python TRAIN_BASELINE_MODELS.py", "CMD")

    result_baseline = subprocess.run(
        [PYTHON_EXE, "TRAIN_BASELINE_MODELS.py"],
        cwd=PROJECT_ROOT,
        capture_output=False
    )

    phase_duration = (datetime.now() - phase_start).total_seconds() / 3600
    training_log["phases"]["baseline"] = {
        "success": result_baseline.returncode == 0,
        "duration_hours": phase_duration,
        "models": 15,
    }

    log_msg(f"Phase 1 completed: {phase_duration:.1f} hours", "PHASE_END")

    # =========================================================================
    # PHASE 2: YOLO-DAM Ablation
    # =========================================================================
    print_header("PHASE 2: YOLO-DAM ABLATION (3 configs)")

    phase_start = datetime.now()

    log_msg("Starting YOLO-DAM ablation training...", "START")
    log_msg(f"Command: python TRAIN_YOLO_DAM_ABLATION.py", "CMD")

    result_yolo_dam = subprocess.run(
        [PYTHON_EXE, "TRAIN_YOLO_DAM_ABLATION.py"],
        cwd=PROJECT_ROOT,
        capture_output=False
    )

    phase_duration = (datetime.now() - phase_start).total_seconds() / 3600
    training_log["phases"]["yolo_dam"] = {
        "success": result_yolo_dam.returncode == 0,
        "duration_hours": phase_duration,
        "models": 3,
    }

    log_msg(f"Phase 2 completed: {phase_duration:.1f} hours", "PHASE_END")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print_header("TRAINING COMPLETE - COMPREHENSIVE ANALYSIS")

    training_log["end_time"] = datetime.now().isoformat()

    total_duration = (
        training_log["phases"]["baseline"]["duration_hours"] +
        training_log["phases"]["yolo_dam"]["duration_hours"]
    )

    print(f"""
PHASE 1: Baseline Models (15 models)
  Status: {'✓ SUCCESS' if training_log["phases"]["baseline"]["success"] else '✗ FAILED'}
  Duration: {training_log["phases"]["baseline"]["duration_hours"]:.1f} hours
  Models: 15

PHASE 2: YOLO-DAM Ablation (3 configs)
  Status: {'✓ SUCCESS' if training_log["phases"]["yolo_dam"]["success"] else '✗ FAILED'}
  Duration: {training_log["phases"]["yolo_dam"]["duration_hours"]:.1f} hours
  Models: 3

TOTAL
  Total Models: 18
  Total Duration: {total_duration:.1f} hours ({total_duration/24:.1f} days)
  Success: {training_log["phases"]["baseline"]["success"] and training_log["phases"]["yolo_dam"]["success"]}
    """)

    # Save training log
    log_file = os.path.join(LOG_DIR, "comprehensive_training_log.json")
    with open(log_file, 'w') as f:
        json.dump(training_log, f, indent=2)

    log_msg(f"Training log saved: {log_file}", "SAVE")

    # =========================================================================
    # NEXT STEP
    # =========================================================================
    print_header("NEXT STEP: Comprehensive Evaluation")

    print("""
Run this to evaluate all 18 trained models:

  D:\\Programy\\anaconda3\\envs\\TF_3_8\\python.exe COMPREHENSIVE_TEST_AND_COMPARE.py

This will:
  ✓ Auto-discover all 18 trained models
  ✓ Test on your test dataset
  ✓ Calculate metrics (precision, recall, F1) by class
  ✓ Generate Excel report with detailed comparisons
  ✓ Show ranking of all models
  ✓ Compare to expected baselines

Output: comparison_report.xlsx in TEST_RESULTS/
    """)

if __name__ == "__main__":
    main()
