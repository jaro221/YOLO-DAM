# -*- coding: utf-8 -*-
"""
YOLO Training from Scratch (No Pretrained Weights)

Uses .yaml configuration files to train models with random initialization.
All models train from scratch on custom dataset.

Modified: 2026-04-08
Author: jarom
"""

from ultralytics import YOLO
from datetime import datetime

# ================================================================================
# Configuration
# ================================================================================

DATA_YAML = "D:/Projekty/2022_01_BattPor/DATA_DEF/YOLOv8/data.yaml"
EPOCHS = 400
IMG_SIZE = 640
BATCH_SIZE = 16
DEVICE = 0


def main():
    """Main training and prediction pipeline."""

    # ================================================================================
    # YOLOv8 - Training from Scratch (.yaml = random initialization)
    # ================================================================================

    print(f"\n{'='*80}")
    print("YOLOv8 Training from Scratch")
    print(f"{'='*80}\n")

    models_v8 = [
        ("yolov8n.yaml", "YOLO8n_from_scratch"),
        ("yolov8m.yaml", "YOLO8m_from_scratch"),
        ("yolov8x.yaml", "YOLO8x_from_scratch"),
    ]

    for yaml_file, run_name in models_v8:
        print(f"[YOLOv8] Training {yaml_file} from scratch...")
        model = YOLO(yaml_file)  # Load architecture only (random weights)
        model.train(
            data=DATA_YAML,
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            device=DEVICE,
            project="D:/Projekty/2022_01_BattPor/DATA_DEF/YOLOv8/",
            name=run_name,
            verbose=True,
            patience=50,  # Early stopping
            save_period=50,  # Save every 50 epochs
        )
        print(f"[YOLOv8] {run_name} completed.\n")

    # ================================================================================
    # YOLOv9 - Training from Scratch (.yaml = random initialization)
    # ================================================================================

    print(f"\n{'='*80}")
    print("YOLOv9 Training from Scratch")
    print(f"{'='*80}\n")

    models_v9 = [
        ("yolov9t.yaml", "YOLO9t_from_scratch"),
        ("yolov9m.yaml", "YOLO9m_from_scratch"),
        ("yolov9e.yaml", "YOLO9e_from_scratch"),
    ]

    for yaml_file, run_name in models_v9:
        print(f"[YOLOv9] Training {yaml_file} from scratch...")
        model = YOLO(yaml_file)  # Load architecture only (random weights)
        model.train(
            data=DATA_YAML,
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            device=DEVICE,
            project="D:/Projekty/2022_01_BattPor/DATA_DEF/YOLOv9/",
            name=run_name,
            verbose=True,
            patience=50,
            save_period=50,
        )
        print(f"[YOLOv9] {run_name} completed.\n")

    # ================================================================================
    # YOLOv10 - Training from Scratch (.yaml = random initialization)
    # ================================================================================

    print(f"\n{'='*80}")
    print("YOLOv10 Training from Scratch")
    print(f"{'='*80}\n")

    models_v10 = [
        ("yolov10n.yaml", "YOLOv10n_from_scratch"),
        ("yolov10m.yaml", "YOLOv10m_from_scratch"),
        ("yolov10l.yaml", "YOLOv10l_from_scratch"),
        ("yolov10x.yaml", "YOLOv10x_from_scratch"),
    ]

    for yaml_file, run_name in models_v10:
        print(f"[YOLOv10] Training {yaml_file} from scratch...")
        model = YOLO(yaml_file)  # Load architecture only (random weights)
        model.train(
            data=DATA_YAML,
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            device=DEVICE,
            project="D:/Projekty/2022_01_BattPor/DATA_DEF/YOLOv10/",
            name=run_name,
            verbose=True,
            patience=50,
            save_period=50,
        )
        print(f"[YOLOv10] {run_name} completed.\n")

    # ================================================================================
    # YOLOv11 - Training from Scratch (.yaml = random initialization)
    # ================================================================================

    print(f"\n{'='*80}")
    print("YOLOv11 Training from Scratch")
    print(f"{'='*80}\n")

    models_v11 = [
        ("yolo11n.yaml", "YOLOv11n_from_scratch"),
        ("yolo11m.yaml", "YOLOv11m_from_scratch"),
        ("yolo11x.yaml", "YOLOv11x_from_scratch"),
    ]

    for yaml_file, run_name in models_v11:
        print(f"[YOLOv11] Training {yaml_file} from scratch...")
        model = YOLO(yaml_file)  # Load architecture only (random weights)
        model.train(
            data=DATA_YAML,
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            device=DEVICE,
            project="D:/Projekty/2022_01_BattPor/DATA_DEF/YOLOv11/",
            name=run_name,
            verbose=True,
            patience=50,
            save_period=50,
        )
        print(f"[YOLOv11] {run_name} completed.\n")

    # ================================================================================
    # YOLO26 - Training from Scratch (.yaml = random initialization)
    # ================================================================================

    print(f"\n{'='*80}")
    print("YOLO26 Training from Scratch")
    print(f"{'='*80}\n")

    models_v26 = [
        ("yolo26n.yaml", "YOLO26n_from_scratch"),
        #("yolo26m.yaml", "YOLO26m_from_scratch"),
        #("yolo26x.yaml", "YOLO26x_from_scratch"),
    ]

    for yaml_file, run_name in models_v26:
        print(f"[YOLO26] Training {yaml_file} from scratch...")
        model = YOLO(yaml_file)  # Load architecture only (random weights)
        model.train(
            data=DATA_YAML,
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            device=DEVICE,
            project="D:/Projekty/2022_01_BattPor/DATA_DEF/YOLO26/",
            name=run_name,
            verbose=True,
            patience=50,
            save_period=50,
        )
        print(f"[YOLO26] {run_name} completed.\n")

    # ================================================================================
    # Summary
    # ================================================================================

    print(f"\n{'='*80}")
    print("ALL MODELS TRAINED FROM SCRATCH")
    print(f"{'='*80}")
    print(f"\nTraining Configuration:")
    print(f"  - Weights: Random initialization (from .yaml configs)")
    print(f"  - Epochs: {EPOCHS}")
    print(f"  - Image Size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"  - Batch Size: {BATCH_SIZE}")
    print(f"  - Device: {DEVICE}")
    print(f"\nResults saved in:")
    print(f"  - YOLOv8:  D:/Projekty/2022_01_BattPor/DATA_DEF/YOLOv8/")
    print(f"  - YOLOv9:  D:/Projekty/2022_01_BattPor/DATA_DEF/YOLOv9/")
    print(f"  - YOLOv10: D:/Projekty/2022_01_BattPor/DATA_DEF/YOLOv10/")
    print(f"  - YOLOv11: D:/Projekty/2022_01_BattPor/DATA_DEF/YOLOv11/")
    print(f"  - YOLO26:  D:/Projekty/2022_01_BattPor/DATA_DEF/YOLO26/")
    print(f"\nTraining completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")

    # ================================================================================
    # PREDICTIONS/TESTING SECTION
    # Run inference on trained from-scratch models
    # ================================================================================

    # Test dataset configuration
    TRUE_DIR = r"D:/Projekty/2022_01_BattPor/DATA_DEF/YOLOv8/test_dataset/labels/test/"
    IMG_DIR = TRUE_DIR.replace("labels", "images")

    print(f"\n{'='*80}")
    print("TESTING TRAINED MODELS (from-scratch)")
    print(f"{'='*80}\n")
    print(f"Test images: {IMG_DIR}\n")

    # ────────────────────────────────────────────────────────────────────────────
    # YOLOv8 - Test from-scratch models
    # ────────────────────────────────────────────────────────────────────────────
    
    print(f"\n{'─'*80}")
    print("YOLOv8 - Predictions from Scratch Models")
    print(f"{'─'*80}\n")
    
    yolov8_models = [
        "D:/Projekty/2022_01_BattPor/DATA_DEF/YOLOv8/YOLO8n_from_scratch/weights/best.pt",
        "D:/Projekty/2022_01_BattPor/DATA_DEF/YOLOv8/YOLO8m_from_scratch/weights/best.pt",
        "D:/Projekty/2022_01_BattPor/DATA_DEF/YOLOv8/YOLO8x_from_scratch/weights/best.pt",
    ]
    
    for model_path in yolov8_models:
        model_name = model_path.split("/")[-3]
        print(f"[YOLOv8] Testing {model_name}...")
        try:
            model = YOLO(model_path)
            results = model.predict(
                source=IMG_DIR,
                save=True,
                save_txt=True,
                save_conf=True,
                project="D:/Projekty/2022_01_BattPor/DATA_DEF/YOLOv8/",
                name=f"{model_name}_predictions",
                imgsz=IMG_SIZE,
                conf=0.20,
                device=DEVICE,
            )
            print(f"[YOLOv8] {model_name} predictions saved.\n")
        except Exception as e:
            print(f"[YOLOv8] Error testing {model_name}: {e}\n")
    
    # ────────────────────────────────────────────────────────────────────────────
    # YOLOv9 - Test from-scratch models
    # ────────────────────────────────────────────────────────────────────────────
    
    print(f"\n{'─'*80}")
    print("YOLOv9 - Predictions from Scratch Models")
    print(f"{'─'*80}\n")
    
    yolov9_models = [
        "D:/Projekty/2022_01_BattPor/DATA_DEF/YOLOv9/YOLO9t_from_scratch/weights/best.pt",
        "D:/Projekty/2022_01_BattPor/DATA_DEF/YOLOv9/YOLO9m_from_scratch/weights/best.pt",
        "D:/Projekty/2022_01_BattPor/DATA_DEF/YOLOv9/YOLO9e_from_scratch/weights/best.pt",
    ]
    
    for model_path in yolov9_models:
        model_name = model_path.split("/")[-3]
        print(f"[YOLOv9] Testing {model_name}...")
        try:
            model = YOLO(model_path)
            results = model.predict(
                source=IMG_DIR,
                save=True,
                save_txt=True,
                save_conf=True,
                project="D:/Projekty/2022_01_BattPor/DATA_DEF/YOLOv9/",
                name=f"{model_name}_predictions",
                imgsz=IMG_SIZE,
                conf=0.20,
                device=DEVICE,
            )
            print(f"[YOLOv9] {model_name} predictions saved.\n")
        except Exception as e:
            print(f"[YOLOv9] Error testing {model_name}: {e}\n")
    
    # ────────────────────────────────────────────────────────────────────────────
    # YOLOv10 - Test from-scratch models
    # ────────────────────────────────────────────────────────────────────────────
    
    print(f"\n{'─'*80}")
    print("YOLOv10 - Predictions from Scratch Models")
    print(f"{'─'*80}\n")
    
    yolov10_models = [
        "D:/Projekty/2022_01_BattPor/DATA_DEF/YOLOv10/YOLOv10n_from_scratch/weights/best.pt",
        "D:/Projekty/2022_01_BattPor/DATA_DEF/YOLOv10/YOLOv10m_from_scratch/weights/best.pt",
        "D:/Projekty/2022_01_BattPor/DATA_DEF/YOLOv10/YOLOv10l_from_scratch/weights/best.pt",
        "D:/Projekty/2022_01_BattPor/DATA_DEF/YOLOv10/YOLOv10x_from_scratch/weights/best.pt",
    ]
    
    for model_path in yolov10_models:
        model_name = model_path.split("/")[-3]
        print(f"[YOLOv10] Testing {model_name}...")
        try:
            model = YOLO(model_path)
            results = model.predict(
                source=IMG_DIR,
                save=True,
                save_txt=True,
                save_conf=True,
                project="D:/Projekty/2022_01_BattPor/DATA_DEF/YOLOv10/",
                name=f"{model_name}_predictions",
                imgsz=IMG_SIZE,
                conf=0.20,
                device=DEVICE,
            )
            print(f"[YOLOv10] {model_name} predictions saved.\n")
        except Exception as e:
            print(f"[YOLOv10] Error testing {model_name}: {e}\n")
    
    # ────────────────────────────────────────────────────────────────────────────
    # YOLOv11 - Test from-scratch models
    # ────────────────────────────────────────────────────────────────────────────
    
    print(f"\n{'─'*80}")
    print("YOLOv11 - Predictions from Scratch Models")
    print(f"{'─'*80}\n")
    
    yolov11_models = [
        "D:/Projekty/2022_01_BattPor/DATA_DEF/YOLOv11/YOLOv11n_from_scratch/weights/best.pt",
        "D:/Projekty/2022_01_BattPor/DATA_DEF/YOLOv11/YOLOv11m_from_scratch/weights/best.pt",
        "D:/Projekty/2022_01_BattPor/DATA_DEF/YOLOv11/YOLOv11x_from_scratch/weights/best.pt",
    ]
    
    for model_path in yolov11_models:
        model_name = model_path.split("/")[-3]
        print(f"[YOLOv11] Testing {model_name}...")
        try:
            model = YOLO(model_path)
            results = model.predict(
                source=IMG_DIR,
                save=True,
                save_txt=True,
                save_conf=True,
                project="D:/Projekty/2022_01_BattPor/DATA_DEF/YOLOv11/",
                name=f"{model_name}_predictions",
                imgsz=IMG_SIZE,
                conf=0.20,
                device=DEVICE,
            )
            print(f"[YOLOv11] {model_name} predictions saved.\n")
        except Exception as e:
            print(f"[YOLOv11] Error testing {model_name}: {e}\n")
    
    # ────────────────────────────────────────────────────────────────────────────
    # YOLO26 - Test from-scratch models
    # ────────────────────────────────────────────────────────────────────────────
    
    print(f"\n{'─'*80}")
    print("YOLO26 - Predictions from Scratch Models")
    print(f"{'─'*80}\n")
    
    yolo26_models = [
        "D:/Projekty/2022_01_BattPor/DATA_DEF/YOLO26/YOLO26n_from_scratch/weights/best.pt",
        "D:/Projekty/2022_01_BattPor/DATA_DEF/YOLO26/YOLO26m_from_scratch/weights/best.pt",
        "D:/Projekty/2022_01_BattPor/DATA_DEF/YOLO26/YOLO26x_from_scratch/weights/best.pt",
    ]
    
    for model_path in yolo26_models:
        model_name = model_path.split("/")[-3]
        print(f"[YOLO26] Testing {model_name}...")
        try:
            model = YOLO(model_path)
            results = model.predict(
                source=IMG_DIR,
                save=True,
                save_txt=True,
                save_conf=True,
                project="D:/Projekty/2022_01_BattPor/DATA_DEF/YOLO26/",
                name=f"{model_name}_predictions",
                imgsz=IMG_SIZE,
                conf=0.20,
                device=DEVICE,
            )
            print(f"[YOLO26] {model_name} predictions saved.\n")
        except Exception as e:
            print(f"[YOLO26] Error testing {model_name}: {e}\n")

    # ────────────────────────────────────────────────────────────────────────────
    # Predictions Complete
    # ────────────────────────────────────────────────────────────────────────────

    print(f"\n{'='*80}")
    print("ALL PREDICTIONS COMPLETED")
    print(f"{'='*80}")
    print(f"\nPrediction results saved in:")
    print(f"  - YOLOv8:  D:/Projekty/2022_01_BattPor/DATA_DEF/YOLOv8/*_predictions/")
    print(f"  - YOLOv9:  D:/Projekty/2022_01_BattPor/DATA_DEF/YOLOv9/*_predictions/")
    print(f"  - YOLOv10: D:/Projekty/2022_01_BattPor/DATA_DEF/YOLOv10/*_predictions/")
    print(f"  - YOLOv11: D:/Projekty/2022_01_BattPor/DATA_DEF/YOLOv11/*_predictions/")
    print(f"  - YOLO26:  D:/Projekty/2022_01_BattPor/DATA_DEF/YOLO26/*_predictions/")
    print(f"\nPredictions completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    """Windows multiprocessing guard.

    Required for Windows systems to prevent RuntimeError when YOLO starts data
    loading workers. Without this guard, child processes re-import the script
    and try to create their own workers, causing infinite recursion.
    """
    main()




