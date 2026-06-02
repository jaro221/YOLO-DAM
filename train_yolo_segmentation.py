from ultralytics import YOLO

print("\n" + "="*70)
print("YOLO SEGMENTATION TRAINING")
print("="*70)

model = YOLO("yolov8m-seg.pt")

results = model.train(
    data=r"D:\Projekty\2022_01_BattPor\2025_12_Dresden\DATASET_SEG\data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
    patience=20,
    save=True,
    verbose=True,
    project=r"D:\Projekty\2022_01_BattPor\2025_12_Dresden\runs",
    name="yolo_segmentation_v1",
)

print("\n" + "="*70)
print("[OK] TRAINING COMPLETE")
print("="*70)
print(f"\nBest model: {results.save_dir}/weights/best.pt")
print(f"Results: {results}")
print("="*70 + "\n")
