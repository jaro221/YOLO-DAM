import os
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

DATASET_DIR = r"D:\Projekty\2022_01_BattPor\DATA_DEF\YOLOv8\dataset"
YOLO_DAM_PATH = r"D:\Projekty\2022_01_BattPor\2025_12_Dresden\Models\YOLODAMv2_best_final.h5"
OUTPUT_SEG_DIR = r"D:\Projekty\2022_01_BattPor\2025_12_Dresden\DATASET_SEG"
IMG_SIZE = 640
GAUSSIAN_SIGMA = 2.0
ERROR_PERCENTILE = 60

def load_yolo_dam():
    try:
        from YOLO_DAM_v2 import model as yolo_model
        yolo_model.load_weights(YOLO_DAM_PATH)
        print("[OK] YOLO-DAM v2 loaded")
        return yolo_model
    except Exception as e:
        print(f"[ERROR] YOLO-DAM: {e}")
        return None

def get_reconstruction_error(model, patch):
    h, w = patch.shape[:2]
    patch_resized = cv2.resize(patch, (IMG_SIZE, IMG_SIZE))
    patch_batch = np.expand_dims(patch_resized, 0).astype(np.float32) / 255.0
    try:
        outputs = model(patch_batch, training=False)
        recon = outputs['auto_reconstruction'].numpy()[0] if 'auto_reconstruction' in outputs else outputs[-2][0]
        recon_resized = cv2.resize(recon, (w, h))
        error = np.sqrt(np.sum((patch.astype(np.float32) / 255 - recon_resized) ** 2, axis=2))
        return error
    except:
        return None

def get_dark_pixels(patch, threshold=30):
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    mask = (gray < threshold).astype(np.uint8)
    return mask

def load_labels(label_file):
    labels = []
    if not os.path.exists(label_file):
        return labels
    with open(label_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                labels.append({
                    "class": int(parts[0]),
                    "x": float(parts[1]),
                    "y": float(parts[2]),
                    "w": float(parts[3]),
                    "h": float(parts[4]),
                })
    return labels

def bbox_to_pixel(x, y, w, h, img_h, img_w):
    x_px = int(x * img_w)
    y_px = int(y * img_h)
    w_px = int(w * img_w)
    h_px = int(h * img_h)
    x1 = max(0, x_px - w_px // 2)
    y1 = max(0, y_px - h_px // 2)
    x2 = min(img_w, x_px + w_px // 2)
    y2 = min(img_h, y_px + h_px // 2)
    return x1, y1, x2, y2

def mask_to_polygon(mask, x1, y1, img_h, img_w):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 5:
        return None
    polygon = []
    for pt in largest[:, 0]:
        x = (float(pt[0]) + x1) / img_w
        y = (float(pt[1]) + y1) / img_h
        polygon.append((x, y))
    return polygon

def process_image(image_path, label_path, model, img_h, img_w):
    image = cv2.imread(image_path)
    if image is None:
        return []
    
    labels = load_labels(label_path)
    segmentations = []
    
    for label in labels:
        class_id = label["class"]
        x1, y1, x2, y2 = bbox_to_pixel(label["x"], label["y"], label["w"], label["h"], img_h, img_w)
        patch = image[y1:y2, x1:x2].copy()
        
        if patch.size == 0:
            continue
        
        error = get_reconstruction_error(model, patch)
        if error is None:
            continue
        
        dark = get_dark_pixels(patch, 30)
        
        error_norm = (error / (np.max(error) + 1e-6) * 255).astype(np.uint8)
        error_thresh = np.percentile(error_norm[error_norm > 0], ERROR_PERCENTILE)
        error_mask = (error_norm >= error_thresh).astype(np.uint8)
        
        combined = (error_mask.astype(float) + dark.astype(float)) / 2.0
        combined = (combined > 0.5).astype(np.uint8)
        
        combined = cv2.GaussianBlur(combined, (5, 5), GAUSSIAN_SIGMA)
        combined = (combined > 127).astype(np.uint8)
        
        polygon = mask_to_polygon(combined, x1, y1, img_h, img_w)
        if polygon:
            segmentations.append({"class_id": class_id, "polygon": polygon})
    
    return segmentations

def main():
    print("\n" + "="*70)
    print("SEGMENTATION GENERATION (YOLO-DAM Reconstruction)")
    print("="*70)
    
    model = load_yolo_dam()
    if not model:
        return False
    
    for split in ["train", "test"]:
        os.makedirs(os.path.join(OUTPUT_SEG_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_SEG_DIR, "labels", split), exist_ok=True)
    
    images_dir = os.path.join(DATASET_DIR, "images", "train")
    labels_dir = os.path.join(DATASET_DIR, "labels", "train")
    
    if not os.path.exists(images_dir):
        print("[ERROR] Images directory not found")
        return False
    
    files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".png"))])
    print(f"Processing {len(files)} images...")
    
    success = 0
    for i, fname in tqdm(enumerate(files)):
        img_path = os.path.join(images_dir, fname)
        base = fname.rsplit(".", 1)[0]
        lbl_path = os.path.join(labels_dir, base + ".txt")
        
        image = cv2.imread(img_path)
        if image is None:
            continue
        
        h, w = image.shape[:2]
        segs = process_image(img_path, lbl_path, model, h, w)
        
        if segs:
            out_img = os.path.join(OUTPUT_SEG_DIR, "images", "train", fname)
            cv2.imwrite(out_img, image)
            
            out_lbl = os.path.join(OUTPUT_SEG_DIR, "labels", "train", base + ".txt")
            with open(out_lbl, "w") as f:
                for seg in segs:
                    coords = " ".join([f"{x:.6f} {y:.6f}" for x, y in seg["polygon"]])
                    f.write(f"{seg['class_id']} {coords}\n")
            
            success += 1
        
        if (i + 1) % 500 == 0 or (i + 1) == len(files):
            print(f"  {i+1}/{len(files)} - {success} segmented")
    
    print(f"\n[OK] Complete: {success}/{len(files)} segmented")
    print(f"[OK] Output: {OUTPUT_SEG_DIR}")
    print("="*70 + "\n")
    return success > 0

if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
