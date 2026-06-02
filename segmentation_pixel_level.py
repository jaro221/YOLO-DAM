import os
import cv2
import numpy as np

DATASET_DIR = r"D:\Projekty\2022_01_BattPor\DATA_DEF\YOLOv8\dataset"
OUTPUT_SEG_DIR = r"D:\Projekty\2022_01_BattPor\2025_12_Dresden\DATASET_SEG"

def load_labels(label_file):
    labels = []
    if not os.path.exists(label_file):
        return labels
    with open(label_file) as f:
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

def extract_dark_pixels(patch, threshold=40):
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    dark_mask = (gray < threshold).astype(np.uint8)
    dark_mask = cv2.GaussianBlur(dark_mask, (3, 3), 1.0)
    dark_mask = (dark_mask > 100).astype(np.uint8)
    return dark_mask

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
        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, 1)
        polygon.append((x, y))
    
    return polygon if len(polygon) >= 3 else None

def main():
    print("\nSIMPLE PIXEL-LEVEL SEGMENTATION (Dark Pixels Only)")
    print("="*70)
    
    for split in ["train", "test"]:
        os.makedirs(os.path.join(OUTPUT_SEG_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_SEG_DIR, "labels", split), exist_ok=True)
    
    images_dir = os.path.join(DATASET_DIR, "images", "train")
    labels_dir = os.path.join(DATASET_DIR, "labels", "train")
    
    files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".png"))])
    print(f"Processing {len(files)} images...\n")
    
    success = 0
    for i, fname in enumerate(files):
        img_path = os.path.join(images_dir, fname)
        base = fname.rsplit(".", 1)[0]
        lbl_path = os.path.join(labels_dir, base + ".txt")
        
        image = cv2.imread(img_path)
        if image is None:
            continue
        
        h, w = image.shape[:2]
        labels = load_labels(lbl_path)
        
        if not labels:
            continue
        
        segs = []
        for label in labels:
            class_id = label["class"]
            x1, y1, x2, y2 = bbox_to_pixel(label["x"], label["y"], label["w"], label["h"], h, w)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            patch = image[y1:y2, x1:x2]
            dark_mask = extract_dark_pixels(patch, 40)
            polygon = mask_to_polygon(dark_mask, x1, y1, h, w)
            
            if polygon:
                segs.append({"class_id": class_id, "polygon": polygon})
        
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
    
    print(f"\n[OK] {success}/{len(files)} segmented")
    print(f"Output: {OUTPUT_SEG_DIR}/labels/train/")
    return success > 0

if __name__ == "__main__":
    main()
