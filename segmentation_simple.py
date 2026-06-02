import os
import cv2

DATASET_DIR = r"D:\Projekty\2022_01_BattPor\DATA_DEF\YOLOv8\dataset"
OUTPUT_SEG_DIR = r"D:\Projekty\2022_01_BattPor\2025_12_Dresden\DATASET_SEG"

def bbox_to_polygon(x, y, w, h):
    x1 = x - w/2
    y1 = y - h/2
    x2 = x + w/2
    y2 = y + h/2
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

def main():
    print("\nSimple Segmentation: BBox to Polygon Conversion")
    
    for split in ["train", "test"]:
        os.makedirs(os.path.join(OUTPUT_SEG_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_SEG_DIR, "labels", split), exist_ok=True)
    
    images_dir = os.path.join(DATASET_DIR, "images", "train")
    labels_dir = os.path.join(DATASET_DIR, "labels", "train")
    
    files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".png"))])
    print(f"Processing {len(files)} images...")
    
    success = 0
    for i, fname in enumerate(files):
        img_path = os.path.join(images_dir, fname)
        base = fname.rsplit(".", 1)[0]
        lbl_path = os.path.join(labels_dir, base + ".txt")
        
        image = cv2.imread(img_path)
        if image is None:
            continue
        
        if not os.path.exists(lbl_path):
            continue
        
        with open(lbl_path) as f:
            lines = f.readlines()
        
        if lines:
            out_img = os.path.join(OUTPUT_SEG_DIR, "images", "train", fname)
            cv2.imwrite(out_img, image)
            
            out_lbl = os.path.join(OUTPUT_SEG_DIR, "labels", "train", base + ".txt")
            with open(out_lbl, "w") as f:
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = parts[0]
                        x, y, w, h = map(float, parts[1:5])
                        polygon = bbox_to_polygon(x, y, w, h)
                        coords = " ".join([f"{px:.6f} {py:.6f}" for px, py in polygon])
                        f.write(f"{class_id} {coords}\n")
            
            success += 1
        
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(files)} - {success} done")
    
    print(f"\n[OK] {success}/{len(files)} segmented")
    return success > 0

if __name__ == "__main__":
    main()
