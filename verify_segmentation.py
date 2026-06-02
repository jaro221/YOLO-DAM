import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATASET_SEG_DIR = r"D:\Projekty\2022_01_BattPor\2025_12_Dresden\DATASET_SEG"
ORIGINAL_DATASET = r"D:\Projekty\2022_01_BattPor\DATA_DEF\YOLOv8\dataset"
OUTPUT_VIZ_DIR = os.path.join(DATASET_SEG_DIR, "visualizations_comparison")

CLASS_NAMES = {
    0: "Agglomerate", 1: "Pinhole-long", 2: "Pinhole-trans", 3: "Pinhole-round",
    4: "Crack-long", 5: "Crack-trans", 6: "Line-long", 7: "Line-trans",
    8: "Line-diag", 9: "Foreign-particle",
}

CLASS_COLORS = {
    0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255), 3: (255, 255, 0),
    4: (255, 0, 255), 5: (0, 255, 255), 6: (128, 0, 0), 7: (0, 128, 0),
    8: (0, 0, 128), 9: (128, 128, 0),
}

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

def load_polygons(label_file):
    polys = []
    if not os.path.exists(label_file):
        return polys
    with open(label_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            class_id = int(parts[0])
            coords = [float(p) for p in parts[1:]]
            points = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
            polys.append({"class_id": class_id, "polygon": points})
    return polys

def visualize_comparison(image_path, orig_label_path, seg_label_path, output_path):
    image = cv2.imread(image_path)
    if image is None:
        return False
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    h, w = image.shape[:2]
    
    orig_bboxes = load_labels(orig_label_path)
    seg_polygons = load_polygons(seg_label_path)
    
    if not orig_bboxes:
        return False
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    ax = axes[0]
    ax.imshow(image_rgb)
    ax.set_title("True BBoxes (Original Labels)", fontsize=12, fontweight="bold")
    
    for bbox in orig_bboxes:
        class_id = bbox["class"]
        x, y, w_norm, h_norm = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
        
        x1 = int((x - w_norm/2) * w)
        y1 = int((y - h_norm/2) * h)
        x2 = int((x + w_norm/2) * w)
        y2 = int((y + h_norm/2) * h)
        
        color = CLASS_COLORS.get(class_id, (255, 255, 255))
        color_rgb = (color[2]/255, color[1]/255, color[0]/255)
        
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor=color_rgb, linewidth=2)
        ax.add_patch(rect)
        
        class_name = CLASS_NAMES.get(class_id, f"C{class_id}")
        ax.text(x1, y1-5, class_name, color=color_rgb, fontsize=9,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    
    ax.axis("off")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    
    ax = axes[1]
    ax.imshow(image_rgb)
    ax.set_title("Generated Polygons (Segmentation Labels)", fontsize=12, fontweight="bold")
    
    for poly in seg_polygons:
        class_id = poly["class_id"]
        polygon = poly["polygon"]
        
        points = []
        for x_norm, y_norm in polygon:
            x_px = int(x_norm * w)
            y_px = int(y_norm * h)
            points.append((x_px, y_px))
        
        points_array = np.array(points, dtype=np.int32)
        
        if len(points_array) < 3:
            continue
        
        color = CLASS_COLORS.get(class_id, (255, 255, 255))
        color_rgb = (color[2]/255, color[1]/255, color[0]/255)
        
        poly_patch = plt.Polygon(points_array, fill=True, alpha=0.3,
                               facecolor=color_rgb, edgecolor=color_rgb, linewidth=2)
        ax.add_patch(poly_patch)
        
        ax.plot(points_array[:, 0], points_array[:, 1], "o", color=color_rgb, markersize=4)
        
        cx = np.mean(points_array[:, 0])
        cy = np.mean(points_array[:, 1])
        class_name = CLASS_NAMES.get(class_id, f"C{class_id}")
        ax.text(int(cx), int(cy), class_name, color=color_rgb, fontsize=8,
               bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
               ha="center", va="center")
    
    ax.axis("off")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    
    return True

def main():
    print("\n" + "="*70)
    print("SEGMENTATION VERIFICATION")
    print("Compare True BBoxes vs Generated Polygons")
    print("="*70)
    
    os.makedirs(OUTPUT_VIZ_DIR, exist_ok=True)
    
    seg_images_dir = os.path.join(DATASET_SEG_DIR, "images", "train")
    seg_labels_dir = os.path.join(DATASET_SEG_DIR, "labels", "train")
    orig_images_dir = os.path.join(ORIGINAL_DATASET, "images", "train")
    orig_labels_dir = os.path.join(ORIGINAL_DATASET, "labels", "train")
    
    if not os.path.exists(seg_images_dir):
        print("[ERROR] Segmentation images not found")
        return False
    
    files = sorted([f for f in os.listdir(seg_labels_dir) if f.endswith(".txt")])[:20]
    
    print(f"Comparing {len(files)} samples...")
    
    success = 0
    for i, fname in enumerate(files):
        base = fname.rsplit(".", 1)[0]
        
        seg_img_path = os.path.join(seg_images_dir, f"{base}.jpg")
        if not os.path.exists(seg_img_path):
            seg_img_path = os.path.join(seg_images_dir, f"{base}.png")
        
        orig_lbl_path = os.path.join(orig_labels_dir, fname)
        seg_lbl_path = os.path.join(seg_labels_dir, fname)
        
        output_path = os.path.join(OUTPUT_VIZ_DIR, f"{i+1:02d}_{base}_comparison.png")
        
        try:
            if visualize_comparison(seg_img_path, orig_lbl_path, seg_lbl_path, output_path):
                success += 1
                print(f"  {i+1:2d}. [OK] {base}")
            else:
                print(f"  {i+1:2d}. [SKIP] {base} - no labels")
        except Exception as e:
            print(f"  {i+1:2d}. [ERROR] {base} - {str(e)[:40]}")
    
    print(f"\n[OK] Created {success} comparison visualizations")
    print(f"[OK] Output: {OUTPUT_VIZ_DIR}\n")
    
    return success > 0

if __name__ == "__main__":
    main()
