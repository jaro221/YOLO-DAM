
import os
import tensorflow as tf
import numpy as np
import cv2
import random

IMG_SIZE = 640
NUM_CLASSES = 10
BATCH_SIZE = 8
EPOCHS = 300
STEPS_PER_EPOCH = 800
LEARNING_RATE = 1e-2

IMG_SIZE = 640
NUM_CLASSES = 10

def load_restored_image(img_path, img_size=640):
    try:
        import cv2
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError("cv2 returned None")
        img = cv2.resize(img, (img_size, img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype(np.float32) / 255.0
    except Exception as e:
        print(f"⚠️ Error loading restored {img_path}: {e}")
        return np.zeros((img_size, img_size, 3), dtype=np.float32) 
    
    
def augment_hsv(image):
    """HSV color space augmentation"""
    # Random hue shift
    if tf.random.uniform([]) > 0.5:
        image = tf.image.random_hue(image, 0.015)
    
    # Random saturation
    if tf.random.uniform([]) > 0.5:
        image = tf.image.random_saturation(image, 0.7, 1.3)
    
    # Random brightness
    if tf.random.uniform([]) > 0.5:
        image = tf.image.random_brightness(image, 0.4)
    
    # Clip to valid range
    image = tf.clip_by_value(image, 0.0, 1.0)
    
    return image


def augment_flip(image, boxes):
    """Random horizontal flip"""
    if tf.random.uniform([]) > 0.5:
        image = tf.image.flip_left_right(image)
        # Flip box x-coordinates
        if len(boxes) > 0:
            boxes_flipped = tf.stack([
                1.0 - boxes[:, 0],  # flip center x
                boxes[:, 1],         # keep y
                boxes[:, 2],         # keep w
                boxes[:, 3]          # keep h
            ], axis=-1)
            return image, boxes_flipped
    return image, boxes

def build_targets_from_boxes_multicell(boxes, classes, img_size=640, num_classes=10, Name=None):
    """
    Improved target assignment: assign large objects to multiple cells
    """
    scales = {"p3": 80, "p4": 40, "p5": 20}
    targets = {}
    
    for scale_name, grid_size in scales.items():
        cls_t = np.zeros((grid_size, grid_size, num_classes), dtype=np.float32)
        reg_t = np.zeros((grid_size, grid_size, 4), dtype=np.float32)
        obj_t = np.zeros((grid_size, grid_size, 1), dtype=np.float32)

        if boxes is None or len(boxes) == 0:
            targets[f"{scale_name}_cls_t"] = cls_t
            targets[f"{scale_name}_reg_t"] = reg_t
            targets[f"{scale_name}_obj_t"] = obj_t
            continue
        
        for (x, y, w, h), cls in zip(boxes, classes):
            # Object size in grid cells
            obj_width_cells = w * grid_size
            obj_height_cells = h * grid_size
            
            # Center cell
            gx_center = x * grid_size
            gy_center = y * grid_size
            gi_center = int(np.clip(np.floor(gx_center), 0, grid_size - 1))
            gj_center = int(np.clip(np.floor(gy_center), 0, grid_size - 1))
            
            # ✅ NEW: Calculate how many cells the object spans
            # For large objects, assign to multiple cells
            max_span = max(obj_width_cells, obj_height_cells)
            
            if max_span > 3.0:  # Large object (spans >3 cells)
                # Assign to center + adjacent cells
                radius = 1  # Assign to 3x3 grid around center
            elif max_span > 6.0:  # Very large object
                radius = 2  # Assign to 5x5 grid around center
            else:
                radius = 0  # Small object, only center cell
            
            # Assign to multiple cells
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    gi = gi_center + dx
                    gj = gj_center + dy
                    
                    # Check bounds
                    if gi < 0 or gi >= grid_size or gj < 0 or gj >= grid_size:
                        continue
                    
                    # Skip if already assigned (keep first assignment)
                    if obj_t[gj, gi, 0] == 1 and False:
                        continue
                    
                    # Assign target
                    cls_t[gj, gi, cls] = 1.0
                    obj_t[gj, gi, 0] = 1.0
                    reg_t[gj, gi] = [x, y, w, h]  # Same box coords for all cells
        
        targets[f"{scale_name}_cls_t"] = cls_t
        targets[f"{scale_name}_reg_t"] = reg_t
        targets[f"{scale_name}_obj_t"] = obj_t
    
    targets["raw"] = ""
    targets["name"] = Name
    return targets

def create_defect_mask(boxes, classes, img_size):
    """
    Create binary mask where:
    - 1 = good area (background)
    - 0 = defect area (inside bounding boxes)
    
    Args:
        boxes: numpy array [N, 4] in normalized format [x_center, y_center, w, h]
        classes: numpy array [N] class indices
        img_size: image size (640)
    
    Returns:
        mask: numpy array [img_size, img_size, 1] with values 0 or 1
    """
    # Start with all good (1s)
    mask = np.ones((img_size, img_size, 1), dtype=np.float32)
    
    if len(boxes) == 0:
        return mask
    
    # Convert normalized boxes to pixel coordinates
    for box in boxes:
        x_center, y_center, w, h = box
        
        # Convert to pixel coordinates
        x_center_px = x_center * img_size
        y_center_px = y_center * img_size
        w_px = w * img_size
        h_px = h * img_size
        
        # Calculate corners
        x1 = int(max(0, x_center_px - w_px / 2))
        y1 = int(max(0, y_center_px - h_px / 2))
        x2 = int(min(img_size, x_center_px + w_px / 2))
        y2 = int(min(img_size, y_center_px + h_px / 2))
        
        # Set defect area to 0
        mask[y1:y2, x1:x2, 0] = 0.0
    
    return mask

def parse_yolo_label_with_augment(label_path, img_size=IMG_SIZE, augment=True):
    """Parse YOLO label and optionally apply augmentations to boxes"""
    boxes   = []
    classes = []

    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            lines = f.readlines()


        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            x, y, w, h = map(float, parts[1:])

            boxes.append([x, y, w, h])
            classes.append(cls)


    return np.array(boxes, dtype=np.float32), np.array(classes, dtype=np.int32)



def yolo_dataset_with_augmentation(images_dir, labels_dir,image_rec, batch_size=BATCH_SIZE, 
                                       img_size=IMG_SIZE, num_classes=NUM_CLASSES, 
                                       augment=True):
    """Dataset generator with augmentation support"""
    image_files = [f for f in os.listdir(images_dir) 
                   if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    
    def _gen():
        shuffled = image_files.copy()
        random.shuffle(shuffled)  # re-shuffle each epoch
        for img_name in shuffled:
            img_path = os.path.join(images_dir, img_name)
            img_rec = os.path.join(image_rec, img_name)
            label_path = os.path.join(labels_dir, img_name.rsplit(".",1)[0] + ".txt")
            
            # Load image
            img = tf.io.read_file(img_path)
            img = tf.image.decode_image(img, channels=3)
            img = tf.image.resize(img, [img_size, img_size])
            img = tf.cast(img, tf.float32) / 255.0
            restored_image = load_restored_image(img_rec, img_size)

            # Parse labels
            boxes, classes = parse_yolo_label_with_augment(label_path, img_size, augment)
            
            # Apply augmentations
            if augment and len(boxes) > 0 and augment:
                # HSV augmentation
                img = augment_hsv(img)
                
                # Horizontal flip
                boxes_tf = tf.constant(boxes, dtype=tf.float32)
                img, boxes_tf = augment_flip(img, boxes_tf)
                
                boxes = boxes_tf
  
                
            
            # Build targets
            targets = build_targets_from_boxes_multicell(boxes, classes, img_size, num_classes, img_name)
            defect_mask = create_defect_mask(boxes, classes, img_size)
            
            yield {
                "image": img,
                "p3_cls_t": targets["p3_cls_t"],
                "p3_reg_t": targets["p3_reg_t"],
                "p3_obj_t": targets["p3_obj_t"],
                "p4_cls_t": targets["p4_cls_t"],
                "p4_reg_t": targets["p4_reg_t"],
                "p4_obj_t": targets["p4_obj_t"],
                "p5_cls_t": targets["p5_cls_t"],
                "p5_reg_t": targets["p5_reg_t"],
                "p5_obj_t": targets["p5_obj_t"],
                "AUTO": restored_image,
                "mask": defect_mask,  # ✅ NEW: Defect mask
                "raw": targets["raw"],
                "name": targets["name"],
            }
    
    out_sig = {
        "image": tf.TensorSpec([img_size, img_size, 3], tf.float32),
        "p3_cls_t": tf.TensorSpec([img_size//8, img_size//8, num_classes], tf.float32),
        "p3_reg_t": tf.TensorSpec([img_size//8, img_size//8, 4], tf.float32),
        "p3_obj_t": tf.TensorSpec([img_size//8, img_size//8, 1], tf.float32),
        "p4_cls_t": tf.TensorSpec([img_size//16, img_size//16, num_classes], tf.float32),
        "p4_reg_t": tf.TensorSpec([img_size//16, img_size//16, 4], tf.float32),
        "p4_obj_t": tf.TensorSpec([img_size//16, img_size//16, 1], tf.float32),
        "p5_cls_t": tf.TensorSpec([img_size//32, img_size//32, num_classes], tf.float32),
        "p5_reg_t": tf.TensorSpec([img_size//32, img_size//32, 4], tf.float32),
        "p5_obj_t": tf.TensorSpec([img_size//32, img_size//32, 1], tf.float32),
        "AUTO": tf.TensorSpec([img_size, img_size, 3], tf.float32),
        "mask": tf.TensorSpec([img_size, img_size, 1], tf.float32),  # ✅ NEW
        "raw": tf.TensorSpec([], tf.string),
        "name": tf.TensorSpec([], tf.string),
    }
    
    ds = tf.data.Dataset.from_generator(_gen, output_signature=out_sig)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds