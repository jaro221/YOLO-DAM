import os
import tensorflow as tf
from datetime import datetime

# ── Local imports ─────────────────────────────────────────────────────────────
from YOLO_DAM_v3 import model
from YOLO_DAM_loss_v2 import detection_loss
from YOLO_DAM_dataset_v2_RELATIVE import make_yolo_dataset

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
IMG_SIZE    = 640
NUM_CLASSES = 10
BATCH_SIZE  = 4
EPOCHS      = 400

DATASET_DIR  = r"D:/Projekty/2022_01_BattPor/2025_12_Dresden/YOLOv8/dataset"
WEIGHTS_PATH = r"D:/Projekty/2022_01_BattPor/2025_12_Dresden/Models/YOLODAM_merged_v26_new.h5"
SAVE_DIR     = r"D:/Projekty/2022_01_BattPor/2025_12_Dresden/Models"
LOG_PATH     = r"D:/Projekty/2022_01_BattPor/2025_12_Dresden/Models/train_log_dam_v3.txt"

# ─────────────────────────────────────────────────────────────────────────────
# Features & Improvements
# ─────────────────────────────────────────────────────────────────────────────
ENABLE_ADVANCED_AUG = True
ENABLE_O2O_MATCHING = True
USE_LABEL_SMOOTHING = 0.01
COSINE_ANNEALING = True
learning_rate = 5e-5

# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
def training(model, epochs=EPOCHS):
    train_ds = make_yolo_dataset(
        os.path.join(DATASET_DIR, "images", "train"),
        os.path.join(DATASET_DIR, "labels", "train"),
        os.path.join(DATASET_DIR, "restored", "train"),
        batch_size=BATCH_SIZE,
        augment=ENABLE_ADVANCED_AUG,
    )

    optimizer     = tf.keras.optimizers.Adam(learning_rate=5e-5)
    best_loss     = float("inf")

    log_file = open(LOG_PATH, "a")
    log_file.write(f"\n{'='*60}\n")
    log_file.write(f"Training YOLO-DAM v3 (C3k2 backbone): {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write(f"LR={learning_rate}  epochs={epochs}\n")
    log_file.write(f"{'='*60}\n")

    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    @tf.function
    def train_step(imgs, batch, epoch_tf):
        with tf.GradientTape() as tape:
            preds = model(imgs, training=True)
            loss, comps = detection_loss(
                preds, batch,
                epoch=epoch_tf,
                total_epochs=epochs,
                label_smoothing=USE_LABEL_SMOOTHING,
                use_l1_loss=False)  # Use CIoU (working better)
        grads, global_norm = tf.clip_by_global_norm(
            tape.gradient(loss, model.trainable_variables), clip_norm=5)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss, comps, global_norm

    for epoch in range(epochs):
        log(f"\nEpoch {epoch+1}/{epochs}")
        running_loss = 0.0
        step = 0

        for batch in train_ds:
            loss, comps, gnorm = train_step(
                batch['image'], batch,
                tf.constant(epoch + 1, dtype=tf.float32))
            running_loss += float(loss)
            step += 1

            if step % 100 == 0:
                scale_str = ""
                for s in ['p2', 'p3', 'p4', 'p5']:
                    if f'{s}_box' not in comps:
                        continue
                    scale_str += (
                        f"    {s}: "
                        f"grad_norm={float(gnorm):.2f}  "
                        f"box={comps[f'{s}_box']:.5f}  "
                        f"obj={comps[f'{s}_obj']:.5f}  "
                        f"cls={comps[f'{s}_cls']:.5f}  "
                        f"pos={comps[f'{s}_pos']:.0f}  "
                        f"pospre={comps[f'{s}_pospre']:.0f}\n"
                    )
                log(f"  Step {step}: Loss={float(loss):.5f}\n{scale_str}")

        epoch_loss = running_loss / max(step, 1)
        log(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")

        # Cosine annealing LR schedule
        if COSINE_ANNEALING:
            import math
            new_lr = 5e-5 * (1 + math.cos(math.pi * (epoch + 1) / epochs)) / 2
            optimizer.learning_rate.assign(new_lr)
            log(f"  LR (cosine): {new_lr:.2e}")

        # Save best
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            path = os.path.join(SAVE_DIR, f"YOLODAM_v3_best_e{epoch+1}.h5")
            model.save_weights(path)
            log(f"  Saved best: {path}  loss={epoch_loss:.4f}")

    log_file.write(f"\nTraining ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.close()
    log("Training complete!")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("YOLO-DAM v3 Training Configuration")
    print("="*60)
    print(f"✅ Upgraded backbone: C3k2 blocks (YOLO26 architecture)")
    print(f"✅ Advanced augmentation — {ENABLE_ADVANCED_AUG}")
    print(f"✅ One-to-One matching — {ENABLE_O2O_MATCHING}")
    print(f"✅ Label smoothing — {USE_LABEL_SMOOTHING}")
    print(f"✅ Cosine annealing LR — {COSINE_ANNEALING}")
    print(f"💾 Model: 67.1M params (width=1.0, depth=1.0)")
    print(f"📊 Backbone: C3k2 (YOLO26 style, better feature extraction)")
    print("="*60 + "\n")

    # Load merged v26 backbone
    print(f"Loading merged weights: {WEIGHTS_PATH}")
    try:
        model.load_weights(WEIGHTS_PATH)
        print("[OK] Loaded merged weights (v26 backbone + new DAM heads)")
    except Exception as e:
        print(f"[WARNING] Could not load weights: {e}")
        print("Starting training with random initialization...")

    print("Starting training...")
    training(model)
