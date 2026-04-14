
import os
import tensorflow as tf
import numpy as np
from YOLO_DAM_dataset import yolo_dataset_with_augmentation
from YOLO_DAM_loss import detection_loss
from YOLO_DAM import model

DATASET_DIR = r"D:\Projekty\2022_01_BattPor\2025_12_Dresden\YOLOv8\dataset"

def training(model, epochs=30):
    """
    Continue training with fixed target assignment
    """

       
    train_ds = yolo_dataset_with_augmentation(
        os.path.join(DATASET_DIR, "images", "train"),
        os.path.join(DATASET_DIR, "labels", "train"),
        os.path.join(DATASET_DIR, "restored", "train"),
        batch_size=4,
        augment=True
    )
    
    loss_all=[]
    loss_comps=[]
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)  # Low LR
    loss_save = np.inf
    
    @tf.function
    def train_step(imgs, batch):
        with tf.GradientTape() as tape:
            preds = model(imgs, training=True)
            loss, comps = detection_loss(preds, batch)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss, comps
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        running_loss = 0.0
        step = 0
        
        for batch in train_ds.take(800):
            loss, comps = train_step(batch['image'], batch)
            running_loss += float(loss)
            step += 1
                               
            
            if step % 100 == 0:
                print(f"  Step {step}: Loss={loss:.3f}, "
                      f"Box={comps['p3_box']:.3f}, "
                      f"Obj={comps['p3_obj']:.3f}, "
                      f"Cls={comps['p3_cls']:.3f}")
            
            if step == 400:
                for scale in ['p3', 'p4', 'p5']:
                    cls_t = batch[f'{scale}_cls_t']
                    obj_t = batch[f'{scale}_obj_t']
                    print(f" Scale = {scale}: cls={tf.reduce_sum(tf.cast(cls_t>0, tf.float32)).numpy()}, "
                          f"obj={tf.reduce_sum(tf.cast(obj_t>0, tf.float32)).numpy()}")
        
        print(f"Epoch {epoch+1} Loss: {running_loss/step:.4f}")
        loss_all.append(running_loss/step)
        loss_comps.append(comps)
        
        if loss_save>np.float32(running_loss/step):
            loss_save=np.float32(running_loss/step)
            print(f"Saved epoch {epoch+1} Loss: {running_loss/step:.4f}")
            model.save_weights(f"D:/Projekty/2022_01_BattPor/2025_12_Dresden/Models/MODEL_Better_v6_e{epoch}.h5")
    
    print("\n✅ Training complete!")
    return loss_all, loss_comps




loss_all, loss_comps = training(model, epochs=500)