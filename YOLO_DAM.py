import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as L, Model
import math
from tqdm import tqdm
import random

# ----------------- Hyperparameters / config -----------------
IMG_SIZE = 640
NUM_CLASSES = 10
BATCH_SIZE = 8
EPOCHS = 300
STEPS_PER_EPOCH = 800
LEARNING_RATE = 1e-2

IMG_SIZE = 640
NUM_CLASSES = 10

# Bias initializer for low confidence predictions
bias_init_low_conf = tf.constant_initializer(-math.log((1 - 0.01) / 0.01))
CLASS_SIZE_CAPS = {9: (16/640, 60/640),
                   0: (16/640, 60/640),# min=0.025, max=0.094
}

# ----------------- Model Building Blocks -----------------
ALPHA_PER_CLASS = [
    0.25,  # 0 Agglomerate
    0.25,  # 1 Pinhole-long
    0.25,  # 2 Pinhole-trans
    0.25,  # 3 Pinhole-round
    0.50,  # 4 Crack-long     ← boost rare class
    0.25,  # 5 Crack-trans
    0.25,  # 6 Line-long
    0.25,  # 7 Line-trans
    0.25,  # 8 Line-diag
    0.75,  # 9 Foreign-particle ← high alpha = penalise FP more
]

CLASS_WEIGHTS = tf.constant([
    1.0,   # 0 Agglomerate    — 1820 instances
    1.0,   # 1 Pinhole-long   — 1851
    1.0,   # 2 Pinhole-trans  — 2516
    1.0,   # 3 Pinhole-round  — 1530
    2.0,   # 4 Crack-long     — 1145 (fewest)
    1.0,   # 5 Crack-trans    — 2229
    1.0,   # 6 Line-long      — 2051
    1.0,   # 7 Line-trans     — 2006
    1.0,   # 8 Line-diag      — 1502
    2.0,   # 9 Foreign-particle — 1576 but hardest
], dtype=tf.float32)


def SiLU(x): 
    return tf.nn.silu(x)


class ConvBNAct(L.Layer):
    """Standard convolution with batch norm and SiLU activation"""
    def __init__(self, filters, k=1, s=1, g=1, act=True, name=None):
        super().__init__(name=name)
        self.conv = L.Conv2D(filters, k, s, padding="same", use_bias=False, groups=g,
                             name=None if name is None else name + "/conv")
        self.bn = L.BatchNormalization(name=None if name is None else name + "/bn")
        self.act = L.Activation(SiLU, name=None if name is None else name + "/silu") if act else None

    def call(self, x, training=None):
        x = self.conv(x)
        x = self.bn(x, training=training)
        if self.act is not None:
            x = self.act(x)
        return x


class Bottleneck(L.Layer):
    """Standard bottleneck block"""
    def __init__(self, c, shortcut=True, e=0.5, name=None):
        super().__init__(name=name)
        hidden = int(c * e)
        self.cv1 = ConvBNAct(hidden, 1, 1, name=None if name is None else name + "/cv1")
        self.cv2 = ConvBNAct(c, 3, 1, name=None if name is None else name + "/cv2")
        self.shortcut = shortcut

    def call(self, x, training=None):
        y = self.cv2(self.cv1(x, training=training), training=training)
        return x + y if self.shortcut else y


class C2fDP(L.Layer):
    """CSP Bottleneck with 2 convolutions (YOLOv8/v11 style)"""
    def __init__(self, c_out, n=2, e=0.5, shortcut=True, name=None):
        super().__init__(name=name)
        hidden = int(c_out * e)
        self.cv1 = ConvBNAct(hidden, 1, 1, name=None if name is None else name + "/cv1")
        self.cv2 = ConvBNAct(hidden, 1, 1, name=None if name is None else name + "/cv2")
        self.blocks = [Bottleneck(hidden, shortcut, e=1.0, 
                                 name=None if name is None else f"{name}/b{i}") 
                      for i in range(n)]
        self.cv3 = ConvBNAct(c_out, 1, 1, name=None if name is None else name + "/cv3")

    def call(self, x, training=None):
        y1 = self.cv1(x, training=training)
        y2 = self.cv2(x, training=training)
        ys = [y1, y2]
        for b in self.blocks:
            y2 = b(y2, training=training)
            ys.append(y2)
        cat = tf.concat(ys, axis=-1)
        return self.cv3(cat, training=training)


class SPPF(L.Layer):
    """SPPF with shortcut — YOLO26 style"""
    def __init__(self, c_out, k=5, name=None):
        super().__init__(name=name)
        self.k   = k
        self.cv1 = None
        self.cv2 = ConvBNAct(c_out, 1, 1,
                              name=None if name is None else name+"/cv2")
        # ✅ YOLO26: projection for shortcut
        self.cv_skip = ConvBNAct(c_out, 1, 1,
                                  name=None if name is None else name+"/cv_skip")

    def build(self, input_shape):
        c_in   = int(input_shape[-1])
        hidden = max(1, c_in // 2)
        self.cv1 = ConvBNAct(hidden, 1, 1,
                               name=self.name+"/cv1")
        super().build(input_shape)

    def call(self, x, training=None):
        skip = self.cv_skip(x, training=training)  # ✅ shortcut from input

        x  = self.cv1(x, training=training)
        y1 = L.MaxPool2D(self.k, strides=1, padding="same")(x)
        y2 = L.MaxPool2D(self.k, strides=1, padding="same")(y1)
        y3 = L.MaxPool2D(self.k, strides=1, padding="same")(y2)

        out = self.cv2(
            tf.concat([x, y1, y2, y3], axis=-1),
            training=training
        )
        return out + skip   # ✅ residual addition


# ----------------- Neck (Feature Pyramid Network) -----------------

class PANetNeck(L.Layer):
    """Path Aggregation Network (PANet) neck — 4 scales P2/P3/P4/P5"""
    def __init__(self, ch, name=None):
        super().__init__(name=name)
        c2, c3, c4, c5 = ch   # ← unpack 4 channels

        # ── Top-down pathway ──────────────────────────────────
        self.l5  = ConvBNAct(c4, 1, 1, name=self.name + "/l5")
        self.l4  = ConvBNAct(c3, 1, 1, name=self.name + "/l4")
        self.l3  = ConvBNAct(c2, 1, 1, name=self.name + "/l3")   # ← NEW

        self.c4  = C2fDP(c4, n=2, name=self.name + "/c4")
        self.c3  = C2fDP(c3, n=2, name=self.name + "/c3")
        self.c2  = C2fDP(c2, n=2, name=self.name + "/c2")         # ← NEW

        # ── Bottom-up pathway ─────────────────────────────────
        self.d3  = ConvBNAct(c2, 3, 2, name=self.name + "/d3")   # ← NEW p2→p3
        self.p3  = C2fDP(c3, n=2, name=self.name + "/p3")         # ← NEW
        self.d4  = ConvBNAct(c3, 3, 2, name=self.name + "/d4")
        self.p4  = C2fDP(c4, n=2, name=self.name + "/p4")
        self.d5  = ConvBNAct(c4, 3, 2, name=self.name + "/d5")
        self.p5  = C2fDP(c5, n=2, name=self.name + "/p5")

        self.up  = L.UpSampling2D(size=2, interpolation="nearest")

    def call(self, feats, training=None):
        c2, c3, c4, c5 = feats   # ← unpack 4 inputs

        # ── Top-down ──────────────────────────────────────────
        p5_lat = self.l5(c5, training=training)
        p4_td  = self.c4(tf.concat([self.up(p5_lat), c4], axis=-1), training=training)
        p4_lat = self.l4(p4_td, training=training)
        p3_td  = self.c3(tf.concat([self.up(p4_lat), c3], axis=-1), training=training)
        p3_lat = self.l3(p3_td, training=training)
        p2_out = self.c2(tf.concat([self.up(p3_lat), c2], axis=-1), training=training)  # ← NEW

        # ── Bottom-up ─────────────────────────────────────────
        n3  = self.p3(tf.concat([self.d3(p2_out, training=training), p3_td], axis=-1),
                      training=training)                                                  # ← NEW
        n4  = self.p4(tf.concat([self.d4(n3,     training=training), p4_td], axis=-1),
                      training=training)
        n5  = self.p5(tf.concat([self.d5(n4,     training=training), c5],   axis=-1),
                      training=training)

        return [p2_out, n3, n4, n5]   # ← 4 output


# ----------------- Detection Head -----------------

class DecoupledHead(tf.keras.layers.Layer):
    def __init__(self, ch_in, num_classes, width_mult=1.0, name=None):
        super().__init__(name=name)
        self.num_classes = num_classes
        c_mid = int(256 * width_mult)

        cls_bias_init = bias_init_low_conf
        obj_bias_inits = [
            tf.constant_initializer(-math.log((1 - 0.004) / 0.004)),  # P2
            tf.constant_initializer(-math.log((1 - 0.016) / 0.016)),  # P3
            tf.constant_initializer(-math.log((1 - 0.060) / 0.060)),  # P4
            tf.constant_initializer(-math.log((1 - 0.250) / 0.250)),  # P5
        ]

        # ── Stems ─────────────────────────────────────────────
        self.stems = [
            ConvBNAct(c_mid, 1, 1, name=self.name + f"/stem{i}")
            for i in range(4)
        ]

        # ── One-to-MANY heads (existing) ──────────────────────
        self.cls_convs_m2m = []
        for i in range(4):
            seq = tf.keras.Sequential([
                ConvBNAct(c_mid, 3, 1, name=f"m2m_cls{i}_conv1"),
                ConvBNAct(c_mid, 3, 1, name=f"m2m_cls{i}_conv2"),
                tf.keras.layers.Conv2D(num_classes, 1, 1,
                    name=f"m2m_cls{i}_conv3",
                    bias_initializer=cls_bias_init)
            ], name=f"m2m_cls_head_{i}")
            self.cls_convs_m2m.append(seq)

        self.reg_convs_m2m = []
        for i in range(4):
            seq = tf.keras.Sequential([
                ConvBNAct(c_mid, 3, 1, name=f"m2m_reg{i}_conv1"),
                ConvBNAct(c_mid, 3, 1, name=f"m2m_reg{i}_conv2"),
                tf.keras.layers.Conv2D(4, 1, 1, name=f"m2m_reg{i}_conv3")
            ], name=f"m2m_reg_head_{i}")
            self.reg_convs_m2m.append(seq)

        self.obj_heads_m2m = [
            tf.keras.layers.Conv2D(1, 1, 1,
                name=f"m2m_obj{i}_conv",
                bias_initializer=obj_bias_inits[i])
            for i in range(4)
        ]

        # ── One-to-ONE heads (new) ────────────────────────────
        self.cls_convs_o2o = []
        for i in range(4):
            seq = tf.keras.Sequential([
                ConvBNAct(c_mid, 3, 1, name=f"o2o_cls{i}_conv1"),
                ConvBNAct(c_mid, 3, 1, name=f"o2o_cls{i}_conv2"),
                tf.keras.layers.Conv2D(num_classes, 1, 1,
                    name=f"o2o_cls{i}_conv3",
                    bias_initializer=cls_bias_init)
            ], name=f"o2o_cls_head_{i}")
            self.cls_convs_o2o.append(seq)

        self.reg_convs_o2o = []
        for i in range(4):
            seq = tf.keras.Sequential([
                ConvBNAct(c_mid, 3, 1, name=f"o2o_reg{i}_conv1"),
                ConvBNAct(c_mid, 3, 1, name=f"o2o_reg{i}_conv2"),
                tf.keras.layers.Conv2D(4, 1, 1, name=f"o2o_reg{i}_conv3")
            ], name=f"o2o_reg_head_{i}")
            self.reg_convs_o2o.append(seq)

        self.obj_heads_o2o = [
            tf.keras.layers.Conv2D(1, 1, 1,
                name=f"o2o_obj{i}_conv",
                bias_initializer=obj_bias_inits[i])
            for i in range(4)
        ]

    def call(self, feats, training=None):
        outs_m2m = []   # one-to-many (training only)
        outs_o2o = []   # one-to-one  (training + inference)

        for i, x in enumerate(feats):
            stem_out = self.stems[i](x, training=training)

            # One-to-many
            outs_m2m.append((
                self.cls_convs_m2m[i](stem_out, training=training),
                self.reg_convs_m2m[i](stem_out, training=training),
                self.obj_heads_m2m[i](stem_out),
            ))

            # One-to-one
            outs_o2o.append((
                self.cls_convs_o2o[i](stem_out, training=training),
                self.reg_convs_o2o[i](stem_out, training=training),
                self.obj_heads_o2o[i](stem_out),
            ))

        return outs_m2m, outs_o2o

# ----------------- Backbone Builder -----------------

def build_backbone(x, width=0.5, depth=0.5, base_c=64):
    """
    Build CSPDarknet backbone

    Returns:
        c2_out, c3_out, c4_out, c5_out: Feature maps at different scales
        c0_out: Early feature for autoencoder
        (c2, c3, c4, c5): Channel counts for each scale
    """
    c1 = int(base_c * width)
    c2 = int(base_c * 2 * width)
    c3 = int(base_c * 4 * width)
    c4 = int(base_c * 8 * width)
    c5 = int(base_c * 16 * width)

    # Stem
    x = ConvBNAct(c1, 3, 2, name="stem0")(x)          # /2  → 320×320
    c0_in  = x
    c0_out = ConvBNAct(c2, 3, 2, name="stem0_conv2")(c0_in)  # C0 → autoencoder

    x = ConvBNAct(c2, 3, 2, name="stem1")(x)          # /4  → 160×160
    c2_out = C2fDP(c2, n=max(1, int(3*depth)), name="c2")(x)  # ← NEW: capture P2

    # Stage 3 (P3)
    x      = ConvBNAct(c3, 3, 2, name="down_c3")(c2_out)     # ← feed c2_out
    c3_out = C2fDP(c3, n=max(1, int(6*depth)), name="c3")(x)

    # Stage 4 (P4)
    x      = ConvBNAct(c4, 3, 2, name="down_c4")(c3_out)
    c4_out = C2fDP(c4, n=max(1, int(6*depth)), name="c4")(x)

    # Stage 5 (P5)
    x      = ConvBNAct(c5, 3, 2, name="down_c5")(c4_out)
    x      = C2fDP(c5, n=max(1, int(3*depth)), name="c5")(x)
    c5_out = SPPF(c5, k=5, name="sppf")(x)

    return c2_out, c3_out, c4_out, c5_out, c0_out, (c2, c3, c4, c5)  # ← updated

    
# ============================================================================
# IMPROVEMENT 5: Enhanced Backbone
# ============================================================================



class MaskHead_V2(L.Layer):
    """
    ✅ CORRECTED: Defect mask head
    
    Input: p3 features [B, 80, 80, C]
    Output: defect mask [B, 640, 640, 1] float32, 1=good, 0=defect
    """
    def __init__(self, width_mult=1.0, name=None):
        super().__init__(name=name)
        self.width_mult = width_mult
        
    def build(self, input_shape):
        c_mid = int(64 * self.width_mult)
        
        # Process features
        self.conv1 = L.Conv2D(c_mid, 3, padding='same', use_bias=False)
        self.bn1 = L.BatchNormalization(momentum=0.03)
        self.act1 = L.Activation('relu')
        
        self.conv2 = L.Conv2D(c_mid//2, 3, padding='same', use_bias=False)
        self.bn2 = L.BatchNormalization(momentum=0.03)
        self.act2 = L.Activation('relu')
        
        # ✅ Upsample to 640x640 (80 → 640 is 8x)
        self.upsample = L.UpSampling2D(size=8, interpolation='bilinear')
        
        # ✅ Final mask prediction (1 channel only!)
        self.mask_conv = L.Conv2D(1, 1, padding='same', activation='sigmoid')
        
        super().build(input_shape)
    
    def call(self, x, training=None):
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.act1(x)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.act2(x)
        
        x = self.upsample(x)      # [B, 640, 640, C]
        mask = self.mask_conv(x)  # [B, 640, 640, 1]
        
        return mask  # float32, values in [0, 1]
    
class AutoHead_V2(L.Layer):
    """
    ✅ CORRECTED: Defect mask head
    
    Input: p3 features [B, 80, 80, C]
    Output: defect mask [B, 640, 640, 1] float32, 1=good, 0=defect
    """
    def __init__(self, width_mult=1.0, name=None):
        super().__init__(name=name)
        self.width_mult = width_mult
        
    def build(self, input_shape):

        
        # Process features
        self.conv1 = L.Conv2DTranspose(128, kernel_size=(5, 5), strides=(1, 1), padding='same', use_bias=False)
        self.bn1 = L.BatchNormalization(momentum=0.03)
        self.act1 = L.Activation('relu')
        
        self.conv2 = L.Conv2DTranspose(64, kernel_size=(3, 3), strides=(2, 2), padding='same', use_bias=False)
        self.bn2 = L.BatchNormalization(momentum=0.03)
        self.act2 = L.Activation('relu')
        
        self.conv2A = L.Conv2DTranspose(64, kernel_size=(3, 3), strides=(2, 2), padding='same', use_bias=False)
        self.bn2A = L.BatchNormalization(momentum=0.03)
        self.act2A = L.Activation('relu')
        
        self.conv3 = L.Conv2DTranspose(32, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)
        self.bn3   = L.BatchNormalization(momentum=0.03)   # ← ADD
        self.act3 = L.Activation('relu')
        self.auto_conv = L.Conv2DTranspose(3, kernel_size=(3, 3),strides=(1, 1), padding='same', activation='sigmoid')
        
        super().build(input_shape)
    
    def call(self, x, training=None):
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.act2(x)

        x = self.conv2A(x)
        x = self.bn2A(x, training=training)
        x = self.act2A(x)

        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.act3(x)


        auto = self.auto_conv(x)  # [B, 640, 640, 3]

        return auto  # float32, RGB values in [0, 1]
     
    


# ============================================================================
# IMPROVEMENT 6: Complete Model Assembly
# ============================================================================

def build_yolo_model(img_size=640, num_classes=10, width=0.5, depth=0.5,
                     use_attention=True, reg_max=16, use_autoencoder=True):
    """
    Build YOLO-DAM with P2 scale (160×160) for small defect detection.
    Heads: Detection (P2/P3/P4/P5) + Mask + Autoencoder
    """

    inputs = L.Input(shape=(img_size, img_size, 3), name="image_input")

    # ── Backbone ─────────────────────────────────────────────
    # Returns: c2_out, c3_out, c4_out, c5_out, c0_out, (c2,c3,c4,c5)
    c2, c3, c4, c5, c0, ch = build_backbone(inputs, width, depth)

    # ── Neck ──────────────────────────────────────────────────
    neck = PANetNeck(ch, name="neck")
    p2, p3, n4, n5 = neck([c2, c3, c4, c5])   # ← 4 scales now

    # ── Detection Head ────────────────────────────────────────
    head = DecoupledHead(ch, num_classes, width_mult=width, name="head")
    det_outputs_m2m, det_outputs_o2o = head([p2, p3, n4, n5])

    # ── Build outputs dict ────────────────────────────────────
    outputs = {}
    for i, scale in enumerate(['p2', 'p3', 'p4', 'p5']):
        outputs[f'{scale}_cls']     = det_outputs_m2m[i][0]
        outputs[f'{scale}_reg']     = det_outputs_m2m[i][1]
        outputs[f'{scale}_obj']     = det_outputs_m2m[i][2]
        
        # One-to-one (used in training loss + inference)
        outputs[f'{scale}_cls_o2o'] = det_outputs_o2o[i][0]
        outputs[f'{scale}_reg_o2o'] = det_outputs_o2o[i][1]
        outputs[f'{scale}_obj_o2o'] = det_outputs_o2o[i][2]

    # ── Auxiliary heads ───────────────────────────────────────
    mask_head = MaskHead_V2(width_mult=width, name="mask_head")
    auto_head = AutoHead_V2(width_mult=width, name="auto_head")

    outputs['auto_masked_recon']  = mask_head(c3)   # C3 → 80×80 → mask
    outputs['auto_reconstruction'] = auto_head(c0)  # C0 → 160×160 → recon

    model = Model(inputs=inputs, outputs=outputs,
                  name="yolo_dam_p2p3p4p5")
    return model

# ----------------- Usage -----------------


# Build model (NEW CONFIG: width=1.0, depth=1.0)
# Expected: 87.4M params, 8-10GB VRAM, +8-12% recall vs width=0.6
model = build_yolo_model(
    img_size=IMG_SIZE,
    num_classes=NUM_CLASSES,
    width=1.0,   # Standard YOLO width (v26 equivalent)
    depth=1.0    # Standard YOLO depth (better feature learning)
)

# Display model summary
model.summary(line_length=160)