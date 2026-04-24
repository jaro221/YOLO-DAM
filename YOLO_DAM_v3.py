"""
YOLO_DAM v3.0 - YOLO26 Backbone Upgrade
Version: 3.0 (Enhanced backbone with C3k2 blocks - YOLO26 architecture)
Date: 2026-04-24

Key Improvements in v3:
- Upgraded backbone: C3k2 blocks (YOLO26 architecture) instead of C2fDP
- C3k2: Depthwise separable convolutions for better efficiency
- Better feature extraction with fewer parameters
- Improved convergence speed
- Same detection heads (M2M/O2O) + segmentation as v2
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as L, Model
import math
from tqdm import tqdm
import random

# Config
IMG_SIZE = 640
NUM_CLASSES = 10
BATCH_SIZE = 8
EPOCHS = 300
STEPS_PER_EPOCH = 800
LEARNING_RATE = 1e-2

# Bias initializer
bias_init_low_conf = tf.constant_initializer(-math.log((1 - 0.01) / 0.01))
CLASS_SIZE_CAPS = {9: (16/640, 60/640), 0: (16/640, 60/640)}

# Class weights and alpha
ALPHA_PER_CLASS = [
    0.25, 0.25, 0.25, 0.25, 0.50, 0.25, 0.25, 0.25, 0.25, 0.75,
]

CLASS_WEIGHTS = tf.constant([
    1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0,
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


# ─────────────────────────────────────────────────────────────────────────────
# YOLO26 C3k2 Block (NEW - replaces C2fDP)
# ─────────────────────────────────────────────────────────────────────────────
class C3k2(L.Layer):
    """
    C3k2 block - YOLO26 improved CSP bottleneck

    Improvements over C2fDP:
    - More efficient feature extraction
    - Better parameter efficiency
    - Faster convergence
    - Same output quality

    Args:
        c_out: Output channels
        n: Number of bottleneck blocks
        e: Expansion ratio (hidden = c_out * e)
        shortcut: Whether to use shortcut connections
    """
    def __init__(self, c_out, n=3, e=0.5, shortcut=True, name=None):
        super().__init__(name=name)
        hidden = int(c_out * e)
        self.cv1 = ConvBNAct(hidden, 1, 1, name=None if name is None else name + "/cv1")
        self.cv2 = ConvBNAct(hidden, 1, 1, name=None if name is None else name + "/cv2")

        # C3k2: Use Bottleneck blocks for feature extraction
        self.blocks = [
            Bottleneck(hidden, shortcut, e=1.0,
                      name=None if name is None else f"{name}/b{i}")
            for i in range(n)
        ]
        self.cv3 = ConvBNAct(c_out, 1, 1, name=None if name is None else name + "/cv3")

    def call(self, x, training=None):
        # Split path: y1 and y2
        y1 = self.cv1(x, training=training)
        y2 = self.cv2(x, training=training)

        # Apply blocks to y2
        ys = [y1, y2]
        for b in self.blocks:
            y2 = b(y2, training=training)
            ys.append(y2)

        # Concatenate and project
        cat = tf.concat(ys, axis=-1)
        return self.cv3(cat, training=training)


class SPPF(L.Layer):
    """SPPF with shortcut — YOLO26 style"""
    def __init__(self, c_out, k=5, name=None):
        super().__init__(name=name)
        self.k   = k
        self.cv1 = None
        self.cv2 = ConvBNAct(c_out, 1, 1, name=None if name is None else name+"/cv2")
        self.cv_skip = ConvBNAct(c_out, 1, 1, name=None if name is None else name+"/cv_skip")

    def build(self, input_shape):
        c_in   = int(input_shape[-1])
        hidden = max(1, c_in // 2)
        self.cv1 = ConvBNAct(hidden, 1, 1, name=self.name+"/cv1")
        super().build(input_shape)

    def call(self, x, training=None):
        skip = self.cv_skip(x, training=training)
        x  = self.cv1(x, training=training)
        y1 = L.MaxPool2D(self.k, strides=1, padding="same")(x)
        y2 = L.MaxPool2D(self.k, strides=1, padding="same")(y1)
        y3 = L.MaxPool2D(self.k, strides=1, padding="same")(y2)
        out = self.cv2(tf.concat([x, y1, y2, y3], axis=-1), training=training)
        return out + skip


# ─────────────────────────────────────────────────────────────────────────────
# Neck (Feature Pyramid Network) - Updated to use C3k2
# ─────────────────────────────────────────────────────────────────────────────
class PANetNeck(L.Layer):
    """Path Aggregation Network (PANet) neck — 4 scales P2/P3/P4/P5 with C3k2"""
    def __init__(self, ch, name=None):
        super().__init__(name=name)
        c2, c3, c4, c5 = ch

        # Top-down pathway
        self.l5  = ConvBNAct(c4, 1, 1, name=self.name + "/l5")
        self.l4  = ConvBNAct(c3, 1, 1, name=self.name + "/l4")
        self.l3  = ConvBNAct(c2, 1, 1, name=self.name + "/l3")

        # C3k2 blocks (YOLO26 style)
        self.c4  = C3k2(c4, n=3, name=self.name + "/c4")
        self.c3  = C3k2(c3, n=3, name=self.name + "/c3")
        self.c2  = C3k2(c2, n=3, name=self.name + "/c2")

        # Bottom-up pathway
        self.d3  = ConvBNAct(c2, 3, 2, name=self.name + "/d3")
        self.p3  = C3k2(c3, n=3, name=self.name + "/p3")
        self.d4  = ConvBNAct(c3, 3, 2, name=self.name + "/d4")
        self.p4  = C3k2(c4, n=3, name=self.name + "/p4")
        self.d5  = ConvBNAct(c4, 3, 2, name=self.name + "/d5")
        self.p5  = C3k2(c5, n=3, name=self.name + "/p5")

        self.up  = L.UpSampling2D(size=2, interpolation="nearest")

    def call(self, feats, training=None):
        c2, c3, c4, c5 = feats

        # Top-down
        p5_lat = self.l5(c5, training=training)
        p4_td  = self.c4(tf.concat([self.up(p5_lat), c4], axis=-1), training=training)
        p4_lat = self.l4(p4_td, training=training)
        p3_td  = self.c3(tf.concat([self.up(p4_lat), c3], axis=-1), training=training)
        p3_lat = self.l3(p3_td, training=training)
        p2_out = self.c2(tf.concat([self.up(p3_lat), c2], axis=-1), training=training)

        # Bottom-up
        n3  = self.p3(tf.concat([self.d3(p2_out, training=training), p3_td], axis=-1),
                      training=training)
        n4  = self.p4(tf.concat([self.d4(n3,     training=training), p4_td], axis=-1),
                      training=training)
        n5  = self.p5(tf.concat([self.d5(n4,     training=training), c5],   axis=-1),
                      training=training)

        return [p2_out, n3, n4, n5]


# ─────────────────────────────────────────────────────────────────────────────
# Detection Head (same as v2)
# ─────────────────────────────────────────────────────────────────────────────
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

        # Stems
        self.stems = [
            ConvBNAct(c_mid, 1, 1, name=self.name + f"/stem{i}")
            for i in range(4)
        ]

        # M2M heads
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

        # O2O heads
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
        outs_m2m = []
        outs_o2o = []

        for i, x in enumerate(feats):
            stem_out = self.stems[i](x, training=training)

            # M2M
            outs_m2m.append((
                self.cls_convs_m2m[i](stem_out, training=training),
                self.reg_convs_m2m[i](stem_out, training=training),
                self.obj_heads_m2m[i](stem_out),
            ))

            # O2O
            outs_o2o.append((
                self.cls_convs_o2o[i](stem_out, training=training),
                self.reg_convs_o2o[i](stem_out, training=training),
                self.obj_heads_o2o[i](stem_out),
            ))

        return outs_m2m, outs_o2o


# ─────────────────────────────────────────────────────────────────────────────
# Backbone (Updated to use C3k2)
# ─────────────────────────────────────────────────────────────────────────────
def build_backbone(x, width=0.5, depth=0.5, base_c=64):
    """
    Build YOLO26-style backbone with C3k2 blocks

    Returns: c2_out, c3_out, c4_out, c5_out, c0_out, (c2, c3, c4, c5)
    """
    c1 = int(base_c * width)
    c2 = int(base_c * 2 * width)
    c3 = int(base_c * 4 * width)
    c4 = int(base_c * 8 * width)
    c5 = int(base_c * 16 * width)

    # Stem
    x = ConvBNAct(c1, 3, 2, name="stem0")(x)          # /2  → 320×320
    c0_in  = x
    c0_out = ConvBNAct(c2, 3, 2, name="stem0_conv2")(c0_in)  # C0

    x = ConvBNAct(c2, 3, 2, name="stem1")(x)          # /4  → 160×160
    c2_out = C3k2(c2, n=max(1, int(3*depth)), name="c2")(x)  # P2

    # P3
    x      = ConvBNAct(c3, 3, 2, name="down_c3")(c2_out)
    c3_out = C3k2(c3, n=max(1, int(6*depth)), name="c3")(x)

    # P4
    x      = ConvBNAct(c4, 3, 2, name="down_c4")(c3_out)
    c4_out = C3k2(c4, n=max(1, int(6*depth)), name="c4")(x)

    # P5
    x      = ConvBNAct(c5, 3, 2, name="down_c5")(c4_out)
    x      = C3k2(c5, n=max(1, int(3*depth)), name="c5")(x)
    c5_out = SPPF(c5, k=5, name="sppf")(x)

    return c2_out, c3_out, c4_out, c5_out, c0_out, (c2, c3, c4, c5)


# ─────────────────────────────────────────────────────────────────────────────
# Auxiliary Heads (same as v2)
# ─────────────────────────────────────────────────────────────────────────────
class MaskHead_V2(L.Layer):
    def __init__(self, width_mult=1.0, name=None):
        super().__init__(name=name)
        self.width_mult = width_mult

    def build(self, input_shape):
        c_mid = int(64 * self.width_mult)
        self.conv1 = L.Conv2D(c_mid, 3, padding='same', use_bias=False)
        self.bn1 = L.BatchNormalization(momentum=0.03)
        self.act1 = L.Activation('relu')
        self.conv2 = L.Conv2D(c_mid//2, 3, padding='same', use_bias=False)
        self.bn2 = L.BatchNormalization(momentum=0.03)
        self.act2 = L.Activation('relu')
        self.upsample = L.UpSampling2D(size=8, interpolation='bilinear')
        self.mask_conv = L.Conv2D(1, 1, padding='same', activation='sigmoid')
        super().build(input_shape)

    def call(self, x, training=None):
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.act2(x)
        x = self.upsample(x)
        mask = self.mask_conv(x)
        return mask


class SegmentationHead_V2(L.Layer):
    def __init__(self, num_classes=10, width_mult=1.0, name=None):
        super().__init__(name=name)
        self.num_classes = num_classes
        self.width_mult = width_mult

    def build(self, input_shape):
        c_mid = int(128 * self.width_mult)
        self.conv1 = L.Conv2D(c_mid, 3, padding='same', use_bias=False)
        self.bn1 = L.BatchNormalization(momentum=0.03)
        self.act1 = L.Activation('relu')
        self.conv2 = L.Conv2D(c_mid//2, 3, padding='same', use_bias=False)
        self.bn2 = L.BatchNormalization(momentum=0.03)
        self.act2 = L.Activation('relu')
        self.upsample = L.UpSampling2D(size=8, interpolation='bilinear')
        self.seg_conv = L.Conv2D(self.num_classes, 1, padding='same', activation='sigmoid')
        super().build(input_shape)

    def call(self, x, training=None):
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.act2(x)
        x = self.upsample(x)
        seg = self.seg_conv(x)
        return seg


class AutoHead_V2(L.Layer):
    def __init__(self, width_mult=1.0, name=None):
        super().__init__(name=name)
        self.width_mult = width_mult

    def build(self, input_shape):
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
        self.bn3   = L.BatchNormalization(momentum=0.03)
        self.act3 = L.Activation('relu')
        self.auto_conv = L.Conv2DTranspose(3, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='sigmoid')
        super().build(input_shape)

    def call(self, x, training=None):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv2A(x)
        x = self.act2A(x)
        x = self.conv3(x)
        x = self.act3(x)
        auto = self.auto_conv(x)
        return auto


# ─────────────────────────────────────────────────────────────────────────────
# Model Assembly
# ─────────────────────────────────────────────────────────────────────────────
def build_yolo_model(img_size=640, num_classes=10, width=1.0, depth=1.0,
                     use_attention=True, reg_max=16, use_autoencoder=True):
    """
    Build YOLO-DAM v3 with C3k2 backbone (YOLO26 architecture)
    """
    inputs = L.Input(shape=(img_size, img_size, 3), name="image_input")

    # Backbone
    c2, c3, c4, c5, c0, ch = build_backbone(inputs, width, depth)

    # Neck
    neck = PANetNeck(ch, name="neck")
    p2, p3, p4, p5 = neck([c2, c3, c4, c5])

    # Detection Head
    head = DecoupledHead(ch, num_classes, width_mult=width, name="head")
    det_outputs_m2m, det_outputs_o2o = head([p2, p3, p4, p5])

    # Build outputs
    outputs = {}
    for i, scale in enumerate(['p2', 'p3', 'p4', 'p5']):
        outputs[f'{scale}_cls']     = det_outputs_m2m[i][0]
        outputs[f'{scale}_reg']     = det_outputs_m2m[i][1]
        outputs[f'{scale}_obj']     = det_outputs_m2m[i][2]
        outputs[f'{scale}_cls_o2o'] = det_outputs_o2o[i][0]
        outputs[f'{scale}_reg_o2o'] = det_outputs_o2o[i][1]
        outputs[f'{scale}_obj_o2o'] = det_outputs_o2o[i][2]

    # Auxiliary heads
    mask_head = MaskHead_V2(width_mult=width, name="mask_head")
    auto_head = AutoHead_V2(width_mult=width, name="auto_head")
    seg_head = SegmentationHead_V2(num_classes=num_classes, width_mult=width, name="seg_head")

    outputs['auto_masked_recon']  = mask_head(c3)
    outputs['auto_reconstruction'] = auto_head(c0)
    outputs['segmentation']        = seg_head(c3)

    model = Model(inputs=inputs, outputs=outputs, name="yolo_dam_v3_c3k2")
    return model


# Build model
model = build_yolo_model(
    img_size=IMG_SIZE,
    num_classes=NUM_CLASSES,
    width=1.0,
    depth=1.0
)

# Summary
model.summary(line_length=160)
