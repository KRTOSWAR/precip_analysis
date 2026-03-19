"""
models.py — All four model architectures for full-image downscaling.

Key change from patch-based version
------------------------------------
The UNet backbone now operates on the full Mozambique domain
(CHIRPS_H × CHIRPS_W) rather than on small patches.  Because the UNet
has 3 MaxPool2D(×2) stages, the spatial dimensions must be divisible by
2³ = 8.  We zero-pad the input by (PAD_H, PAD_W) = (4, 2) before the
first conv and crop back after the last decoder block.  This is invisible
to the caller — input and output always have shape (CHIRPS_H, CHIRPS_W).

Model summary
─────────────
build_unet_mse()      → (H, W, 1)   ReLU            [MSE loss]
build_unet_bg()       → (H, W, 3)   custom acts      [Bernoulli-Gamma NLL]
build_unet_compound() → (H, W, 1)   ReLU            [Compound FSS+MSE]
build_wgan()          → generator (H,W,1) + discriminator (PatchGAN critic)
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import tensorflow.keras.backend as K

from . import config as C


# ─────────────────────────────────────────────────────────────────────────────
# Residual block
# ─────────────────────────────────────────────────────────────────────────────
def _res_conv_block(x, filters, kernel_size=3, dropout=0.0,
                    batchnorm=False, name_prefix="res"):
    """
    Residual convolutional block with learned 1×1 shortcut connection.
    (Ascenso et al. 2024 / RA-UNet design.)
    """
    init = "he_normal"

    h = layers.Conv2D(filters, kernel_size, padding="same",
                      kernel_initializer=init,
                      name=f"{name_prefix}_conv1")(x)
    if batchnorm:
        h = layers.BatchNormalization(name=f"{name_prefix}_bn1")(h)
    h = layers.Activation("relu", name=f"{name_prefix}_act1")(h)

    h = layers.Conv2D(filters, kernel_size, padding="same",
                      kernel_initializer=init,
                      name=f"{name_prefix}_conv2")(h)
    if batchnorm:
        h = layers.BatchNormalization(name=f"{name_prefix}_bn2")(h)
    if dropout > 0:
        h = layers.Dropout(dropout, name=f"{name_prefix}_drop")(h)

    sc = layers.Conv2D(filters, 1, padding="same",
                       kernel_initializer=init,
                       name=f"{name_prefix}_sc_conv")(x)
    if batchnorm:
        sc = layers.BatchNormalization(name=f"{name_prefix}_sc_bn")(sc)
    sc = layers.Activation("relu", name=f"{name_prefix}_sc_act")(sc)

    return layers.add([h, sc], name=f"{name_prefix}_add")


# ─────────────────────────────────────────────────────────────────────────────
# Attention gate
# ─────────────────────────────────────────────────────────────────────────────
def _gating_signal(x, out_size, batchnorm=False, name_prefix="gate"):
    g = layers.Conv2D(out_size, 1, padding="same",
                      kernel_initializer="he_normal",
                      name=f"{name_prefix}_conv")(x)
    if batchnorm:
        g = layers.BatchNormalization(name=f"{name_prefix}_bn")(g)
    return layers.Activation("relu", name=f"{name_prefix}_act")(g)


def _attention_block(x, gating, inter_shape, name_prefix="att"):
    """
    Soft attention gate (Oktay et al. 2018 / RA-UNet).
    Selectively emphasises encoder features most relevant to the decoder query.
    """
    init     = "he_normal"
    shape_x  = K.int_shape(x)
    shape_g  = K.int_shape(gating)

    theta_x     = layers.Conv2D(inter_shape, 2, strides=2, padding="same",
                                 kernel_initializer=init,
                                 name=f"{name_prefix}_theta")(x)
    shape_theta = K.int_shape(theta_x)

    phi_g = layers.Conv2D(inter_shape, 1, padding="same",
                           kernel_initializer=init,
                           name=f"{name_prefix}_phi")(gating)

    up_factor = (
        max(1, shape_theta[1] // shape_g[1]),
        max(1, shape_theta[2] // shape_g[2])
    )
    phi_g_up = layers.Conv2DTranspose(
        inter_shape, 3, strides=up_factor, padding="same",
        kernel_initializer=init,
        name=f"{name_prefix}_phi_up"
    )(phi_g)

    concat = layers.add([phi_g_up, theta_x], name=f"{name_prefix}_add")
    concat = layers.Activation("relu",        name=f"{name_prefix}_relu")(concat)

    psi = layers.Conv2D(1, 1, padding="same", kernel_initializer=init,
                         name=f"{name_prefix}_psi")(concat)
    psi = layers.Activation("sigmoid", name=f"{name_prefix}_sigmoid")(psi)

    shape_sig = K.int_shape(psi)
    up2 = layers.UpSampling2D(
        size=(shape_x[1] // shape_sig[1], shape_x[2] // shape_sig[2]),
        name=f"{name_prefix}_up_psi"
    )(psi)
    up2 = layers.Lambda(
        lambda t: K.repeat_elements(t, shape_x[3], axis=3),
        name=f"{name_prefix}_repeat"
    )(up2)

    attended = layers.multiply([up2, x], name=f"{name_prefix}_mul")
    out      = layers.Conv2D(shape_x[3], 1, padding="same",
                              kernel_initializer=init,
                              name=f"{name_prefix}_out_conv")(attended)
    return layers.BatchNormalization(name=f"{name_prefix}_bn")(out)


# ─────────────────────────────────────────────────────────────────────────────
# RA-UNet backbone (full-image, with pad / crop wrapper)
# ─────────────────────────────────────────────────────────────────────────────
def _build_ra_unet_backbone(input_tensor, filters, dropout=0.0, batchnorm=False):
    """
    Shared encoder-decoder backbone.

    Pads the input to the nearest multiple of 8 required by the three
    MaxPool2D stages, then crops the decoder output back to the original
    spatial dimensions.

    Returns
    -------
    features : tensor (B, CHIRPS_H, CHIRPS_W, filters[0])
    """
    f = filters   # [32, 64, 128, 256]

    # ── Pad to multiple of 8 ─────────────────────────────────────────────
    # Padding is added on the bottom / right so the top-left corner
    # (Mozambique's northernmost / westernmost extent) stays aligned.
    x = input_tensor
    if C.PAD_H > 0 or C.PAD_W > 0:
        x = layers.ZeroPadding2D(
            padding=((0, C.PAD_H), (0, C.PAD_W)),
            name="pad_input"
        )(x)
    # After padding: (B, CHIRPS_H + PAD_H, CHIRPS_W + PAD_W, C)
    # e.g. (B, 344, 232, 7)

    # ── Encoder ──────────────────────────────────────────────────────────
    d1 = _res_conv_block(x,  f[0], dropout=dropout, batchnorm=batchnorm,
                          name_prefix="enc1")
    p1 = layers.MaxPooling2D(2, name="pool1")(d1)

    d2 = _res_conv_block(p1, f[1], dropout=dropout, batchnorm=batchnorm,
                          name_prefix="enc2")
    p2 = layers.MaxPooling2D(2, name="pool2")(d2)

    d3 = _res_conv_block(p2, f[2], dropout=dropout, batchnorm=batchnorm,
                          name_prefix="enc3")
    p3 = layers.MaxPooling2D(2, name="pool3")(d3)

    # ── Bottleneck ────────────────────────────────────────────────────────
    bn = _res_conv_block(p3, f[3], dropout=dropout, batchnorm=batchnorm,
                          name_prefix="bottleneck")

    # ── Decoder ───────────────────────────────────────────────────────────
    g3   = _gating_signal(bn,  f[2], batchnorm=batchnorm, name_prefix="gate3")
    att3 = _attention_block(d3, g3,  f[2],                name_prefix="att3")
    u3   = layers.UpSampling2D(2, name="up3")(bn)
    u3   = layers.concatenate([u3, att3], axis=3, name="cat3")
    d3u  = _res_conv_block(u3, f[2], dropout=dropout, batchnorm=batchnorm,
                            name_prefix="dec3")

    g2   = _gating_signal(d3u, f[1], batchnorm=batchnorm, name_prefix="gate2")
    att2 = _attention_block(d2, g2,  f[1],                name_prefix="att2")
    u2   = layers.UpSampling2D(2, name="up2")(d3u)
    u2   = layers.concatenate([u2, att2], axis=3, name="cat2")
    d2u  = _res_conv_block(u2, f[1], dropout=dropout, batchnorm=batchnorm,
                            name_prefix="dec2")

    g1   = _gating_signal(d2u, f[0], batchnorm=batchnorm, name_prefix="gate1")
    att1 = _attention_block(d1, g1,  f[0],                name_prefix="att1")
    u1   = layers.UpSampling2D(2, name="up1")(d2u)
    u1   = layers.concatenate([u1, att1], axis=3, name="cat1")
    d1u  = _res_conv_block(u1, f[0], dropout=dropout, batchnorm=batchnorm,
                            name_prefix="dec1")
    # d1u shape: (B, 344, 232, f[0])

    # ── Crop back to original spatial size ───────────────────────────────
    if C.PAD_H > 0 or C.PAD_W > 0:
        d1u = layers.Cropping2D(
            cropping=((0, C.PAD_H), (0, C.PAD_W)),
            name="crop_output"
        )(d1u)
    # d1u shape: (B, CHIRPS_H, CHIRPS_W, f[0])  e.g. (B, 340, 230, 32)

    return d1u


# ─────────────────────────────────────────────────────────────────────────────
# Public model builders
# ─────────────────────────────────────────────────────────────────────────────

def build_unet_mse(input_shape=C.UNET_INPUT_SHAPE,
                   filters=C.UNET_FILTERS,
                   dropout=0.1) -> Model:
    """
    UNet with single ReLU output.  Trained with masked MSE.
    Light dropout added because MSE tends to overfit.
    """
    inp      = layers.Input(shape=input_shape, name="era5_input")
    features = _build_ra_unet_backbone(inp, filters, dropout=dropout)
    out      = layers.Conv2D(1, 1, kernel_initializer="he_normal",
                              name="output_conv")(features)
    out      = layers.Activation("relu", name="output_relu")(out)
    return Model(inputs=inp, outputs=out, name="UNet_MSE")


def build_unet_bg(input_shape=C.UNET_INPUT_SHAPE,
                  filters=C.UNET_FILTERS,
                  dropout=0.1) -> Model:
    """
    UNet with 3-channel output for Bernoulli-Gamma NLL:
      ch 0: p     (sigmoid) — P(rain > threshold)
      ch 1: alpha (softplus) — Gamma shape
      ch 2: beta  (softplus) — Gamma rate
    """
    inp      = layers.Input(shape=input_shape, name="era5_input")
    features = _build_ra_unet_backbone(inp, filters, dropout=dropout)

    p_out = layers.Conv2D(1, 1, kernel_initializer="he_normal",
                           name="p_conv")(features)
    p_out = layers.Activation("sigmoid",  name="p_out")(p_out)

    alpha_out = layers.Conv2D(1, 1, kernel_initializer="he_normal",
                               name="alpha_conv")(features)
    alpha_out = layers.Activation("softplus", name="alpha_out")(alpha_out)

    beta_out  = layers.Conv2D(1, 1, kernel_initializer="he_normal",
                               name="beta_conv")(features)
    beta_out  = layers.Activation("softplus", name="beta_out")(beta_out)

    combined  = layers.concatenate([p_out, alpha_out, beta_out],
                                    axis=-1, name="bg_output")
    return Model(inputs=inp, outputs=combined, name="UNet_BG")


def build_unet_compound(input_shape=C.UNET_INPUT_SHAPE,
                        filters=C.UNET_FILTERS) -> Model:
    """
    UNet with single ReLU output.  Trained with compound (FSS'+MSE) loss.
    No dropout — compound loss provides sufficient regularisation.
    """
    inp      = layers.Input(shape=input_shape, name="era5_input")
    features = _build_ra_unet_backbone(inp, filters, dropout=0.0)
    out      = layers.Conv2D(1, 1, kernel_initializer="he_normal",
                              name="output_conv")(features)
    out      = layers.Activation("relu", name="output_relu")(out)
    return Model(inputs=inp, outputs=out, name="UNet_Compound")


def build_wgan_generator(input_shape=C.UNET_INPUT_SHAPE,
                          filters=C.UNET_FILTERS) -> Model:
    """
    WGAN generator = RA-UNet with compound loss guidance.
    Architecture identical to build_unet_compound.
    """
    inp      = layers.Input(shape=input_shape, name="era5_input")
    features = _build_ra_unet_backbone(inp, filters, dropout=0.0)
    out      = layers.Conv2D(1, 1, kernel_initializer="he_normal",
                              name="output_conv")(features)
    out      = layers.Activation("relu", name="output_relu")(out)
    return Model(inputs=inp, outputs=out, name="WGAN_Generator")


def build_wgan_discriminator(hr_shape=C.HR_SHAPE,
                              era5_shape=C.UNET_INPUT_SHAPE) -> Model:
    """
    PatchGAN discriminator (no sigmoid — outputs raw Wasserstein scores).

    Jointly conditions on the upsampled ERA5 context and the rainfall field
    so the critic can assess physical consistency, not just visual realism.

    Architecture: 4 strided Conv2D blocks (stride 2 each) → linear output.
    Final receptive field covers a substantial portion of the Mozambique domain.
    """
    era5_inp = layers.Input(shape=era5_shape, name="era5_cond")
    rain_inp = layers.Input(shape=hr_shape,   name="rain_input")

    x = layers.concatenate([era5_inp, rain_inp], axis=-1, name="joint_input")

    def _disc_block(x, filters, stride, name, batchnorm=True):
        x = layers.Conv2D(filters, 4, strides=stride, padding="same",
                           use_bias=not batchnorm,
                           kernel_initializer="he_normal",
                           name=f"{name}_conv")(x)
        if batchnorm:
            x = layers.LayerNormalization(name=f"{name}_ln")(x)
        return layers.LeakyReLU(0.2, name=f"{name}_lrelu")(x)

    x = _disc_block(x,  32, 2, "db1", batchnorm=False)   # H/2
    x = _disc_block(x,  64, 2, "db2")                     # H/4
    x = _disc_block(x, 128, 2, "db3")                     # H/8
    x = _disc_block(x, 256, 2, "db4")                     # H/16

    # Linear output — no sigmoid for WGAN
    out = layers.Conv2D(1, 4, strides=1, padding="same",
                         kernel_initializer="he_normal",
                         name="disc_output")(x)

    return Model(inputs=[era5_inp, rain_inp], outputs=out,
                  name="WGAN_Discriminator")


# ─────────────────────────────────────────────────────────────────────────────
# Helper: expected rainfall from Bernoulli-Gamma output
# ─────────────────────────────────────────────────────────────────────────────
def bg_expected_rainfall(y_pred_bg):
    """
    Convert the 3-channel B-G output to expected rainfall.
    E[R] = p * alpha / beta
    """
    p     = y_pred_bg[..., 0:1]
    alpha = y_pred_bg[..., 1:2]
    beta  = y_pred_bg[..., 2:3] + 1e-7
    return p * (alpha / beta)
