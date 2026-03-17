"""
models.py — All four model architectures.

All UNet variants share the same RA-UNet backbone (residual blocks +
attention gates, following Ascenso et al. 2024 / Jin et al. 2020).
They differ only in their output head and loss function.

Model summary
─────────────
build_unet_mse()      → output: (H, W, 1), ReLU        [trained with MSE]
build_unet_bg()       → output: (H, W, 3), custom act.  [Bernoulli-Gamma]
build_unet_compound() → output: (H, W, 1), ReLU        [compound loss]
build_wgan()          → returns (generator, discriminator)
                         generator = UNet (H, W, 1), ReLU
                         discriminator = PatchGAN critic

Architecture notes
──────────────────
- Input: (HR_PATCH_SIZE, HR_PATCH_SIZE, N_ERA5_VARS) — ERA5 already
  bilinearly upsampled to CHIRPS resolution by the data loader.
- The UNet encoder has 3 downsampling stages (MaxPool2D ×2 each time),
  with residual blocks at every level and attention gates on the skip
  connections, matching Ascenso et al. Table 1 (filters [32,64,128,256]).
- No BatchNorm / Dropout in the base model (Ascenso et al. found no
  overfitting with compound loss; we add dropout only for MSE/BG variants
  where overfitting is more likely).
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import tensorflow.keras.backend as K
import config as C


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

def _res_conv_block(x, filters, kernel_size=3, dropout=0.0,
                    batchnorm=False, name_prefix="res"):
    """
    Residual convolutional block (Fig 3 / code snippet from Ascenso et al.).
    Two Conv2D layers with a learned 1×1 shortcut connection.
    """
    init = "he_normal"

    # Main path
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

    # Shortcut path
    sc = layers.Conv2D(filters, 1, padding="same",
                       kernel_initializer=init,
                       name=f"{name_prefix}_sc_conv")(x)
    if batchnorm:
        sc = layers.BatchNormalization(name=f"{name_prefix}_sc_bn")(sc)
    sc = layers.Activation("relu", name=f"{name_prefix}_sc_act")(sc)

    return layers.add([h, sc], name=f"{name_prefix}_add")


def _gating_signal(x, out_size, batchnorm=False, name_prefix="gate"):
    """1×1 conv gating signal for attention unit."""
    g = layers.Conv2D(out_size, 1, padding="same",
                      kernel_initializer="he_normal",
                      name=f"{name_prefix}_conv")(x)
    if batchnorm:
        g = layers.BatchNormalization(name=f"{name_prefix}_bn")(g)
    return layers.Activation("relu", name=f"{name_prefix}_act")(g)


def _attention_block(x, gating, inter_shape, name_prefix="att"):
    """
    Soft attention gate (Oktay et al. 2018, as used in RA-UNet).
    Selectively emphasises spatially relevant features from the encoder.
    """
    init = "he_normal"
    shape_x = K.int_shape(x)
    shape_g = K.int_shape(gating)

    # Downsample x to match gating signal resolution
    theta_x = layers.Conv2D(inter_shape, 2, strides=2, padding="same",
                             kernel_initializer=init,
                             name=f"{name_prefix}_theta")(x)
    shape_theta = K.int_shape(theta_x)

    # Upsample gating signal to match theta_x resolution
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

    # Additive attention
    concat = layers.add([phi_g_up, theta_x], name=f"{name_prefix}_add")
    concat = layers.Activation("relu", name=f"{name_prefix}_relu")(concat)

    psi = layers.Conv2D(1, 1, padding="same", kernel_initializer=init,
                         name=f"{name_prefix}_psi")(concat)
    psi = layers.Activation("sigmoid", name=f"{name_prefix}_sigmoid")(psi)

    # Upsample attention map back to x resolution and broadcast
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
    out = layers.Conv2D(shape_x[3], 1, padding="same",
                         kernel_initializer=init,
                         name=f"{name_prefix}_out_conv")(attended)
    return layers.BatchNormalization(name=f"{name_prefix}_bn")(out)


# ─────────────────────────────────────────────────────────────────────────────
# RA-UNet encoder/decoder (shared backbone)
# ─────────────────────────────────────────────────────────────────────────────

def _build_ra_unet_backbone(input_tensor, filters, dropout=0.0,
                             batchnorm=False):
    """
    Shared encoder-decoder backbone returning the final feature map
    before the output head.

    Returns
    -------
    features : tensor (B, H, W, filters[0])  ready for output head
    """
    f = filters   # [32, 64, 128, 256]

    # ── Encoder ──────────────────────────────────────────────
    d1 = _res_conv_block(input_tensor, f[0], dropout=dropout,
                          batchnorm=batchnorm, name_prefix="enc1")
    p1 = layers.MaxPooling2D(2, name="pool1")(d1)

    d2 = _res_conv_block(p1, f[1], dropout=dropout,
                          batchnorm=batchnorm, name_prefix="enc2")
    p2 = layers.MaxPooling2D(2, name="pool2")(d2)

    d3 = _res_conv_block(p2, f[2], dropout=dropout,
                          batchnorm=batchnorm, name_prefix="enc3")
    p3 = layers.MaxPooling2D(2, name="pool3")(d3)

    # ── Bottleneck ───────────────────────────────────────────
    bn = _res_conv_block(p3, f[3], dropout=dropout,
                          batchnorm=batchnorm, name_prefix="bottleneck")

    # ── Decoder ──────────────────────────────────────────────
    # Level 3
    g3   = _gating_signal(bn, f[2], batchnorm=batchnorm, name_prefix="gate3")
    att3 = _attention_block(d3, g3, f[2], name_prefix="att3")
    u3   = layers.UpSampling2D(2, data_format="channels_last", name="up3")(bn)
    u3   = layers.concatenate([u3, att3], axis=3, name="cat3")
    d3u  = _res_conv_block(u3, f[2], dropout=dropout,
                            batchnorm=batchnorm, name_prefix="dec3")

    # Level 2
    g2   = _gating_signal(d3u, f[1], batchnorm=batchnorm, name_prefix="gate2")
    att2 = _attention_block(d2, g2, f[1], name_prefix="att2")
    u2   = layers.UpSampling2D(2, data_format="channels_last", name="up2")(d3u)
    u2   = layers.concatenate([u2, att2], axis=3, name="cat2")
    d2u  = _res_conv_block(u2, f[1], dropout=dropout,
                            batchnorm=batchnorm, name_prefix="dec2")

    # Level 1
    g1   = _gating_signal(d2u, f[0], batchnorm=batchnorm, name_prefix="gate1")
    att1 = _attention_block(d1, g1, f[0], name_prefix="att1")
    u1   = layers.UpSampling2D(2, data_format="channels_last", name="up1")(d2u)
    u1   = layers.concatenate([u1, att1], axis=3, name="cat1")
    d1u  = _res_conv_block(u1, f[0], dropout=dropout,
                            batchnorm=batchnorm, name_prefix="dec1")

    return d1u   # (B, H, W, f[0])


# ─────────────────────────────────────────────────────────────────────────────
# Public model builders
# ─────────────────────────────────────────────────────────────────────────────

def build_unet_mse(input_shape=C.UNET_INPUT_SHAPE,
                   filters=C.UNET_FILTERS,
                   dropout=0.1) -> Model:
    """
    UNet with single ReLU output — trained with masked MSE.
    Slight dropout added because MSE tends to overfit.
    """
    inp      = layers.Input(shape=input_shape, name="era5_input")
    features = _build_ra_unet_backbone(inp, filters, dropout=dropout,
                                        batchnorm=False)
    out = layers.Conv2D(1, 1, kernel_initializer="he_normal",
                         name="output_conv")(features)
    out = layers.Activation("relu", name="output_relu")(out)
    return Model(inputs=inp, outputs=out, name="UNet_MSE")


def build_unet_bg(input_shape=C.UNET_INPUT_SHAPE,
                  filters=C.UNET_FILTERS,
                  dropout=0.1) -> Model:
    """
    UNet with 3-channel output for Bernoulli-Gamma:
      ch 0 : p     (sigmoid)  — P(rain > threshold)
      ch 1 : alpha (softplus) — Gamma shape
      ch 2 : beta  (softplus) — Gamma rate
    """
    inp      = layers.Input(shape=input_shape, name="era5_input")
    features = _build_ra_unet_backbone(inp, filters, dropout=dropout,
                                        batchnorm=False)

    # Three separate output streams sharing the same backbone
    p_out = layers.Conv2D(1, 1, kernel_initializer="he_normal",
                           name="p_conv")(features)
    p_out = layers.Activation("sigmoid", name="p_out")(p_out)

    alpha_out = layers.Conv2D(1, 1, kernel_initializer="he_normal",
                               name="alpha_conv")(features)
    alpha_out = layers.Activation("softplus", name="alpha_out")(alpha_out)

    beta_out = layers.Conv2D(1, 1, kernel_initializer="he_normal",
                              name="beta_conv")(features)
    beta_out = layers.Activation("softplus", name="beta_out")(beta_out)

    # Concatenate to (B, H, W, 3)
    combined = layers.concatenate([p_out, alpha_out, beta_out],
                                    axis=-1, name="bg_output")
    return Model(inputs=inp, outputs=combined, name="UNet_BG")


def build_unet_compound(input_shape=C.UNET_INPUT_SHAPE,
                        filters=C.UNET_FILTERS) -> Model:
    """
    UNet with single ReLU output — trained with compound (FSS+MSE) loss.
    No dropout following Ascenso et al. (compound loss regularises enough).
    """
    inp      = layers.Input(shape=input_shape, name="era5_input")
    features = _build_ra_unet_backbone(inp, filters, dropout=0.0,
                                        batchnorm=False)
    out = layers.Conv2D(1, 1, kernel_initializer="he_normal",
                         name="output_conv")(features)
    out = layers.Activation("relu", name="output_relu")(out)
    return Model(inputs=inp, outputs=out, name="UNet_Compound")


def build_wgan_generator(input_shape=C.UNET_INPUT_SHAPE,
                          filters=C.UNET_FILTERS) -> Model:
    """
    WGAN generator = RA-UNet with compound loss guidance.
    Identical architecture to build_unet_compound.
    """
    inp      = layers.Input(shape=input_shape, name="era5_input")
    features = _build_ra_unet_backbone(inp, filters, dropout=0.0,
                                        batchnorm=False)
    out = layers.Conv2D(1, 1, kernel_initializer="he_normal",
                         name="output_conv")(features)
    out = layers.Activation("relu", name="output_relu")(out)
    return Model(inputs=inp, outputs=out, name="WGAN_Generator")


def build_wgan_discriminator(hr_shape=(C.HR_PATCH_SIZE, C.HR_PATCH_SIZE, 1),
                              era5_shape=C.UNET_INPUT_SHAPE) -> Model:
    """
    PatchGAN discriminator (no sigmoid — outputs raw Wasserstein scores).

    Takes (ERA5_upsampled, CHIRPS_or_pred) as joint input so the critic
    can assess whether the rainfall is consistent with the atmospheric state,
    not just whether it looks realistic in isolation.

    Architecture: 4 strided Conv2D blocks → linear output.
    Deliberately lightweight for CPU feasibility.
    """
    era5_inp   = layers.Input(shape=era5_shape,  name="era5_cond")
    rain_inp   = layers.Input(shape=hr_shape,    name="rain_input")

    # Concatenate ERA5 context with rainfall field
    x = layers.concatenate([era5_inp, rain_inp], axis=-1, name="joint_input")

    # 4 strided conv blocks (LeakyReLU, no BN in first layer per DCGAN advice)
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

    # Linear output (no sigmoid — WGAN uses raw scores)
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
    Convert the 3-channel B-G output to expected rainfall E[R].

    E[R] = p * E[Gamma(alpha, beta)]
         = p * alpha / beta

    Useful for inference / evaluation.
    """
    p     = y_pred_bg[..., 0:1]
    alpha = y_pred_bg[..., 1:2]
    beta  = y_pred_bg[..., 2:3] + 1e-7
    return p * (alpha / beta)
