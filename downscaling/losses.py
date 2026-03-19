"""
losses.py — All loss functions for the four experiments.
No tensorflow_probability dependency — pure TF/numpy only.

All losses operate on full-image batches (B, H_hr, W_hr, 1).
Ocean pixels carry the sentinel value -1 and are excluded from every loss
via the land mask computed inside each function from y_true.
"""

import math
import numpy as np
import tensorflow as tf
from . import config as C


# ─────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────────────────────────────────────
def _land_mask(y_true):
    """Float mask: 1 on land pixels (y_true >= 0), 0 on ocean (-1 sentinel)."""
    return tf.cast(y_true >= 0.0, tf.float32)


def _tf_percentile(tensor, q):
    """
    q-th percentile of a flat tensor — fully differentiable, no tfp needed.
    """
    flat    = tf.reshape(tensor, [-1])
    sorted_ = tf.sort(flat)
    n       = tf.cast(tf.shape(sorted_)[0], tf.float32)
    idx     = tf.cast(tf.math.floor(q / 100.0 * (n - 1.0)), tf.int32)
    idx     = tf.clip_by_value(idx, 0, tf.shape(sorted_)[0] - 1)
    return sorted_[idx]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Masked MSE
# ─────────────────────────────────────────────────────────────────────────────
def masked_mse(y_true, y_pred):
    """MSE computed only over land pixels."""
    mask   = _land_mask(y_true)
    n_land = tf.reduce_sum(mask) + 1e-8
    return tf.reduce_sum((y_true - y_pred) ** 2 * mask) / n_land


# ─────────────────────────────────────────────────────────────────────────────
# 2. Bernoulli-Gamma NLL  (Rampal et al. 2022)
# ─────────────────────────────────────────────────────────────────────────────
def bernoulli_gamma_nll(y_true, y_pred_combined):
    """
    Negative log-likelihood for a mixed Bernoulli-Gamma distribution.

    y_pred_combined : (B, H, W, 3)
        ch 0  p     — P(rain > threshold)     [sigmoid output]
        ch 1  alpha — Gamma shape parameter   [softplus output]
        ch 2  beta  — Gamma rate parameter    [softplus output]
    """
    p     = y_pred_combined[..., 0:1]
    alpha = y_pred_combined[..., 1:2]
    beta  = y_pred_combined[..., 2:3]

    mask  = _land_mask(y_true)
    r_obs = tf.maximum(y_true, 0.0)
    wet   = tf.cast(r_obs > C.RAIN_THRESHOLD, tf.float32)
    dry   = 1.0 - wet

    eps       = 1e-7
    p_clipped = tf.clip_by_value(p, eps, 1.0 - eps)
    ll_bern   = (wet * tf.math.log(p_clipped)
                 + dry * tf.math.log(1.0 - p_clipped))

    r_safe   = tf.maximum(r_obs, eps)
    ll_gamma = ((alpha - 1.0) * tf.math.log(r_safe)
                - beta  * r_safe
                + alpha * tf.math.log(beta + eps)
                - tf.math.lgamma(alpha + eps))

    ll     = ll_bern + wet * ll_gamma
    n_land = tf.reduce_sum(mask) + 1e-8
    return -tf.reduce_sum(ll * mask) / n_land


# ─────────────────────────────────────────────────────────────────────────────
# 3. Compound loss: FSS' + MSE  (Ascenso et al. 2024)
# ─────────────────────────────────────────────────────────────────────────────
def _soft_binary(x, threshold):
    """Differentiable soft-binary field via an arctan approximation."""
    eps = 1e-7
    return eps + (1.0 - 2.0 * eps) * (
        0.5 + tf.math.atan(x - threshold) / tf.constant(math.pi)
    )


def _normalise_map(x):
    xmin = tf.reduce_min(x)
    xmax = tf.reduce_max(x)
    return (x - xmin) / (xmax - xmin + 1e-7)


def _fss_single_3d(yt_3d, yp_3d, q, n):
    """
    FSS' for one sample.

    Accepts rank-3 tensors (H, W, 1) — as delivered by tf.map_fn —
    adds/removes the batch dimension internally for conv2d.
    """
    yt = tf.expand_dims(yt_3d, axis=0)   # (1, H, W, 1)
    yp = tf.expand_dims(yp_3d, axis=0)

    thr_t = _tf_percentile(yt, q) + 1e-7
    thr_p = _tf_percentile(yp, q) + 1e-7

    bin_t = _normalise_map(_soft_binary(yt, thr_t))
    bin_p = _normalise_map(_soft_binary(yp, thr_p))

    kernel = tf.constant(
        np.ones((n, n, 1, 1), dtype=np.float32) / float(n * n),
        dtype=tf.float32
    )
    frac_t = tf.nn.conv2d(bin_t, kernel, strides=[1,1,1,1], padding="SAME")
    frac_p = tf.nn.conv2d(bin_p, kernel, strides=[1,1,1,1], padding="SAME")

    num   = tf.reduce_mean(tf.square(frac_p - frac_t))
    denom = tf.reduce_mean(tf.square(frac_p) + tf.square(frac_t)) + 1e-7
    return num / denom


def fss_loss_batch(y_true, y_pred, q, n=C.FSS_N):
    """Mean FSS' across all samples in the batch (ocean already zeroed)."""
    fss_vals = tf.map_fn(
        fn=lambda pair: _fss_single_3d(pair[0], pair[1], q, n),
        elems=(y_true, y_pred),
        fn_output_signature=tf.float32
    )
    return tf.reduce_mean(fss_vals)


def compound_loss(y_true, y_pred,
                  fss_weight=C.COMPOUND_FSS_WEIGHT,
                  mse_weight=C.COMPOUND_MSE_WEIGHT,
                  percentiles=C.FSS_PERCENTILES,
                  n=C.FSS_N):
    """
    L_cmpd = fss_weight * Σ FSS'_q  +  mse_weight * MSE
    (Ascenso et al. 2024, eq. 2)
    """
    mask    = tf.cast(y_true >= 0, tf.float32)
    yt_land = y_true * mask
    yp_land = y_pred * mask

    mse_term  = masked_mse(y_true, y_pred)
    fss_total = tf.constant(0.0)
    for q in percentiles:
        fss_total = fss_total + fss_loss_batch(yt_land, yp_land, q=q, n=n)

    return fss_weight * fss_total + mse_weight * mse_term


# ─────────────────────────────────────────────────────────────────────────────
# 4. WGAN-GP losses
# ─────────────────────────────────────────────────────────────────────────────
def wasserstein_generator_loss(fake_logits):
    return -tf.reduce_mean(fake_logits)


def wasserstein_discriminator_loss(real_logits, fake_logits):
    return tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)


def gradient_penalty(discriminator_fn, real_samples, fake_samples,
                     lam=C.WGAN_GP_LAMBDA):
    B     = tf.shape(real_samples)[0]
    alpha = tf.random.uniform([B, 1, 1, 1], 0.0, 1.0)
    interp = real_samples + alpha * (fake_samples - real_samples)

    with tf.GradientTape() as tape:
        tape.watch(interp)
        pred = discriminator_fn(interp)

    grads   = tape.gradient(pred, interp)
    norm    = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1,2,3]) + 1e-12)
    penalty = tf.reduce_mean((norm - 1.0) ** 2)
    return lam * penalty


def wgan_generator_loss_total(fake_logits, y_true, y_pred_rainfall,
                               adv_weight=1.0, cmpd_weight=10.0):
    """Combined adversarial + compound loss for the WGAN generator."""
    adv_loss = wasserstein_generator_loss(fake_logits)
    cmpd     = compound_loss(y_true, y_pred_rainfall)
    return adv_weight * adv_loss + cmpd_weight * cmpd
