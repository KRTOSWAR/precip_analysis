"""
data_loader.py — Full-image tf.data pipeline.

Each training sample is one month:
  X : (H_lr, W_lr, C)  → bilinearly upsampled to (H_hr, W_hr, C)  
  y : (H_hr, W_hr, 1)  CHIRPS normalised target  (ocean pixels = -1 sentinel)

The upsampling is done lazily inside a tf.data.map so the heavy (N, H_hr, W_hr, C)
array is never materialised in RAM — only the small LR array is stored.
"""

import numpy as np
import tensorflow as tf
from . import config as C

# ---------------------------------------------------------------------------
# Load preprocessed .npz splits
# ---------------------------------------------------------------------------
def load_split(path: str):
    """
    Load (X, y, times) from a .npz file produced by training_data_prep.py.

    X     : (N, H_lr, W_lr, C)  normalised ERA5  (float32)
    y     : (N, H_hr, W_hr, 1)  log1p-normalised CHIRPS  (ocean → -1 sentinel)
    times : (N,)  string timestamps
    """
    data  = np.load(path, allow_pickle=True)
    X     = data["X"].astype(np.float32)
    y     = data["y"].astype(np.float32)
    times = data["times"]
    # Replace NaN ocean pixels with the -1 sentinel so losses can mask them
    y = np.where(np.isnan(y), -1.0, y)
    print(f"  Loaded {path}:  X={X.shape}  y={y.shape}")
    return X, y, times


def load_split_raw(path: str):
    """
    Load y_raw (mm/month) for evaluation — keeps ocean pixels as NaN.
    Only the y_raw field is returned; X and times come from load_split.
    """
    data  = np.load(path, allow_pickle=True)
    y_raw = data["y_raw"].astype(np.float32)   # (N, H_hr, W_hr, 1)  in mm
    return y_raw


def load_land_mask(meta_path: str) -> np.ndarray:
    """Return (H_hr, W_hr) float32 land mask from the metadata file."""
    meta = np.load(meta_path, allow_pickle=True)
    return meta["land_mask_hr"].astype(np.float32)


def load_norm_stats(meta_path: str) -> dict:
    """
    Return normalisation statistics needed to invert predictions.

    Keys
    ----
    chirps_log_mean : float   (log-space mean used for CHIRPS z-score)
    chirps_log_std  : float
    era5_mean       : (C,)    per-channel z-score mean for ERA5
    era5_std        : (C,)    per-channel z-score std  for ERA5
    """
    meta = np.load(meta_path, allow_pickle=True)
    return {
        "chirps_log_mean" : float(meta["chirps_log_mean"]),
        "chirps_log_std"  : float(meta["chirps_log_std"]),
        "era5_mean"       : meta["norm_mean"].astype(np.float32),
        "era5_std"        : meta["norm_std"].astype(np.float32),
    }


# ---------------------------------------------------------------------------
# Inverse transform helpers
# ---------------------------------------------------------------------------
def invert_chirps_norm(y_norm: np.ndarray, norm_stats: dict) -> np.ndarray:
    """
    Convert log1p-normalised predictions back to mm/month.

    y_norm → (y_norm * log_std + log_mean) → expm1 → mm
    Ocean/boundary pixels (y_norm == -1 sentinel) become NaN.
    """
    mu, sigma = norm_stats["chirps_log_mean"], norm_stats["chirps_log_std"]
    y_log = y_norm * (sigma + 1e-8) + mu
    y_mm  = np.expm1(y_log)
    y_mm  = np.maximum(y_mm, 0.0)     # clip any tiny negatives from float math
    return y_mm


# ---------------------------------------------------------------------------
# tf.data.Dataset  (full-image, no patches)
# ---------------------------------------------------------------------------
def build_dataset(X: np.ndarray, y: np.ndarray,
                  batch_size: int,
                  shuffle: bool = True,
                  seed: int = None) -> tf.data.Dataset:
    """
    Build a batched tf.data.Dataset of (era5_upsampled, chirps) full images.

    Parameters
    ----------
    X          : (N, H_lr, W_lr, C)  normalised ERA5 — stored LR, upsampled on-the-fly
    y          : (N, H_hr, W_hr, 1)  normalised CHIRPS, ocean = -1
    batch_size : months per mini-batch
    shuffle    : shuffle samples each epoch
    seed       : RNG seed for reproducibility

    Returns
    -------
    tf.data.Dataset  yielding
        x_hr : (batch, H_hr, W_hr, C)  bilinearly upsampled ERA5
        y_hr : (batch, H_hr, W_hr, 1)  CHIRPS target
    """
    ds = tf.data.Dataset.from_tensor_slices((X, y))

    # Upsample ERA5 patch from LR to CHIRPS resolution — lazy, per sample
    def _upsample(x_lr, y_hr):
        x_hr = tf.image.resize(
            x_lr,
            [C.CHIRPS_H, C.CHIRPS_W],
            method=tf.image.ResizeMethod.BILINEAR
        )
        return x_hr, y_hr

    ds = ds.map(_upsample, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(
            buffer_size=X.shape[0],
            seed=seed,
            reshuffle_each_iteration=True
        )

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ---------------------------------------------------------------------------
# Inference helper: single month forward pass
# ---------------------------------------------------------------------------
def predict_full(model, X_single: np.ndarray) -> np.ndarray:
    """
    Run a single forward pass for one month.

    Parameters
    ----------
    model    : tf.keras.Model  (any trained downscaling model)
    X_single : (H_lr, W_lr, C)  single month ERA5, normalised

    Returns
    -------
    pred : (H_hr, W_hr, 1)  model output, negatives clipped to 0
    """
    x_hr = tf.image.resize(
        X_single,
        [C.CHIRPS_H, C.CHIRPS_W],
        method=tf.image.ResizeMethod.BILINEAR
    ).numpy()
    x_batch = x_hr[np.newaxis]                          # (1, H_hr, W_hr, C)
    pred    = model(x_batch, training=False).numpy()    # (1, H_hr, W_hr, 1)
    return np.maximum(pred[0], 0.0)                     # (H_hr, W_hr, 1)


def predict_all(model, X: np.ndarray, exp_name: str,
                norm_stats: dict = None) -> np.ndarray:
    """
    Run predict_full over every month in X.

    Returns
    -------
    preds : (N, H_hr, W_hr, 1)  in normalised space
            (call invert_chirps_norm afterwards for mm)
    """
    from . import models   # local import avoids circular dep
    preds = []
    N     = X.shape[0]
    for i in range(N):
        if (i + 1) % 12 == 0 or i == N - 1:
            print(f"    Predicting month {i+1}/{N} ...")
        pred = predict_full(model, X[i])
        if exp_name == "unet_bg":
            pred = models.bg_expected_rainfall(pred)
        preds.append(pred)
    return np.stack(preds, axis=0)   # (N, H_hr, W_hr, 1)
