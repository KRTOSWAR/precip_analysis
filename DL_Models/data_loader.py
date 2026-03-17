"""
data_loader.py — Patch-based tf.data pipeline.

Why patches?
  The full domain is 320×300 ERA5 → 1600×1500 CHIRPS. A single full-image
  training sample would require ~1.4 GB of GPU memory. Instead we:
    1. Load full arrays into RAM once (they fit: ~2 GB for the training set).
    2. At each training step, randomly sample (LR_PATCH_SIZE × LR_PATCH_SIZE)
       crops from ERA5 and the corresponding (HR_PATCH_SIZE × HR_PATCH_SIZE)
       crops from CHIRPS.
    3. Bilinearly upsample each ERA5 patch to HR resolution so the UNet
       operates entirely at the CHIRPS scale.

Patch strategy:
  Only sample patches where ≥ 30% of the CHIRPS pixels are valid land
  (non-NaN). This avoids wasting compute on ocean patches.
"""

import numpy as np
import tensorflow as tf
import config as C


# ---------------------------------------------------------------------------
# Load preprocessed .npz arrays
# ---------------------------------------------------------------------------
def load_split(path: str):
    """Return (X, y, times) from a .npz file saved by preprocess_pp_data.py."""
    data = np.load(path, allow_pickle=True)
    X = data["X"].astype(np.float32)   # (N, H_lr, W_lr, C)
    y = data["y"].astype(np.float32)   # (N, H_hr, W_hr, 1)
    times = data["times"]
    # Replace NaN in y with -1 so we can detect ocean pixels in the loss
    # (we use -1 as the sentinel — real rainfall is always >= 0)
    y = np.where(np.isnan(y), -1.0, y)
    print(f"  Loaded {path}: X={X.shape}  y={y.shape}")
    return X, y, times


def load_land_mask(meta_path: str) -> np.ndarray:
    """Return the land mask (H_hr, W_hr) from the metadata file."""
    meta = np.load(meta_path, allow_pickle=True)
    return meta["land_mask"].astype(np.float32)


# ---------------------------------------------------------------------------
# Patch extraction
# ---------------------------------------------------------------------------
def _extract_patch(X_sample, y_sample, land_mask_hr,
                   rng: np.random.Generator, min_land_frac=0.30):
    """
    Draw one random valid patch from a single (X, y) month.

    Returns
    -------
    x_patch : (HR_PATCH_SIZE, HR_PATCH_SIZE, C)  bilinearly upsampled ERA5
    y_patch : (HR_PATCH_SIZE, HR_PATCH_SIZE, 1)  CHIRPS target
    """
    H_lr, W_lr = X_sample.shape[:2]
    H_hr, W_hr = y_sample.shape[:2]
    ps_lr = C.LR_PATCH_SIZE
    ps_hr = C.HR_PATCH_SIZE
    margin = C.PATCH_MARGIN

    max_attempts = 50
    for _ in range(max_attempts):
        r = rng.integers(margin, H_lr - ps_lr - margin)
        c = rng.integers(margin, W_lr - ps_lr - margin)

        # Corresponding high-res crop
        r_hr = r * C.SCALE_FACTOR
        c_hr = c * C.SCALE_FACTOR

        y_patch  = y_sample[r_hr : r_hr + ps_hr, c_hr : c_hr + ps_hr, :]
        lm_patch = land_mask_hr[r_hr : r_hr + ps_hr, c_hr : c_hr + ps_hr]

        # Check land fraction (use land mask, not NaN — more stable)
        if lm_patch.mean() < min_land_frac:
            continue

        x_patch_lr = X_sample[r : r + ps_lr, c : c + ps_lr, :]

        # Bilinear upsample ERA5 patch to HR resolution
        # tf.image.resize expects (H, W, C) → add/remove batch dim
        x_patch_hr = tf.image.resize(
            x_patch_lr,
            [ps_hr, ps_hr],
            method=tf.image.ResizeMethod.BILINEAR
        ).numpy()

        return x_patch_hr.astype(np.float32), y_patch.astype(np.float32)

    # If no valid patch found after max_attempts, return centre patch
    r = (H_lr - ps_lr) // 2
    c = (W_lr - ps_lr) // 2
    r_hr = r * C.SCALE_FACTOR
    c_hr = c * C.SCALE_FACTOR
    x_patch_lr = X_sample[r : r + ps_lr, c : c + ps_lr, :]
    x_patch_hr = tf.image.resize(x_patch_lr, [ps_hr, ps_hr],
                                  method=tf.image.ResizeMethod.BILINEAR).numpy()
    y_patch = y_sample[r_hr : r_hr + ps_hr, c_hr : c_hr + ps_hr, :]
    return x_patch_hr.astype(np.float32), y_patch.astype(np.float32)


# ---------------------------------------------------------------------------
# Generator function for tf.data.Dataset.from_generator
# ---------------------------------------------------------------------------
def _make_generator(X, y, land_mask_hr, n_patches, seed=None):
    """
    Returns a Python generator that yields (x_patch, y_patch) pairs.
    Each call draws n_patches patches sampled uniformly across all months.
    """
    rng = np.random.default_rng(seed)
    N = X.shape[0]  # number of months

    def generator():
        for _ in range(n_patches):
            idx = rng.integers(0, N)
            xp, yp = _extract_patch(X[idx], y[idx], land_mask_hr, rng)
            yield xp, yp

    return generator


# ---------------------------------------------------------------------------
# Public API: build tf.data.Dataset
# ---------------------------------------------------------------------------
def build_dataset(X, y, land_mask_hr, n_patches, batch_size,
                  shuffle=True, seed=None) -> tf.data.Dataset:
    """
    Build a batched tf.data.Dataset of (ERA5_patch, CHIRPS_patch) pairs.

    Parameters
    ----------
    X           : ndarray (N, H_lr, W_lr, C)  normalised ERA5
    y           : ndarray (N, H_hr, W_hr, 1)  CHIRPS, NaN→-1
    land_mask_hr: ndarray (H_hr, W_hr)        boolean/float land mask
    n_patches   : int  number of patches per epoch
    batch_size  : int
    shuffle     : bool  shuffle buffer
    seed        : int or None

    Returns
    -------
    tf.data.Dataset yielding batches of shape
        x: (batch_size, HR_PATCH_SIZE, HR_PATCH_SIZE, N_ERA5_VARS)
        y: (batch_size, HR_PATCH_SIZE, HR_PATCH_SIZE, 1)
    """
    gen = _make_generator(X, y, land_mask_hr, n_patches, seed=seed)

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(C.HR_PATCH_SIZE, C.HR_PATCH_SIZE, C.N_ERA5_VARS),
                          dtype=tf.float32),
            tf.TensorSpec(shape=(C.HR_PATCH_SIZE, C.HR_PATCH_SIZE, 1),
                          dtype=tf.float32),
        )
    )

    if shuffle:
        ds = ds.shuffle(buffer_size=min(n_patches, 500), seed=seed)

    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ---------------------------------------------------------------------------
# Inference: sliding-window patch reconstruction
# ---------------------------------------------------------------------------
def reconstruct_full(model, X_single, scale=C.SCALE_FACTOR,
                     lr_patch=C.LR_PATCH_SIZE,
                     hr_patch=C.HR_PATCH_SIZE) -> np.ndarray:
    """
    Run model over a single month using a sliding window and stitch together.

    Parameters
    ----------
    model    : tf.keras.Model  (any of the 4 trained models)
    X_single : ndarray (H_lr, W_lr, C)  single month ERA5 (normalised)

    Returns
    -------
    pred : ndarray (H_hr, W_hr, 1)  reconstructed prediction
    """
    H_lr, W_lr, C_in = X_single.shape
    H_hr = H_lr * scale
    W_hr = W_lr * scale

    pred_sum = np.zeros((H_hr, W_hr, 1), dtype=np.float32)
    count    = np.zeros((H_hr, W_hr, 1), dtype=np.float32)

    # Stride = half patch size for overlap averaging (reduces edge artefacts)
    stride_lr = lr_patch // 2
    stride_hr = stride_lr * scale

    r_starts = list(range(0, H_lr - lr_patch + 1, stride_lr))
    c_starts = list(range(0, W_lr - lr_patch + 1, stride_lr))
    # Ensure last row/col is included
    if r_starts[-1] + lr_patch < H_lr:
        r_starts.append(H_lr - lr_patch)
    if c_starts[-1] + lr_patch < W_lr:
        c_starts.append(W_lr - lr_patch)

    for r in r_starts:
        for c in c_starts:
            x_lr = X_single[r : r + lr_patch, c : c + lr_patch, :]
            x_hr = tf.image.resize(x_lr, [hr_patch, hr_patch],
                                   method="bilinear").numpy()
            x_batch = x_hr[np.newaxis]   # (1, H, W, C)

            out = model(x_batch, training=False).numpy()  # (1, H, W, 1)
            out = np.maximum(out, 0.0)  # clip negatives

            r_hr = r * scale
            c_hr = c * scale
            pred_sum[r_hr : r_hr + hr_patch, c_hr : c_hr + hr_patch] += out[0]
            count   [r_hr : r_hr + hr_patch, c_hr : c_hr + hr_patch] += 1.0

    # Average overlapping regions
    pred = np.divide(pred_sum, count, where=count > 0)
    return pred
