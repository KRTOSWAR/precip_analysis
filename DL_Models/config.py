"""
config.py — Central configuration for all experiments.
Edit this file to change paths, patch sizes, and training settings.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR    = r"C:\Users\HP ZBOOK\RAINFALL_ DONWSCALLING\DATA\PROCESSED"
TRAIN_FILE  = os.path.join(DATA_DIR, "train.npz")
VAL_FILE    = os.path.join(DATA_DIR, "val.npz")
TEST_FILE   = os.path.join(DATA_DIR, "test.npz")
META_FILE   = os.path.join(DATA_DIR, "metadata.npz")
OUTPUT_DIR  = r"C:\Users\HP ZBOOK\RAINFALL_ DONWSCALLING\RESULTS"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Spatial configuration
# ---------------------------------------------------------------------------
SCALE_FACTOR   = 5          # ERA5 0.25 deg -> CHIRPS 0.05 deg
LR_PATCH_SIZE  = 32         # ERA5 patch (low-resolution)
HR_PATCH_SIZE  = LR_PATCH_SIZE * SCALE_FACTOR   # = 160  (CHIRPS patch)
N_ERA5_VARS    = 6          # Q, T, U, V, W, Z

# After bilinear upsampling ERA5 to HR_PATCH_SIZE, both inputs share this shape
# Input to UNet: (HR_PATCH_SIZE, HR_PATCH_SIZE, N_ERA5_VARS)
UNET_INPUT_SHAPE = (HR_PATCH_SIZE, HR_PATCH_SIZE, N_ERA5_VARS)

# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------
BATCH_SIZE        = 8      # Keep small for CPU
EPOCHS            = 100
LR_INIT           = 1e-4
EARLY_STOP_PATIENCE = 15

# Patches sampled per training epoch
# (each month has many possible patch locations; we sample a subset)
PATCHES_PER_EPOCH_TRAIN = 1000
PATCHES_PER_EPOCH_VAL   = 200

# Patch sampling: avoid edges (at least 1 cell margin)
PATCH_MARGIN = 2

# UNet filter sizes (following Ascenso et al. Table 1)
UNET_FILTERS = [32, 64, 128, 256]

# ---------------------------------------------------------------------------
# Loss-specific configuration
# ---------------------------------------------------------------------------

# Compound loss weights (Ascenso eq. 2)
COMPOUND_FSS_WEIGHT = 0.7   # weight on sum of FSS' terms
COMPOUND_MSE_WEIGHT = 1.0   # weight on MSE term
FSS_N              = 15     # neighbourhood size for FSS (must be odd)
FSS_PERCENTILES    = [80, 95, 99]

# Bernoulli-Gamma: minimum valid rainfall threshold (mm/month)
# Pixels below this are treated as "dry" for the Bernoulli component
RAIN_THRESHOLD = 1.0

# WGAN-GP settings
WGAN_GP_LAMBDA     = 10     # gradient penalty weight
WGAN_N_CRITIC      = 5      # discriminator updates per generator update
WGAN_GEN_LR        = 1e-4
WGAN_DIS_LR        = 1e-4

# ---------------------------------------------------------------------------
# Experiment names (used for saving checkpoints and results)
# ---------------------------------------------------------------------------
EXPERIMENTS = {
    "unet_mse"      : "UNet + MSE",
    "unet_bg"       : "UNet + Bernoulli-Gamma",
    "unet_compound" : "UNet + Compound (FSS+MSE)",
    "wgan_compound" : "WGAN + Compound",
}
