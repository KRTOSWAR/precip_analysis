"""
config.py — Central configuration for all experiments.
Full-image mode: the entire Mozambique domain is fed to the UNet at once.
No patching. Adjust DATA_DIR / OUTPUT_DIR to match your environment.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR   = '../data/processed/training_data'
TRAIN_FILE = os.path.join(DATA_DIR, "train.npz")
VAL_FILE   = os.path.join(DATA_DIR, "val.npz")
TEST_FILE  = os.path.join(DATA_DIR, "test.npz")
META_FILE  = os.path.join(DATA_DIR, "metadata.npz")
OUTPUT_DIR = '../output/training_results' 

# ---------------------------------------------------------------------------
# Spatial configuration
# These come from the training_data_prep.py pipeline output.
# Update if you change the domain or resolution.
# ---------------------------------------------------------------------------
SCALE_FACTOR = 10           # ERA5 0.50° → CHIRPS 0.05°
ERA5_H       = 34           # low-res grid height  (H_lr)
ERA5_W       = 23           # low-res grid width   (W_lr)
CHIRPS_H     = 340          # high-res grid height (ERA5_H × SCALE_FACTOR)
CHIRPS_W     = 230          # high-res grid width  (ERA5_W × SCALE_FACTOR)
N_ERA5_VARS  = 7            # q, t, u, v, w, z, tp

# Add this line to config.py alongside SCALE_FACTOR
SCALE_FACTOR    = 10
ERA5_TARGET_RES = 0.50     # deg — add this line

# Full-image shapes (ERA5 bilinearly upsampled to CHIRPS resolution)
UNET_INPUT_SHAPE = (CHIRPS_H, CHIRPS_W, N_ERA5_VARS)
HR_SHAPE         = (CHIRPS_H, CHIRPS_W, 1)

# UNet requires H and W divisible by 2^(n_pooling_stages) = 2^3 = 8.
# We zero-pad the input and crop the output back to the original size.
_POOL_MULT = 8
PAD_H      = (_POOL_MULT - CHIRPS_H % _POOL_MULT) % _POOL_MULT   # 4  → padded H = 344
PAD_W      = (_POOL_MULT - CHIRPS_W % _POOL_MULT) % _POOL_MULT   # 2  → padded W = 232

# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------
BATCH_SIZE          = 4     # months per batch; full images fit easily in RAM
EPOCHS              = 30
LR_INIT             = 1e-1
EARLY_STOP_PATIENCE = 5

# UNet filter sizes  
UNET_FILTERS = [32, 64, 128, 256]

# ---------------------------------------------------------------------------
# Loss-specific configuration
# ---------------------------------------------------------------------------
COMPOUND_FSS_WEIGHT = 0.7
COMPOUND_MSE_WEIGHT = 1.0
FSS_N               = 15        # neighbourhood size for FSS (must be odd)
FSS_PERCENTILES     = [80, 95, 99]
RAIN_THRESHOLD      = 1.0       # mm/month — dry/wet split for Bernoulli-Gamma

# WGAN-GP
WGAN_GP_LAMBDA = 10
WGAN_N_CRITIC  = 5
WGAN_GEN_LR    = 1e-4
WGAN_DIS_LR    = 1e-4

# ---------------------------------------------------------------------------
# Experiment registry
# ---------------------------------------------------------------------------
EXPERIMENTS = {
    "unet_mse"      : "UNet + MSE",
    "unet_bg"       : "UNet + Bernoulli-Gamma",
    "unet_compound" : "UNet + Compound (FSS+MSE)",
    "wgan_compound" : "WGAN + Compound",
}
