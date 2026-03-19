"""
Mozambique Precipitation Downscaling Package
============================================
ERA5 (0.50°) → CHIRPS (0.05°) statistical downscaling using deep learning.

Modules
-------
config        Central configuration (paths, grid sizes, hyperparameters)
data_loader   tf.data pipeline, normalisation helpers, inference utilities
models        RA-UNet and WGAN-GP architectures
losses        Masked MSE, Bernoulli-Gamma NLL, Compound FSS+MSE, WGAN-GP
train         Training entry point  (python train.py --exp <name>)
evaluate      Evaluation entry point (python evaluate.py --exp <name>)
plot_results  Visualisation entry point (python plot_results.py --month <i>)

Quick start
-----------
    # 1. Prepare training data
    python training_data_prep.py

    # 2. Train all four experiments
    python train.py

    # 3. Evaluate on the test set
    python evaluate.py --plot

    # 4. Plot maps for a specific month
    python plot_results.py --month 0

Experiments
-----------
    unet_mse       RA-UNet trained with masked MSE
    unet_bg        RA-UNet trained with Bernoulli-Gamma NLL
    unet_compound  RA-UNet trained with compound FSS' + MSE loss
    wgan_compound  WGAN-GP generator trained with compound + adversarial loss
"""

# ---------------------------------------------------------------------------
# Public API — import the most commonly used symbols at package level
# ---------------------------------------------------------------------------

# Replace all bare imports in __init__.py with relative imports

# In __init__.py, change this block:
from .config import (
    SCALE_FACTOR,
    ERA5_TARGET_RES,      # ← remove this line
    UNET_INPUT_SHAPE,
    HR_SHAPE,
    N_ERA5_VARS,
    CHIRPS_H,
    CHIRPS_W,
    ERA5_H,
    ERA5_W,
    EXPERIMENTS,
)

from .data_loader import (
    load_split,
    load_split_raw,
    load_land_mask,
    load_norm_stats,
    build_dataset,
    invert_chirps_norm,
    predict_full,
    predict_all,
)

from .models import (
    build_unet_mse,
    build_unet_bg,
    build_unet_compound,
    build_wgan_generator,
    build_wgan_discriminator,
    bg_expected_rainfall,
)

from .losses import (
    masked_mse,
    bernoulli_gamma_nll,
    compound_loss,
    fss_loss_batch,
    wasserstein_generator_loss,
    wasserstein_discriminator_loss,
    gradient_penalty,
    wgan_generator_loss_total,
)

__all__ = [
    # config
    "SCALE_FACTOR", "ERA5_TARGET_RES",
    "UNET_INPUT_SHAPE", "HR_SHAPE",
    "N_ERA5_VARS", "CHIRPS_H", "CHIRPS_W", "ERA5_H", "ERA5_W",
    "EXPERIMENTS",
    # data_loader
    "load_split", "load_split_raw", "load_land_mask", "load_norm_stats",
    "build_dataset", "invert_chirps_norm", "predict_full", "predict_all",
    # models
    "build_unet_mse", "build_unet_bg", "build_unet_compound",
    "build_wgan_generator", "build_wgan_discriminator",
    "bg_expected_rainfall",
    # losses
    "masked_mse", "bernoulli_gamma_nll", "compound_loss", "fss_loss_batch",
    "wasserstein_generator_loss", "wasserstein_discriminator_loss",
    "gradient_penalty", "wgan_generator_loss_total",
]

__version__ = "1.0.0"
__author__  = "Mozambique Downscaling Project"
