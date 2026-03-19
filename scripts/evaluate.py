"""
evaluate.py — Post-training evaluation on the held-out test set.
Full-image mode: one forward pass per test month, no sliding window.

Metrics (per month, then averaged over test set)
─────────────────────────────────────────────────
Pixel-wise (land only, in mm/month after inverse normalisation):
  MAE        mean absolute error
  RMSE       root mean squared error
  Bias       mean bias (pred − obs)
  Corr       Pearson spatial correlation

Spatial extremes:
  FSS_80/95/99   Fractions Skill Score at three percentiles
  R95p_bias      bias in the 95th-percentile value
  Peak_err       error in the maximum pixel value

Usage
-----
  python evaluate.py                      # all trained models
  python evaluate.py --exp unet_compound  # single model
  python evaluate.py --plot               # also save spatial maps
"""
"""
evaluate.py — Post-training evaluation on the held-out test set.
Full-image mode: one forward pass per test month, no sliding window.

Metrics (per month, then averaged over test set)
─────────────────────────────────────────────────
Pixel-wise (land only, in mm/month after inverse normalisation):
  MAE        mean absolute error
  RMSE       root mean squared error
  Bias       mean bias (pred − obs)
  Corr       Pearson spatial correlation

Spatial extremes:
  FSS_80/95/99   Fractions Skill Score at three percentiles
  R95p_bias      bias in the 95th-percentile value
  Peak_err       error in the maximum pixel value

Usage
-----
  python evaluate.py                      # all trained models
  python evaluate.py --exp unet_compound  # single model
  python evaluate.py --plot               # also save spatial maps
"""

"""
plot_results.py — Downscaling comparison plots (full-image mode).

The ERA5 TP comparison panel is sourced directly from the last channel
of X_test (denormalised from z-score) — no external .nc file needed.
This guarantees the displayed ERA5 TP is exactly what the model saw.

Usage
-----
  python plot_results.py                  # first test month
  python plot_results.py --month 6        # month index 6
  python plot_results.py --all            # every test month
"""

import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import downscaling.config as C
from downscaling.data_loader import (
    load_split, load_split_raw, load_land_mask,
    load_norm_stats, invert_chirps_norm, predict_full
)
from downscaling.models import (
    build_unet_mse, build_unet_compound, build_wgan_generator
)

PLOT_DIR = '../output/plots'
os.makedirs(PLOT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Custom precipitation colormap
# ─────────────────────────────────────────────────────────────────────────────
def make_cmap(high_vals=False, low_vals=False):
    precip_clevs = [0, 1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 70, 100, 150]
    if high_vals:
        precip_clevs = [0, 20, 40, 60, 80, 100, 125, 150,
                        175, 200, 225, 250, 300, 350, 400, 500]
    if low_vals:
        precip_clevs = [0, 2, 4, 6, 8, 10, 12, 14,
                        16, 19, 20, 22, 24, 26, 28, 30]
    tc_colours = [
        (255/255, 255/255, 255/255), (169/255, 209/255, 222/255),
        (137/255, 190/255, 214/255), (105/255, 160/255, 194/255),
        ( 93/255, 168/255,  98/255), (128/255, 189/255, 100/255),
        (165/255, 196/255, 134/255), (233/255, 245/255, 105/255),
        (245/255, 191/255, 105/255), (245/255, 112/255, 105/255),
        (245/255, 105/255, 149/255), (240/255,  93/255, 154/255),
        (194/255,  89/255, 188/255), ( 66/255,  57/255, 230/255),
        ( 24/255,  17/255, 153/255), (  9/255,   5/255,  87/255),
    ]
    cmap = ListedColormap(tc_colours[:len(precip_clevs)])
    norm = BoundaryNorm(precip_clevs, cmap.N, extend="max")
    return cmap, norm


# ─────────────────────────────────────────────────────────────────────────────
# Denormalise ERA5 TP from X  (z-score inverse, then upsample)
# ─────────────────────────────────────────────────────────────────────────────
def extract_era5_tp_mm(X_single: np.ndarray,
                        norm_stats: dict,
                        target_h: int,
                        target_w: int) -> np.ndarray:
    """
    Reverse the z-score on the TP channel (last channel of X) and
    bilinearly upsample to the CHIRPS display grid.

    Note: TP in X uses plain z-score — NOT log1p — because it is an
    input feature, not a target. The inverse is simply:
        tp_mm = tp_norm * (std + 1e-8) + mean

    Returns
    -------
    tp_mm : (target_h, target_w)  in mm/month, clipped >= 0
    """
    tp_idx  = X_single.shape[-1] - 1
    tp_norm = X_single[:, :, tp_idx]
    mu      = norm_stats["era5_mean"][tp_idx]
    sigma   = norm_stats["era5_std"][tp_idx]
    tp_real = tp_norm * (sigma + 1e-8) + mu

    tp_hr = tf.image.resize(
        tp_real[:, :, np.newaxis],
        [target_h, target_w],
        method=tf.image.ResizeMethod.BILINEAR
    ).numpy()[:, :, 0]

    return np.maximum(tp_hr, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Load trained model
# ─────────────────────────────────────────────────────────────────────────────
def load_model(exp_name: str) -> tf.keras.Model:
    builders   = {
        "unet_mse"      : build_unet_mse,
        "unet_compound" : build_unet_compound,
        "wgan_compound" : build_wgan_generator,
    }
    ckpt_names = {
        "unet_mse"      : "best_model.keras",
        "unet_compound" : "best_model.keras",
        "wgan_compound" : "best_generator.keras",
    }
    model = builders[exp_name]()
    ckpt  = os.path.join(C.OUTPUT_DIR, exp_name, ckpt_names[exp_name])
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    model.load_weights(ckpt)
    print(f"  Loaded {exp_name}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Inference: single month → mm/month
# ─────────────────────────────────────────────────────────────────────────────
def predict_month_mm(model, X_single: np.ndarray,
                     norm_stats: dict) -> np.ndarray:
    """Single forward pass → invert log1p normalisation → mm/month."""
    pred_norm = predict_full(model, X_single)       # (H_hr, W_hr, 1)
    pred_mm   = invert_chirps_norm(pred_norm, norm_stats)
    return pred_mm[:, :, 0]                         # (H_hr, W_hr)


# ─────────────────────────────────────────────────────────────────────────────
# Shared axis styling
# ─────────────────────────────────────────────────────────────────────────────
def _style_ax(ax, lons, lats, title):
    ax.set_title(title, fontsize=9, fontweight="bold", pad=4)
    ax.set_xlabel("Longitude", fontsize=8)
    ax.set_ylabel("Latitude",  fontsize=8)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.tick_params(labelsize=7)
    ax.set_xlim(lons.min(), lons.max())
    ax.set_ylim(lats.min(), lats.max())
    ax.grid(True, linewidth=0.3, alpha=0.4, color="grey")


def _add_cbar(fig, ax, im, label):
    divider = make_axes_locatable(ax)
    cax     = divider.append_axes("right", size="4%", pad=0.06)
    cb      = fig.colorbar(im, cax=cax, extend="max")
    cb.set_label(label, fontsize=7)
    cb.ax.tick_params(labelsize=6)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — Absolute rainfall maps
# ─────────────────────────────────────────────────────────────────────────────
def plot_absolute(chirps_mm, era5_tp_mm, pred_mse_mm, pred_wgan_mm,
                  land_mask, lats, lons, month_label, out_path):
    """
    4-panel figure.
    - CHIRPS truth      → high_vals colormap  (0–500 mm)
    - ERA5 TP (from X)  → low_vals  colormap  (0–30 mm)  — LR, full domain
    - UNet prediction   → high_vals colormap  (0–500 mm)
    - WGAN prediction   → high_vals colormap  (0–500 mm)
    """
    cmap_hi,  norm_hi  = make_cmap(high_vals=True)
    cmap_low, norm_low = make_cmap(low_vals=True)
    ext = [lons.min(), lons.max(), lats.min(), lats.max()]

    panels = [
        (np.where(land_mask, chirps_mm,   np.nan), cmap_hi,  norm_hi,
         "CHIRPS (truth)\n0.05° target"),
        (era5_tp_mm,                                cmap_low, norm_low,
         "ERA5 TP (input ch.6)\n0.50° low-res"),
        (np.where(land_mask, pred_mse_mm,  np.nan), cmap_hi,  norm_hi,
         "UNet-Compound\n0.05° prediction"),
        (np.where(land_mask, pred_wgan_mm, np.nan), cmap_hi,  norm_hi,
         "WGAN+Compound\n0.05° prediction"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(24, 7), constrained_layout=True)
    for ax, (data, cmap, norm, title) in zip(axes, panels):
        im = ax.imshow(data, cmap=cmap, norm=norm,
                        extent=ext, origin="upper", aspect="auto")
        _style_ax(ax, lons, lats, title)
        _add_cbar(fig, ax, im, "mm/month")

    fig.suptitle(f"Precipitation Downscaling — {month_label}  "
                 f"(ERA5 0.50° → CHIRPS 0.05°)",
                 fontsize=13, fontweight="bold")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — Bias maps  (prediction − CHIRPS, land only)
# ─────────────────────────────────────────────────────────────────────────────
def plot_bias(chirps_mm, era5_tp_mm, pred_mse_mm, pred_wgan_mm,
              land_mask, lats, lons, month_label, out_path):
    chirps_land = np.where(land_mask, chirps_mm, np.nan)
    era5_land   = np.where(land_mask, era5_tp_mm, np.nan)

    panels = [
        (pred_mse_mm  - chirps_land, "UNet-Compound − CHIRPS"),
        (pred_wgan_mm - chirps_land, "WGAN+Compound − CHIRPS"),
        (era5_land    - chirps_land, "ERA5 TP − CHIRPS  (baseline)"),
    ]
    panels = [(np.where(land_mask, b, np.nan), t) for b, t in panels]

    all_vals = np.concatenate([b[np.isfinite(b)] for b, _ in panels])
    abs_max  = np.nanpercentile(np.abs(all_vals), 98)
    ext      = [lons.min(), lons.max(), lats.min(), lats.max()]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    for ax, (bias, title) in zip(axes, panels):
        im   = ax.imshow(bias, cmap=plt.cm.RdBu_r,
                          vmin=-abs_max, vmax=abs_max,
                          extent=ext, origin="upper", aspect="auto")
        land = bias[np.isfinite(bias)]
        _style_ax(ax, lons, lats,
                  f"{title}\nbias={np.nanmean(land):+.1f}  "
                  f"RMSE={np.sqrt(np.nanmean(land**2)):.1f} mm/month")
        _add_cbar(fig, ax, im, "mm/month")

    fig.suptitle(f"Bias Maps (Pred − CHIRPS) — {month_label}",
                 fontsize=13, fontweight="bold")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--month", type=int, default=0)
    parser.add_argument("--all",   action="store_true")
    args = parser.parse_args()

    print("Loading test data ...")
    X_test, _, times_test = load_split(C.TEST_FILE)
    y_test_raw            = load_split_raw(C.TEST_FILE)
    land_mask             = load_land_mask(C.META_FILE).astype(bool)
    norm_stats            = load_norm_stats(C.META_FILE)
    meta                  = dict(np.load(C.META_FILE, allow_pickle=True))

    chirps_lats = meta["chirps_lats"]
    chirps_lons = meta["chirps_lons"]
    test_times  = pd.DatetimeIndex(times_test)

    print(f"  Test months  : {len(test_times)}")
    print(f"  X channels   : {list(meta['channel_names'])}")
    print(f"  TP channel   : index {X_test.shape[-1]-1}  "
          f"(last channel — will be denormed for display)")

    print("\nLoading models ...")
    model_compound = load_model("unet_compound")
    model_wgan     = load_model("wgan_compound")

    month_indices = list(range(len(test_times))) if args.all else [args.month]

    for idx in month_indices:
        if idx >= len(test_times):
            print(f"  Skipping {idx} — only {len(test_times)} test months")
            continue

        month_label = str(test_times[idx])[:7]
        print(f"\nPlotting month {idx}: {month_label}")

        chirps_map   = y_test_raw[idx, :, :, 0]
        era5_tp_map  = extract_era5_tp_mm(
            X_test[idx], norm_stats, C.CHIRPS_H, C.CHIRPS_W
        )

        pred_compound = predict_month_mm(model_compound, X_test[idx], norm_stats)
        pred_wgan     = predict_month_mm(model_wgan,     X_test[idx], norm_stats)

        out1 = os.path.join(PLOT_DIR, f"plot1_absolute_{month_label}.png")
        plot_absolute(chirps_map, era5_tp_map, pred_compound, pred_wgan,
                      land_mask, chirps_lats, chirps_lons, month_label, out1)

        out2 = os.path.join(PLOT_DIR, f"plot2_bias_{month_label}.png")
        plot_bias(chirps_map, era5_tp_map, pred_compound, pred_wgan,
                  land_mask, chirps_lats, chirps_lons, month_label, out2)

    print(f"\nAll plots saved to: {PLOT_DIR}")


if __name__ == "__main__":
    main()