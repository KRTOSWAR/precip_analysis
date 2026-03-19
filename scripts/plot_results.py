"""
plot_results.py — Downscaling comparison plots (full-image mode).
No sliding window. No Gaussian smoothing. One forward pass per month.

Usage
-----
  python plot_results.py                  # first test month
  python plot_results.py --month 6        # month index 6
  python plot_results.py --all            # every test month
"""

import os
import argparse
import numpy as np
import xarray as xr
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import downscaling.config as C
from downscaling.data_loader import (
    load_split, load_split_raw, load_land_mask,
    load_norm_stats, invert_chirps_norm, predict_full
)
from downscaling.models import build_unet_mse, build_unet_compound, build_wgan_generator

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ERA5_TP_FILE = '../data/processed/era5_tp_mozambique.nc'
PLOT_DIR     = '../output/plots'
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
        (255/255, 255/255, 255/255),
        (169/255, 209/255, 222/255),
        (137/255, 190/255, 214/255),
        (105/255, 160/255, 194/255),
        ( 93/255, 168/255,  98/255),
        (128/255, 189/255, 100/255),
        (165/255, 196/255, 134/255),
        (233/255, 245/255, 105/255),
        (245/255, 191/255, 105/255),
        (245/255, 112/255, 105/255),
        (245/255, 105/255, 149/255),
        (240/255,  93/255, 154/255),
        (194/255,  89/255, 188/255),
        ( 66/255,  57/255, 230/255),
        ( 24/255,  17/255, 153/255),
        (  9/255,   5/255,  87/255),
    ]
    cmap = ListedColormap(tc_colours[:len(precip_clevs)])
    norm = BoundaryNorm(precip_clevs, cmap.N, extend="max")
    return cmap, norm


# ─────────────────────────────────────────────────────────────────────────────
# Load trained model from checkpoint
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
# Load ERA5 TP (low-res reference, shown without land mask)
# ─────────────────────────────────────────────────────────────────────────────
def load_era5_tp(meta: dict, test_times: pd.DatetimeIndex) -> np.ndarray:
    """Interpolate ERA5 TP to CHIRPS grid for each test month."""
    print("Loading ERA5 TP for comparison ...")
    ds = xr.open_dataset(ERA5_TP_FILE)

    for old in ["valid_time", "latitude", "longitude"]:
        new = {"valid_time": "time", "latitude": "lat", "longitude": "lon"}
        if old in ds.dims:
            ds = ds.rename({old: new[old]})

    ds["time"] = pd.to_datetime(ds["time"].values)

    tp_var = next(
        (v for v in ds.data_vars
         if any(k in v.lower() for k in ["tp", "precip", "precipitation"])),
        list(ds.data_vars)[0]
    )
    tp = ds[tp_var].clip(min=0)

    # Unit check — convert m → mm if needed
    if float(tp.isel(time=0).mean()) < 2.0:
        tp = tp * 1000.0
        print("  Converted m → mm (×1000)")

    chirps_lats = meta["chirps_lats"].astype(np.float64)
    chirps_lons = meta["chirps_lons"].astype(np.float64)

    era5_maps = []
    for t in test_times:
        try:
            arr = tp.sel(time=t, method="nearest").interp(
                lat=chirps_lats, lon=chirps_lons, method="linear"
            ).values.astype(np.float32)
        except Exception as e:
            print(f"  Warning {t}: {e}")
            arr = np.full((len(chirps_lats), len(chirps_lons)), np.nan, np.float32)
        era5_maps.append(arr)

    result = np.stack(era5_maps, axis=0)
    print(f"  ERA5 TP shape: {result.shape}  "
          f"range: {np.nanmin(result):.1f}–{np.nanmax(result):.1f} mm")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Inference: full-image, single forward pass + inverse norm
# ─────────────────────────────────────────────────────────────────────────────
def predict_month_mm(model, X_single: np.ndarray,
                     norm_stats: dict) -> np.ndarray:
    """
    Predict one month and return result in mm/month.

    Parameters
    ----------
    X_single   : (H_lr, W_lr, C)
    norm_stats : dict from load_norm_stats

    Returns
    -------
    pred_mm : (H_hr, W_hr)  in mm/month, non-negative
    """
    pred_norm = predict_full(model, X_single)   # (H_hr, W_hr, 1)
    pred_mm   = invert_chirps_norm(pred_norm, norm_stats)
    return pred_mm[:, :, 0]                     # (H_hr, W_hr)


# ─────────────────────────────────────────────────────────────────────────────
# Shared axis styling
# ─────────────────────────────────────────────────────────────────────────────
def _style_ax(ax, lons, lats, title):
    ax.set_title(title, fontsize=10, fontweight="bold", pad=4)
    ax.set_xlabel("Longitude", fontsize=8)
    ax.set_ylabel("Latitude",  fontsize=8)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.tick_params(labelsize=7)
    ax.set_xlim(lons.min(), lons.max())
    ax.set_ylim(lats.min(), lats.max())
    ax.grid(True, linewidth=0.3, alpha=0.5, color="grey")


def _add_cbar(fig, ax, im, label):
    divider = make_axes_locatable(ax)
    cax     = divider.append_axes("right", size="4%", pad=0.06)
    cb      = fig.colorbar(im, cax=cax, extend="max")
    cb.set_label(label, fontsize=7)
    cb.ax.tick_params(labelsize=6)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — Absolute rainfall maps
# ─────────────────────────────────────────────────────────────────────────────
def plot_absolute(chirps_mm, era5_mm, pred_mse_mm, pred_wgan_mm,
                  land_mask, lats, lons, month_label, out_path):
    precip_cmap, precip_norm = make_cmap(high_vals=True)
    ext = [lons.min(), lons.max(), lats.min(), lats.max()]

    panels = [
        (np.where(land_mask, chirps_mm,    np.nan), "CHIRPS (truth)"),
        (np.where(era5_mm >= 0, era5_mm,   np.nan), "ERA5 TP (raw)"),
        (np.where(land_mask, pred_mse_mm,  np.nan), "UNet-MSE"),
        (np.where(land_mask, pred_wgan_mm, np.nan), "WGAN+Compound"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(22, 6), constrained_layout=True)
    for ax, (data, title) in zip(axes, panels):
        im = ax.imshow(data, cmap=precip_cmap, norm=precip_norm,
                        extent=ext, origin="upper", aspect="auto")
        _style_ax(ax, lons, lats, title)
        _add_cbar(fig, ax, im, "mm/month")

    fig.suptitle(f"PP Downscaling — Absolute Rainfall — {month_label}",
                 fontsize=13, fontweight="bold")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — Bias maps  (prediction − CHIRPS)
# ─────────────────────────────────────────────────────────────────────────────
def plot_bias(chirps_mm, era5_mm, pred_mse_mm, pred_wgan_mm,
              land_mask, lats, lons, month_label, out_path):
    chirps_land = np.where(land_mask, chirps_mm, np.nan)

    panels = [
        (pred_mse_mm  - chirps_land, "UNet-MSE − CHIRPS"),
        (pred_wgan_mm - chirps_land, "WGAN+Compound − CHIRPS"),
        (era5_mm      - chirps_land, "ERA5 − CHIRPS"),
    ]
    panels = [(np.where(land_mask, b, np.nan), t) for b, t in panels]

    all_vals = np.concatenate([b[np.isfinite(b)] for b, _ in panels])
    abs_max  = np.nanpercentile(np.abs(all_vals), 98)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    ext       = [lons.min(), lons.max(), lats.min(), lats.max()]

    for ax, (bias, title) in zip(axes, panels):
        im   = ax.imshow(bias, cmap=plt.cm.RdBu_r,
                          vmin=-abs_max, vmax=abs_max,
                          extent=ext, origin="upper", aspect="auto")
        land = bias[np.isfinite(bias)]
        _style_ax(ax, lons, lats,
                  f"{title}\nbias={np.nanmean(land):+.1f}  "
                  f"RMSE={np.sqrt(np.nanmean(land**2)):.1f} mm/month")
        _add_cbar(fig, ax, im, "mm/month")

    fig.suptitle(f"PP Downscaling — Bias (Pred − CHIRPS) — {month_label}",
                 fontsize=13, fontweight="bold")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--month", type=int, default=0,
                        help="Single test month index to plot (default 0)")
    parser.add_argument("--all",   action="store_true",
                        help="Plot all test months")
    args = parser.parse_args()

    print("Loading test data ...")
    X_test, _, times_test = load_split(C.TEST_FILE)
    y_test_raw            = load_split_raw(C.TEST_FILE)   # (N, H, W, 1) mm
    land_mask             = load_land_mask(C.META_FILE).astype(bool)
    norm_stats            = load_norm_stats(C.META_FILE)
    meta                  = dict(np.load(C.META_FILE, allow_pickle=True))

    chirps_lats  = meta["chirps_lats"]
    chirps_lons  = meta["chirps_lons"]
    test_times   = pd.DatetimeIndex(times_test)

    # ERA5 TP on CHIRPS grid for comparison panel
    era5_tp = load_era5_tp(meta, test_times)

    # Load models
    print("\nLoading models ...")
    model_mse  = load_model("unet_compound")
    model_wgan = load_model("wgan_compound")

    month_indices = list(range(len(test_times))) if args.all else [args.month]

    for idx in month_indices:
        if idx >= len(test_times):
            print(f"  Skipping {idx} — only {len(test_times)} test months")
            continue

        month_label = str(test_times[idx])[:7]
        print(f"\nPlotting month {idx}: {month_label}")

        chirps_map    = y_test_raw[idx, :, :, 0]     # mm, NaN on ocean
        era5_map      = era5_tp[idx]

        pred_mse_mm   = predict_month_mm(model_mse,  X_test[idx], norm_stats)
        pred_wgan_mm  = predict_month_mm(model_wgan, X_test[idx], norm_stats)

        out1 = os.path.join(PLOT_DIR, f"plot1_absolute_{month_label}.png")
        plot_absolute(chirps_map, era5_map, pred_mse_mm, pred_wgan_mm,
                      land_mask, chirps_lats, chirps_lons, month_label, out1)

        out2 = os.path.join(PLOT_DIR, f"plot2_bias_{month_label}.png")
        plot_bias(chirps_map, era5_map, pred_mse_mm, pred_wgan_mm,
                  land_mask, chirps_lats, chirps_lons, month_label, out2)

    print(f"\nAll plots saved to: {PLOT_DIR}")


if __name__ == "__main__":
    main()
