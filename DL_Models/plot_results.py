"""
plot_results.py  v2 — PP Downscaling comparison plots.

Fixes over v1
-------------
  - ERA5 TP loading: robust dimension-name handling (latitude/longitude
    vs lat/lon) so the field is never blank
  - ERA5 displayed without land-masking (it covers ocean too)
  - Gaussian smoothing applied to stitched predictions to remove patch seams
  - Custom precipitation colormap (make_cmap) used for absolute maps
  - Diverging bias colormap unchanged

Usage
-----
  python plot_results.py                  # first test month
  python plot_results.py --month 6        # month index 6
  python plot_results.py --all            # every test month
"""

import os
import re
import argparse
import numpy as np
import xarray as xr
import pandas as pd
import tensorflow as tf
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import config as C
from data_loader import load_split, load_land_mask, reconstruct_full
from models import build_unet_mse, build_unet_compound, build_wgan_generator

# ─────────────────────────────────────────────────────────────────────────────
# Paths  — edit ERA5_TP_FILE if needed
# ─────────────────────────────────────────────────────────────────────────────
ERA5_TP_FILE = r"C:\Users\HP ZBOOK\RAINFALL_ DONWSCALLING\DATA\INPUT\ERA5_TP.nc"
PLOT_DIR     = os.path.join(C.OUTPUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Custom precipitation colormap  (provided by user)
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
    precip_cmap = ListedColormap(tc_colours[:len(precip_clevs)])
    precip_norm = BoundaryNorm(precip_clevs, precip_cmap.N, extend="max")
    return precip_cmap, precip_norm


# ─────────────────────────────────────────────────────────────────────────────
# Load trained model
# ─────────────────────────────────────────────────────────────────────────────
def load_model(exp_name):
    builders   = {"unet_mse": build_unet_mse,
                  "unet_compound": build_unet_compound,
                  "wgan_compound": build_wgan_generator}
    ckpt_names = {"unet_mse": "best_model.keras",
                  "unet_compound": "best_model.keras",
                  "wgan_compound": "best_generator.keras"}
    model = builders[exp_name]()
    ckpt  = os.path.join(C.OUTPUT_DIR, exp_name, ckpt_names[exp_name])
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    model.load_weights(ckpt)
    print(f"  Loaded {exp_name}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Load ERA5 TP  — robust to lat/lon vs latitude/longitude naming
# ─────────────────────────────────────────────────────────────────────────────
def load_era5_tp(meta, test_times):
    print("Loading ERA5 TP ...")
    ds = xr.open_dataset(ERA5_TP_FILE)
    print(f"  Raw dims : {list(ds.dims)}")
    print(f"  Raw vars : {list(ds.data_vars)}")

    # ── 1. Normalise time dimension ───────────────────────────────────────
    for old in ["valid_time", "time"]:
        if old in ds.dims and old != "time":
            ds = ds.rename({old: "time"})
            break
    ds["time"] = pd.to_datetime(ds["time"].values)

    # ── 2. Normalise spatial dimensions ──────────────────────────────────
    # Handle all common ERA5 naming conventions
    rename_map = {}
    if "latitude"  in ds.dims: rename_map["latitude"]  = "lat"
    if "longitude" in ds.dims: rename_map["longitude"] = "lon"
    if rename_map:
        ds = ds.rename(rename_map)
    print(f"  Dims after rename: {list(ds.dims)}")

    # ── 3. Find precipitation variable ───────────────────────────────────
    tp_var = None
    for c in ["tp", "total_precipitation", "precip", "pr", "precipitation"]:
        if c in ds:
            tp_var = c
            break
    if tp_var is None:
        raise KeyError(f"Cannot find TP. Available: {list(ds.data_vars)}")
    print(f"  Using variable: '{tp_var}'")

    # ── 4. Unit conversion ────────────────────────────────────────────────
    sample_val = float(ds[tp_var].isel(time=0).mean())
    print(f"  Sample mean value before conversion: {sample_val:.6f}")
    if sample_val < 2.0:     # still in metres
        print("  Converting m → mm  (×1000)")
        tp_mm = ds[tp_var] * 1000.0
    else:
        print("  Already in mm — no conversion")
        tp_mm = ds[tp_var]

    # Clip negative artefacts
    tp_mm = tp_mm.clip(min=0)

    # ── 5. Target grid ────────────────────────────────────────────────────
    chirps_lats = meta["chirps_lats"].astype(np.float64)
    chirps_lons = meta["chirps_lons"].astype(np.float64)

    era5_maps = []
    for t in test_times:
        try:
            era5_t      = tp_mm.sel(time=t, method="nearest")
            era5_interp = era5_t.interp(lat=chirps_lats,
                                         lon=chirps_lons,
                                         method="linear")
            arr = era5_interp.values.astype(np.float32)
        except Exception as e:
            print(f"  Warning: {t} → {e}")
            arr = np.full((len(chirps_lats), len(chirps_lons)),
                          np.nan, dtype=np.float32)
        era5_maps.append(arr)

    era5_arr = np.stack(era5_maps, axis=0)
    print(f"  ERA5 TP shape: {era5_arr.shape}  "
          f"range: {np.nanmin(era5_arr):.1f}–{np.nanmax(era5_arr):.1f} mm")
    return era5_arr


# ─────────────────────────────────────────────────────────────────────────────
# Inference with Gaussian smoothing to remove patch seams
# ─────────────────────────────────────────────────────────────────────────────
def predict_month(model, X_single, smooth_sigma=1.5):
    """
    Sliding-window inference + Gaussian smoothing.

    smooth_sigma : std of the Gaussian kernel in pixels (CHIRPS pixels).
                   1.5 px ≈ 7.5 km — smooths seam artefacts without blurring
                   real rainfall gradients.  Set to 0 to disable.
    """
    pred = reconstruct_full(model, X_single)   # (H_hr, W_hr, 1)
    arr  = pred[:, :, 0]                        # (H_hr, W_hr)
    if smooth_sigma > 0:
        arr = gaussian_filter(arr, sigma=smooth_sigma)
    arr = np.maximum(arr, 0.0)                  # clip any negatives post-blur
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# Masking helpers
# ─────────────────────────────────────────────────────────────────────────────
def _mask_land_only(arr, land_mask):
    """Mask ocean pixels with NaN — for CHIRPS and predictions."""
    out = arr.copy().astype(np.float32)
    out[~land_mask] = np.nan
    return out


def _mask_valid(arr):
    """Mask only fill-value pixels (<0) — for ERA5 which includes ocean."""
    out = arr.copy().astype(np.float32)
    out[out < 0] = np.nan
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Shared axis styling
# ─────────────────────────────────────────────────────────────────────────────
def _style_ax(ax, lons, lats, title):
    ax.set_title(title, fontsize=10, fontweight="bold", pad=4)
    ax.set_xlabel("Longitude", fontsize=8)
    ax.set_ylabel("Latitude",  fontsize=8)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.tick_params(labelsize=7)
    ax.set_xlim(lons.min(), lons.max())
    ax.set_ylim(lats.min(), lats.max())
    ax.grid(True, linewidth=0.3, alpha=0.5, color="grey")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — Absolute rainfall maps (custom colormap)
# ─────────────────────────────────────────────────────────────────────────────
def plot_absolute(chirps, era5, pred_mse, pred_wgan,
                  land_mask, lats, lons, month_label, out_path):
    """
    4-panel figure using the custom precipitation colormap.
    ERA5 shown over full domain (land + ocean).
    CHIRPS and predictions shown over land only.
    """
    precip_cmap, precip_norm = make_cmap(high_vals=True)

    ext = [lons.min(), lons.max(), lats.min(), lats.max()]

    panels = [
        (_mask_land_only(chirps,    land_mask), "CHIRPS\n(ground truth)",      True),
        (_mask_valid(era5),                      "ERA5 TP\n(raw reanalysis)",   False),
        (_mask_land_only(pred_mse,  land_mask), "UNet-MSE\n(prediction)",       True),
        (_mask_land_only(pred_wgan, land_mask), "WGAN+Compound\n(prediction)",  True),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(22, 6),
                              constrained_layout=True)

    for ax, (data, title, _) in zip(axes, panels):
        im = ax.imshow(data,
                       cmap=precip_cmap, norm=precip_norm,
                       extent=ext, origin="upper", aspect="auto")
        _style_ax(ax, lons, lats, title)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.06)
        cb  = plt.colorbar(im, cax=cax, extend="max")
        cb.set_label("mm / month", fontsize=7)
        cb.ax.tick_params(labelsize=6)

    fig.suptitle(f"PP Downscaling — Absolute Rainfall — {month_label}",
                 fontsize=13, fontweight="bold")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — Bias maps (prediction − CHIRPS)
# ─────────────────────────────────────────────────────────────────────────────
def plot_bias(chirps, era5, pred_mse, pred_wgan,
              land_mask, lats, lons, month_label, out_path):
    """
    3-panel diverging bias figure (land only).
    Blue = underestimate, Red = overestimate.
    """
    # Only compare over land where CHIRPS is valid
    chirps_land = np.where(land_mask, chirps, np.nan)
    era5_land   = np.where(land_mask & np.isfinite(era5), era5, np.nan)

    bias_mse  = _mask_land_only(pred_mse  - chirps_land, land_mask)
    bias_wgan = _mask_land_only(pred_wgan - chirps_land, land_mask)
    bias_era5 = np.where(np.isfinite(era5_land),
                          era5_land - chirps_land, np.nan)

    panels = [
        (bias_mse,  "UNet-MSE − CHIRPS"),
        (bias_wgan, "WGAN+Compound − CHIRPS"),
        (bias_era5, "ERA5 − CHIRPS"),
    ]

    # Symmetric colour limit at 98th percentile of absolute bias
    all_bias = np.concatenate([
        b[np.isfinite(b)] for b, _ in panels
    ])
    abs_max = np.nanpercentile(np.abs(all_bias), 98)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6),
                              constrained_layout=True)
    ext      = [lons.min(), lons.max(), lats.min(), lats.max()]
    cmap_div = plt.cm.RdBu_r

    for ax, (bias, title) in zip(axes, panels):
        im = ax.imshow(bias,
                       cmap=cmap_div,
                       vmin=-abs_max, vmax=abs_max,
                       extent=ext, origin="upper", aspect="auto")

        land_vals = bias[np.isfinite(bias)]
        mean_b    = np.nanmean(land_vals)
        rmse      = np.sqrt(np.nanmean(land_vals**2))

        _style_ax(ax, lons, lats,
                  f"{title}\nbias={mean_b:+.1f}  RMSE={rmse:.1f}  mm/month")

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.06)
        cb  = plt.colorbar(im, cax=cax)
        cb.set_label("mm / month", fontsize=7)
        cb.ax.tick_params(labelsize=6)

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
    parser.add_argument("--month", type=int, default=0)
    parser.add_argument("--all",   action="store_true")
    parser.add_argument("--sigma", type=float, default=1.5,
                        help="Gaussian smoothing sigma in CHIRPS pixels "
                             "(default 1.5, set 0 to disable)")
    args = parser.parse_args()

    # ── Load processed test data ──────────────────────────────────────────
    print("Loading test data ...")
    X_test, y_test, times_test = load_split(C.TEST_FILE)
    land_mask = load_land_mask(C.META_FILE).astype(bool)
    meta      = dict(np.load(C.META_FILE, allow_pickle=True))

    chirps_lats = meta["chirps_lats"]
    chirps_lons = meta["chirps_lons"]

    # CHIRPS: sentinel -1 → NaN
    y_test_mm = np.where(y_test >= 0, y_test, np.nan)[:, :, :, 0]

    # ── Load ERA5 TP ──────────────────────────────────────────────────────
    test_times_pd = pd.DatetimeIndex(times_test)
    era5_tp = load_era5_tp(meta, test_times_pd)

    # ── Load models ───────────────────────────────────────────────────────
    print("\nLoading models ...")
    model_mse  = load_model("unet_mse")
    model_wgan = load_model("wgan_compound")

    # ── Plot ──────────────────────────────────────────────────────────────
    month_indices = (list(range(len(times_test)))
                     if args.all else [args.month])

    for idx in month_indices:
        if idx >= len(times_test):
            print(f"  Skipping {idx} — only {len(times_test)} test months")
            continue

        month_label = str(test_times_pd[idx])[:7]
        print(f"\nPlotting month {idx}: {month_label}")

        chirps_map = y_test_mm[idx]
        era5_map   = era5_tp[idx]

        print("  UNet-MSE inference ...")
        pred_mse  = predict_month(model_mse,  X_test[idx], args.sigma)

        print("  WGAN inference ...")
        pred_wgan = predict_month(model_wgan, X_test[idx], args.sigma)

        # Plot 1
        out1 = os.path.join(PLOT_DIR, f"plot1_absolute_{month_label}.png")
        plot_absolute(chirps_map, era5_map, pred_mse, pred_wgan,
                      land_mask, chirps_lats, chirps_lons,
                      month_label, out1)

        # Plot 2
        out2 = os.path.join(PLOT_DIR, f"plot2_bias_{month_label}.png")
        plot_bias(chirps_map, era5_map, pred_mse, pred_wgan,
                  land_mask, chirps_lats, chirps_lons,
                  month_label, out2)

    print(f"\nAll plots saved to: {PLOT_DIR}")


if __name__ == "__main__":
    main()
