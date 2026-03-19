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

import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from downscaling import config as C
from downscaling.data_loader import (
    load_split, load_split_raw, load_land_mask,
    load_norm_stats, invert_chirps_norm, predict_all
)
from downscaling.models import (
    build_unet_mse, build_unet_bg, build_unet_compound,
    build_wgan_generator
)
from downscaling.losses import _fss_single_3d


# ─────────────────────────────────────────────────────────────────────────────
# Custom precipitation colormap  (shared with plot_results.py)
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
# Denormalise ERA5 TP channel from X
# ─────────────────────────────────────────────────────────────────────────────
def extract_era5_tp_mm(X_single: np.ndarray, norm_stats: dict,
                        target_h: int, target_w: int) -> np.ndarray:
    """
    Extract the TP channel from a single normalised ERA5 sample,
    reverse the z-score, and bilinearly upsample to the CHIRPS grid.

    Parameters
    ----------
    X_single  : (H_lr, W_lr, C)  normalised ERA5 — TP is channel index -1
    norm_stats: dict with 'era5_mean' and 'era5_std'  shape (C,)
    target_h  : CHIRPS grid height  (C.CHIRPS_H)
    target_w  : CHIRPS grid width   (C.CHIRPS_W)

    Returns
    -------
    tp_mm : (target_h, target_w)  ERA5 TP in mm/month, clipped >= 0
    """
    tp_idx  = X_single.shape[-1] - 1           # last channel is always TP
    tp_norm = X_single[:, :, tp_idx]           # (H_lr, W_lr)

    # Reverse z-score:  x_real = x_norm * std + mean
    mu      = norm_stats["era5_mean"][tp_idx]
    sigma   = norm_stats["era5_std"][tp_idx]
    tp_real = tp_norm * (sigma + 1e-8) + mu    # mm/month at LR resolution

    # Upsample to CHIRPS grid for display
    tp_tensor = tf.image.resize(
        tp_real[:, :, np.newaxis],              # (H_lr, W_lr, 1)
        [target_h, target_w],
        method=tf.image.ResizeMethod.BILINEAR
    ).numpy()[:, :, 0]                          # (H_hr, W_hr)

    return np.maximum(tp_tensor, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Load a trained model from checkpoint
# ─────────────────────────────────────────────────────────────────────────────
def load_model(exp_name: str) -> tf.keras.Model:
    builders = {
        "unet_mse"      : build_unet_mse,
        "unet_bg"       : build_unet_bg,
        "unet_compound" : build_unet_compound,
        "wgan_compound" : build_wgan_generator,
    }
    ckpt_names = {
        "unet_mse"      : "best_model.keras",
        "unet_bg"       : "best_model.keras",
        "unet_compound" : "best_model.keras",
        "wgan_compound" : "best_generator.keras",
    }
    model     = builders[exp_name]()
    ckpt_path = os.path.join(C.OUTPUT_DIR, exp_name, ckpt_names[exp_name])
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. Run train.py first."
        )
    model.load_weights(ckpt_path)
    print(f"  Loaded {exp_name} from {ckpt_path}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Metric computation  (all values in mm/month, land pixels only)
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(y_true_mm: np.ndarray,
                    y_pred_mm: np.ndarray,
                    land_mask: np.ndarray) -> dict:
    lm = land_mask.astype(bool)
    N  = y_true_mm.shape[0]

    mae_list, mse_list, bias_list, corr_list = [], [], [], []
    fss80_list, fss95_list, fss99_list       = [], [], []
    r95p_bias_list, peak_err_list            = [], []

    for i in range(N):
        yt = y_true_mm[i, :, :, 0]
        yp = y_pred_mm[i, :, :, 0]

        valid = lm & np.isfinite(yt) & (yt >= 0)
        if valid.sum() < 50:
            continue

        yt_l = yt[valid]
        yp_l = yp[valid]

        mae_list.append(np.mean(np.abs(yp_l - yt_l)))
        mse_list.append(np.mean((yp_l - yt_l) ** 2))
        bias_list.append(np.mean(yp_l - yt_l))

        if np.std(yt_l) > 0 and np.std(yp_l) > 0:
            corr_list.append(np.corrcoef(yt_l, yp_l)[0, 1])

        r95p_bias_list.append(np.percentile(yp_l, 95) - np.percentile(yt_l, 95))
        peak_err_list.append(yp_l.max() - yt_l.max())

        lm_f   = lm.astype(np.float32)
        yt_fss = tf.constant((yt * lm_f)[np.newaxis, :, :, np.newaxis], tf.float32)
        yp_fss = tf.constant((yp * lm_f)[np.newaxis, :, :, np.newaxis], tf.float32)
        fss80_list.append(float(_fss_single_3d(yt_fss[0], yp_fss[0], q=80, n=C.FSS_N)))
        fss95_list.append(float(_fss_single_3d(yt_fss[0], yp_fss[0], q=95, n=C.FSS_N)))
        fss99_list.append(float(_fss_single_3d(yt_fss[0], yp_fss[0], q=99, n=C.FSS_N)))

    return {
        "MAE"      : np.mean(mae_list),
        "RMSE"     : np.sqrt(np.mean(mse_list)),
        "Bias"     : np.mean(bias_list),
        "Corr"     : np.mean(corr_list),
        "FSS_80"   : np.mean(fss80_list),
        "FSS_95"   : np.mean(fss95_list),
        "FSS_99"   : np.mean(fss99_list),
        "R95p_bias": np.mean(r95p_bias_list),
        "Peak_err" : np.mean(peak_err_list),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Spatial maps
# ─────────────────────────────────────────────────────────────────────────────
def _add_cbar(fig, ax, im, label, extend="max"):
    divider = make_axes_locatable(ax)
    cax     = divider.append_axes("right", size="4%", pad=0.06)
    cb      = fig.colorbar(im, cax=cax, extend=extend)
    cb.set_label(label, fontsize=7)
    cb.ax.tick_params(labelsize=6)


def plot_maps(y_true_mm, X_samples, pred_dict_mm,
              land_mask, norm_stats, meta, out_dir, sample_idx=0):
    """
    4-panel map: CHIRPS truth | ERA5 TP (from X, denormed) | model predictions.

    ERA5 TP uses the low_vals colormap (0–30 mm range) — consistent with
    the coarse 0.5° resolution which smooths out rainfall peaks.
    CHIRPS and predictions use the high_vals colormap (0–500 mm range).
    """
    lats = meta["chirps_lats"]
    lons = meta["chirps_lons"]
    lm   = land_mask.astype(bool)
    ext  = [lons.min(), lons.max(), lats.min(), lats.max()]

    # ── ERA5 TP: denorm from X, upsample, no land mask (covers full domain)
    era5_tp_mm = extract_era5_tp_mm(
        X_samples[sample_idx], norm_stats, C.CHIRPS_H, C.CHIRPS_W
    )

    # ── Colormaps
    cmap_hi,  norm_hi  = make_cmap(high_vals=True)   # CHIRPS + predictions
    cmap_low, norm_low = make_cmap(low_vals=True)     # ERA5 LR TP

    n_models  = len(pred_dict_mm)
    n_panels  = 2 + n_models      # truth + era5 + one per model
    fig, axes = plt.subplots(1, n_panels,
                              figsize=(4 * n_panels, 6),
                              constrained_layout=True)

    # Panel 0 — CHIRPS truth
    yt = np.where(lm, y_true_mm[sample_idx, :, :, 0], np.nan)
    im_hi = axes[0].imshow(yt, cmap=cmap_hi, norm=norm_hi,
                            extent=ext, origin="upper", aspect="auto")
    axes[0].set_title("CHIRPS (truth)\nhigh-res 0.05°", fontsize=9)
    _add_cbar(fig, axes[0], im_hi, "mm/month")

    # Panel 1 — ERA5 TP (low-res, from X_test channel 6, denormed)
    im_low = axes[1].imshow(era5_tp_mm, cmap=cmap_low, norm=norm_low,
                             extent=ext, origin="upper", aspect="auto")
    axes[1].set_title("ERA5 TP (input channel)\nlow-res 0.50°", fontsize=9)
    _add_cbar(fig, axes[1], im_low, "mm/month")

    # Panels 2+ — model predictions
    for ax, (name, preds_mm) in zip(axes[2:], pred_dict_mm.items()):
        yp = np.where(lm, preds_mm[sample_idx, :, :, 0], np.nan)
        im = ax.imshow(yp, cmap=cmap_hi, norm=norm_hi,
                        extent=ext, origin="upper", aspect="auto")
        ax.set_title(f"{C.EXPERIMENTS[name]}\nhigh-res 0.05°", fontsize=9)
        _add_cbar(fig, ax, im, "mm/month")

    for ax in axes:
        ax.set_xlabel("Lon", fontsize=8)
        ax.set_ylabel("Lat", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, linewidth=0.3, alpha=0.4, color="grey")

    plt.savefig(os.path.join(out_dir, f"map_sample_{sample_idx}.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved map → {out_dir}/map_sample_{sample_idx}.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="all",
                        choices=list(C.EXPERIMENTS.keys()) + ["all"])
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    print("Loading test data ...")
    X_test, _, times_test = load_split(C.TEST_FILE)
    y_test_raw            = load_split_raw(C.TEST_FILE)
    land_mask             = load_land_mask(C.META_FILE)
    norm_stats            = load_norm_stats(C.META_FILE)
    meta                  = dict(np.load(C.META_FILE, allow_pickle=True))

    print(f"  Test months  : {X_test.shape[0]}")
    print(f"  X shape (LR) : {X_test.shape[1:]}  "
          f"channels: {list(meta['channel_names'])}")
    print(f"  Upsampled to : {C.UNET_INPUT_SHAPE}")

    exps_to_run  = list(C.EXPERIMENTS.keys()) if args.exp == "all" else [args.exp]
    all_results  = {}
    pred_dict_mm = {}

    for exp_name in exps_to_run:
        print(f"\nEvaluating: {C.EXPERIMENTS[exp_name]}")
        try:
            model      = load_model(exp_name)
            preds_norm = predict_all(model, X_test, exp_name, norm_stats)
            preds_mm   = invert_chirps_norm(preds_norm, norm_stats)
            pred_dict_mm[exp_name] = preds_mm

            metrics = compute_metrics(y_test_raw, preds_mm, land_mask)
            all_results[exp_name] = metrics

            print("  Metrics (mm/month):")
            for k, v in metrics.items():
                print(f"    {k:12s}: {v:.4f}")
        except FileNotFoundError as e:
            print(f"  SKIPPED — {e}")

    if not all_results:
        print("\nNo trained models found. Run train.py first.")
        return

    df       = pd.DataFrame(all_results).T
    df.index = [C.EXPERIMENTS[e] for e in df.index]
    out_csv  = os.path.join(C.OUTPUT_DIR, "evaluation_results.csv")
    df.to_csv(out_csv, float_format="%.4f")
    print(f"\nResults saved to {out_csv}")
    print("\n" + df.to_string())

    if args.plot and pred_dict_mm:
        maps_dir = os.path.join(C.OUTPUT_DIR, "maps")
        os.makedirs(maps_dir, exist_ok=True)
        for idx in [0, 6, 12]:
            if idx < X_test.shape[0]:
                plot_maps(y_test_raw, X_test, pred_dict_mm,
                          land_mask, norm_stats, meta,
                          maps_dir, sample_idx=idx)


if __name__ == "__main__":
    main()