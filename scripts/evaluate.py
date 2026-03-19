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

from downscaling import config as C
from downscaling.data_loader import( load_split, load_split_raw, load_land_mask,
    load_norm_stats, invert_chirps_norm, predict_all)
from downscaling.models import (
    build_unet_mse, build_unet_bg, build_unet_compound,
    build_wgan_generator
)
from downscaling.losses import masked_mse, compound_loss, _fss_single_3d

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
# Metric computation (all in mm/month, land pixels only)
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(y_true_mm: np.ndarray,
                    y_pred_mm: np.ndarray,
                    land_mask: np.ndarray) -> dict:
    """
    Parameters
    ----------
    y_true_mm : (N, H, W, 1)  CHIRPS in mm/month, ocean = NaN
    y_pred_mm : (N, H, W, 1)  predicted rainfall in mm/month
    land_mask : (H, W)        boolean land mask
    """
    lm = land_mask.astype(bool)
    N  = y_true_mm.shape[0]

    mae_list, mse_list, bias_list, corr_list     = [], [], [], []
    fss80_list, fss95_list, fss99_list           = [], [], []
    r95p_bias_list, peak_err_list                = [], []

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

        q95_t = np.percentile(yt_l, 95)
        q95_p = np.percentile(yp_l, 95)
        r95p_bias_list.append(q95_p - q95_t)
        peak_err_list.append(yp_l.max() - yt_l.max())

        # FSS — zero out ocean, operate on 2-D fields
        lm_f   = lm.astype(np.float32)
        yt_fss = tf.constant((yt * lm_f)[np.newaxis, :, :, np.newaxis], tf.float32)
        yp_fss = tf.constant((yp * lm_f)[np.newaxis, :, :, np.newaxis], tf.float32)
        fss80_list.append(float(_fss_single_3d(yt_fss[0], yp_fss[0], q=80,  n=C.FSS_N)))
        fss95_list.append(float(_fss_single_3d(yt_fss[0], yp_fss[0], q=95,  n=C.FSS_N)))
        fss99_list.append(float(_fss_single_3d(yt_fss[0], yp_fss[0], q=99,  n=C.FSS_N)))

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
# Quick spatial maps for a chosen month
# ─────────────────────────────────────────────────────────────────────────────
def plot_maps(y_true_mm, pred_dict_mm, land_mask, meta, out_dir, sample_idx=0):
    lats = meta["chirps_lats"]
    lons = meta["chirps_lons"]
    lm   = land_mask.astype(bool)

    n_models = len(pred_dict_mm)
    fig, axes = plt.subplots(1, n_models + 1,
                              figsize=(4 * (n_models + 1), 5),
                              constrained_layout=True)

    yt      = np.where(lm, y_true_mm[sample_idx, :, :, 0], np.nan)
    vmax    = np.nanpercentile(yt, 99)
    norm    = mcolors.Normalize(vmin=0, vmax=vmax)
    cmap    = "YlGnBu"

    im = axes[0].imshow(yt, origin="upper", cmap=cmap, norm=norm,
                         extent=[lons.min(), lons.max(), lats.min(), lats.max()],
                         aspect="auto")
    axes[0].set_title("CHIRPS (truth)", fontsize=10)

    for ax, (name, preds_mm) in zip(axes[1:], pred_dict_mm.items()):
        yp = np.where(lm, preds_mm[sample_idx, :, :, 0], np.nan)
        ax.imshow(yp, origin="upper", cmap=cmap, norm=norm,
                   extent=[lons.min(), lons.max(), lats.min(), lats.max()],
                   aspect="auto")
        ax.set_title(C.EXPERIMENTS[name], fontsize=10)

    for ax in axes:
        ax.set_xlabel("Lon", fontsize=8)
        ax.set_ylabel("Lat", fontsize=8)
        ax.tick_params(labelsize=7)

    fig.colorbar(im, ax=axes, label="mm/month", shrink=0.6)
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
    y_test_raw            = load_split_raw(C.TEST_FILE)   # mm/month, ocean=NaN
    land_mask             = load_land_mask(C.META_FILE)
    norm_stats            = load_norm_stats(C.META_FILE)
    meta                  = dict(np.load(C.META_FILE, allow_pickle=True))

    print(f"  Test months : {X_test.shape[0]}")
    print(f"  LR shape    : {X_test.shape[1:]}  → upsampled to {C.UNET_INPUT_SHAPE}")

    exps_to_run = (list(C.EXPERIMENTS.keys())
                   if args.exp == "all" else [args.exp])

    all_results   = {}
    pred_dict_mm  = {}

    for exp_name in exps_to_run:
        print(f"\nEvaluating: {C.EXPERIMENTS[exp_name]}")
        try:
            model      = load_model(exp_name)
            # Predict in normalised space, then invert to mm
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

    # Summary table
    df      = pd.DataFrame(all_results).T
    df.index = [C.EXPERIMENTS[e] for e in df.index]
    out_csv  = os.path.join(C.OUTPUT_DIR, "evaluation_results.csv")
    df.to_csv(out_csv, float_format="%.4f")
    print(f"\nResults saved to {out_csv}")
    print("\n" + df.to_string())

    # Spatial maps
    if args.plot and pred_dict_mm:
        maps_dir = os.path.join(C.OUTPUT_DIR, "maps")
        os.makedirs(maps_dir, exist_ok=True)
        for idx in [0, 6, 12]:
            if idx < X_test.shape[0]:
                plot_maps(y_test_raw, pred_dict_mm, land_mask, meta,
                          maps_dir, sample_idx=idx)


if __name__ == "__main__":
    main()
