"""
evaluate.py — Post-training evaluation on the held-out test set.

Metrics computed (per month, then averaged)
───────────────────────────────────────────
Pixel-wise:
  MAE        mean absolute error              (mm/month)
  MSE        mean squared error               (mm²/month²)
  RMSE       root mean squared error          (mm/month)
  Bias       mean bias (pred − obs)           (mm/month)
  Corr       Pearson spatial correlation

Spatial:
  FSS_80/95/99  Fractions Skill Score at 3 percentiles (lower = worse)

Extreme:
  R95p_bias  bias in the 95th percentile of rainfall
  Peak_err   error in peak (max) pixel value

Usage
-----
  python evaluate.py                     # evaluates all trained models
  python evaluate.py --exp unet_compound # single experiment
  python evaluate.py --plot              # also save spatial maps
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

import config as C
from data_loader import load_split, load_land_mask, reconstruct_full
from models import (
    build_unet_mse, build_unet_bg, build_unet_compound,
    build_wgan_generator, bg_expected_rainfall
)
from losses import _fss_single_3d


# ─────────────────────────────────────────────────────────────────────────────
# Load a trained model from its checkpoint
# ─────────────────────────────────────────────────────────────────────────────
def load_model(exp_name: str) -> tf.keras.Model:
    if exp_name == "unet_mse":
        model = build_unet_mse()
    elif exp_name == "unet_bg":
        model = build_unet_bg()
    elif exp_name in ("unet_compound", "wgan_compound"):
        model = build_unet_compound() if exp_name == "unet_compound" \
                else build_wgan_generator()
    else:
        raise ValueError(f"Unknown experiment: {exp_name}")

    ckpt_name = ("best_generator.keras"
                 if exp_name == "wgan_compound" else "best_model.keras")
    ckpt_path = os.path.join(C.OUTPUT_DIR, exp_name, ckpt_name)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {ckpt_path}. "
            "Did you run train.py first?"
        )
    model.load_weights(ckpt_path)
    print(f"  Loaded {exp_name} from {ckpt_path}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Inference: predict all test months
# ─────────────────────────────────────────────────────────────────────────────
def predict_all(model, X_test, exp_name: str) -> np.ndarray:
    """
    Run sliding-window inference for every month in the test set.
    Returns predictions shape (N, H_hr, W_hr, 1).
    """
    N = X_test.shape[0]
    preds = []
    for i in range(N):
        if (i + 1) % 12 == 0:
            print(f"    Predicting month {i+1}/{N} ...")
        pred = reconstruct_full(model, X_test[i])

        # For Bernoulli-Gamma, convert to expected rainfall
        if exp_name == "unet_bg":
            pred = bg_expected_rainfall(pred).numpy()

        preds.append(pred)
    return np.stack(preds, axis=0)   # (N, H_hr, W_hr, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Metric computation
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(y_true_all, y_pred_all, land_mask):
    """
    Compute all metrics over the test set.

    y_true_all : (N, H_hr, W_hr, 1)  CHIRPS, ocean = -1
    y_pred_all : (N, H_hr, W_hr, 1)  model predictions
    land_mask  : (H_hr, W_hr)        boolean land mask
    """
    results = {}
    N = y_true_all.shape[0]

    mae_list, mse_list, bias_list, corr_list = [], [], [], []
    fss80_list, fss95_list, fss99_list       = [], [], []
    r95p_bias_list, peak_err_list            = [], []

    lm = land_mask.astype(bool)

    for i in range(N):
        yt = y_true_all[i, :, :, 0]   # (H, W)
        yp = y_pred_all[i, :, :, 0]

        # Apply land mask and valid-data mask
        valid = lm & (yt >= 0)
        if valid.sum() < 100:
            continue

        yt_l = yt[valid]
        yp_l = yp[valid]

        mae_list.append(np.mean(np.abs(yp_l - yt_l)))
        mse_list.append(np.mean((yp_l - yt_l) ** 2))
        bias_list.append(np.mean(yp_l - yt_l))

        if np.std(yt_l) > 0 and np.std(yp_l) > 0:
            corr_list.append(np.corrcoef(yt_l, yp_l)[0, 1])

        # Extreme: 95th percentile bias
        q95_true = np.percentile(yt_l, 95)
        q95_pred = np.percentile(yp_l, 95)
        r95p_bias_list.append(q95_pred - q95_true)

        # Peak error
        peak_err_list.append(yp_l.max() - yt_l.max())

        # FSS (use masked 2D fields, shape HxW; tensor ops)
        yt_t = tf.constant(yt[np.newaxis, :, :, np.newaxis], dtype=tf.float32)
        yp_t = tf.constant(yp[np.newaxis, :, :, np.newaxis], dtype=tf.float32)
        # Zero out ocean for FSS
        lm_t = tf.constant(lm[np.newaxis, :, :, np.newaxis], dtype=tf.float32)
        yt_t = yt_t * lm_t
        yp_t = yp_t * lm_t

        fss80_list.append(float(_fss_single(yt_t, yp_t, q=80,  n=C.FSS_N)))
        fss95_list.append(float(_fss_single(yt_t, yp_t, q=95,  n=C.FSS_N)))
        fss99_list.append(float(_fss_single(yt_t, yp_t, q=99,  n=C.FSS_N)))

    results["MAE"]       = np.mean(mae_list)
    results["RMSE"]      = np.sqrt(np.mean(mse_list))
    results["Bias"]      = np.mean(bias_list)
    results["Corr"]      = np.mean(corr_list)
    results["FSS_80"]    = np.mean(fss80_list)   # lower = worse in FSS'
    results["FSS_95"]    = np.mean(fss95_list)
    results["FSS_99"]    = np.mean(fss99_list)
    results["R95p_bias"] = np.mean(r95p_bias_list)
    results["Peak_err"]  = np.mean(peak_err_list)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation: spatial maps for a chosen month
# ─────────────────────────────────────────────────────────────────────────────
def plot_maps(y_true_all, pred_dict, land_mask, meta, out_dir,
              sample_idx=0):
    """
    Plot side-by-side maps: CHIRPS truth + prediction from each model.
    """
    lats = meta["chirps_lats"]
    lons = meta["chirps_lons"]
    lm   = land_mask.astype(float)

    n_models = len(pred_dict)
    fig, axes = plt.subplots(1, n_models + 1,
                              figsize=(4 * (n_models + 1), 4),
                              constrained_layout=True)

    vmax = np.nanpercentile(
        np.where(land_mask, y_true_all[sample_idx, :, :, 0], np.nan), 99
    )
    norm = mcolors.Normalize(vmin=0, vmax=vmax)
    cmap = "YlGnBu"

    # Truth
    yt = np.where(land_mask, y_true_all[sample_idx, :, :, 0], np.nan)
    im = axes[0].imshow(yt, origin="upper", cmap=cmap, norm=norm)
    axes[0].set_title("CHIRPS (truth)")
    axes[0].axis("off")

    # Predictions
    for ax, (name, preds) in zip(axes[1:], pred_dict.items()):
        yp = np.where(land_mask, preds[sample_idx, :, :, 0], np.nan)
        ax.imshow(yp, origin="upper", cmap=cmap, norm=norm)
        ax.set_title(C.EXPERIMENTS[name])
        ax.axis("off")

    fig.colorbar(im, ax=axes, label="Rainfall (mm/month)", shrink=0.6)
    fig.suptitle(f"Test sample {sample_idx}", fontsize=12)
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
    parser.add_argument("--plot", action="store_true",
                        help="Save spatial map figures")
    args = parser.parse_args()

    print("Loading test data ...")
    X_test, y_test, times_test = load_split(C.TEST_FILE)
    land_mask = load_land_mask(C.META_FILE)
    meta      = dict(np.load(C.META_FILE, allow_pickle=True))

    exps_to_run = (list(C.EXPERIMENTS.keys())
                   if args.exp == "all" else [args.exp])

    all_results = {}
    pred_dict   = {}

    for exp_name in exps_to_run:
        print(f"\nEvaluating: {C.EXPERIMENTS[exp_name]}")
        try:
            model  = load_model(exp_name)
            preds  = predict_all(model, X_test, exp_name)
            pred_dict[exp_name] = preds
            metrics = compute_metrics(y_test, preds, land_mask)
            all_results[exp_name] = metrics
            print("  Metrics:")
            for k, v in metrics.items():
                print(f"    {k:12s}: {v:.4f}")
        except FileNotFoundError as e:
            print(f"  SKIPPED — {e}")

    if not all_results:
        print("\nNo trained models found. Run train.py first.")
        return

    # ── Summary table ──────────────────────────────────────────────────────
    df = pd.DataFrame(all_results).T
    df.index = [C.EXPERIMENTS[e] for e in df.index]
    out_csv = os.path.join(C.OUTPUT_DIR, "evaluation_results.csv")
    df.to_csv(out_csv, float_format="%.4f")
    print(f"\nSummary table saved to {out_csv}")
    print("\n" + df.to_string())

    # ── Spatial maps ───────────────────────────────────────────────────────
    if args.plot and pred_dict:
        maps_dir = os.path.join(C.OUTPUT_DIR, "maps")
        os.makedirs(maps_dir, exist_ok=True)
        for idx in [0, 6, 12]:   # Jan, Jul, Jan of following year
            if idx < len(y_test):
                plot_maps(y_test, pred_dict, land_mask, meta,
                          maps_dir, sample_idx=idx)


if __name__ == "__main__":
    main()
