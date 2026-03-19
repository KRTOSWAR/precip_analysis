"""
train.py — Training pipeline for all four downscaling experiments.
Full-image mode: each mini-batch contains BATCH_SIZE full monthly fields.

Usage
-----
  python train.py                      # trains all 4 experiments
  python train.py --exp unet_mse       # one experiment only
  python train.py --exp wgan_compound  # WGAN only

Output per experiment (RESULTS/<exp_name>/)
-------------------------------------------
  best_model.keras     best checkpoint by val loss
  history.csv          per-epoch train / val loss
  training_curve.png   loss curves
"""

import os
import argparse
import csv
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau
)
import matplotlib.pyplot as plt

from downscaling import config as C
from downscaling.data_loader import load_split, load_land_mask, build_dataset
from downscaling.models import(
    build_unet_mse, build_unet_bg, build_unet_compound,
    build_wgan_generator, build_wgan_discriminator
)
from downscaling.losses import (
    masked_mse, bernoulli_gamma_nll, compound_loss,
    wasserstein_discriminator_loss, gradient_penalty,
    wgan_generator_loss_total
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _plot_history(history_dict, out_dir, title):
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, values in history_dict.items():
        ax.plot(values, label=label, linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_curve.png"), dpi=120)
    plt.close()
    print(f"  Saved curve → {out_dir}/training_curve.png")


def _save_history_csv(history_dict, out_dir):
    path = os.path.join(out_dir, "history.csv")
    keys = list(history_dict.keys())
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch"] + keys)
        for i in range(len(history_dict[keys[0]])):
            writer.writerow([i + 1] + [history_dict[k][i] for k in keys])
    print(f"  Saved history → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Standard Keras training  (unet_mse, unet_bg, unet_compound)
# ─────────────────────────────────────────────────────────────────────────────
def train_keras_model(exp_name: str,
                      X_train, y_train,
                      X_val,   y_val):
    print(f"\n{'='*60}")
    print(f"  Experiment : {C.EXPERIMENTS[exp_name]}")
    print(f"  Input shape: {C.UNET_INPUT_SHAPE}  (after ERA5 upsample)")
    print(f"  Batch size : {C.BATCH_SIZE} months")
    print(f"{'='*60}")

    out_dir = os.path.join(C.OUTPUT_DIR, exp_name)
    os.makedirs(out_dir, exist_ok=True)

    # ── Model + loss ────────────────────────────────────────────────────
    if exp_name == "unet_mse":
        model   = build_unet_mse()
        loss_fn = masked_mse
    elif exp_name == "unet_bg":
        model   = build_unet_bg()
        loss_fn = bernoulli_gamma_nll
    elif exp_name == "unet_compound":
        model   = build_unet_compound()
        loss_fn = compound_loss
    else:
        raise ValueError(f"Unknown experiment: {exp_name}")

    model.summary(line_length=80)
    model.compile(
        optimizer=Adam(learning_rate=C.LR_INIT),
        loss=loss_fn,
    )

    # ── Datasets ─────────────────────────────────────────────────────────
    # Each epoch sees every training month exactly once (shuffled).
    train_ds = build_dataset(X_train, y_train,
                              batch_size=C.BATCH_SIZE,
                              shuffle=True, seed=42)
    val_ds   = build_dataset(X_val, y_val,
                              batch_size=C.BATCH_SIZE,
                              shuffle=False)

    # ── Callbacks ─────────────────────────────────────────────────────────
    ckpt_path = os.path.join(out_dir, "best_model.keras")
    callbacks = [
        ModelCheckpoint(ckpt_path, monitor="val_loss",
                         save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss",
                       patience=C.EARLY_STOP_PATIENCE,
                       restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                           patience=5, min_lr=1e-6, verbose=1),
        CSVLogger(os.path.join(out_dir, "history.csv"), append=False),
    ]

    # ── Train ─────────────────────────────────────────────────────────────
    t0   = time.time()
    hist = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=C.EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )
    print(f"\n  Training finished in {(time.time()-t0)/60:.1f} min")

    _plot_history(
        {"train": hist.history["loss"],
         "val":   hist.history["val_loss"]},
        out_dir,
        title=C.EXPERIMENTS[exp_name]
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Custom WGAN-GP training loop
# ─────────────────────────────────────────────────────────────────────────────
def train_wgan(X_train, y_train, X_val, y_val):
    exp_name = "wgan_compound"
    print(f"\n{'='*60}")
    print(f"  Experiment : {C.EXPERIMENTS[exp_name]}")
    print(f"  n_critic   : {C.WGAN_N_CRITIC}")
    print(f"{'='*60}")

    out_dir = os.path.join(C.OUTPUT_DIR, exp_name)
    os.makedirs(out_dir, exist_ok=True)

    generator     = build_wgan_generator()
    discriminator = build_wgan_discriminator()
    generator.summary(line_length=80)
    discriminator.summary(line_length=80)

    g_opt = Adam(C.WGAN_GEN_LR, beta_1=0.0, beta_2=0.9)
    d_opt = Adam(C.WGAN_DIS_LR, beta_1=0.0, beta_2=0.9)

    train_ds = build_dataset(X_train, y_train,
                              batch_size=C.BATCH_SIZE,
                              shuffle=True, seed=42)
    val_ds   = build_dataset(X_val, y_val,
                              batch_size=C.BATCH_SIZE,
                              shuffle=False)

    # ── tf.function step functions ────────────────────────────────────────
    @tf.function
    def train_discriminator_step(x_batch, y_batch):
        fake = generator(x_batch, training=False)
        with tf.GradientTape() as tape:
            real_logits = discriminator([x_batch, y_batch], training=True)
            fake_logits = discriminator([x_batch, fake],    training=True)
            d_loss      = wasserstein_discriminator_loss(real_logits, fake_logits)
            # Gradient penalty
            B      = tf.shape(y_batch)[0]
            alpha  = tf.random.uniform([B, 1, 1, 1], 0.0, 1.0)
            interp = y_batch + alpha * (fake - y_batch)
            with tf.GradientTape() as gp_tape:
                gp_tape.watch(interp)
                interp_logit = discriminator([x_batch, interp], training=True)
            grads  = gp_tape.gradient(interp_logit, interp)
            norm   = tf.sqrt(tf.reduce_sum(grads**2, axis=[1,2,3]) + 1e-12)
            gp     = C.WGAN_GP_LAMBDA * tf.reduce_mean((norm - 1.0) ** 2)
            d_total = d_loss + gp
        grads_d = tape.gradient(d_total, discriminator.trainable_variables)
        d_opt.apply_gradients(zip(grads_d, discriminator.trainable_variables))
        return d_total

    @tf.function
    def train_generator_step(x_batch, y_batch):
        with tf.GradientTape() as tape:
            fake        = generator(x_batch, training=True)
            fake_logits = discriminator([x_batch, fake], training=False)
            g_loss      = wgan_generator_loss_total(fake_logits, y_batch, fake)
        grads_g = tape.gradient(g_loss, generator.trainable_variables)
        g_opt.apply_gradients(zip(grads_g, generator.trainable_variables))
        return g_loss

    @tf.function
    def val_step(x_batch, y_batch):
        fake = generator(x_batch, training=False)
        return masked_mse(y_batch, fake)

    # ── Training loop ─────────────────────────────────────────────────────
    history    = {"train_d_loss": [], "train_g_loss": [], "val_loss": []}
    best_val   = np.inf
    no_improve = 0
    ckpt_path  = os.path.join(out_dir, "best_generator.keras")

    t0 = time.time()
    for epoch in range(1, C.EPOCHS + 1):
        d_losses, g_losses = [], []
        step = 0

        for x_batch, y_batch in train_ds:
            # n_critic discriminator steps, then 1 generator step
            d_loss = train_discriminator_step(x_batch, y_batch)
            d_losses.append(float(d_loss))
            step += 1

            if step % C.WGAN_N_CRITIC == 0:
                g_loss = train_generator_step(x_batch, y_batch)
                g_losses.append(float(g_loss))

        val_loss = float(np.mean([
            float(val_step(xb, yb)) for xb, yb in val_ds
        ]))

        d_mean = np.mean(d_losses) if d_losses else 0.0
        g_mean = np.mean(g_losses) if g_losses else 0.0

        history["train_d_loss"].append(d_mean)
        history["train_g_loss"].append(g_mean)
        history["val_loss"].append(val_loss)

        print(f"  Epoch {epoch:3d}/{C.EPOCHS}  "
              f"D={d_mean:.4f}  G={g_mean:.4f}  val_MSE={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            generator.save(ckpt_path)
            no_improve = 0
            print(f"    ✓ Best val loss: {best_val:.4f}  checkpoint saved")
        else:
            no_improve += 1
            if no_improve >= C.EARLY_STOP_PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    print(f"\n  WGAN training finished in {(time.time()-t0)/60:.1f} min")

    _save_history_csv(history, out_dir)
    _plot_history(
        {"train_G": history["train_g_loss"],
         "val_MSE": history["val_loss"]},
        out_dir,
        title="WGAN + Compound (Generator loss + Val MSE)"
    )

    generator.load_weights(ckpt_path)
    return generator


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="all",
                        choices=list(C.EXPERIMENTS.keys()) + ["all"])
    args = parser.parse_args()

    print("Loading preprocessed data ...")
    X_train, y_train, _ = load_split(C.TRAIN_FILE)
    X_val,   y_val,   _ = load_split(C.VAL_FILE)

    print(f"  Train: X={X_train.shape}  y={y_train.shape}  "
          f"(stored LR, upsampled to {C.UNET_INPUT_SHAPE} on-the-fly)")
    print(f"  Val  : X={X_val.shape}    y={y_val.shape}")

    run_all = args.exp == "all"

    if run_all or args.exp == "unet_mse":
        train_keras_model("unet_mse", X_train, y_train, X_val, y_val)

    if run_all or args.exp == "unet_bg":
        train_keras_model("unet_bg",  X_train, y_train, X_val, y_val)

    if run_all or args.exp == "unet_compound":
        train_keras_model("unet_compound", X_train, y_train, X_val, y_val)

    if run_all or args.exp == "wgan_compound":
        train_wgan(X_train, y_train, X_val, y_val)

    print("\nAll experiments complete. Results in:", C.OUTPUT_DIR)


if __name__ == "__main__":
    main()
