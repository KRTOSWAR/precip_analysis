"""
train.py — Training pipeline for all four PP downscaling experiments.

Usage
-----
  python train.py                     # trains all 4 experiments in sequence
  python train.py --exp unet_mse      # trains one experiment only
  python train.py --exp wgan_compound # trains WGAN only

Output (per experiment, saved to RESULTS/<exp_name>/)
------
  best_model.keras       best checkpoint (lowest val loss)
  history.csv            per-epoch train/val loss
  training_curve.png     loss curves plot

CPU note
--------
On CPU the WGAN will be very slow because of the n_critic inner loop.
Reduce WGAN_N_CRITIC in config.py to 2 if it is too slow,
or increase BATCH_SIZE to 8 on machines with ≥16 GB RAM.
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

import config as C
from data_loader import load_split, load_land_mask, build_dataset
from models import (
    build_unet_mse, build_unet_bg, build_unet_compound,
    build_wgan_generator, build_wgan_discriminator
)
from losses import (
    masked_mse, bernoulli_gamma_nll, compound_loss,
    wasserstein_discriminator_loss, gradient_penalty,
    wgan_generator_loss_total
)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: plotting
# ─────────────────────────────────────────────────────────────────────────────
def _plot_history(history_dict, out_dir, title):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history_dict["train_loss"], label="Train", linewidth=1.5)
    ax.plot(history_dict["val_loss"],   label="Val",   linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_curve.png"), dpi=120)
    plt.close()
    print(f"  Saved training curve → {out_dir}/training_curve.png")


def _save_history_csv(history_dict, out_dir):
    path = os.path.join(out_dir, "history.csv")
    keys = list(history_dict.keys())
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch"] + keys)
        n = len(history_dict[keys[0]])
        for i in range(n):
            writer.writerow([i + 1] + [history_dict[k][i] for k in keys])
    print(f"  Saved history → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Standard Keras training (unet_mse, unet_bg, unet_compound)
# ─────────────────────────────────────────────────────────────────────────────
def train_keras_model(exp_name: str, X_train, y_train, X_val, y_val,
                       land_mask_hr):
    print(f"\n{'='*60}")
    print(f"  Experiment: {C.EXPERIMENTS[exp_name]}")
    print(f"{'='*60}")

    out_dir = os.path.join(C.OUTPUT_DIR, exp_name)
    os.makedirs(out_dir, exist_ok=True)

    # ── Build model and set loss ──────────────────────────────────────────
    if exp_name == "unet_mse":
        model = build_unet_mse()
        loss_fn = masked_mse

    elif exp_name == "unet_bg":
        model = build_unet_bg()
        loss_fn = bernoulli_gamma_nll

    elif exp_name == "unet_compound":
        model = build_unet_compound()
        loss_fn = compound_loss

    else:
        raise ValueError(f"Unknown experiment name: {exp_name}")

    model.summary(line_length=80)

    model.compile(
        optimizer=Adam(learning_rate=C.LR_INIT),
        loss=loss_fn,
        metrics=["mae"]   # always track MAE as a secondary metric
    )

    # ── Build tf.data pipelines ──────────────────────────────────────────
    train_ds = build_dataset(
        X_train, y_train, land_mask_hr,
        n_patches=C.PATCHES_PER_EPOCH_TRAIN,
        batch_size=C.BATCH_SIZE,
        shuffle=True, seed=42
    )
    val_ds = build_dataset(
        X_val, y_val, land_mask_hr,
        n_patches=C.PATCHES_PER_EPOCH_VAL,
        batch_size=C.BATCH_SIZE,
        shuffle=False, seed=0
    )

    # ── Callbacks ────────────────────────────────────────────────────────
    ckpt_path = os.path.join(out_dir, "best_model.keras")
    callbacks = [
        ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True,
                         verbose=1),
        EarlyStopping(monitor="val_loss", patience=C.EARLY_STOP_PATIENCE,
                       restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5,
                           min_lr=1e-6, verbose=1),
        CSVLogger(os.path.join(out_dir, "history.csv"), append=False),
    ]

    # ── Train ─────────────────────────────────────────────────────────────
    t0 = time.time()
    hist = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=C.EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )
    elapsed = time.time() - t0
    print(f"\n  Training finished in {elapsed/60:.1f} min")

    # ── Save curve ────────────────────────────────────────────────────────
    _plot_history(
        {"train_loss": hist.history["loss"],
         "val_loss"  : hist.history["val_loss"]},
        out_dir,
        title=C.EXPERIMENTS[exp_name]
    )

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Custom WGAN-GP training loop
# ─────────────────────────────────────────────────────────────────────────────
def train_wgan(X_train, y_train, X_val, y_val, land_mask_hr):
    exp_name = "wgan_compound"
    print(f"\n{'='*60}")
    print(f"  Experiment: {C.EXPERIMENTS[exp_name]}")
    print(f"{'='*60}")

    out_dir = os.path.join(C.OUTPUT_DIR, exp_name)
    os.makedirs(out_dir, exist_ok=True)

    generator     = build_wgan_generator()
    discriminator = build_wgan_discriminator()
    generator.summary(line_length=80)
    discriminator.summary(line_length=80)

    g_optimizer = Adam(C.WGAN_GEN_LR, beta_1=0.0, beta_2=0.9)
    d_optimizer = Adam(C.WGAN_DIS_LR, beta_1=0.0, beta_2=0.9)

    # ── Build datasets ────────────────────────────────────────────────────
    # For WGAN we need more patches per step to feed n_critic inner loop
    n_total_train = C.PATCHES_PER_EPOCH_TRAIN * C.WGAN_N_CRITIC
    train_ds = build_dataset(
        X_train, y_train, land_mask_hr,
        n_patches=n_total_train,
        batch_size=C.BATCH_SIZE,
        shuffle=True, seed=42
    )
    val_ds = build_dataset(
        X_val, y_val, land_mask_hr,
        n_patches=C.PATCHES_PER_EPOCH_VAL,
        batch_size=C.BATCH_SIZE,
        shuffle=False, seed=0
    )

    # ── Step functions ────────────────────────────────────────────────────
    @tf.function
    def train_discriminator_step(x_batch, y_batch):
        """One discriminator update."""
        fake = generator(x_batch, training=False)
        with tf.GradientTape() as tape:
            real_logits = discriminator([x_batch, y_batch],   training=True)
            fake_logits = discriminator([x_batch, fake],       training=True)
            d_loss = wasserstein_discriminator_loss(real_logits, fake_logits)
            gp     = gradient_penalty(
                lambda inp: discriminator([x_batch, inp], training=True),
                y_batch, fake
            )
            d_loss_total = d_loss + gp
        grads = tape.gradient(d_loss_total, discriminator.trainable_variables)
        d_optimizer.apply_gradients(
            zip(grads, discriminator.trainable_variables)
        )
        return d_loss_total

    @tf.function
    def train_generator_step(x_batch, y_batch):
        """One generator update."""
        with tf.GradientTape() as tape:
            fake        = generator(x_batch, training=True)
            fake_logits = discriminator([x_batch, fake], training=False)
            g_loss      = wgan_generator_loss_total(fake_logits, y_batch, fake)
        grads = tape.gradient(g_loss, generator.trainable_variables)
        g_optimizer.apply_gradients(zip(grads, generator.trainable_variables))
        return g_loss

    @tf.function
    def val_step(x_batch, y_batch):
        fake   = generator(x_batch, training=False)
        return masked_mse(y_batch, fake)

    # ── gradient_penalty wrapper that accepts a lambda ────────────────────
    # (Redefine so @tf.function can trace it cleanly)
    @tf.function
    def _gp_step(x_batch, real_y, fake_y):
        with tf.GradientTape() as tape:
            alpha        = tf.random.uniform([tf.shape(real_y)[0], 1, 1, 1])
            interp       = real_y + alpha * (fake_y - real_y)
            tape.watch(interp)
            interp_logit = discriminator([x_batch, interp], training=True)
        grads  = tape.gradient(interp_logit, interp)
        norm   = tf.sqrt(tf.reduce_sum(grads**2, axis=[1,2,3]) + 1e-12)
        return C.WGAN_GP_LAMBDA * tf.reduce_mean((norm - 1.0)**2)

    @tf.function
    def train_discriminator_step_v2(x_batch, y_batch):
        fake = generator(x_batch, training=False)
        with tf.GradientTape() as tape:
            real_logits = discriminator([x_batch, y_batch],  training=True)
            fake_logits = discriminator([x_batch, fake],      training=True)
            d_loss      = (tf.reduce_mean(fake_logits)
                           - tf.reduce_mean(real_logits))
            gp          = _gp_step(x_batch, y_batch, fake)
            d_total     = d_loss + gp
        grads = tape.gradient(d_total, discriminator.trainable_variables)
        d_optimizer.apply_gradients(
            zip(grads, discriminator.trainable_variables)
        )
        return d_total

    # ── Training loop ─────────────────────────────────────────────────────
    history = {"train_d_loss": [], "train_g_loss": [], "val_loss": []}
    best_val = np.inf
    no_improve = 0
    ckpt_path  = os.path.join(out_dir, "best_generator.keras")

    t0 = time.time()
    for epoch in range(1, C.EPOCHS + 1):
        d_losses, g_losses = [], []
        batches = list(train_ds)

        # Split batches: first n_critic batches go to discriminator,
        # last batch goes to generator (repeats every n_critic+1 batches)
        step = 0
        while step < len(batches):
            # Discriminator inner loop
            for _ in range(C.WGAN_N_CRITIC):
                if step >= len(batches):
                    break
                xb, yb = batches[step]
                d_loss = train_discriminator_step_v2(xb, yb)
                d_losses.append(float(d_loss))
                step += 1

            # Generator step
            if step < len(batches):
                xb, yb = batches[step]
                g_loss = train_generator_step(xb, yb)
                g_losses.append(float(g_loss))
                step += 1

        # Validation
        val_losses = [float(val_step(xb, yb)) for xb, yb in val_ds]
        val_loss   = np.mean(val_losses)

        d_mean = np.mean(d_losses) if d_losses else 0.0
        g_mean = np.mean(g_losses) if g_losses else 0.0

        history["train_d_loss"].append(d_mean)
        history["train_g_loss"].append(g_mean)
        history["val_loss"].append(val_loss)

        print(f"  Epoch {epoch:3d}/{C.EPOCHS}  "
              f"D={d_mean:.4f}  G={g_mean:.4f}  val_MSE={val_loss:.4f}")

        # Save best
        if val_loss < best_val:
            best_val = val_loss
            generator.save(ckpt_path)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= C.EARLY_STOP_PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    elapsed = time.time() - t0
    print(f"\n  WGAN training finished in {elapsed/60:.1f} min")

    _save_history_csv(history, out_dir)
    _plot_history(
        {"train_g_loss": history["train_g_loss"],
         "val_loss":     history["val_loss"]},
        out_dir,
        title="WGAN + Compound (Generator loss + Val MSE)"
    )

    # Load best weights
    generator.load_weights(ckpt_path)
    return generator


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="all",
                        choices=list(C.EXPERIMENTS.keys()) + ["all"],
                        help="Which experiment to run (default: all)")
    args = parser.parse_args()

    # ── Load data ─────────────────────────────────────────────────────────
    print("Loading preprocessed data ...")
    X_train, y_train, _ = load_split(C.TRAIN_FILE)
    X_val,   y_val,   _ = load_split(C.VAL_FILE)
    land_mask_hr         = load_land_mask(C.META_FILE)

    print(f"  Train: X={X_train.shape}  y={y_train.shape}")
    print(f"  Val  : X={X_val.shape}    y={y_val.shape}")
    print(f"  Land mask: {land_mask_hr.shape}  "
          f"({100*land_mask_hr.mean():.1f}% land)")

    # ── Run experiments ───────────────────────────────────────────────────
    run_all = args.exp == "all"

    if run_all or args.exp == "unet_mse":
        train_keras_model("unet_mse", X_train, y_train, X_val, y_val,
                           land_mask_hr)

    if run_all or args.exp == "unet_bg":
        train_keras_model("unet_bg", X_train, y_train, X_val, y_val,
                           land_mask_hr)

    if run_all or args.exp == "unet_compound":
        train_keras_model("unet_compound", X_train, y_train, X_val, y_val,
                           land_mask_hr)

    if run_all or args.exp == "wgan_compound":
        train_wgan(X_train, y_train, X_val, y_val, land_mask_hr)

    print("\nAll experiments complete. Results saved to:", C.OUTPUT_DIR)


if __name__ == "__main__":
    main()
