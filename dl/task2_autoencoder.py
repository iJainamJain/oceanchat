"""
Task 2 - Profile autoencoder and anomaly detection.

Compress a whole temperature/salinity profile (100 depth levels x 2 variables = 200
numbers) into a small latent vector and reconstruct it. Two things come out of this:

  * a learned low-dimensional description of "what an Indian Ocean profile looks like",
    which can be compared against PCA - the linear equivalent;
  * an anomaly score. Profiles the autoencoder reconstructs badly are the unusual ones,
    which is how this is used operationally for quality control.

Protocol: grouped split by float. The reconstruction error is reported on profiles from
floats never seen in training.
"""

import numpy as np
from sklearn.decomposition import PCA
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Flatten, Reshape

from . import common, data

LATENT_DIM = 12
MAX_EPOCHS = 400


def build_autoencoder(n_levels, n_vars, latent_dim=LATENT_DIM):
    inp = Input(shape=(n_levels, n_vars))
    x = Flatten()(inp)
    x = Dense(256, activation="relu")(x)
    x = Dense(64, activation="relu")(x)
    latent = Dense(latent_dim, activation="linear", name="latent")(x)

    x = Dense(64, activation="relu")(latent)
    x = Dense(256, activation="relu")(x)
    x = Dense(n_levels * n_vars, activation="linear")(x)
    out = Reshape((n_levels, n_vars))(x)

    autoencoder = Model(inp, out, name="profile_autoencoder")
    encoder = Model(inp, latent, name="encoder")
    autoencoder.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return autoencoder, encoder


def run():
    common.set_seed()
    print("\n" + "=" * 68)
    print(f"TASK 2 - Profile autoencoder (latent dim {LATENT_DIM})")
    print("=" * 68)

    X, meta = data.build_profiles()
    groups = meta["platform"]
    print(f"{len(X)} profiles x {X.shape[1]} levels x {X.shape[2]} variables "
          f"({X.shape[1] * X.shape[2]} numbers per profile)")

    train, val, test = data.make_split(groups, meta["time"], seed=common.SEED)
    data.split_report("profiles", groups, train, val, test)

    # Standardise each variable using training statistics only.
    mu = X[train].mean(axis=(0, 1), keepdims=True)
    sd = X[train].std(axis=(0, 1), keepdims=True)
    Xs = (X - mu) / sd

    autoencoder, encoder = build_autoencoder(X.shape[1], X.shape[2])
    history = autoencoder.fit(
        Xs[train], Xs[train],
        validation_data=(Xs[val], Xs[val]),
        epochs=MAX_EPOCHS, batch_size=32, verbose=0,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=12, min_lr=1e-5),
        ],
    )

    recon = autoencoder.predict(Xs[test], verbose=0)
    # Report in physical units, per variable.
    recon_phys = recon * sd + mu
    true_phys = X[test]

    per_var = {}
    for i, name in enumerate(["TEMP", "PSAL"]):
        per_var[name] = common.regression_metrics(
            true_phys[:, :, i].ravel(), recon_phys[:, :, i].ravel()
        )

    # PCA with the same number of components is the linear yardstick.
    flat_tr = Xs[train].reshape(len(train), -1)
    flat_te = Xs[test].reshape(len(test), -1)
    pca = PCA(n_components=LATENT_DIM, random_state=common.SEED).fit(flat_tr)
    pca_recon = pca.inverse_transform(pca.transform(flat_te)).reshape(Xs[test].shape)
    pca_phys = pca_recon * sd + mu

    pca_var = {}
    for i, name in enumerate(["TEMP", "PSAL"]):
        pca_var[name] = common.regression_metrics(
            true_phys[:, :, i].ravel(), pca_phys[:, :, i].ravel()
        )

    print()
    common.print_table(
        {f"Autoencoder  {k}": v for k, v in per_var.items()}
        | {f"PCA ({LATENT_DIM} comp) {k}": v for k, v in pca_var.items()}
    )

    # Anomaly score: per-profile reconstruction error on held-out floats.
    err = np.mean((recon - Xs[test]) ** 2, axis=(1, 2))
    order = np.argsort(err)[::-1]
    print(f"\nreconstruction error  median {np.median(err):.4f}  "
          f"p95 {np.percentile(err, 95):.4f}  max {err.max():.4f}")
    print("most anomalous held-out profiles:")
    for rank, j in enumerate(order[:5], 1):
        print(f"  {rank}. float {groups[test][j]}  lat {meta['lat'][test][j]:7.2f}  "
              f"lon {meta['lon'][test][j]:7.2f}  error {err[j]:.4f}")

    autoencoder.save(common.out_path("profile_autoencoder.keras"))
    encoder.save(common.out_path("profile_encoder.keras"))
    np.savez(
        common.out_path("autoencoder_norm.npz"), mu=mu, sd=sd,
        test_error=err, test_platform=groups[test],
        test_lat=meta["lat"][test], test_lon=meta["lon"][test],
    )

    common.save_result("autoencoder", {
        "description": f"Compress a 100-level T/S profile to {LATENT_DIM} numbers and rebuild it",
        "latent_dim": LATENT_DIM,
        "input_shape": list(X.shape[1:]),
        "compression_ratio": round(X.shape[1] * X.shape[2] / LATENT_DIM, 1),
        "split": data.split_label(),
        "n_train": int(len(train)), "n_val": int(len(val)), "n_test": int(len(test)),
        # Reaching the cap is usually harmless here: by then ReduceLROnPlateau has hit
        # min_lr and val_loss is flat. The final learning rate in history shows which.
        "epochs": len(history.history["loss"]),
        "hit_epoch_cap": len(history.history["loss"]) == MAX_EPOCHS,
        "autoencoder": per_var,
        "pca_baseline": pca_var,
        "anomaly": {
            "median_error": float(np.median(err)),
            "p95_error": float(np.percentile(err, 95)),
            "max_error": float(err.max()),
        },
        "history": common.history_curves(history),
    })
    return per_var
