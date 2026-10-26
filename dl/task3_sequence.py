"""
Task 3 - Sequence model: reconstruct a full depth profile from surface data.

Given only what a ship or satellite can see near the surface (temperature and salinity
in the top ~50 db) plus position and day of year, predict the entire temperature
profile down to 2000 db - all 100 levels at once.

This is where architecture genuinely matters, which is the point of the exercise. Three
decoders are compared on identical inputs and splits:

  * Dense    - flat baseline, no notion that the output is ordered by depth
  * 1D-CNN   - upsampling convolutions, local smoothness along the depth axis
  * LSTM     - generates the profile as a sequence, one depth step at a time

Protocol: grouped split by float, plus a climatology baseline (predict the training-set
mean profile every time) so the reported R2 is measured against a real reference.
"""

import numpy as np
import pandas as pd
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (LSTM, Conv1D, Dense, Reshape, TimeDistributed,
                                     UpSampling1D)
from tensorflow.keras.optimizers import Adam

from . import common, data

SURFACE_LEVELS = 3  # top 3 grid levels ~ 10-50 db

# Applied to all three decoders so the comparison stays fair. Without them the LSTM on the
# 2-year temporal split peaked at epoch 4, its val loss then spiked 7x, and early stopping
# kept the undertrained epoch-4 weights.
CLIPNORM = 1.0
MIN_EPOCHS = 20  # early stopping cannot fire before this


def compile_decoder(model):
    model.compile(optimizer=Adam(clipnorm=CLIPNORM), loss="mse", metrics=["mae"])
    return model


def make_inputs(X, meta):
    """Surface T/S plus position and season -> the conditioning vector."""
    surface = X[:, :SURFACE_LEVELS, :].reshape(len(X), -1)
    doy = pd.to_datetime(meta["time"]).dayofyear.to_numpy()
    season = np.stack([np.sin(2 * np.pi * doy / 365.25),
                       np.cos(2 * np.pi * doy / 365.25)], axis=1)
    return np.concatenate(
        [surface, meta["lat"][:, None], meta["lon"][:, None], season], axis=1
    ).astype("float32")


def build_dense(n_in, n_levels):
    inp = Input(shape=(n_in,))
    x = Dense(256, activation="relu")(inp)
    x = Dense(256, activation="relu")(x)
    out = Dense(n_levels, activation="linear")(x)
    return compile_decoder(Model(inp, out, name="dense_decoder"))


def build_cnn(n_in, n_levels):
    """Project to a short sequence, then upsample with convolutions to full depth."""
    seed_len = n_levels // 4
    inp = Input(shape=(n_in,))
    x = Dense(seed_len * 32, activation="relu")(inp)
    x = Reshape((seed_len, 32))(x)
    x = Conv1D(64, 5, padding="same", activation="relu")(x)
    x = UpSampling1D(2)(x)
    x = Conv1D(32, 5, padding="same", activation="relu")(x)
    x = UpSampling1D(2)(x)
    x = Conv1D(16, 5, padding="same", activation="relu")(x)
    x = Conv1D(1, 5, padding="same", activation="linear")(x)
    out = Reshape((n_levels,))(x)
    return compile_decoder(Model(inp, out, name="cnn_decoder"))


def build_lstm(n_in, n_levels):
    """Repeat the conditioning vector at every depth step and unroll an LSTM."""
    inp = Input(shape=(n_in,))
    x = Dense(64, activation="relu")(inp)
    x = Reshape((1, 64))(x)
    x = tf_repeat(x, n_levels)
    x = LSTM(96, return_sequences=True)(x)
    x = LSTM(64, return_sequences=True)(x)
    x = TimeDistributed(Dense(1, activation="linear"))(x)
    out = Reshape((n_levels,))(x)
    return compile_decoder(Model(inp, out, name="lstm_decoder"))


def tf_repeat(x, times):
    from tensorflow.keras.layers import Lambda
    import tensorflow as tf
    return Lambda(lambda t: tf.repeat(t, times, axis=1),
                  output_shape=lambda s: (s[0], times, s[2]))(x)


def run():
    common.set_seed()
    print("\n" + "=" * 68)
    print("TASK 3 - Reconstruct the full profile from surface data")
    print("=" * 68)

    X, meta = data.build_profiles()
    groups = meta["platform"]
    inputs = make_inputs(X, meta)
    target = X[:, :, 0]  # temperature profile

    print(f"{len(X)} profiles | input dim {inputs.shape[1]} -> output {target.shape[1]} levels")

    train, val, test = data.make_split(groups, meta["time"], seed=common.SEED)
    data.split_report("profiles", groups, train, val, test)

    in_mu, in_sd = inputs[train].mean(0), inputs[train].std(0) + 1e-8
    Xin = (inputs - in_mu) / in_sd
    t_mu, t_sd = target[train].mean(), target[train].std()
    Yt = (target - t_mu) / t_sd

    results, histories = {}, {}

    # Climatology: the mean training profile, regardless of input.
    clim = target[train].mean(axis=0, keepdims=True).repeat(len(test), axis=0)
    results["Climatology (mean profile)"] = common.regression_metrics(target[test], clim)

    builders = {
        "Dense decoder": build_dense,
        "1D-CNN decoder": build_cnn,
        "LSTM decoder": build_lstm,
    }
    for name, builder in builders.items():
        common.set_seed()
        model = builder(Xin.shape[1], target.shape[1])
        history = model.fit(
            Xin[train], Yt[train],
            validation_data=(Xin[val], Yt[val]),
            epochs=400, batch_size=32, verbose=0,
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True,
                              start_from_epoch=MIN_EPOCHS),
                ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=12, min_lr=1e-5),
            ],
        )
        pred = model.predict(Xin[test], verbose=0) * t_sd + t_mu
        results[name] = common.regression_metrics(target[test], pred)
        histories[name] = common.history_curves(history)
        results[name]["params"] = int(model.count_params())
        results[name]["epochs"] = len(history.history["loss"])
        model.save(common.out_path(f"seq_{model.name}.keras"))

    print()
    common.print_table({k: v for k, v in results.items()})
    print()
    for k, v in results.items():
        if "params" in v:
            print(f"  {k:<26} {v['params']:>9,} params, {v['epochs']:>3} epochs")

    np.savez(common.out_path("sequence_norm.npz"), in_mu=in_mu, in_sd=in_sd,
             t_mu=t_mu, t_sd=t_sd, depth_grid=data.DEPTH_GRID)

    common.save_result("sequence", {
        "description": "Predict the full 0-2000 db temperature profile from surface T/S, position and season",
        "input_dim": int(inputs.shape[1]),
        "output_levels": int(target.shape[1]),
        "optimizer": f"Adam(clipnorm={CLIPNORM})",
        "min_epochs": MIN_EPOCHS,
        "split": data.split_label(),
        "n_train": int(len(train)), "n_val": int(len(val)), "n_test": int(len(test)),
        "models": results,
        "history": histories,
        "units": "degree_Celsius",
    })
    return results
