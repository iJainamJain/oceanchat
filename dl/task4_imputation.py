"""
Task 4 - Self-supervised masked-depth imputation (BERT-style).

Randomly mask a fraction of the depth levels in a profile and train the network to
reconstruct the missing values from the surrounding context. No labels are needed - the
data supervises itself, which is the same objective that trains masked language models.

This is the practically useful one: real Argo profiles have gaps where a sensor dropped
out, and filling them well matters. The natural baseline is linear interpolation across
the gap, which is what operational systems actually do, so the comparison is fair and
meaningful.

Masking is applied in contiguous blocks rather than at isolated levels. Isolated masking
is trivially solved by averaging the two neighbours; block masking forces the model to
learn the shape of a profile.

Protocol: grouped split by float. Evaluation uses fixed masks generated with a separate
seed so every model sees exactly the same gaps.
"""

import numpy as np
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Concatenate, Conv1D

from . import common, data

MASK_FRACTION = 0.25
BLOCK_SIZE = 8


def make_block_mask(shape, rng, fraction=MASK_FRACTION, block=BLOCK_SIZE):
    """Boolean mask, True where a level is hidden. Masks whole contiguous blocks."""
    n_samples, n_levels = shape
    mask = np.zeros((n_samples, n_levels), dtype=bool)
    n_blocks = max(1, int(round(fraction * n_levels / block)))
    for i in range(n_samples):
        for _ in range(n_blocks):
            start = rng.integers(0, n_levels - block)
            mask[i, start:start + block] = True
    return mask


def build_imputer(n_levels, n_vars):
    """Dilated 1D convolutions: a wide receptive field over depth, cheaply."""
    values = Input(shape=(n_levels, n_vars), name="masked_values")
    flags = Input(shape=(n_levels, 1), name="mask_flag")
    x = Concatenate()([values, flags])
    x = Conv1D(64, 5, padding="same", dilation_rate=1, activation="relu")(x)
    x = Conv1D(64, 5, padding="same", dilation_rate=2, activation="relu")(x)
    x = Conv1D(64, 5, padding="same", dilation_rate=4, activation="relu")(x)
    x = Conv1D(64, 5, padding="same", dilation_rate=8, activation="relu")(x)
    x = Conv1D(32, 5, padding="same", activation="relu")(x)
    out = Conv1D(n_vars, 5, padding="same", activation="linear")(x)
    model = Model([values, flags], out, name="masked_imputer")
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def apply_mask(X, mask):
    """Zero out masked levels and hand the model an explicit mask channel."""
    masked = X.copy()
    masked[mask] = 0.0
    return masked, mask[..., None].astype("float32")


def linear_interp_baseline(X, mask):
    """What operational systems do: interpolate across the gap from known levels."""
    filled = X.copy()
    n_levels = X.shape[1]
    grid = np.arange(n_levels)
    for i in range(len(X)):
        known = ~mask[i]
        if known.sum() < 2:
            continue
        for v in range(X.shape[2]):
            filled[i, mask[i], v] = np.interp(grid[mask[i]], grid[known], X[i, known, v])
    return filled


def run():
    common.set_seed()
    print("\n" + "=" * 68)
    print(f"TASK 4 - Masked-depth imputation ({int(MASK_FRACTION * 100)}% masked, "
          f"blocks of {BLOCK_SIZE} levels)")
    print("=" * 68)

    X, meta = data.build_profiles()
    groups = meta["platform"]
    train, val, test = data.make_split(groups, meta["time"], seed=common.SEED)
    data.split_report("profiles", groups, train, val, test)

    mu = X[train].mean(axis=(0, 1), keepdims=True)
    sd = X[train].std(axis=(0, 1), keepdims=True)
    Xs = (X - mu) / sd

    rng = np.random.default_rng(0)
    mask_tr = make_block_mask(Xs[train].shape[:2], rng)
    mask_va = make_block_mask(Xs[val].shape[:2], np.random.default_rng(1))
    mask_te = make_block_mask(Xs[test].shape[:2], np.random.default_rng(2))

    tr_in = apply_mask(Xs[train], mask_tr)
    va_in = apply_mask(Xs[val], mask_va)
    te_in = apply_mask(Xs[test], mask_te)

    model = build_imputer(X.shape[1], X.shape[2])
    history = model.fit(
        tr_in, Xs[train],
        validation_data=(va_in, Xs[val]),
        epochs=400, batch_size=32, verbose=0,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=12, min_lr=1e-5),
        ],
    )

    pred = model.predict(te_in, verbose=0) * sd + mu
    interp = linear_interp_baseline(X[test], mask_te)
    truth = X[test]

    results = {}
    for i, name in enumerate(["TEMP", "PSAL"]):
        hidden = mask_te  # score only the levels that were actually hidden
        results[f"Dilated CNN  {name}"] = common.regression_metrics(
            truth[:, :, i][hidden], pred[:, :, i][hidden]
        )
        results[f"Linear interp {name}"] = common.regression_metrics(
            truth[:, :, i][hidden], interp[:, :, i][hidden]
        )

    print()
    common.print_table(results)
    print(f"\nscored on {int(mask_te.sum()):,} hidden levels across {len(test)} held-out profiles")
    print(f"epochs trained: {len(history.history['loss'])}, params: {model.count_params():,}")

    model.save(common.out_path("imputer.keras"))
    common.save_result("imputation", {
        "description": f"Reconstruct {int(MASK_FRACTION * 100)}% of depth levels hidden in blocks of {BLOCK_SIZE}",
        "mask_fraction": MASK_FRACTION,
        "block_size": BLOCK_SIZE,
        "objective": "self-supervised (no labels)",
        "split": data.split_label(),
        "n_train": int(len(train)), "n_val": int(len(val)), "n_test": int(len(test)),
        "n_hidden_levels_scored": int(mask_te.sum()),
        "params": int(model.count_params()),
        "models": results,
        "history": common.history_curves(history),
    })
    return results
