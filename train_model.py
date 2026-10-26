"""
Train the temperature model used by the Streamlit Predictions tab.

Protocol note - this matters more than the architecture:

Measurements are NOT independent. Each profile contributes ~790 rows, one per depth
level, and adjacent levels differ by a median of 0.008 degrees C. A random
train_test_split therefore scatters near-duplicate rows across train and test, and the
resulting score measures interpolation between memorised neighbours rather than
generalisation. On this dataset that inflates the headline figure from MAE 0.30 to 0.21.

So the split here is grouped by float: no float appears in more than one split, and the
test set is floats the model has never seen. Classical baselines are trained on the same
split, because a neural network is only worth using if it beats them.
"""

import json
import os
import pickle

import numpy as np
import xarray as xr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential

from dl.common import regression_metrics
from dl.data import grouped_split

DATA_CANDIDATES = [
    os.path.join("data", "argo_indian_ocean.nc"),
    os.path.join("data", "sample_argo.nc"),
]

FEATURES = ["PRES", "PSAL", "LATITUDE", "LONGITUDE"]
TARGET = "TEMP"

MODEL_PATH = os.path.join("models", "ocean_model.keras")
SCALER_PATH = os.path.join("models", "scaler.pkl")
META_PATH = os.path.join("models", "model_meta.json")


def resolve_data_path():
    for path in DATA_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


def build_model(n_in):
    model = Sequential([
        Dense(256, activation="relu", input_shape=(n_in,)),
        Dropout(0.1),
        Dense(128, activation="relu"),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def main():
    data_path = resolve_data_path()
    if data_path is None:
        print("Error: no dataset found. Run 'python build_dataset.py' first.")
        return

    print(f"Loading ARGO data from {data_path} ...")
    df = xr.open_dataset(data_path).to_dataframe().reset_index()

    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        print(f"Error: dataset is missing required columns: {missing}")
        return

    df = df.dropna(subset=FEATURES + [TARGET])
    X = df[FEATURES].to_numpy("float32")
    y = df[TARGET].to_numpy("float32")

    if "PLATFORM" in df.columns:
        groups = df["PLATFORM"].to_numpy()
        split_kind = "grouped by float (no float in more than one split)"
    else:
        # The old synthetic sample file has no float id; fall back honestly.
        groups = np.arange(len(df))
        split_kind = "random (dataset has no PLATFORM column)"

    train, val, test = grouped_split(groups)
    n_floats = {k: int(len(np.unique(groups[ix])))
                for k, ix in (("train", train), ("val", val), ("test", test))}

    print(f"Data loaded. {len(df):,} measurements, features {FEATURES}.")
    print(f"Split: {split_kind}")
    print(f"  train {len(train):,} ({n_floats['train']} floats) | "
          f"val {len(val):,} ({n_floats['val']}) | test {len(test):,} ({n_floats['test']})")

    scaler = StandardScaler().fit(X[train])
    Xtr, Xva, Xte = (scaler.transform(X[i]) for i in (train, val, test))
    ytr, yva, yte = y[train], y[val], y[test]

    # ---------- baselines on the identical split ----------
    print("\nTraining baselines...")
    baselines = {}
    baselines["Linear regression"] = regression_metrics(
        yte, LinearRegression().fit(Xtr, ytr).predict(Xte))
    baselines["k-NN (k=10)"] = regression_metrics(
        yte, KNeighborsRegressor(n_neighbors=10, n_jobs=-1).fit(Xtr, ytr).predict(Xte))
    baselines["Gradient boosting"] = regression_metrics(
        yte, HistGradientBoostingRegressor(max_iter=300, random_state=42)
        .fit(X[train], ytr).predict(X[test]))

    # ---------- the network ----------
    print("Training neural network...")
    model = build_model(len(FEATURES))
    history = model.fit(
        Xtr, ytr, validation_data=(Xva, yva),
        epochs=150, batch_size=512, verbose=2,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5),
        ],
    )
    nn_metrics = regression_metrics(yte, model.predict(Xte, verbose=0))

    print("\n--- Held-out test performance (unseen floats) ---")
    print(f"{'model':<24}{'MAE':>9}{'RMSE':>9}{'R2':>9}")
    for name, m in {**baselines, "Neural network (MLP)": nn_metrics}.items():
        print(f"{name:<24}{m['mae']:>9.4f}{m['rmse']:>9.4f}{m['r2']:>9.4f}")

    best_baseline = min(baselines.items(), key=lambda kv: kv[1]["mae"])
    if nn_metrics["mae"] < best_baseline[1]["mae"]:
        gain = 100 * (1 - nn_metrics["mae"] / best_baseline[1]["mae"])
        print(f"\nNetwork beats the best baseline ({best_baseline[0]}) by {gain:.1f}% MAE.")
    else:
        print(f"\nNote: {best_baseline[0]} matches or beats the network here.")

    os.makedirs("models", exist_ok=True)
    model.save(MODEL_PATH)
    with open(SCALER_PATH, "wb") as fh:
        pickle.dump(scaler, fh)

    meta = {
        "features": FEATURES,
        "target": TARGET,
        "split": split_kind,
        "n_train": int(len(train)), "n_val": int(len(val)), "n_test": int(len(test)),
        "n_floats": n_floats,
        "mae": nn_metrics["mae"],
        "rmse": nn_metrics["rmse"],
        "r2": nn_metrics["r2"],
        "baselines": baselines,
        "epochs": len(history.history["loss"]),
        "history": {k: [float(x) for x in v] for k, v in history.history.items()},
        "data_path": data_path,
    }
    with open(META_PATH, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Scaler saved to {SCALER_PATH}")
    print(f"Metadata saved to {META_PATH}")
    print("Training complete! You can now run Streamlit.")


if __name__ == "__main__":
    main()
