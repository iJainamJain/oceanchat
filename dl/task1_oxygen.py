"""
Task 1 - Predict dissolved oxygen from physics.

Learn DOX2 from (PRES, TEMP, PSAL, LATITUDE, LONGITUDE). This is a real problem in
operational oceanography: oxygen sensors are expensive and only a minority of floats
carry one, so inferring oxygen from the physical variables every float measures would
extend coverage enormously.

The scientific difficulty is what makes it a good DL task: the oxygen minimum zone is
a sharp, non-monotonic feature that a linear model cannot represent at all.

Protocol: grouped split by float, so the test set is floats the model has never seen.
Baselines (linear, k-NN, gradient boosting) are reported alongside, because a neural
network is only justified if it beats them.
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from tensorflow.keras.models import Sequential

from . import common, data

FEATURES = ["PRES", "TEMP", "PSAL", "LATITUDE", "LONGITUDE"]
TARGET = "DOX2"


def build_model(n_in):
    model = Sequential([
        Dense(256, activation="relu", input_shape=(n_in,)),
        BatchNormalization(),
        Dropout(0.15),
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.15),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def run():
    common.set_seed()
    print("\n" + "=" * 68)
    print("TASK 1 - Dissolved oxygen from physical variables")
    print("=" * 68)

    flat = data.load_flat()
    n_total = len(flat)
    df = flat.dropna(subset=FEATURES + [TARGET])
    del flat
    X = df[FEATURES].to_numpy("float32")
    y = df[TARGET].to_numpy("float32")
    groups = df["PLATFORM"].to_numpy()

    print(f"{len(df):,} labelled measurements from {df.PLATFORM.nunique()} floats "
          f"({100 * len(df) / n_total:.2f}% of the dataset carry oxygen)")

    train, val, test = data.make_split(groups, df["TIME"].to_numpy(), seed=common.SEED)
    data.split_report("oxygen", groups, train, val, test)

    scaler = StandardScaler().fit(X[train])
    Xtr, Xva, Xte = (scaler.transform(X[i]) for i in (train, val, test))
    ytr, yva, yte = y[train], y[val], y[test]

    results = {}

    lin = LinearRegression().fit(Xtr, ytr)
    results["Linear regression"] = common.regression_metrics(yte, lin.predict(Xte))

    knn = KNeighborsRegressor(n_neighbors=10, n_jobs=-1).fit(Xtr, ytr)
    results["k-NN (k=10)"] = common.regression_metrics(yte, knn.predict(Xte))

    gbm = HistGradientBoostingRegressor(max_iter=400, random_state=common.SEED).fit(X[train], ytr)
    results["Gradient boosting"] = common.regression_metrics(yte, gbm.predict(X[test]))

    model = build_model(len(FEATURES))
    history = model.fit(
        Xtr, ytr, validation_data=(Xva, yva),
        epochs=200, batch_size=256, verbose=0,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-5),
        ],
    )
    results["Neural network (MLP)"] = common.regression_metrics(
        yte, model.predict(Xte, verbose=0)
    )

    print()
    common.print_table(results)
    print(f"epochs trained: {len(history.history['loss'])}")

    model.save(common.out_path("oxygen_model.keras"))
    common.save_result("oxygen", {
        "description": "Predict dissolved oxygen (umol/kg) from PRES/TEMP/PSAL/LAT/LON",
        "features": FEATURES,
        "target": TARGET,
        "n_labelled": int(len(df)),
        "n_floats": int(df.PLATFORM.nunique()),
        "split": data.split_label(),
        "n_train": int(len(train)), "n_val": int(len(val)), "n_test": int(len(test)),
        "models": results,
        "history": common.history_curves(history),
        "units": "umol/kg",
    })
    return results
