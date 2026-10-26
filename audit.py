"""Audit the DL claim: is R^2=0.9976 real generalisation, and does the NN earn its place?

    python audit.py                                              # the one-month dataset
    python audit.py --data data/argo_indian_ocean_2yr.nc --cutoff 2025-07-01

--cutoff adds a third split: held-out floats AND held-out time.
--seed sets every split, the models and TensorFlow, and names the output after the seed.

The MLP trains until early stopping (max 200 epochs). An earlier fixed 40-epoch budget left
it undertrained on the 9M-row random split, so it scored worse there than on held-out
floats - an artefact, not a finding.
"""
import os, warnings, json, argparse
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import xarray as xr
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

DEFAULT_DATA = os.path.join("data", "argo_indian_ocean.nc")

parser = argparse.ArgumentParser(description="Random vs float-grouped split audit.")
parser.add_argument("--data", default=DEFAULT_DATA)
parser.add_argument("--out", help="results JSON (default audit_results[_<dataset>].json)")
parser.add_argument("--cutoff", help="YYYY-MM-DD; also run the held-out floats + time split")
parser.add_argument("--seed", type=int, help="default 42; when given, output is models/<dataset>/audit/seed_<n>.json")
args = parser.parse_args()
stem = os.path.splitext(os.path.basename(args.data))[0]
if args.out is None:
    if args.seed is not None:
        args.out = os.path.join("models", stem, "audit", f"seed_{args.seed}.json")
    else:
        same = os.path.normpath(args.data) == os.path.normpath(DEFAULT_DATA)
        args.out = "audit_results.json" if same else f"audit_results_{stem}.json"
SEED = 42 if args.seed is None else args.seed

import tensorflow as tf
tf.keras.utils.set_random_seed(SEED)

FEATURES = ["PRES", "PSAL", "LATITUDE", "LONGITUDE"]

df = xr.open_dataset(args.data).to_dataframe().reset_index()
df = df.dropna(subset=FEATURES + ["TEMP"])
X = df[FEATURES].values
y = df["TEMP"].values
groups = df["PLATFORM"].values

print(f"{args.data}: rows={len(df):,}  floats={df.PLATFORM.nunique()}  "
      f"profiles~{df.groupby(['PLATFORM','TIME']).ngroups}")


def mlp(n_in):
    m = Sequential([
        Dense(256, activation="relu", input_shape=(n_in,)),
        Dropout(0.1),
        Dense(128, activation="relu"),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    m.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return m


def evaluate(name, tr, te):
    Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    out = {}

    lin = LinearRegression().fit(Xtr_s, ytr)
    p = lin.predict(Xte_s)
    out["Linear regression"] = (mean_absolute_error(yte, p), r2_score(yte, p))

    gbm = HistGradientBoostingRegressor(max_iter=300, random_state=SEED).fit(Xtr, ytr)
    p = gbm.predict(Xte)
    out["Gradient boosting"] = (mean_absolute_error(yte, p), r2_score(yte, p))

    net = mlp(Xtr.shape[1])
    net.fit(Xtr_s, ytr, validation_split=0.1, epochs=200, batch_size=1024,
            callbacks=[EarlyStopping(monitor="val_loss", patience=6,
                                     restore_best_weights=True)], verbose=0)
    p = net.predict(Xte_s, verbose=0).ravel()
    out["Neural network (MLP)"] = (mean_absolute_error(yte, p), r2_score(yte, p))

    print(f"\n===== {name} =====")
    print(f"train={len(tr):,}  test={len(te):,}")
    print(f"{'model':<24}{'MAE (C)':>10}{'R2':>10}")
    for k, (mae, r2) in out.items():
        print(f"{k:<24}{mae:>10.3f}{r2:>10.4f}")
    return out


results = {}

# 1) What train_model.py does today: random split over individual measurements.
tr, te = train_test_split(np.arange(len(X)), test_size=0.2, random_state=SEED)
results["random"] = evaluate("RANDOM SPLIT (what the project does now)", tr, te)

# 2) Honest split: no float appears in both train and test.
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
tr, te = next(gss.split(X, y, groups))
results["grouped"] = evaluate("GROUPED SPLIT BY FLOAT (held-out floats)", tr, te)

# 3) Stricter still: held-out floats from a later period than training.
if args.cutoff:
    from dl.data import temporal_split
    tr, va, te = temporal_split(groups, df["TIME"].values, args.cutoff, seed=SEED)
    results["temporal"] = evaluate(
        f"HELD-OUT FLOATS AND TIME (train < {args.cutoff} <= test)",
        np.concatenate([tr, va]), te,
    )

print("\n\n===== SUMMARY (R2) =====")
print(f"{'model':<24}" + "".join(f"{s:>12}" for s in results))
for k in results["random"]:
    print(f"{k:<24}" + "".join(f"{results[s][k][1]:>12.4f}" for s in results))

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
with open(args.out, "w", encoding="utf-8") as fh:
    json.dump(results, fh, indent=2)
print(f"\n-> {args.out}")
