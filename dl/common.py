"""Shared helpers: seeding, metrics, and the results file the Streamlit app reads."""

import json
import os
import random

import numpy as np
import tensorflow as tf

# Where models and dl_results.json go. Set by run_dl.py --out; the default is the
# folder the Streamlit app reads, so other datasets must write somewhere else.
OUT_DIR = "models"

# Seed for weight init, data splits and baselines. Set by run_dl.py --seeds.
SEED = 42


def out_path(name):
    os.makedirs(OUT_DIR, exist_ok=True)
    return os.path.join(OUT_DIR, name)


def results_path():
    return os.path.join(OUT_DIR, "dl_results.json")


def set_seed(seed=None):
    seed = SEED if seed is None else seed
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype="float64").ravel()
    y_pred = np.asarray(y_pred, dtype="float64").ravel()
    err = y_pred - y_true
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
    }


def history_curves(history):
    """Learning curves, so the report can show them rather than assert convergence."""
    return {k: [float(x) for x in v] for k, v in history.history.items()}


def load_results():
    path = results_path()
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except json.JSONDecodeError:
            return {}
    return {}


def save_result(task, payload):
    """Merge one task's results into <OUT_DIR>/dl_results.json."""
    path = out_path("dl_results.json")
    results = load_results()
    results[task] = payload
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    os.replace(tmp, path)
    print(f"  -> saved results['{task}'] to {path}")


def print_table(rows, headers=("model", "MAE", "RMSE", "R2")):
    print(f"{headers[0]:<28}{headers[1]:>10}{headers[2]:>10}{headers[3]:>10}")
    for name, m in rows.items():
        print(f"{name:<28}{m['mae']:>10.4f}{m['rmse']:>10.4f}{m['r2']:>10.4f}")
