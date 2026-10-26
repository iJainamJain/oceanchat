"""
Shared data preparation for the deep-learning experiments.

Two views of the same dataset:

  * measurement view  - one row per (profile, depth level). Used by the oxygen model.
  * profile view      - one sample per profile, interpolated onto a fixed depth grid,
                        shape (n_profiles, N_LEVELS, n_vars). Used by the autoencoder,
                        the sequence model and the masked-imputation model.

Every split in this package is grouped by float (PLATFORM). Measurements inside one
profile are almost identical to their neighbours - adjacent depth levels differ by a
median of 0.008 degrees C - so a random split leaks near-duplicates into the test set
and inflates the score. Grouping by float is the honest protocol.
"""

import os

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.model_selection import GroupShuffleSplit

# Set by run_dl.py --data. Read at call time, so reassigning it before a task runs works.
DATA_PATH = os.path.join("data", "argo_indian_ocean.nc")

# Split protocol, set by run_dl.py --split / --cutoff.
#   "float"    - held-out floats (the default, and what the app's results use)
#   "temporal" - held-out floats AND held-out time: train on profiles before
#                TIME_CUTOFF, test on later profiles from floats never seen before it
SPLIT_MODE = "float"
TIME_CUTOFF = None

# 100 levels at 20 db spacing. Profiles must span this range to be included, so no
# sample is ever built by extrapolating beyond what the float actually measured.
N_LEVELS = 100
DEPTH_GRID = np.linspace(10.0, 1990.0, N_LEVELS)
SURFACE_MAX_DB = 30.0
DEEP_MIN_DB = 1900.0

PROFILE_KEYS = ["PLATFORM", "TIME"]


def load_flat():
    """The measurement-level table."""
    df = xr.open_dataset(DATA_PATH).to_dataframe().reset_index()
    return df.drop(columns=["OBS"], errors="ignore")


def build_profiles(df=None, variables=("TEMP", "PSAL")):
    """Interpolate each profile onto DEPTH_GRID.

    Returns (X, meta) where X has shape (n_profiles, N_LEVELS, len(variables)) and
    meta is a dict of per-profile arrays: platform, lat, lon, time.
    """
    if df is None:
        df = load_flat()

    needed = list(variables)
    df = df.dropna(subset=needed + ["PRES"])

    profiles, platforms, lats, lons, times = [], [], [], [], []

    for (platform, time), grp in df.groupby(PROFILE_KEYS, sort=False):
        grp = grp.sort_values("PRES")
        pres = grp["PRES"].to_numpy()

        # Only keep profiles that actually span the grid.
        if pres.min() > SURFACE_MAX_DB or pres.max() < DEEP_MIN_DB or len(pres) < 50:
            continue

        cols = []
        for var in variables:
            cols.append(np.interp(DEPTH_GRID, pres, grp[var].to_numpy()))
        profiles.append(np.stack(cols, axis=-1))

        platforms.append(platform)
        lats.append(grp["LATITUDE"].iloc[0])
        lons.append(grp["LONGITUDE"].iloc[0])
        times.append(time)

    X = np.asarray(profiles, dtype="float32")
    meta = {
        "platform": np.asarray(platforms),
        "lat": np.asarray(lats, dtype="float32"),
        "lon": np.asarray(lons, dtype="float32"),
        "time": np.asarray(times),
    }
    return X, meta


def grouped_split(groups, test_size=0.2, val_size=0.15, seed=42):
    """Train/val/test indices such that no float appears in more than one split."""
    idx = np.arange(len(groups))

    outer = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval, test = next(outer.split(idx, groups=groups))

    inner_frac = val_size / (1.0 - test_size)
    inner = GroupShuffleSplit(n_splits=1, test_size=inner_frac, random_state=seed)
    tr_rel, va_rel = next(inner.split(trainval, groups=groups[trainval]))

    return trainval[tr_rel], trainval[va_rel], test


def temporal_split(groups, times, cutoff, val_size=0.15, seed=42):
    """Train/val before `cutoff`, test after it on floats never seen before it.

    Checks that a model trained on one period still works on the next. Post-cutoff
    samples from floats that were already in the training period are dropped rather
    than tested, so the no-shared-floats rule still holds.
    """
    t = pd.to_datetime(np.asarray(times))
    cut = pd.Timestamp(cutoff)
    before = np.flatnonzero(t < cut)
    after = np.flatnonzero(t >= cut)

    seen = np.unique(groups[before])
    test = after[~np.isin(groups[after], seen)]
    if len(before) == 0 or len(test) == 0:
        raise ValueError(
            f"temporal split at {cut.date()} leaves {len(before):,} samples before it and "
            f"{len(test):,} from unseen floats after it - pick another cutoff"
        )

    inner = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    tr_rel, va_rel = next(inner.split(before, groups=groups[before]))

    print(f"temporal split at {cut.date()}: dropped {len(after) - len(test):,} post-cutoff "
          f"samples whose float was already seen before the cutoff")
    return before[tr_rel], before[va_rel], test


def make_split(groups, times, seed=42):
    """Train/val/test indices under the protocol chosen by SPLIT_MODE."""
    if SPLIT_MODE == "float":
        return grouped_split(groups, seed=seed)
    if SPLIT_MODE == "temporal":
        return temporal_split(groups, times, TIME_CUTOFF, seed=seed)
    raise ValueError(f"unknown SPLIT_MODE {SPLIT_MODE!r}")


def split_label():
    """How the split is described in the results file."""
    if SPLIT_MODE == "temporal":
        return f"held-out floats and time (train < {TIME_CUTOFF}, test >= {TIME_CUTOFF})"
    return "grouped by float"


def split_report(name, groups, train, val, test):
    def floats(ix):
        return len(np.unique(groups[ix]))

    print(
        f"{name}: train {len(train):,} ({floats(train)} floats) | "
        f"val {len(val):,} ({floats(val)}) | test {len(test):,} ({floats(test)})"
    )
    overlap = (
        set(groups[train]) & set(groups[test])
        or set(groups[train]) & set(groups[val])
        or set(groups[val]) & set(groups[test])
    )
    assert not overlap, "float appears in more than one split"
