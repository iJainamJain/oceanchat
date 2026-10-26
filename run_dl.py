"""
Run the deep-learning experiment suite.

    python run_dl.py                 # all four tasks on the default (one-month) dataset
    python run_dl.py oxygen          # one task
    python run_dl.py --data data/argo_indian_ocean_2yr.nc
    python run_dl.py --data data/argo_indian_ocean_2yr.nc --split temporal --cutoff 2025-07-01
    python run_dl.py --data data/argo_indian_ocean_2yr.nc --seeds 1 2 3 4 5

--seeds reruns everything per seed (weight init, the float-grouped split, baselines) into
<out>/seed_<n>/; aggregate_seeds.py turns those into mean +/- std.

Every split keeps floats disjoint between train, val and test, and every task reports a
non-neural baseline alongside the network.

Results go to <out>/dl_results.json. The default out folder, models/, is the one the
Streamlit app reads, so any other dataset or split writes to
models/<dataset name>[_temporal]/ unless --out says otherwise.
"""

import argparse
import json
import os
import time

import pandas as pd

from dl import common, data, task1_oxygen, task2_autoencoder, task3_sequence, task4_imputation

TASKS = {
    "oxygen": task1_oxygen.run,
    "autoencoder": task2_autoencoder.run,
    "sequence": task3_sequence.run,
    "imputation": task4_imputation.run,
}


def default_out(data_path, split):
    """models/ only for the app's own dataset and split; a subfolder for anything else."""
    if os.path.normpath(data_path) == os.path.normpath(data.DATA_PATH) and split == "float":
        return "models"
    name = os.path.splitext(os.path.basename(data_path))[0]
    return os.path.join("models", name + ("_temporal" if split == "temporal" else ""))


def main():
    parser = argparse.ArgumentParser(description="Run the deep-learning experiment suite.")
    parser.add_argument("tasks", nargs="*", help=f"any of {list(TASKS)} (default: all)")
    parser.add_argument("--data", default=data.DATA_PATH, help="flat NetCDF from build_dataset.py")
    parser.add_argument("--out", help="output folder for models and dl_results.json")
    parser.add_argument("--split", choices=["float", "temporal"], default="float",
                        help="float: held-out floats. temporal: held-out floats and time")
    parser.add_argument("--cutoff", help="YYYY-MM-DD; train before it, test after it")
    parser.add_argument("--seeds", nargs="+", type=int,
                        help="run once per seed into <out>/seed_<n>/ (default: seed 42 into <out>)")
    args = parser.parse_args()

    wanted = args.tasks or list(TASKS)
    unknown = [t for t in wanted if t not in TASKS]
    if unknown:
        parser.error(f"unknown task(s) {unknown}; choose from {list(TASKS)}")
    if not os.path.exists(args.data):
        parser.error(f"no such dataset: {args.data}")
    if args.split == "temporal":
        if not args.cutoff:
            parser.error("--split temporal needs --cutoff YYYY-MM-DD")
        try:
            pd.Timestamp(args.cutoff)
        except ValueError:
            parser.error(f"--cutoff {args.cutoff!r} is not a date")

    out = args.out or default_out(args.data, args.split)
    data.DATA_PATH = args.data
    data.SPLIT_MODE = args.split
    data.TIME_CUTOFF = args.cutoff if args.split == "temporal" else None

    started = time.time()
    for seed in args.seeds or [common.SEED]:
        common.SEED = seed
        common.OUT_DIR = os.path.join(out, f"seed_{seed}") if args.seeds else out

        print(f"data  {args.data}\nsplit {data.split_label()}\nseed  {seed}\nout   {common.OUT_DIR}")
        with open(common.out_path("run_config.json"), "w", encoding="utf-8") as fh:
            json.dump({
                "data": args.data,
                "split": args.split,
                "cutoff": data.TIME_CUTOFF,
                "seed": seed,
                "tasks": wanted,
                "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, fh, indent=2)

        for name in wanted:
            task_start = time.time()
            TASKS[name]()
            print(f"[seed {seed}: {name} finished in {time.time() - task_start:.0f}s]")

        print(f"\nseed {seed} done after {time.time() - started:.0f}s -> {common.results_path()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
