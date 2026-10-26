"""
Merge a Coriolis/CMEMS Argo subsetting request into one flat NetCDF.

    python build_dataset.py                                   # the app's one-month dataset
    python build_dataset.py --raw-dir data/raw_2yr --out data/argo_indian_ocean_2yr.nc

Input : <raw-dir>/**/GL_PR_PF_<WMO>.nc   (per-float files, TIME x DEPTH; subfolders ok)
Output: <out>                            (flat OBS table ready for app.py / run_dl.py)

Rules applied:
  * floats without PSAL are skipped entirely (temperature-only floats)
  * delayed-mode *_ADJUSTED values are preferred per-cell, raw used as fallback
  * only QC flags 1 (good) and 2 (probably good) are kept, for TEMP/PSAL/PRES
  * profiles with bad TIME_QC or POSITION_QC are dropped

Biogeochemistry:
  A minority of floats carry BGC sensors. Those variables are attached to the same
  rows as the core measurements and left as NaN elsewhere, so the core table stays
  one consistent shape. Note most BGC parameters are distributed *_ADJUSTED only.

Columns are accumulated as float32 (TIME stays float64), matching what is written, so a
multi-year request fits in memory.
"""

import argparse
import glob
import os

import netCDF4
import numpy as np
import pandas as pd

RAW_DIR = os.path.join("data", "raw")
OUT_PATH = os.path.join("data", "argo_indian_ocean.nc")
GOOD_QC = (1, 2)

# name -> (units, long_name). All are sparse; see the coverage report at the end.
BGC_VARS = {
    "DOX2": ("umol kg-1", "Dissolved oxygen"),
    "CPHL": ("mg m-3", "Chlorophyll-a"),
    "BBP700": ("m-1", "Particle backscattering at 700 nm"),
    "NTAW": ("umol kg-1", "Nitrate (NO3-N)"),
    "PHPH": ("1", "pH (total scale)"),
}

CORE_SPECS = [
    ("PRES", "f4", "decibar", "Sea water pressure, equals 0 at sea-level"),
    ("TEMP", "f4", "degree_Celsius", "Sea temperature in-situ ITS-90 scale"),
    ("PSAL", "f4", "psu", "Practical salinity"),
    ("LATITUDE", "f4", "degree_north", "Latitude of the profile"),
    ("LONGITUDE", "f4", "degree_east", "Longitude of the profile"),
    ("TIME", "f8", "seconds since 1970-01-01T00:00:00Z", "Time of the profile"),
    ("PLATFORM", "i4", "1", "WMO identifier of the Argo float"),
]


def _read(ds, name):
    """Return a float64 array with masked/fill values as NaN, or None if absent."""
    if name not in ds.variables:
        return None
    return np.ma.filled(ds[name][:].astype("f8"), np.nan)


def _pick(ds, base, shape):
    """Prefer ADJUSTED values where present, fall back to raw. Returns (values, qc)."""
    raw, raw_qc = _read(ds, base), _read(ds, f"{base}_QC")
    adj, adj_qc = _read(ds, f"{base}_ADJUSTED"), _read(ds, f"{base}_ADJUSTED_QC")

    if raw is None:
        raw = np.full(shape, np.nan)
    if raw_qc is None:
        raw_qc = np.full(shape, 9.0)

    if adj is None:
        return np.broadcast_to(raw, shape).copy(), np.broadcast_to(raw_qc, shape).copy()

    if adj_qc is None:
        adj_qc = np.full(shape, 9.0)

    adj = np.broadcast_to(adj, shape)
    adj_qc = np.broadcast_to(adj_qc, shape)
    raw = np.broadcast_to(raw, shape)
    raw_qc = np.broadcast_to(raw_qc, shape)

    use_adj = ~np.isnan(adj)
    return np.where(use_adj, adj, raw), np.where(use_adj, adj_qc, raw_qc)


def _bgc(ds, base, shape, keep):
    """QC-filtered BGC column for the kept rows, or None if this float lacks the sensor."""
    present = any(
        n in ds.variables for n in (base, f"{base}_ADJUSTED")
    )
    if not present:
        return None
    values, qc = _pick(ds, base, shape)
    values = np.where(np.isin(qc, GOOD_QC), values, np.nan)
    return values[keep]


def time_coverage(platform, time_s):
    """Profiles per month, so a multi-year request can be checked for gaps at a glance."""
    prof = pd.DataFrame({"p": platform, "t": time_s}).drop_duplicates()
    months = pd.to_datetime(prof["t"], unit="s").dt.to_period("M")
    per_month = months.value_counts().sort_index()
    span = pd.period_range(per_month.index.min(), per_month.index.max(), freq="M")
    missing = [str(m) for m in span if m not in per_month.index]

    print(f"\nTime span: {per_month.index.min()} to {per_month.index.max()} "
          f"({len(per_month)} of {len(span)} months have data)")
    for month, n in per_month.items():
        print(f"  {month}  {n:>6,} profiles")
    if missing:
        print(f"  WARNING - no profiles in: {', '.join(missing)}")


def main():
    parser = argparse.ArgumentParser(description="Flatten per-float Argo files into one NetCDF.")
    parser.add_argument("--raw-dir", default=RAW_DIR, help=f"input folder (default {RAW_DIR})")
    parser.add_argument("--out", default=OUT_PATH, help=f"output file (default {OUT_PATH})")
    args = parser.parse_args()
    raw_dir, out_path = args.raw_dir, args.out

    files = sorted(glob.glob(os.path.join(raw_dir, "**", "*.nc"), recursive=True))
    if not files:
        print(f"Error: no NetCDF files found in {raw_dir}/")
        return

    print(f"Scanning {len(files)} float files from {raw_dir}/ ...")

    core_names = [s[0] for s in CORE_SPECS]
    cols = {k: [] for k in core_names}
    cols.update({k: [] for k in BGC_VARS})

    skipped_no_psal = 0
    n_profiles = 0
    bgc_floats = {k: 0 for k in BGC_VARS}

    for i, path in enumerate(files, 1):
        ds = netCDF4.Dataset(path)
        try:
            if "PSAL" not in ds.variables and "PSAL_ADJUSTED" not in ds.variables:
                skipped_no_psal += 1
                continue

            shape = ds["TEMP"][:].shape  # (TIME, DEPTH)

            temp, temp_qc = _pick(ds, "TEMP", shape)
            psal, psal_qc = _pick(ds, "PSAL", shape)
            pres, pres_qc = _pick(ds, "PRES", shape)

            # Per-profile fields, broadcast down the DEPTH axis.
            lat = np.broadcast_to(_read(ds, "LATITUDE").reshape(-1, 1), shape)
            lon = np.broadcast_to(_read(ds, "LONGITUDE").reshape(-1, 1), shape)

            tvar = ds["TIME"]
            time_num = netCDF4.date2num(
                netCDF4.num2date(tvar[:], tvar.units),
                "seconds since 1970-01-01T00:00:00Z",
            )
            time_col = np.broadcast_to(np.asarray(time_num, dtype="f8").reshape(-1, 1), shape)

            # Profile-level QC gates.
            time_qc = _read(ds, "TIME_QC")
            pos_qc = _read(ds, "POSITION_QC")
            prof_ok = np.ones(shape[0], dtype=bool)
            if time_qc is not None:
                prof_ok &= np.isin(time_qc, GOOD_QC)
            if pos_qc is not None:
                prof_ok &= np.isin(pos_qc, GOOD_QC)
            prof_ok = np.broadcast_to(prof_ok.reshape(-1, 1), shape)

            keep = (
                ~np.isnan(temp) & ~np.isnan(psal) & ~np.isnan(pres)
                & ~np.isnan(lat) & ~np.isnan(lon)
                & np.isin(temp_qc, GOOD_QC)
                & np.isin(psal_qc, GOOD_QC)
                & np.isin(pres_qc, GOOD_QC)
                & prof_ok
            )

            if not keep.any():
                continue

            wmo = int("".join(c for c in os.path.basename(path) if c.isdigit()))
            n = int(keep.sum())

            cols["PRES"].append(pres[keep].astype("f4"))
            cols["TEMP"].append(temp[keep].astype("f4"))
            cols["PSAL"].append(psal[keep].astype("f4"))
            cols["LATITUDE"].append(lat[keep].astype("f4"))
            cols["LONGITUDE"].append(lon[keep].astype("f4"))
            cols["TIME"].append(time_col[keep])
            cols["PLATFORM"].append(np.full(n, wmo, dtype="i4"))

            for base in BGC_VARS:
                column = _bgc(ds, base, shape, keep)
                if column is None:
                    column = np.full(n, np.nan, dtype="f4")
                elif np.isfinite(column).any():
                    bgc_floats[base] += 1
                cols[base].append(column.astype("f4", copy=False))

            n_profiles += int(keep.any(axis=1).sum())
        finally:
            ds.close()

        if i % 100 == 0:
            print(f"  ...{i}/{len(files)} files")

    print(f"\nSkipped {skipped_no_psal} floats with no salinity.")
    finish(
        cols, n_profiles, bgc_floats, out_path,
        source_files=f"{len(files) - skipped_no_psal} float files from {raw_dir}/",
        qc_policy="TEMP/PSAL/PRES QC flags 1-2 only; ADJUSTED preferred over raw",
    )


def finish(cols, n_profiles, bgc_floats, out_path, source_files, qc_policy):
    """Concatenate, sort, report and write the flat table.

    Shared with build_dataset_gdac.py so both input formats produce the same file.
    """
    if not cols["TEMP"]:
        print("Error: no usable observations survived filtering.")
        return

    data = {k: np.concatenate(v) for k, v in cols.items()}
    cols.clear()
    n_obs = data["TEMP"].size

    # Sort by float then time then depth so the file reads sensibly.
    order = np.lexsort((data["PRES"], data["TIME"], data["PLATFORM"]))
    data = {k: v[order] for k, v in data.items()}

    print(f"Kept {n_obs:,} observations from {n_profiles:,} profiles "
          f"({len(np.unique(data['PLATFORM']))} floats).")

    time_coverage(data["PLATFORM"], data["TIME"])

    print("\nBiogeochemistry coverage:")
    for base, (units, long_name) in BGC_VARS.items():
        valid = int(np.isfinite(data[base]).sum())
        pct = 100.0 * valid / n_obs
        print(
            f"  {base:<8} {valid:>9,} obs ({pct:5.2f}%)  "
            f"from {bgc_floats[base]:>3} floats  [{units}]"
        )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with netCDF4.Dataset(out_path, "w", format="NETCDF4") as out:
        out.createDimension("OBS", n_obs)

        specs = list(CORE_SPECS) + [
            (name, "f4", units, long_name) for name, (units, long_name) in BGC_VARS.items()
        ]
        for name, dtype, units, long_name in specs:
            var = out.createVariable(name, dtype, ("OBS",), zlib=True, complevel=4)
            var.units = units
            var.long_name = long_name
            if name == "TIME":
                var.calendar = "standard"
            var[:] = data[name]

        out.title = "Argo Indian Ocean subset, flattened"
        out.source_files = source_files
        out.qc_policy = qc_policy
        out.bgc_note = (
            "BGC columns are sparse and mostly distributed as *_ADJUSTED only; "
            "NaN means the float carried no such sensor or the value failed QC."
        )

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"\nWrote {out_path} ({size_mb:.1f} MB)")
    if os.path.normpath(out_path) == os.path.normpath(OUT_PATH):
        print("Next: python train_model.py")
    else:
        print(f"Next: python run_dl.py --data {out_path}")


if __name__ == "__main__":
    main()
