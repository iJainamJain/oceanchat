"""
Merge an Argo GDAC-format request (one file per profile) into the same flat NetCDF that
build_dataset.py produces, so app.py and run_dl.py can read either.

    python build_dataset_gdac.py --raw-dir data/raw_2yr --out data/argo_indian_ocean_2yr.nc

Input : <raw-dir>/<dac>/<wmo>/profiles/{R,D}<wmo>_<cycle>.nc    core T/S profiles
        <raw-dir>/<dac>/<wmo>/profiles/{SR,SD}<wmo>_<cycle>.nc  synthetic BGC profiles

The Euro-Argo Data Selection tool delivers this layout for larger requests; the first
(one-month) request came as CMEMS per-float files instead, which build_dataset.py reads.

Rules applied (Argo user manual conventions):
  * ascending profiles only - descending ones (<cycle>D.nc) are skipped
  * a core file can hold several profiles per cycle; only the primary sampling one is used
  * DATA_MODE R uses raw values and QC; A or D uses *_ADJUSTED values and QC only. There is
    deliberately no per-cell fallback to raw: delayed-mode operators blank adjusted values
    they reject, and falling back would bring the rejected raw value back.
  * TEMP/PSAL/PRES QC flags 1-2 only; JULD_QC and POSITION_QC 1-2 only

Biogeochemistry:
  * read from the S-file of the same cycle, choosing raw or adjusted per parameter from
    PARAMETER_DATA_MODE, QC flags 1-2 only
  * attached only where the S-file level has exactly the pressure of a kept core level.
    About 40% of BGC levels sit between CTD levels, where the S-file carries interpolated
    T/S (QC 8); those are dropped rather than paired with estimated inputs.
  * renamed to the CMEMS names the rest of the project uses:
    DOXY->DOX2, CHLA->CPHL, BBP700, NITRATE->NTAW, PH_IN_SITU_TOTAL->PHPH
"""

import argparse
import glob
import os
import re

import netCDF4
import numpy as np

from build_dataset import BGC_VARS, CORE_SPECS, finish

NAME = re.compile(r"^(SR|SD|R|D)(\d+)_(\d+)(D?)\.nc$")
GOOD_QC = (b"1", b"2")
BGC_SOURCE = {
    "DOX2": "DOXY",
    "CPHL": "CHLA",
    "BBP700": "BBP700",
    "NTAW": "NITRATE",
    "PHPH": "PH_IN_SITU_TOTAL",
}
# JULD is days since 1950-01-01; the flat file stores seconds since 1970-01-01.
DAYS_1950_TO_1970 = 7305
PRES_MATCH_DBAR = 0.005


def _chars(ds, name):
    a = ds[name][:]
    return np.asarray(np.ma.filled(a, b" ") if np.ma.isMaskedArray(a) else a)


def _text(chars):
    return b"".join(np.ravel(chars).tolist()).decode(errors="replace").strip()


def _num(ds, name):
    return np.ma.filled(ds[name][:].astype("f8"), np.nan)


def _values(ds, base, prof, adjusted):
    """(values, qc) for one profile, from *_ADJUSTED when adjusted, else raw."""
    name = f"{base}_ADJUSTED" if adjusted else base
    if name not in ds.variables:
        return None, None
    return _num(ds, name)[prof], _chars(ds, f"{name}_QC")[prof]


def index_files(raw_dir):
    """{(wmo, cycle): path} for ascending core and S files; plus a count of descending."""
    core, bgc, descending = {}, {}, 0
    for path in glob.glob(os.path.join(raw_dir, "**", "*.nc"), recursive=True):
        m = NAME.match(os.path.basename(path))
        if not m:
            continue
        if m.group(4):
            descending += 1
            continue
        kind, key = m.group(1), (int(m.group(2)), int(m.group(3)))
        if kind in ("R", "D"):
            # One core file per cycle is expected; if both exist, delayed mode wins.
            if kind == "D" or key not in core:
                core[key] = path
        else:
            if kind == "SD" or key not in bgc:
                bgc[key] = path
    return core, bgc, descending


def read_core(path):
    """Kept levels of the primary profile, or a string saying why the profile was skipped."""
    with netCDF4.Dataset(path) as ds:
        ds.set_auto_chartostring(False)

        prof = 0
        if "VERTICAL_SAMPLING_SCHEME" in ds.variables:
            schemes = [_text(r) for r in _chars(ds, "VERTICAL_SAMPLING_SCHEME")]
            prof = next((i for i, s in enumerate(schemes) if s.startswith("Primary")), 0)

        if (_chars(ds, "JULD_QC")[prof] not in GOOD_QC
                or _chars(ds, "POSITION_QC")[prof] not in GOOD_QC):
            return "bad time/position QC"

        adjusted = _chars(ds, "DATA_MODE")[prof] in (b"A", b"D")
        pres, pres_qc = _values(ds, "PRES", prof, adjusted)
        temp, temp_qc = _values(ds, "TEMP", prof, adjusted)
        psal, psal_qc = _values(ds, "PSAL", prof, adjusted)
        if psal is None or temp is None or pres is None:
            return "no salinity"

        juld = ds["JULD"]
        assert "1950-01-01" in juld.units, juld.units
        days = float(np.ma.filled(juld[prof], np.nan))
        lat = float(np.ma.filled(ds["LATITUDE"][prof], np.nan))
        lon = float(np.ma.filled(ds["LONGITUDE"][prof], np.nan))
        if not (np.isfinite(days) and np.isfinite(lat) and np.isfinite(lon)):
            return "bad time/position QC"

        keep = (
            np.isfinite(pres) & np.isfinite(temp) & np.isfinite(psal)
            & np.isin(pres_qc, GOOD_QC) & np.isin(temp_qc, GOOD_QC) & np.isin(psal_qc, GOOD_QC)
        )
        if not keep.any():
            return "no good levels"

        return {
            "PRES": pres[keep], "TEMP": temp[keep], "PSAL": psal[keep],
            "raw_pres": _num(ds, "PRES")[prof][keep],
            "LATITUDE": lat, "LONGITUDE": lon,
            "TIME": (days - DAYS_1950_TO_1970) * 86400.0,
        }


def read_bgc(path, core_raw_pres):
    """{column: values aligned to the core levels} for the BGC parameters this S-file has."""
    with netCDF4.Dataset(path) as ds:
        ds.set_auto_chartostring(False)
        params = [_text(r) for r in _chars(ds, "STATION_PARAMETERS")[0]]
        modes = _text(_chars(ds, "PARAMETER_DATA_MODE")[0])

        # Match each S-file level to the core level with the same raw pressure.
        s_pres = _num(ds, "PRES")[0]
        order = np.argsort(core_raw_pres)
        sorted_pres = core_raw_pres[order]
        pos = np.clip(np.searchsorted(sorted_pres, s_pres), 1, len(sorted_pres) - 1) \
            if len(sorted_pres) > 1 else np.zeros(len(s_pres), dtype=int)
        left = np.maximum(pos - 1, 0)
        nearest = np.where(
            np.abs(sorted_pres[left] - s_pres) <= np.abs(sorted_pres[pos] - s_pres), left, pos
        )
        matched = np.abs(sorted_pres[nearest] - s_pres) < PRES_MATCH_DBAR

        out = {}
        for column, source in BGC_SOURCE.items():
            if source not in params:
                continue
            i = params.index(source)
            adjusted = i < len(modes) and modes[i] in "AD"
            values, qc = _values(ds, source, 0, adjusted)
            if values is None:
                continue
            use = matched & np.isfinite(values) & np.isin(qc, GOOD_QC)
            if not use.any():
                continue
            aligned = np.full(len(core_raw_pres), np.nan, dtype="f4")
            aligned[order[nearest[use]]] = values[use]
            out[column] = aligned
        return out


def main():
    parser = argparse.ArgumentParser(description="Flatten GDAC-format Argo profiles into one NetCDF.")
    parser.add_argument("--raw-dir", required=True, help="extracted request folder")
    parser.add_argument("--out", required=True, help="output NetCDF")
    args = parser.parse_args()

    core, bgc, descending = index_files(args.raw_dir)
    if not core:
        print(f"Error: no core profile files (R*/D*.nc) found in {args.raw_dir}/")
        return
    print(f"Found {len(core):,} ascending core profiles and {len(bgc):,} BGC profiles "
          f"in {args.raw_dir}/ (skipping {descending:,} descending files)")

    cols = {name: [] for name, *_ in CORE_SPECS}
    cols.update({k: [] for k in BGC_VARS})
    skipped = {}
    bgc_float_sets = {k: set() for k in BGC_VARS}
    n_profiles = 0

    for i, (key, path) in enumerate(sorted(core.items()), 1):
        prof = read_core(path)
        if isinstance(prof, str):
            skipped[prof] = skipped.get(prof, 0) + 1
            continue

        wmo = key[0]
        n = len(prof["PRES"])
        cols["PRES"].append(prof["PRES"].astype("f4"))
        cols["TEMP"].append(prof["TEMP"].astype("f4"))
        cols["PSAL"].append(prof["PSAL"].astype("f4"))
        cols["LATITUDE"].append(np.full(n, prof["LATITUDE"], dtype="f4"))
        cols["LONGITUDE"].append(np.full(n, prof["LONGITUDE"], dtype="f4"))
        cols["TIME"].append(np.full(n, prof["TIME"], dtype="f8"))
        cols["PLATFORM"].append(np.full(n, wmo, dtype="i4"))

        extra = read_bgc(bgc[key], prof["raw_pres"]) if key in bgc else {}
        for column in BGC_VARS:
            if column in extra:
                bgc_float_sets[column].add(wmo)
                cols[column].append(extra[column])
            else:
                cols[column].append(np.full(n, np.nan, dtype="f4"))

        n_profiles += 1
        if i % 2000 == 0:
            print(f"  ...{i:,}/{len(core):,} profiles")

    for reason, count in sorted(skipped.items()):
        print(f"Skipped {count:,} profiles: {reason}.")
    finish(
        cols, n_profiles, {k: len(v) for k, v in bgc_float_sets.items()}, args.out,
        source_files=f"{n_profiles:,} GDAC core profiles (+ S-files) from {args.raw_dir}/",
        qc_policy=("TEMP/PSAL/PRES QC flags 1-2 only; ADJUSTED values only when DATA_MODE "
                   "(or PARAMETER_DATA_MODE for BGC) is A or D; BGC kept only on core CTD levels"),
    )


if __name__ == "__main__":
    main()
