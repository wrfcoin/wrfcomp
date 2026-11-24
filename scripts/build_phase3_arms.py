"""Build the Phase 3 arms B and C for one year from the frozen v3 stores. Builds only: nothing is scored.

Inputs under the data directory: ``candidates/gefs-station-corrected-v3/<year>-MM.zarr`` (arm A),
``truth/isd_conus_<year>.nc`` and ``citizen/conus/issue_extract_<year>.nc`` (from ``run_phase3_qc.py``).

Outputs: ``candidates/gefs-station-corrected-own/`` and ``candidates/gefs-station-corrected-citizen/``
(one store a month), and under ``phase3/<year>/``: ``build.json`` (paths, content hashes, learned gains,
counts) and ``covered_stations.csv`` with its sha256. Official station ids only; no citizen identifier
or position is written or printed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import xarray as xr

from wrfcomp.baselines.io import write
from wrfcomp.harness.hashing import content_sha256
from wrfcomp.models import citizen_arms as arms
from wrfcomp.models import gefs_station_v2 as v2

A_DIR = "gefs-station-corrected-v3"


def load_extract(path: Path):
    with xr.open_dataset(path, engine="h5netcdf") as ds:
        ds = ds.load()
    names = ds.attrs["variables"].split(",")
    if tuple(names) != arms.VARIABLES or int(ds.attrs["window_minutes"]) != 15:
        raise ValueError("extract does not match the registered rule")
    n = ds.sizes["station"]
    order = np.argsort(ds.row_issue.values, kind="stable")
    issue = ds.row_issue.values[order]
    bounds = np.searchsorted(issue, np.arange(ds.sizes["issue"] + 1))
    columns = {k: ds[k].values[order] for k in ("row_station", "row_var", "row_value", "row_weight")}
    issue_time = ds.issue_time.values.astype(np.int64).astype("datetime64[s]").astype("datetime64[ns]")

    def snapshot(i: int) -> dict:
        rows = slice(bounds[i], bounds[i + 1])
        out = {}
        for v, var in enumerate(arms.VARIABLES):
            value, weight = np.full(n, np.nan), np.zeros(n)
            pick = columns["row_var"][rows] == v
            station = columns["row_station"][rows][pick]
            value[station] = columns["row_value"][rows][pick]
            weight[station] = columns["row_weight"][rows][pick]
            out[var] = (value, weight)
        return out

    return ds, issue_time, snapshot


def publish(parts, directory: Path, period: str) -> dict:
    store = xr.concat(parts, dim="init_time", coords="minimal", compat="identical", join="exact")
    for var in store.variables.values():
        var.encoding.clear()
    dest = directory / f"{period}.zarr"
    if dest.exists():
        with xr.open_zarr(dest, chunks=None) as previous:
            xr.testing.assert_identical(store, previous.load())
    else:
        write(store, dest)
    with xr.open_zarr(dest, chunks=None) as saved:
        return {
            "period": period,
            "path": str(dest),
            "content_sha256": content_sha256(saved),
            "n_init_times": int(saved.sizes["init_time"]),
        }


def build(data: Path, year: int) -> dict:
    start, _ = v2.year_bounds(year)
    extract, issue_time, snapshot = load_extract(data / "citizen" / "conus" / f"issue_extract_{year}.nc")
    months = sorted((data / "candidates" / A_DIR).glob(f"{year}-??.zarr"))
    if not months:
        raise ValueError("no v3 stores for the year")
    with xr.open_dataset(data / "truth" / f"isd_conus_{year}.nc") as archive:
        truth = archive.sel(
            valid_time=slice(np.datetime64(start, "ns"), None)
        ).load()  # nothing from before 1 January
    pairs = arms.neighbour_pairs(
        extract.lat.values,
        extract.lon.values,
        extract.elevation_m.values,
        truth.latitude.values,
        truth.longitude.values,
        truth.elevation_m.values,
    )
    replay = arms.Replay(year, truth, pairs, flagged=extract.ever_copycat.values)
    out = {"year": year, "arm_a": [], "arm_b": [], "arm_c": [], "issues": []}
    n_issues = 0
    for path in months:
        with xr.open_zarr(path, chunks=None) as source:
            a_month = source.load()
        out["arm_a"].append(
            {"period": path.stem, "path": str(path), "content_sha256": content_sha256(a_month)}
        )
        parts_b, parts_c = [], []
        for t in a_month.init_time.values:
            i = int(np.searchsorted(issue_time, t))
            if i >= len(issue_time) or issue_time[i] != t:
                raise ValueError("an issue of the v3 store is not in the citizen extract schedule")
            b, c, diag = replay.issue(a_month.sel(init_time=[t]), snapshot(i))
            parts_b.append(b)
            parts_c.append(c)
            out["issues"].append(asdict(diag))
            n_issues += 1
        out["arm_b"].append(publish(parts_b, data / "candidates" / arms.MODEL_OWN, path.stem))
        out["arm_c"].append(publish(parts_c, data / "candidates" / arms.MODEL_CITIZEN, path.stem))
        print(
            json.dumps(
                {
                    "month": path.stem,
                    "b": out["arm_b"][-1]["content_sha256"],
                    "c": out["arm_c"][-1]["content_sha256"],
                }
            ),
            flush=True,
        )
    if n_issues != len(issue_time):
        raise ValueError("the v3 stores do not cover the year's schedule")
    covered = arms.covered_stations(truth.station_id.values, replay.citizen_defined_count, n_issues)
    folder = data / "phase3" / f"{year}"
    folder.mkdir(parents=True, exist_ok=True)
    text = "station_id\n" + "".join(f"{s}\n" for s in covered)
    (folder / "covered_stations.csv").write_text(text)
    total = sum(d["weight_total"].get("t2m", 0.0) for d in out["issues"])
    flagged = sum(d["weight_flagged"].get("t2m", 0.0) for d in out["issues"])
    out.update(
        n_issues=n_issues,
        citizen_stations=int(extract.sizes["station"]),
        neighbour_pairs=int(pairs.citizen.size),
        covered_count=len(covered),
        covered_sha256=hashlib.sha256(text.encode()).hexdigest(),
        copycat_weight_share_t2m=(flagged / total) if total else None,
    )
    (folder / "build.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2024, choices=(2024,))
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("WRFCOMP_DATA_DIR") or Path.home() / "wrfcomp-data"),
    )
    args = parser.parse_args()
    out = build(args.data_dir.resolve(), args.year)
    print(
        json.dumps(
            {
                k: out[k]
                for k in (
                    "n_issues",
                    "citizen_stations",
                    "neighbour_pairs",
                    "covered_count",
                    "covered_sha256",
                    "copycat_weight_share_t2m",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
