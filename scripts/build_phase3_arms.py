"""Build the Phase 3 arms B and C for one year from the frozen v3 stores. Builds only: nothing is scored.

Inputs under the data directory: ``candidates/gefs-station-corrected-v3/<year>-MM.zarr`` (arm A),
``truth/isd_conus_<year>.nc`` and ``citizen/conus/issue_extract_<year>.nc`` (from ``run_phase3_qc.py``).

Everything is checked before anything is written: the extract must be complete and made under the
current arrival policy, and the extract, the truth file and the v3 stores must all be on exactly the
year's issue schedule and the stations of ``configs/stations_conus.csv``.

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
import pandas as pd
import xarray as xr

from wrfcomp.assimilation import phase3_qc as p3
from wrfcomp.assimilation.madis_archive import ARRIVAL_POLICY
from wrfcomp.baselines.io import CONFIG, write
from wrfcomp.harness.config import load_config
from wrfcomp.harness.hashing import content_sha256
from wrfcomp.models import citizen_arms as arms
from wrfcomp.models import gefs_station_v2 as v2

A_DIR = "gefs-station-corrected-v3"


class BuildRefusedError(RuntimeError):
    """An input is incomplete, stale, or not on the registered schedule or station list."""


def registered_stations() -> list[str]:
    path = CONFIG.parent / Path(load_config(CONFIG).evaluation.stations_file).name
    return pd.read_csv(path, dtype=str, usecols=["station_id"])["station_id"].tolist()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_extract(path: Path, year: int, issues: np.ndarray, stations: list[str]):
    if not path.is_file():
        raise BuildRefusedError("there is no citizen extract for the year")
    with xr.open_dataset(path, engine="h5netcdf") as ds:
        ds = ds.load()
    a = ds.attrs
    if a.get("format") != p3.EXTRACT_FORMAT or a.get("arrival_policy") != ARRIVAL_POLICY:
        raise BuildRefusedError("the extract was not made under the current format and arrival policy")
    if int(a.get("complete", 0)) != 1 or int(a.get("year", 0)) != year or not a.get("run_digest"):
        raise BuildRefusedError("the extract is not marked as a complete QC run of this year")
    if (
        tuple(str(a.get("variables", "")).split(",")) != arms.VARIABLES
        or int(a.get("window_minutes", 0)) != 15
    ):
        raise BuildRefusedError("the extract does not match the registered rule")
    issue_time = ds.issue_time.values.astype(np.int64).astype("datetime64[s]").astype("datetime64[ns]")
    if not np.array_equal(issue_time, np.asarray(issues, dtype="datetime64[ns]")):
        raise BuildRefusedError("the extract is not on the year's issue schedule")
    if ds.official_station_id.values.astype(str).tolist() != list(stations):
        raise BuildRefusedError("the extract was not made for the registered official stations")
    n = ds.sizes["station"]
    order = np.argsort(ds.row_issue.values, kind="stable")
    issue = ds.row_issue.values[order]
    bounds = np.searchsorted(issue, np.arange(ds.sizes["issue"] + 1))
    names = ("row_station", "row_var", "row_value", "row_weight", "row_age_ns")
    columns = {k: ds[k].values[order] for k in names}

    def snapshot(i: int) -> dict:
        rows = slice(bounds[i], bounds[i + 1])
        out = {}
        for v, var in enumerate(arms.VARIABLES):
            value, weight, age = np.full(n, np.nan), np.zeros(n), np.zeros(n, dtype=np.int64)
            pick = columns["row_var"][rows] == v
            station = columns["row_station"][rows][pick]
            value[station] = columns["row_value"][rows][pick]
            weight[station] = columns["row_weight"][rows][pick]
            age[station] = columns["row_age_ns"][rows][pick]
            out[var] = (value, weight, age)
        return out

    return ds, snapshot


def verify_extract_is_current(data: Path, year: int, extract_path: Path, extract: xr.Dataset) -> None:
    """Refuse an extract that is not the published one or was not made from the files now present.

    Old QC weights must never be combined with other official readings, other GEFS inputs, other
    citizen tiles or other QC code than the ones they were computed with.
    """
    summary = extract_path.with_name(f"issue_extract_{year}.summary.json")
    if not summary.is_file() or json.loads(summary.read_text()).get("extract_sha256") != sha256_file(
        extract_path
    ):
        raise BuildRefusedError("the extract is not the one its published summary describes")
    if extract.attrs.get("truth_sha256") != sha256_file(data / "truth" / f"isd_conus_{year}.nc"):
        raise BuildRefusedError("the official readings have changed since the QC that made the extract")
    try:
        current = p3.current_run_record(data, year)
    except p3.IncompleteError as exc:
        raise BuildRefusedError(f"the QC inputs behind the extract are not intact: {exc}") from None
    if current["digest"] != extract.attrs.get("run_digest"):
        raise BuildRefusedError(
            "the extract was not made from the tiles, truth, GEFS and QC code now present"
        )


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


def build(data: Path, year: int, *, issues=None, stations=None) -> dict:
    """``issues`` and ``stations`` default to the registered schedule and stations; tests pass their own."""
    start, _ = v2.year_bounds(year)
    start = np.datetime64(start, "ns")
    issues = p3.issue_times(year) if issues is None else np.asarray(issues, dtype="datetime64[ns]")
    stations = registered_stations() if stations is None else list(stations)
    extract_path = data / "citizen" / "conus" / f"issue_extract_{year}.nc"
    extract, snapshot = load_extract(extract_path, year, issues, stations)
    verify_extract_is_current(data, year, extract_path, extract)
    months = sorted((data / "candidates" / A_DIR).glob(f"{year}-??.zarr"))
    seen = []
    for path in months:
        with xr.open_zarr(path, chunks=None) as source:
            if source.station_id.values.astype(str).tolist() != stations:
                raise BuildRefusedError(f"v3 store {path.stem} is not on the registered stations")
            seen.append(source.init_time.values.astype("datetime64[ns]"))
    if not seen or not np.array_equal(np.concatenate(seen), issues):
        raise BuildRefusedError("the v3 stores are not exactly the year's issue schedule")
    with xr.open_dataset(data / "truth" / f"isd_conus_{year}.nc") as archive:
        # No label before 1 January; a reading measured before it is refused where it is read.
        truth = archive.sel(valid_time=slice(start, None)).load()
    if truth.station_id.values.astype(str).tolist() != stations:
        raise BuildRefusedError("the truth file is not on the registered stations")
    pairs = arms.neighbour_pairs(
        extract.lat.values,
        extract.lon.values,
        extract.elevation_m.values,
        truth.latitude.values,
        truth.longitude.values,
        truth.elevation_m.values,
    )
    replay = arms.Replay(
        year, truth, pairs, flagged=extract.ever_copycat.values, token=extract.station_token.values
    )
    out = {
        "year": year,
        "extract_sha256": sha256_file(extract_path),
        "extract_run_digest": str(extract.attrs["run_digest"]),
        "arrival_policy": ARRIVAL_POLICY,
        "arm_a": [],
        "arm_b": [],
        "arm_c": [],
        "issues": [],
    }
    i = 0
    for path in months:
        with xr.open_zarr(path, chunks=None) as source:
            a_month = source.load()
        out["arm_a"].append(
            {"period": path.stem, "path": str(path), "content_sha256": content_sha256(a_month)}
        )
        parts_b, parts_c = [], []
        for t in a_month.init_time.values:
            b, c, diag = replay.issue(a_month.sel(init_time=[t]), snapshot(i))
            parts_b.append(b)
            parts_c.append(c)
            out["issues"].append(asdict(diag))
            i += 1
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
    covered = arms.covered_stations(truth.station_id.values, replay.citizen_defined_count, len(issues))
    folder = data / "phase3" / f"{year}"
    folder.mkdir(parents=True, exist_ok=True)
    text = "station_id\n" + "".join(f"{s}\n" for s in covered)
    (folder / "covered_stations.csv").write_text(text)
    total = sum(d["weight_total"].get("t2m", 0.0) for d in out["issues"])
    flagged = sum(d["weight_flagged"].get("t2m", 0.0) for d in out["issues"])
    out.update(
        n_issues=len(issues),
        official_stations=len(stations),
        citizen_station_rows=int(extract.sizes["station"]),
        citizen_tokens=int(len(set(extract.station_token.values.tolist()))),
        neighbour_pairs=int(pairs.citizen.size),
        covered_count=len(covered),
        covered_sha256=hashlib.sha256(text.encode()).hexdigest(),
        copycat_weight_share_t2m=(flagged / total) if total else None,
    )
    (folder / "build.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--year", type=int, default=2024, choices=(2024,))
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("WRFCOMP_DATA_DIR") or Path.home() / "wrfcomp-data"),
    )
    args = parser.parse_args()
    out = build(args.data_dir.resolve(), args.year)
    keys = (
        "n_issues", "official_stations", "citizen_station_rows", "citizen_tokens", "neighbour_pairs",
        "covered_count", "covered_sha256", "copycat_weight_share_t2m", "extract_sha256",
    )  # fmt: skip
    print(json.dumps({k: out[k] for k in keys}))


if __name__ == "__main__":
    main()
