"""Build candidate v4 for one year: Phase 3 arm B (v3 plus the airport's own reading), and nothing else.

v4 is arm B of ``docs/PHASE3_DESIGN.md`` (rule 1.2), built by the same code that built and scored it:
``wrfcomp.models.citizen_arms.Replay`` and the publishing, station and schedule checks of
``scripts/build_phase3_arms.py``. Arm B uses official data only, so this script needs no citizen extract
and never opens a citizen file: the replay is given no citizen station and an empty snapshot, which
leaves B unchanged (B never reads the neighbourhood term; arm C is computed and thrown away).

Inputs under the data directory: ``candidates/gefs-station-corrected-v3/<year>-MM.zarr`` and
``truth/isd_conus_<year>.nc``. Everything is checked before anything is written: the v3 stores and
the truth file must be on exactly the year's issue schedule and the stations of
``configs/stations_conus.csv``.

Outputs, in ``--out`` (default ``candidates/gefs-station-corrected-v4/`` under the data directory): one
store a month, with the B stores' own ``model_id`` and ``model_version``, and ``build.json`` (input and
output paths with content hashes, learned gains per issue). Nothing is scored.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import xarray as xr

from wrfcomp.assimilation import phase3_qc as p3
from wrfcomp.harness.hashing import content_sha256
from wrfcomp.harness.metrics import evaluation_init_times
from wrfcomp.models import citizen_arms as arms
from wrfcomp.models import gefs_station_v2 as v2


def _phase3_builder():
    path = Path(__file__).resolve().with_name("build_phase3_arms.py")
    spec = importlib.util.spec_from_file_location("wrfcomp_build_phase3_arms", path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load scripts/build_phase3_arms.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


phase3 = _phase3_builder()
BuildRefusedError = phase3.BuildRefusedError
A_DIR = phase3.A_DIR
OUT_DIR = "gefs-station-corrected-v4"
# The last issue of each year that may be built. 2024 is the Phase 3 schedule; 2023 ends at
# 2023-12-15T12Z as accepted for the confirmation period (docs/CONFIRMATION_2023.md, section 1).
LAST_ISSUE = {2023: "2023-12-15T12", **p3.YEAR_END}


def issue_times(year: int) -> np.ndarray:
    """The 00Z and 12Z issue times of the year's schedule."""
    if year in p3.YEAR_END:
        return p3.issue_times(year)
    return evaluation_init_times(np.datetime64(f"{year}-01-01T00"), np.datetime64(LAST_ISSUE[year]), [0, 12])


def no_citizen_snapshot() -> dict:
    """Arm B reads no citizen report: zero citizen station rows at every issue."""
    return {v: (np.empty(0), np.empty(0), np.empty(0, dtype=np.int64)) for v in arms.VARIABLES}


NO_PAIRS = arms.Pairs(
    citizen=np.empty(0, int), official=np.empty(0, int), distance_km=np.empty(0), cell=np.empty(0, int)
)


def code_state() -> dict:
    root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"code_commit": None, "clean_tree": None}
    return {"code_commit": commit, "clean_tree": status == ""}


def build(data: Path, year: int, out_dir: Path | None = None, *, issues=None, stations=None) -> dict:
    """``issues`` and ``stations`` default to the registered schedule and stations; tests pass their own."""
    start, _ = v2.year_bounds(year)
    start = np.datetime64(start, "ns")
    issues = issue_times(year) if issues is None else np.asarray(issues, dtype="datetime64[ns]")
    stations = phase3.registered_stations() if stations is None else list(stations)
    out_dir = data / "candidates" / OUT_DIR if out_dir is None else out_dir
    months = sorted((data / "candidates" / A_DIR).glob(f"{year}-??.zarr"))
    seen = []
    for path in months:
        with xr.open_zarr(path, chunks=None) as source:
            if source.station_id.values.astype(str).tolist() != stations:
                raise BuildRefusedError(f"v3 store {path.stem} is not on the registered stations")
            seen.append(source.init_time.values.astype("datetime64[ns]"))
    if not seen or not np.array_equal(np.concatenate(seen), issues):
        raise BuildRefusedError("the v3 stores are not exactly the year's issue schedule")
    truth_path = data / "truth" / f"isd_conus_{year}.nc"
    if not truth_path.is_file():
        raise BuildRefusedError("there is no truth file for the year")
    with xr.open_dataset(truth_path) as archive:
        # No label before 1 January; a reading measured before it is refused where it is read.
        truth = archive.sel(valid_time=slice(start, None)).load()
    if truth.station_id.values.astype(str).tolist() != stations:
        raise BuildRefusedError("the truth file is not on the registered stations")
    replay = arms.Replay(year, truth, NO_PAIRS)
    out = {
        "year": year,
        "candidate": "v4 = Phase 3 arm B (v3 + the airport's own reading)",
        "model_id": arms.MODEL_OWN,
        "model_version": arms.VERSION,
        **code_state(),
        "truth": {"path": str(truth_path), "sha256": phase3.sha256_file(truth_path)},
        "v3": [],
        "v4": [],
        "issues": [],
    }
    i = 0
    for path in months:
        with xr.open_zarr(path, chunks=None) as source:
            a_month = source.load()
        out["v3"].append({"period": path.stem, "path": str(path), "content_sha256": content_sha256(a_month)})
        parts = []
        for t in a_month.init_time.values:
            b, _, diag = replay.issue(a_month.sel(init_time=[t]), no_citizen_snapshot())
            parts.append(b)
            out["issues"].append({"init": diag.init, "phi": diag.phi, "own_defined": diag.own_defined})
            i += 1
        out["v4"].append(phase3.publish(parts, out_dir, path.stem))
        print(json.dumps({"month": path.stem, "v4": out["v4"][-1]["content_sha256"]}), flush=True)
    out.update(n_issues=i, official_stations=len(stations))
    (out_dir / "build.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--year", type=int, default=2024, choices=sorted(LAST_ISSUE))
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("WRFCOMP_DATA_DIR") or Path.home() / "wrfcomp-data"),
    )
    parser.add_argument("--out", type=Path, default=None, help=f"default: <data-dir>/candidates/{OUT_DIR}")
    args = parser.parse_args()
    data = args.data_dir.resolve()
    out = build(data, args.year, None if args.out is None else args.out.resolve())
    print(
        json.dumps({k: out[k] for k in ("year", "n_issues", "official_stations", "model_id", "code_commit")})
    )


if __name__ == "__main__":
    main()
