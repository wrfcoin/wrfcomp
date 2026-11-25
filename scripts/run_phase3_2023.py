"""Schedule-only wrapper for Design 1 standard QC & B/C1 builds (2023 + 2024 isolated replay).

Leaves all frozen Design 1 files at ``ec736ee`` (the seven ``phase3_qc.QC_CODE`` files,
``build_phase3_arms.py``, and ``citizen_arms.py``) 100% untouched so 2024's ``code_digest()``
and ``issue_extract_2024.nc`` remain valid without regenerating 2024 standard QC.

Supports:
- ``--year 2024 --stop-after-month 2024-02 --out-dir <scratch>``: replays Design 1 arms B and C1
  into isolated scratch from cold January 1 start and compares ``content_sha256`` against the
  already-scored read-only ``phase3/2024/build.json`` (Stage 2a pre-fetch parity proof).
- ``--year 2023``: builds 2023 standard B and C1 on the registered 698-issue schedule
  (``2023-01-01T00Z`` to ``2023-12-15T12Z``) once the 2023 extract exists.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import xarray as xr

from wrfcomp.assimilation import phase3_qc as p3
from wrfcomp.assimilation import phase3_qc_loao as loao_qc
from wrfcomp.assimilation.madis_archive import ARRIVAL_POLICY
from wrfcomp.models import citizen_arms as arms
from wrfcomp.models import gefs_station_v2 as v2


def _load_frozen_builder():
    path = Path(__file__).resolve().with_name("build_phase3_arms.py")
    spec = importlib.util.spec_from_file_location("wrfcomp_build_phase3_arms_w", path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load scripts/build_phase3_arms.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


phase3_frozen = _load_frozen_builder()
BuildRefusedError = phase3_frozen.BuildRefusedError


def wrapper_digest() -> str:
    return p3.digest_of(
        "phase3-2023-wrapper",
        p3.code_digest(),
        p3.sha256_file(Path(__file__).resolve()),
    )


def build_design1_arms(
    data: Path,
    year: int,
    *,
    out_dir: Path,
    stop_after_month: str | None = None,
    issues=None,
    stations=None,
    verify_extract: bool = True,
) -> dict:
    """Replay untouched ``citizen_arms.Replay`` into isolated ``out_dir`` (read-only live stores)."""
    start, _ = v2.year_bounds(year)
    start = np.datetime64(start, "ns")
    issues = loao_qc.issue_times(year) if issues is None else np.asarray(issues, dtype="datetime64[ns]")
    stations = phase3_frozen.registered_stations() if stations is None else list(stations)
    extract_path = data / "citizen" / "conus" / f"issue_extract_{year}.nc"
    extract, snapshot = phase3_frozen.load_extract(extract_path, year, issues, stations)
    if verify_extract and year == 2024:
        phase3_frozen.verify_extract_is_current(data, year, extract_path, extract)
    months = sorted((data / "candidates" / phase3_frozen.A_DIR).glob(f"{year}-??.zarr"))
    seen = []
    for path in months:
        with xr.open_zarr(path, chunks=None) as source:
            if source.station_id.values.astype(str).tolist() != stations:
                raise BuildRefusedError(f"v3 store {path.stem} is not on the registered stations")
            seen.append(source.init_time.values.astype("datetime64[ns]"))
    if not seen or not np.array_equal(np.concatenate(seen), issues):
        raise BuildRefusedError("the v3 stores are not exactly the year's issue schedule")
    with xr.open_dataset(data / "truth" / f"isd_conus_{year}.nc") as archive:
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
        "stop_after_month": stop_after_month,
        "frozen_code_digest": p3.code_digest(),
        "wrapper_digest": wrapper_digest(),
        "extract_sha256": p3.sha256_file(extract_path),
        "arrival_policy": ARRIVAL_POLICY,
        "arm_b": [],
        "arm_c": [],
        "issues": [],
    }
    i = 0
    for path in months:
        if stop_after_month is not None and path.stem > stop_after_month:
            break
        with xr.open_zarr(path, chunks=None) as source:
            a_month = source.load()
        parts_b, parts_c = [], []
        for t in a_month.init_time.values:
            b, c, diag = replay.issue(a_month.sel(init_time=[t]), snapshot(i))
            parts_b.append(b)
            parts_c.append(c)
            out["issues"].append(asdict(diag))
            i += 1
        out["arm_b"].append(phase3_frozen.publish(parts_b, out_dir / arms.MODEL_OWN, path.stem))
        out["arm_c"].append(phase3_frozen.publish(parts_c, out_dir / arms.MODEL_CITIZEN, path.stem))
    out["n_issues"] = i
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "build.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def verify_against_build_json(replayed: dict, reference_build_json: Path) -> dict:
    """Compare replayed monthly B/C1 content hashes against read-only ``phase3/2024/build.json``."""
    ref = json.loads(reference_build_json.read_text())
    ref_b = {m["period"]: m["content_sha256"] for m in ref["arm_b"]}
    ref_c = {m["period"]: m["content_sha256"] for m in ref["arm_c"]}
    mismatches = []
    for entry in replayed["arm_b"]:
        if ref_b.get(entry["period"]) != entry["content_sha256"]:
            mismatches.append(f"arm_b {entry['period']}: hash mismatch")
    for entry in replayed["arm_c"]:
        if ref_c.get(entry["period"]) != entry["content_sha256"]:
            mismatches.append(f"arm_c {entry['period']}: hash mismatch")
    if mismatches:
        raise BuildRefusedError("; ".join(mismatches))
    return {
        "verified_months": [m["period"] for m in replayed["arm_b"]],
        "arm_b_sha256": {m["period"]: m["content_sha256"] for m in replayed["arm_b"]},
        "arm_c_sha256": {m["period"]: m["content_sha256"] for m in replayed["arm_c"]},
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--year", type=int, default=2024, choices=sorted(loao_qc.LAST_ISSUE))
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("WRFCOMP_DATA_DIR") or Path.home() / "wrfcomp-data"),
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="isolated output directory")
    parser.add_argument("--stop-after-month", type=str, default=None, help="stop after YYYY-MM")
    parser.add_argument(
        "--verify-against",
        type=Path,
        default=None,
        help="read-only reference build.json (e.g. phase3/2024/build.json) for hash-parity check",
    )
    args = parser.parse_args()
    out = build_design1_arms(
        args.data_dir.resolve(),
        args.year,
        out_dir=args.out_dir.resolve(),
        stop_after_month=args.stop_after_month,
    )
    if args.verify_against is not None:
        summary = verify_against_build_json(out, args.verify_against.resolve())
        print(json.dumps(summary), flush=True)
    else:
        print(
            json.dumps(
                {
                    "year": out["year"],
                    "n_issues": out["n_issues"],
                    "frozen_code_digest": out["frozen_code_digest"],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
