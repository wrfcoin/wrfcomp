"""Run the shipped station QC over the stored CONUS citizen year and keep what Phase 3 needs.

Three steps:

``partition``  daily files -> one file per one-degree tile and month (same anonymised columns).
``run``        per tile: QC every station that can be a neighbour of an official station, and keep,
               for each 00Z/12Z issue time, its latest positive-weight report of the last 15 minutes.
``merge``      one extract file for the forecast builder, and a summary of counts.

Nothing incomplete or stale can be merged. Each product records what it was made from:

- ``partition`` needs a daily file for every day of the year's schedule, each written under the
  current arrival policy. A missing *source hour* is recorded (it is a fact about the archive); a
  missing *day file* is work that has not run, and is refused.
- the tiles carry a digest of the daily files; the QC outputs carry a digest of the tiles, the truth
  file, the GEFS cycles and the QC code. A product with another digest is refused, never reused.
- ``merge`` needs an output for every tile of the manifest. A timing run on some tiles
  (``--only-tiles``) writes to its own folder and cannot be merged.

Logs carry one-degree tile labels, counts and aggregates. No identifier or position is printed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import multiprocessing as mp
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from wrfcomp.assimilation import phase3_qc as p3
from wrfcomp.assimilation.madis_archive import (
    ARRIVAL_POLICY,
    CONTENT_COLUMNS,
    VALUE_COLUMNS,
    arrived_by_next_issue_ns,
    latest_version_rows,
    load_table,
)
from wrfcomp.assimilation.qc import distances_km

NEIGHBOUR_KM = p3.NEIGHBOUR_KM
WINDOW_MIN = p3.WINDOW_MIN
YEAR_END = p3.YEAR_END
EXTRACT_FORMAT = p3.EXTRACT_FORMAT
IncompleteError = p3.IncompleteError
sha256_file, digest_of, read_manifest = p3.sha256_file, p3.digest_of, p3.read_manifest
_G: dict = {}


def issue_times(year: int) -> np.ndarray:
    return p3.issue_times(year)


def expected_days(year: int) -> list[str]:
    """Every UTC day that holds an hour before the year's last issue time."""
    last = dt.datetime.fromisoformat(YEAR_END[year]) - dt.timedelta(hours=1)
    day, out = dt.date(year, 1, 1), []
    while day <= last.date():
        out.append(day.isoformat())
        day += dt.timedelta(days=1)
    return out


def tile_of(lat, lon):
    return np.floor(lat).astype(int), np.floor(lon).astype(int)


# ----------------------------------------------------------------------------- partition


def partition(citizen_dir: Path, year: int) -> dict:
    src, dst = citizen_dir / "conus" / f"{year}", citizen_dir / "conus" / f"{year}-tiles"
    days = expected_days(year)
    absent = [d for d in days if not (src / f"{d}.nc").is_file()]
    if absent:
        raise IncompleteError(
            f"{len(absent)} of {len(days)} day files are missing (first: {absent[0]}); fetch them first"
        )
    start = np.datetime64(f"{year}-01-01T00:00:00", "ns")
    inputs = [(d, sha256_file(src / f"{d}.nc")) for d in days]
    digest = digest_of("tiles", year, ARRIVAL_POLICY, inputs)
    if dst.exists():
        if read_manifest(dst / "manifest.json", "tiles")["digest"] != digest:
            raise IncompleteError(
                "the tiles were made from other daily files; remove them and partition again"
            )
        manifest = p3.verify_tiles(dst)  # every tile-month present and unchanged, or refused
        print("tiles already match the daily files", flush=True)
        return manifest
    building = dst.with_name(dst.name + ".building")
    if building.exists():
        shutil.rmtree(building)  # an interrupted run of this step
    missing_hours, tiles, n_reports, parts = {}, set(), 0, {}
    columns = ("station_id", "obs_time", "received_time", "lat", "lon", "elevation_m", *VALUE_COLUMNS)
    for month in sorted({d[:7] for d in days}):
        tables = []
        for d in (d for d in days if d.startswith(month)):
            table, attrs = load_table(src / f"{d}.nc", with_attrs=True)  # refuses another arrival policy
            if attrs.get("day") != d:
                raise IncompleteError(f"day file {d} does not say which day it holds")
            if attrs.get("hours_missing"):
                missing_hours[d] = str(attrs["hours_missing"]).split(",")
            tables.append(table)
        cols = {name: np.concatenate([getattr(t, name) for t in tables]) for name in columns}
        obs = cols["obs_time"].astype("datetime64[ns]").astype(np.int64)
        received = cols["received_time"].astype("datetime64[ns]").astype(np.int64)
        if np.any(cols["obs_time"] < start) or not arrived_by_next_issue_ns(obs, received).all():
            raise IncompleteError(
                "a daily file holds a report from before the year or one not arrived in time"
            )
        ti, tj = tile_of(cols["lat"].astype(np.float64), cols["lon"].astype(np.float64))
        code = ti * 1000 + tj
        order = np.argsort(code, kind="stable")
        bounds = np.flatnonzero(np.r_[True, code[order][1:] != code[order][:-1], True])
        for a, b in zip(bounds[:-1], bounds[1:], strict=True):
            rows = order[a:b]
            name = f"{ti[rows[0]]:+03d}_{tj[rows[0]]:+04d}"
            folder = building / name
            folder.mkdir(parents=True, exist_ok=True)
            # A station that changes its stored position is a new station (token and position together).
            key = pd.DataFrame({k: cols[k][rows] for k in ("station_id", "lat", "lon", "elevation_m")})
            ident = key.groupby(["station_id", "lat", "lon", "elevation_m"], sort=True).ngroup().to_numpy()
            content = [cols[c][rows] for c in CONTENT_COLUMNS]
            pick = latest_version_rows(ident, obs[rows], received[rows], content)  # by station, time
            rows, ident = rows[pick], ident[pick]
            head = rows[np.r_[True, ident[1:] != ident[:-1]]]
            np.savez_compressed(
                folder / f"{month}.npz",
                ident_token=cols["station_id"][head].astype("S11"),
                ident_lat=cols["lat"][head],
                ident_lon=cols["lon"][head],
                ident_elevation_m=cols["elevation_m"][head],
                row_ident=ident.astype(np.int32),
                obs_time=obs[rows],
                received_time=received[rows],
                **{c: cols[c][rows] for c in VALUE_COLUMNS},
            )
            parts[f"{name}/{month}.npz"] = sha256_file(folder / f"{month}.npz")
            tiles.add(name)
            n_reports += len(rows)
        print(f"partitioned {month}: {len(obs)} reports", flush=True)
    manifest = {
        "digest": digest,
        "year": year,
        "arrival_policy": ARRIVAL_POLICY,
        "days": len(days),
        "missing_source_hours": missing_hours,
        "tiles": sorted(tiles),
        "parts": dict(sorted(parts.items())),
        "reports": n_reports,
    }
    (building / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    building.rename(dst)
    return manifest


# ----------------------------------------------------------------------------- run


def _parts(tiles_dir: Path, ti: int, tj: int) -> list[Path]:
    paths = []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            paths.extend(sorted((tiles_dir / f"{ti + di:+03d}_{tj + dj:+04d}").glob("*.npz")))
    return paths


def write_tile_output(path: Path, run_digest: str, token, lat, lon, height, stats, rows: dict) -> None:
    tmp = path.with_name(f".{path.stem}.part.npz")
    np.savez_compressed(
        tmp,
        run_digest=np.array(run_digest),
        token=np.asarray(token).astype("U11"),
        lat=np.asarray(lat, dtype=np.float32),
        lon=np.asarray(lon, dtype=np.float32),
        elevation_m=np.asarray(height, dtype=np.float32),
        stats=np.asarray(stats, dtype=np.int64).reshape(len(token), 6),
        **{k: np.asarray(v) for k, v in rows.items()},
    )
    tmp.replace(path)


def run_tile(name: str) -> dict:
    ti, tj = int(name[:3]), int(name[4:])
    t0 = time.monotonic()
    out_path = _G["out"] / f"{name}.npz"
    if out_path.exists():
        with np.load(out_path) as z:
            if str(z["run_digest"]) != _G["digest"]:
                raise IncompleteError(f"tile output {name} was made from other inputs")
        return {"tile": name, "skipped": True}
    trial, official, forecasts, issues = _G["trial"], _G["official"], _G["forecasts"], _G["issues"]
    start_ns = int(_G["start"].astype("datetime64[ns]").astype(np.int64))
    # Station tables first; report rows are read only for the stations that are needed.
    index: dict[tuple, int] = {}
    part_ids = []
    paths = _parts(_G["tiles"], ti, tj)
    for path in paths:
        with np.load(path) as z:
            keys = zip(
                z["ident_token"].tolist(),
                z["ident_lat"].tolist(),
                z["ident_lon"].tolist(),
                z["ident_elevation_m"].tolist(),
                strict=True,
            )
            part_ids.append(np.array([index.setdefault(key, len(index)) for key in keys], dtype=np.int64))
    n = len(index)
    table = list(index)
    token = np.array([k[0] for k in table], dtype="S11")
    lat = np.array([k[1] for k in table], dtype=np.float32).astype(np.float64)
    lon = np.array([k[2] for k in table], dtype=np.float32).astype(np.float64)
    height = np.array([k[3] for k in table], dtype=np.float32).astype(np.float64)
    to_official = np.array([distances_km(lat[k], lon[k], official.lat, official.lon).min() for k in range(n)])
    si, sj = tile_of(lat, lon)
    core = np.flatnonzero((si == ti) & (sj == tj) & (to_official <= NEIGHBOUR_KM))
    radius = trial.CFG.radius_km
    near = {}
    for k in core:
        d = distances_km(lat[k], lon[k], lat, lon)
        near[k] = np.flatnonzero((d <= radius) & (np.arange(n) != k))
    needed = sorted(set(core.tolist()) | {int(j) for v in near.values() for j in v})
    wanted = np.zeros(n, dtype=bool)
    wanted[needed] = True
    chunks: dict[int, list] = {k: [] for k in needed}
    for path, ids in zip(paths, part_ids, strict=True):
        if not wanted[ids].any():
            continue
        with np.load(path) as z:
            gid = ids[z["row_ident"]]
            obs, received = z["obs_time"], z["received_time"]
            # Year isolation and Amendment 1 again, on what is actually read.
            ok = wanted[gid] & (obs >= start_ns) & arrived_by_next_issue_ns(obs, received)
            keep = np.flatnonzero(ok)
            gid = gid[keep]
            arrays = [obs[keep], received[keep], *(z[c][keep] for c in VALUE_COLUMNS)]
        if not keep.size:
            continue
        cuts = np.flatnonzero(np.r_[True, gid[1:] != gid[:-1], True])
        for a, b in zip(cuts[:-1], cuts[1:], strict=True):
            chunks[int(gid[a])].append([x[a:b] for x in arrays])
    # A station with no usable report left is not a station.
    needed = [k for k in needed if chunks[k]]
    alive = set(needed)
    core = np.array([k for k in core if k in alive], dtype=int)
    near = {k: np.array([j for j in near[k] if j in alive], dtype=int) for k in core}

    def station(k: int) -> p3.StationData:
        parts = chunks.pop(k)
        cols = [np.concatenate([c[i] for c in parts]) for i in range(2 + len(VALUE_COLUMNS))]
        # A report repeated across months: one version, chosen as everywhere else (all arrived in time).
        pick = latest_version_rows(np.zeros(len(cols[0]), dtype=np.int8), cols[0], cols[1], cols[2:])
        when = cols[0][pick].astype("datetime64[ns]")
        return p3.StationData(lat[k], lon[k], height[k], when, *(c[pick] for c in cols[2:]))

    data, sun, source = {}, {}, {}
    for k in needed:
        data[k] = station(k)
        sun[k] = p3._sun(data[k].times, data[k].lat, data[k].lon, trial.CFG)
        source[k] = p3.station_source(data[k], official, forecasts, trial, sun=sun[k])
    rec = {key: [] for key in ("station", "issue", "var", "value", "weight", "earned", "age_ns")}
    stats = []
    n_reports = 0
    for local, k in enumerate(core):
        result = p3.station_weights(
            data[k],
            [(data[j], source[j][0]) for j in near[k]],
            source[k][1],
            official,
            forecasts,
            trial,
            sun=sun[k],
        )
        row = [0, 0, 0, 0, False, False]
        for v, var in enumerate(p3.THERMAL):
            if var not in result:
                continue
            channel = result[var]
            has, value, weight, earned, age = p3.issue_snapshot(channel, issues, WINDOW_MIN)
            ix = np.flatnonzero(has)
            rec["station"].append(np.full(ix.size, local, dtype=np.int32))
            rec["issue"].append(ix.astype(np.int32))
            rec["var"].append(np.full(ix.size, v, dtype=np.int8))
            rec["value"].append(value[ix].astype(np.float32))
            rec["weight"].append(weight[ix].astype(np.float32))
            rec["earned"].append(earned[ix])
            rec["age_ns"].append(age[ix].astype(np.int64))
            row[v] = len(channel.weights)
            row[2 + v] = int(np.count_nonzero(channel.weights > 0))
            row[4] |= bool(channel.copycat.any())
            row[5] |= bool(channel.earned.any())
            n_reports += len(channel.weights)
        row[4] |= bool(len(source[k][1]))
        stats.append(row)
    write_tile_output(
        out_path,
        _G["digest"],
        token[core].astype("U11"),
        lat[core],
        lon[core],
        height[core],
        stats,
        {k: (np.concatenate(v) if v else np.empty(0)) for k, v in rec.items()},
    )
    return {
        "tile": name,
        "stations": int(len(core)),
        "first_pass_stations": len(needed),
        "channel_reports": n_reports,
        "seconds": round(time.monotonic() - t0, 1),
    }


def run(data_dir: Path, year: int, processes: int, only: int | None) -> None:
    citizen = data_dir / "citizen" / "conus"
    tiles_dir = citizen / f"{year}-tiles"
    manifest = p3.verify_tiles(tiles_dir)  # every tile-month present and unchanged, or refused
    start = np.datetime64(f"{year}-01-01T00", "ns")
    issues = issue_times(year)
    t0 = time.monotonic()
    trial, truth, forecasts = p3.load_year_inputs(data_dir, year)
    inputs = p3.run_record(data_dir, year, manifest, forecasts)
    # A run on some tiles is for timing only. It has its own folder and can never be merged.
    out = citizen / (f"{year}-qc" if only is None else f"{year}-qc.timing")
    out.mkdir(parents=True, exist_ok=True)
    record_path = out / "run.json"
    if record_path.exists() and json.loads(record_path.read_text())["digest"] != inputs["digest"]:
        raise IncompleteError("the QC outputs present were made from other inputs or code; remove them first")
    record_path.write_text(json.dumps(inputs, indent=1) + "\n")
    official = p3.official_from_truth(truth, forecasts, trial)
    print(
        f"official series and {len(forecasts.issued)} forecast entries ready in {time.monotonic() - t0:.0f}s",
        flush=True,
    )
    _G.update(
        trial=trial,
        official=official,
        forecasts=forecasts,
        issues=issues,
        tiles=tiles_dir,
        out=out,
        digest=inputs["digest"],
        start=start,
    )
    tiles = manifest["tiles"] if only is None else manifest["tiles"][:only]
    t0 = time.monotonic()
    total = 0
    with mp.get_context("fork").Pool(processes) as pool, (out / "run.jsonl").open("a") as log:
        for record in pool.imap_unordered(run_tile, tiles):
            log.write(json.dumps(record) + "\n")
            log.flush()
            total += record.get("channel_reports", 0)
    print(
        f"{len(tiles)} tiles, {total} channel reports weighted, "
        f"{time.monotonic() - t0:.0f}s wall, {processes} processes",
        flush=True,
    )


# ----------------------------------------------------------------------------- merge


def write_extract(path: Path, cat: dict, issues: np.ndarray, official_ids, attrs: dict) -> None:
    ds = xr.Dataset(
        {
            "station_token": ("station", np.asarray(cat["token"]).astype("S11")),
            "lat": ("station", np.asarray(cat["lat"], dtype=np.float32)),
            "lon": ("station", np.asarray(cat["lon"], dtype=np.float32)),
            "elevation_m": ("station", np.asarray(cat["elevation_m"], dtype=np.float32)),
            "ever_copycat": ("station", np.asarray(cat["stats"])[:, 4].astype(bool)),
            "ever_credential": ("station", np.asarray(cat["stats"])[:, 5].astype(bool)),
            "row_station": ("row", np.asarray(cat["station"]).astype(np.int32)),
            "row_issue": ("row", np.asarray(cat["issue"]).astype(np.int32)),
            "row_var": ("row", np.asarray(cat["var"]).astype(np.int8)),
            "row_value": ("row", np.asarray(cat["value"]).astype(np.float32)),
            "row_weight": ("row", np.asarray(cat["weight"]).astype(np.float32)),
            "row_credential": ("row", np.asarray(cat["earned"]).astype(bool)),
            "row_age_ns": ("row", np.asarray(cat["age_ns"]).astype(np.int64)),
            "issue_time": ("issue", np.asarray(issues).astype("datetime64[s]").astype(np.int64)),
            "official_station_id": ("official", np.asarray(official_ids).astype("S11")),
        },
        attrs={
            **attrs,
            "format": EXTRACT_FORMAT,
            "variables": ",".join(p3.THERMAL),
            "window_minutes": WINDOW_MIN,
            "arrival_policy": ARRIVAL_POLICY,
            "privacy": "keyed 11-char station tokens; lat/lon rounded to 0.1 deg, elevation to 10 m",
        },
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.part")
    ds.to_netcdf(tmp, engine="h5netcdf", encoding={k: {"zlib": True, "complevel": 4} for k in ds.data_vars})
    tmp.replace(path)


def merge(data_dir: Path, year: int) -> dict:
    citizen = data_dir / "citizen" / "conus"
    out = citizen / f"{year}-qc"
    record = read_manifest(out / "run.json", "QC run")
    current = p3.current_run_record(data_dir, year)  # verifies the tiles, hashes truth and GEFS content
    manifest = read_manifest(citizen / f"{year}-tiles" / "manifest.json", "tiles")
    if record["digest"] != current["digest"]:
        raise IncompleteError("the QC outputs were made from other inputs or code than the ones present")
    absent = [name for name in manifest["tiles"] if not (out / f"{name}.npz").is_file()]
    if absent:
        raise IncompleteError(
            f"{len(absent)} of {len(manifest['tiles'])} tiles have no QC output; the run is not complete"
        )
    names = (
        "token", "lat", "lon", "elevation_m", "stats",
        "station", "issue", "var", "value", "weight", "earned", "age_ns",
    )  # fmt: skip
    parts = {k: [] for k in names}
    offset = 0
    for name in manifest["tiles"]:
        with np.load(out / f"{name}.npz") as z:
            if str(z["run_digest"]) != record["digest"]:
                raise IncompleteError(f"tile output {name} was made from other inputs")
            if not len(z["token"]):
                continue
            for k in names:
                if k != "station":
                    parts[k].append(z[k])
            parts["station"].append(z["station"].astype(np.int64) + offset)
            offset += len(z["token"])
    cat = {k: np.concatenate(v) for k, v in parts.items()}
    with xr.open_dataset(data_dir / "truth" / f"isd_conus_{year}.nc") as ds:
        official_ids = ds.station_id.values.astype(str)
    path = citizen / f"issue_extract_{year}.nc"
    write_extract(
        path,
        cat,
        issue_times(year),
        official_ids,
        {
            "complete": 1,
            "year": year,
            "run_digest": record["digest"],
            "tiles_digest": manifest["digest"],
            "truth_sha256": record["truth_sha256"],
            "gefs_digest": record["gefs_digest"],
            "code_digest": record["code_digest"],
            "tiles": len(manifest["tiles"]),
            "days": manifest["days"],
            "missing_source_hours": json.dumps(manifest["missing_source_hours"], sort_keys=True),
        },
    )
    stats = cat["stats"]
    summary = {
        "year": year,
        "run_digest": record["digest"],
        "extract_sha256": sha256_file(path),
        "stations_weighted": int(len(cat["token"])),
        "distinct_tokens": int(len(set(cat["token"].tolist()))),
        "channel_reports": {"t2m": int(stats[:, 0].sum()), "td2m": int(stats[:, 1].sum())},
        "positive_weight_share": {
            "t2m": float(stats[:, 2].sum() / max(stats[:, 0].sum(), 1)),
            "td2m": float(stats[:, 3].sum() / max(stats[:, 1].sum(), 1)),
        },
        "stations_ever_copycat": int(stats[:, 4].sum()),
        "stations_ever_credential": int(stats[:, 5].sum()),
        "issue_rows": int(len(cat["issue"])),
        "missing_source_hours": sum(len(v) for v in manifest["missing_source_hours"].values()),
        "extract_bytes": path.stat().st_size,
    }
    (citizen / f"issue_extract_{year}.summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("step", choices=("partition", "run", "merge"))
    parser.add_argument("--year", type=int, default=2024, choices=sorted(YEAR_END))
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("WRFCOMP_DATA_DIR") or Path.home() / "wrfcomp-data"),
    )
    parser.add_argument("--processes", type=int, default=8)
    parser.add_argument(
        "--only-tiles", type=int, default=None, help="timing run on the first N tiles; never merged"
    )
    args = parser.parse_args()
    if args.step == "partition":
        partition(args.data_dir / "citizen", args.year)
    elif args.step == "run":
        run(args.data_dir, args.year, args.processes, args.only_tiles)
    else:
        merge(args.data_dir, args.year)


if __name__ == "__main__":
    main()
