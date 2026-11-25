"""Run the official-free citizen QC pass for Phase 3 Design 2 Track B over existing CONUS tiles.

Two steps:

``run``    per tile: run :mod:`wrfcomp.assimilation.phase3_qc_loao` (zero ISD official readings
           loaded) on every citizen station within 50 km of any static station coordinate, and
           keep each 00Z/12Z issue's latest positive-weight report of the last 15 minutes.
``merge``  merge all tile outputs into ``citizen/conus/issue_extract_loao_<year>.nc``
           (format ``phase3-extract-loao.v1``) and write an aggregate-only summary JSON.

Logs carry opaque tile ordinals, counts, and aggregates only. No citizen identifier or
coordinate-derived tile label is ever printed or logged.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import time
from pathlib import Path

import numpy as np
import xarray as xr

from wrfcomp.assimilation import phase3_qc as p3
from wrfcomp.assimilation import phase3_qc_loao as loao
from wrfcomp.assimilation.madis_archive import (
    ARRIVAL_POLICY,
    VALUE_COLUMNS,
    arrived_by_next_issue_ns,
    latest_version_rows,
)
from wrfcomp.assimilation.qc import distances_km

STOP_FREE_BYTES = 40 * 10**9
_G: dict = {}


def check_free_space(data_dir: Path, min_free_bytes: int = STOP_FREE_BYTES) -> None:
    if min_free_bytes <= 0:
        return
    free = shutil.disk_usage(data_dir).free
    if free < min_free_bytes:
        raise RuntimeError(f"free space {free / 1e9:.1f} GB is below {min_free_bytes / 1e9:.1f} GB stop line")


def _tile_of(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.floor(lat).astype(int), np.floor(lon).astype(int)


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
        stats=np.asarray(stats, dtype=np.int64).reshape(len(token), 4),
        **{k: np.asarray(v) for k, v in rows.items()},
    )
    tmp.replace(path)


def run_tile(task: tuple[int, str]) -> dict:
    tile_ordinal, name = task
    ti, tj = int(name[:3]), int(name[4:])
    t0 = time.monotonic()
    out_path = _G["out"] / f"{name}.npz"
    if out_path.exists():
        with np.load(out_path) as z:
            if str(z["run_digest"]) != _G["digest"]:
                raise p3.IncompleteError(f"tile ordinal {tile_ordinal} output was made from other inputs")
        return {"tile_ordinal": tile_ordinal, "skipped": True}
    trial, sites, forecasts, issues = _G["trial"], _G["sites"], _G["forecasts"], _G["issues"]
    start_ns = int(_G["start"].astype("datetime64[ns]").astype(np.int64))
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
    to_site = np.array([distances_km(lat[k], lon[k], sites.lat, sites.lon).min() for k in range(n)])
    si, sj = _tile_of(lat, lon)
    core = np.flatnonzero((si == ti) & (sj == tj) & (to_site <= loao.NEIGHBOUR_KM))
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
            ok = wanted[gid] & (obs >= start_ns) & arrived_by_next_issue_ns(obs, received)
            keep = np.flatnonzero(ok)
            gid = gid[keep]
            arrays = [obs[keep], received[keep], *(z[c][keep] for c in VALUE_COLUMNS)]
        if not keep.size:
            continue
        cuts = np.flatnonzero(np.r_[True, gid[1:] != gid[:-1], True])
        for a, b in zip(cuts[:-1], cuts[1:], strict=True):
            chunks[int(gid[a])].append([x[a:b] for x in arrays])
    needed = [k for k in needed if chunks[k]]
    alive = set(needed)
    core = np.array([k for k in core if k in alive], dtype=int)
    near = {k: np.array([j for j in near[k] if j in alive], dtype=int) for k in core}

    def station(k: int) -> p3.StationData:
        parts = chunks.pop(k)
        cols = [np.concatenate([c[i] for c in parts]) for i in range(2 + len(VALUE_COLUMNS))]
        pick = latest_version_rows(np.zeros(len(cols[0]), dtype=np.int8), cols[0], cols[1], cols[2:])
        when = cols[0][pick].astype("datetime64[ns]")
        return p3.StationData(lat[k], lon[k], height[k], when, *(c[pick] for c in cols[2:]))

    data, sun, source = {}, {}, {}
    for k in needed:
        data[k] = station(k)
        sun[k] = p3._sun(data[k].times, data[k].lat, data[k].lon, trial.CFG)
        source[k] = loao.station_source_loao(data[k], sites, forecasts, trial)
    rec = {key: [] for key in ("station", "issue", "var", "value", "weight", "age_ns")}
    stats = []
    n_reports = 0
    for local, k in enumerate(core):
        result = loao.station_weights_loao(
            data[k],
            [(data[j], source[j]) for j in near[k]],
            sites,
            forecasts,
            trial,
            sun=sun[k],
        )
        row = [0, 0, 0, 0]
        for v, var in enumerate(loao.THERMAL):
            if var not in result:
                continue
            channel = result[var]
            has, value, weight, _, age = p3.issue_snapshot(channel, issues, loao.WINDOW_MIN)
            ix = np.flatnonzero(has)
            rec["station"].append(np.full(ix.size, local, dtype=np.int32))
            rec["issue"].append(ix.astype(np.int32))
            rec["var"].append(np.full(ix.size, v, dtype=np.int8))
            rec["value"].append(value[ix].astype(np.float32))
            rec["weight"].append(weight[ix].astype(np.float32))
            rec["age_ns"].append(age[ix].astype(np.int64))
            row[v] = len(channel.weights)
            row[2 + v] = int(np.count_nonzero(channel.weights > 0))
            n_reports += len(channel.weights)
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
        "tile_ordinal": tile_ordinal,
        "stations": int(len(core)),
        "first_pass_stations": len(needed),
        "channel_reports": n_reports,
        "seconds": round(time.monotonic() - t0, 1),
    }


def run(
    data_dir: Path,
    year: int,
    processes: int,
    only: int | None,
    *,
    stations_csv: Path = loao.STATIONS_CSV,
    min_free_bytes: int = STOP_FREE_BYTES,
) -> None:
    check_free_space(data_dir, min_free_bytes)
    citizen = data_dir / "citizen" / "conus"
    tiles_dir = citizen / f"{year}-tiles"
    manifest = p3.verify_tiles(tiles_dir)
    start = np.datetime64(f"{year}-01-01T00", "ns")
    issues = loao.issue_times(year)
    t0 = time.monotonic()
    trial, sites, forecasts = loao.load_loao_inputs(data_dir, year, stations_csv=stations_csv)
    inputs = loao.run_record(year, manifest, forecasts, stations_csv=stations_csv)
    out = citizen / (f"{year}-qc-loao" if only is None else f"{year}-qc-loao.timing")
    out.mkdir(parents=True, exist_ok=True)
    record_path = out / "run.json"
    if record_path.exists() and json.loads(record_path.read_text())["digest"] != inputs["digest"]:
        raise p3.IncompleteError(
            "the LOAO QC outputs present were made from other inputs or code; remove them first"
        )
    record_path.write_text(json.dumps(inputs, indent=1) + "\n")
    print(
        f"static sites ({len(sites.station_id)}) and {len(forecasts.issued)} forecast entries "
        f"ready in {time.monotonic() - t0:.0f}s (no ISD truth loaded)",
        flush=True,
    )
    _G.update(
        trial=trial,
        sites=sites,
        forecasts=forecasts,
        issues=issues,
        tiles=tiles_dir,
        out=out,
        digest=inputs["digest"],
        start=start,
    )
    tiles = manifest["tiles"] if only is None else manifest["tiles"][:only]
    tasks = list(enumerate(tiles))
    t0 = time.monotonic()
    total = 0
    with mp.get_context("fork").Pool(processes) as pool, (out / "run.jsonl").open("a") as log:
        for record in pool.imap_unordered(run_tile, tasks):
            log.write(json.dumps(record) + "\n")
            log.flush()
            total += record.get("channel_reports", 0)
    print(
        f"{len(tiles)} tiles, {total} channel reports weighted, "
        f"{time.monotonic() - t0:.0f}s wall, {processes} processes",
        flush=True,
    )


def write_extract(path: Path, cat: dict, issues: np.ndarray, official_ids, attrs: dict) -> None:
    ds = xr.Dataset(
        {
            "station_token": ("station", np.asarray(cat["token"]).astype("S11")),
            "lat": ("station", np.asarray(cat["lat"], dtype=np.float32)),
            "lon": ("station", np.asarray(cat["lon"], dtype=np.float32)),
            "elevation_m": ("station", np.asarray(cat["elevation_m"], dtype=np.float32)),
            "row_station": ("row", np.asarray(cat["station"]).astype(np.int32)),
            "row_issue": ("row", np.asarray(cat["issue"]).astype(np.int32)),
            "row_var": ("row", np.asarray(cat["var"]).astype(np.int8)),
            "row_value": ("row", np.asarray(cat["value"]).astype(np.float32)),
            "row_weight": ("row", np.asarray(cat["weight"]).astype(np.float32)),
            "row_age_ns": ("row", np.asarray(cat["age_ns"]).astype(np.int64)),
            "issue_time": ("issue", np.asarray(issues).astype("datetime64[s]").astype(np.int64)),
            "official_station_id": ("official", np.asarray(official_ids).astype("S11")),
        },
        attrs={
            **attrs,
            "format": loao.EXTRACT_FORMAT,
            "variables": ",".join(loao.THERMAL),
            "window_minutes": loao.WINDOW_MIN,
            "arrival_policy": ARRIVAL_POLICY,
            "official_observations_used": 0,
            "privacy": "keyed 11-char station tokens; lat/lon rounded to 0.1 deg, elevation to 10 m",
        },
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.part")
    ds.to_netcdf(tmp, engine="h5netcdf", encoding={k: {"zlib": True, "complevel": 4} for k in ds.data_vars})
    tmp.replace(path)


def merge(
    data_dir: Path,
    year: int,
    *,
    stations_csv: Path = loao.STATIONS_CSV,
    min_free_bytes: int = STOP_FREE_BYTES,
) -> dict:
    check_free_space(data_dir, min_free_bytes)
    citizen = data_dir / "citizen" / "conus"
    out = citizen / f"{year}-qc-loao"
    record = p3.read_manifest(out / "run.json", "LOAO QC run")
    current = loao.current_run_record(data_dir, year, stations_csv=stations_csv)
    manifest = p3.read_manifest(citizen / f"{year}-tiles" / "manifest.json", "tiles")
    if record["digest"] != current["digest"]:
        raise p3.IncompleteError(
            "the LOAO QC outputs were made from other inputs or code than the ones present"
        )
    absent = [name for name in manifest["tiles"] if not (out / f"{name}.npz").is_file()]
    if absent:
        raise p3.IncompleteError(
            f"{len(absent)} of {len(manifest['tiles'])} tiles have no LOAO QC output; run is not complete"
        )
    names = (
        "token",
        "lat",
        "lon",
        "elevation_m",
        "stats",
        "station",
        "issue",
        "var",
        "value",
        "weight",
        "age_ns",
    )
    parts: dict[str, list] = {k: [] for k in names}
    offset = 0
    for tile_ordinal, name in enumerate(manifest["tiles"]):
        with np.load(out / f"{name}.npz") as z:
            if str(z["run_digest"]) != record["digest"]:
                raise p3.IncompleteError(f"tile ordinal {tile_ordinal} output was made from other inputs")
            if not len(z["token"]):
                continue
            for k in names:
                if k != "station":
                    parts[k].append(z[k])
            parts["station"].append(z["station"].astype(np.int64) + offset)
            offset += len(z["token"])
    cat = {k: np.concatenate(v) if v else np.empty(0) for k, v in parts.items()}
    sites = loao.load_static_sites(stations_csv)
    path = citizen / f"issue_extract_loao_{year}.nc"
    write_extract(
        path,
        cat,
        loao.issue_times(year),
        sites.station_id,
        {
            "complete": 1,
            "year": year,
            "run_digest": record["digest"],
            "tiles_digest": manifest["digest"],
            "stations_sha256": record["stations_sha256"],
            "gefs_digest": record["gefs_digest"],
            "code_digest": record["code_digest"],
            "tiles": len(manifest["tiles"]),
            "days": manifest["days"],
            "missing_source_hours": json.dumps(manifest["missing_source_hours"], sort_keys=True),
        },
    )
    stats = cat["stats"].reshape(-1, 4) if len(cat["stats"]) else np.zeros((0, 4), dtype=np.int64)
    summary = {
        "year": year,
        "format": loao.EXTRACT_FORMAT,
        "run_digest": record["digest"],
        "extract_sha256": p3.sha256_file(path),
        "stations_weighted": int(len(cat["token"])),
        "distinct_tokens": int(len(set(cat["token"].tolist()))),
        "channel_reports": {"t2m": int(stats[:, 0].sum()), "td2m": int(stats[:, 1].sum())},
        "positive_weight_share": {
            "t2m": float(stats[:, 2].sum() / max(stats[:, 0].sum(), 1)),
            "td2m": float(stats[:, 3].sum() / max(stats[:, 1].sum(), 1)),
        },
        "issue_rows": int(len(cat["issue"])),
        "missing_source_hours": sum(len(v) for v in manifest["missing_source_hours"].values()),
        "extract_bytes": path.stat().st_size,
    }
    (citizen / f"issue_extract_loao_{year}.summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("step", choices=("run", "merge"))
    parser.add_argument("--year", type=int, default=2024, choices=sorted(loao.LAST_ISSUE))
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
    data = args.data_dir.resolve()
    if args.step == "run":
        run(data, args.year, args.processes, args.only_tiles)
    else:
        merge(data, args.year)


if __name__ == "__main__":
    main()
