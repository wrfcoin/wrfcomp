"""Run the shipped station QC over the stored CONUS citizen year and keep what Phase 3 needs.

Three steps, each restartable:

``partition``  daily files -> one file per one-degree tile and month (same anonymised columns).
``run``        per tile: QC every station that can be a neighbour of an official station, and keep,
               for each 00Z/12Z issue time, its latest report of the last 15 minutes with its weight.
``merge``      one extract file for the forecast builder, and a summary of counts.

Output and logs carry tokens' counts and aggregates only. No identifier or position is printed.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from wrfcomp.assimilation import phase3_qc as p3
from wrfcomp.assimilation.madis_archive import VALUE_COLUMNS, load_table
from wrfcomp.assimilation.qc import distances_km
from wrfcomp.harness.metrics import evaluation_init_times

NEIGHBOUR_KM = 50.0  # docs/PHASE3_DESIGN.md, rule 4
WINDOW_MIN = 15  # rule 3
YEAR_END = {2024: "2024-12-17T12"}
_G: dict = {}


def issue_times(year: int) -> np.ndarray:
    return evaluation_init_times(np.datetime64(f"{year}-01-01T00"), np.datetime64(YEAR_END[year]), [0, 12])


def tile_of(lat, lon):
    return np.floor(lat).astype(int), np.floor(lon).astype(int)


# ----------------------------------------------------------------------------- partition


def partition(citizen_dir: Path, year: int) -> None:
    src, dst = citizen_dir / "conus" / f"{year}", citizen_dir / "conus" / f"{year}-tiles"
    days = sorted(p for p in src.glob(f"{year}-*.nc"))
    for month in sorted({p.name[:7] for p in days}):
        marker = dst / f".done-{month}"
        if marker.exists():
            continue
        tables = [load_table(p) for p in days if p.name.startswith(month)]
        cols = {
            name: np.concatenate([getattr(t, name) for t in tables])
            for name in (
                "station_id",
                "obs_time",
                "lat",
                "lon",
                "elevation_m",
                "received_delay_s",
                *VALUE_COLUMNS,
            )
        }
        ti, tj = tile_of(cols["lat"].astype(np.float64), cols["lon"].astype(np.float64))
        code = ti * 1000 + tj
        order = np.argsort(code, kind="stable")
        bounds = np.flatnonzero(np.r_[True, code[order][1:] != code[order][:-1], True])
        for a, b in zip(bounds[:-1], bounds[1:], strict=True):
            rows = order[a:b]
            folder = dst / f"{ti[rows[0]]:+03d}_{tj[rows[0]]:+04d}"
            folder.mkdir(parents=True, exist_ok=True)
            # A station that changes its stored position is a new station (token and position together).
            key = pd.DataFrame({k: cols[k][rows] for k in ("station_id", "lat", "lon", "elevation_m")})
            ident = key.groupby(["station_id", "lat", "lon", "elevation_m"], sort=True).ngroup().to_numpy()
            inner = np.lexsort((cols["obs_time"][rows], ident))
            rows, ident = rows[inner], ident[inner]
            head = rows[np.r_[True, ident[1:] != ident[:-1]]]
            tmp = folder / f".{month}.part.npz"
            np.savez_compressed(
                tmp,
                ident_token=cols["station_id"][head].astype("S11"),
                ident_lat=cols["lat"][head],
                ident_lon=cols["lon"][head],
                ident_elevation_m=cols["elevation_m"][head],
                row_ident=ident.astype(np.int32),
                obs_time=cols["obs_time"][rows].astype("datetime64[s]").astype(np.int64),
                received_delay_s=cols["received_delay_s"][rows],
                **{c: cols[c][rows] for c in VALUE_COLUMNS},
            )
            tmp.replace(folder / f"{month}.npz")
        marker.touch()
        print(f"partitioned {month}: {len(cols['obs_time'])} reports in {len(bounds) - 1} tiles", flush=True)


# ----------------------------------------------------------------------------- run


def _parts(tiles_dir: Path, ti: int, tj: int) -> list[Path]:
    paths = []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            paths.extend(sorted((tiles_dir / f"{ti + di:+03d}_{tj + dj:+04d}").glob("*.npz")))
    return paths


def run_tile(args) -> dict:
    ti, tj = args
    t0 = time.monotonic()
    out_path = _G["out"] / f"{ti:+03d}_{tj:+04d}.npz"
    if out_path.exists():
        return {"tile": out_path.stem, "skipped": True}
    trial, official, forecasts, issues = _G["trial"], _G["official"], _G["forecasts"], _G["issues"]
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
            keep = np.flatnonzero(wanted[gid])
            # Amendment 1: a report that had not arrived by the next issue time is never read.
            keep = keep[p3.arrived_by_next_issue(z["obs_time"][keep], z["received_delay_s"][keep])]
            gid = gid[keep]
            arrays = [z["obs_time"][keep], *(z[c][keep] for c in VALUE_COLUMNS)]
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
        parts = sorted(chunks.pop(k), key=lambda c: c[0][0])  # months in time order
        when = np.concatenate([c[0] for c in parts]).astype("datetime64[s]").astype("datetime64[ns]")
        keep = np.r_[True, when[1:] > when[:-1]]
        values = [np.concatenate([c[i] for c in parts])[keep] for i in range(1, 1 + len(VALUE_COLUMNS))]
        return p3.StationData(lat[k], lon[k], height[k], when[keep], *values)

    data, sun, source = {}, {}, {}
    for k in needed:
        data[k] = station(k)
        sun[k] = p3._sun(data[k].times, data[k].lat, data[k].lon, trial.CFG)
        source[k] = p3.station_source(data[k], official, forecasts, trial, sun=sun[k])
    rec = {key: [] for key in ("station", "issue", "var", "value", "weight", "earned")}
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
            has, value, weight, earned = p3.issue_snapshot(channel, issues, WINDOW_MIN)
            ix = np.flatnonzero(has)
            rec["station"].append(np.full(ix.size, local, dtype=np.int32))
            rec["issue"].append(ix.astype(np.int32))
            rec["var"].append(np.full(ix.size, v, dtype=np.int8))
            rec["value"].append(value[ix].astype(np.float32))
            rec["weight"].append(weight[ix].astype(np.float32))
            rec["earned"].append(earned[ix])
            row[v] = len(channel.weights)
            row[2 + v] = int(np.count_nonzero(channel.weights > 0))
            row[4] |= bool(channel.copycat.any())
            row[5] |= bool(channel.earned.any())
            n_reports += len(channel.weights)
        row[4] |= bool(len(source[k][1]))
        stats.append(row)
    tmp = out_path.with_name(f".{out_path.stem}.part.npz")
    np.savez_compressed(
        tmp,
        token=token[core].astype("U11"),
        lat=lat[core].astype(np.float32),
        lon=lon[core].astype(np.float32),
        elevation_m=height[core].astype(np.float32),
        stats=np.array(stats, dtype=np.int64).reshape(len(core), 6),
        **{k: (np.concatenate(v) if v else np.empty(0)) for k, v in rec.items()},
    )
    tmp.replace(out_path)
    return {
        "tile": out_path.stem,
        "stations": int(len(core)),
        "first_pass_stations": len(needed),
        "channel_reports": n_reports,
        "seconds": round(time.monotonic() - t0, 1),
    }


def run(data_dir: Path, year: int, processes: int, only: int | None) -> None:
    citizen = data_dir / "citizen" / "conus"
    tiles_dir, out = citizen / f"{year}-tiles", citizen / f"{year}-qc"
    out.mkdir(parents=True, exist_ok=True)
    start = np.datetime64(f"{year}-01-01T00", "ns")
    issues = issue_times(year)
    trial = p3.load_trial_module(start)
    t0 = time.monotonic()
    with xr.open_dataset(data_dir / "truth" / f"isd_conus_{year}.nc") as ds:
        truth = ds.sel(valid_time=slice(start, issues[-1])).load()  # nothing from before 1 January
    forecasts = p3.load_forecasts(
        data_dir / "baselines" / "gefs", truth.station_id.values, start, issues[-1], trial
    )
    official = p3.official_from_truth(truth, forecasts, trial)
    print(
        f"official series and {len(forecasts.issued)} forecast entries ready in {time.monotonic() - t0:.0f}s",
        flush=True,
    )
    _G.update(trial=trial, official=official, forecasts=forecasts, issues=issues, tiles=tiles_dir, out=out)
    tiles = sorted((int(p.name[:3]), int(p.name[4:])) for p in tiles_dir.iterdir() if p.is_dir())
    if only is not None:
        tiles = tiles[:only]
    t0 = time.monotonic()
    total = 0
    with mp.get_context("fork").Pool(processes) as pool, (out / "run.jsonl").open("a") as log:
        for record in pool.imap_unordered(run_tile, tiles):
            log.write(json.dumps(record) + "\n")
            log.flush()
            total += record.get("channel_reports", 0)
    print(
        f"{len(tiles)} tiles, {total} channel reports weighted, "
        f"{time.monotonic() - t0:.0f}s wall, {processes} processes"
    )


# ----------------------------------------------------------------------------- merge


def merge(data_dir: Path, year: int) -> None:
    citizen = data_dir / "citizen" / "conus"
    out = citizen / f"{year}-qc"
    parts = {
        k: []
        for k in (
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
            "earned",
        )
    }
    offset = 0
    for path in sorted(out.glob("[+-]*.npz")):
        with np.load(path) as z:
            if not len(z["token"]):
                continue
            for k in (
                "token",
                "lat",
                "lon",
                "elevation_m",
                "stats",
                "issue",
                "var",
                "value",
                "weight",
                "earned",
            ):
                parts[k].append(z[k])
            parts["station"].append(z["station"].astype(np.int64) + offset)
            offset += len(z["token"])
    cat = {k: np.concatenate(v) for k, v in parts.items()}
    issues = issue_times(year)
    ds = xr.Dataset(
        {
            "station_token": ("station", cat["token"].astype("S11")),
            "lat": ("station", cat["lat"]),
            "lon": ("station", cat["lon"]),
            "elevation_m": ("station", cat["elevation_m"]),
            "ever_copycat": ("station", cat["stats"][:, 4].astype(bool)),
            "ever_credential": ("station", cat["stats"][:, 5].astype(bool)),
            "row_station": ("row", cat["station"].astype(np.int32)),
            "row_issue": ("row", cat["issue"].astype(np.int32)),
            "row_var": ("row", cat["var"].astype(np.int8)),
            "row_value": ("row", cat["value"].astype(np.float32)),
            "row_weight": ("row", cat["weight"].astype(np.float32)),
            "row_credential": ("row", cat["earned"].astype(bool)),
            "issue_time": ("issue", issues.astype("datetime64[s]").astype(np.int64)),
        },
        attrs={
            "variables": ",".join(p3.THERMAL),
            "window_minutes": WINDOW_MIN,
            "privacy": "keyed 11-char station tokens; lat/lon rounded to 0.1 deg, elevation to 10 m",
        },
    )
    path = citizen / f"issue_extract_{year}.nc"
    ds.to_netcdf(path, engine="h5netcdf", encoding={k: {"zlib": True, "complevel": 4} for k in ds.data_vars})
    stats = cat["stats"]
    summary = {
        "year": year,
        "stations_weighted": int(len(cat["token"])),
        "channel_reports": {"t2m": int(stats[:, 0].sum()), "td2m": int(stats[:, 1].sum())},
        "positive_weight_share": {
            "t2m": float(stats[:, 2].sum() / max(stats[:, 0].sum(), 1)),
            "td2m": float(stats[:, 3].sum() / max(stats[:, 1].sum(), 1)),
        },
        "stations_ever_copycat": int(stats[:, 4].sum()),
        "stations_ever_credential": int(stats[:, 5].sum()),
        "issue_rows": int(len(cat["issue"])),
        "issue_rows_positive_weight": int(np.count_nonzero(cat["weight"] > 0)),
        "extract_bytes": path.stat().st_size,
    }
    (citizen / f"issue_extract_{year}.summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("partition", "run", "merge"))
    parser.add_argument("--year", type=int, default=2024, choices=sorted(YEAR_END))
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("WRFCOMP_DATA_DIR") or Path.home() / "wrfcomp-data"),
    )
    parser.add_argument("--processes", type=int, default=8)
    parser.add_argument("--only-tiles", type=int, default=None, help="run the first N tiles only (timing)")
    args = parser.parse_args()
    if args.step == "partition":
        partition(args.data_dir / "citizen", args.year)
    elif args.step == "run":
        run(args.data_dir, args.year, args.processes, args.only_tiles)
    else:
        merge(args.data_dir, args.year)


if __name__ == "__main__":
    main()
