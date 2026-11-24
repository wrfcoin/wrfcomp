"""Build monthly candidates using authorized GEFS schedules and local, same-year inputs."""

import argparse
import json
import os
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from wrfcomp.baselines.io import CONFIG, UNITS, write
from wrfcomp.harness.hashing import content_sha256
from wrfcomp.harness.metrics import evaluation_init_times
from wrfcomp.models import gefs_station_v2 as v2
from wrfcomp.models import gefs_station_v3 as v3
from wrfcomp.models.gefs_station import MAX_OFFSET, MODEL_ID, START, WINDOW, forecast, summarize


def build(period, data, version=1):
    if version == 3:
        return build_v3([period], data, CONFIG.parents[1] / "experiments")
    if version not in (1, 2):
        raise ValueError("unknown candidate version")
    if period not in {f"2024-{m:02}" for m in range(1, 13)}:
        raise ValueError("expected a 2024 calendar month")
    record = json.loads((CONFIG.parents[1] / "experiments" / f"gefs_{period}.json").read_text())
    targets = evaluation_init_times(
        np.datetime64(record["period_start"]),
        np.datetime64(record["period_end"]),
        record["init_hours_scored"],
    )
    if len(targets) != record["n_init_times"] or targets[-1] > np.datetime64("2024-12-17T12"):
        raise ValueError("invalid reference schedule")
    window = v2.WINDOW if version == 2 else WINDOW
    start = max(START, targets[0] - window - np.timedelta64(max(record["leads_scored"]), "h") - MAX_OFFSET)
    inits = evaluation_init_times(start, targets[-1], record["init_hours_scored"])
    parts = []
    counts = dict.fromkeys(UNITS, 0)
    history = deque()
    with xr.open_dataset(data / "truth/isd_conus_2024.nc") as archive:
        truth = archive.sel(valid_time=slice(max(START, start - np.timedelta64(3, "h")), targets[-1])).load()
    for t in inits:
        stamp = pd.Timestamp(t).strftime("%Y%m%d%H")
        with xr.open_zarr(data / "baselines/gefs" / f"{stamp}.zarr", chunks=None) as source:
            if not np.array_equal(source.init_time.values, [t]):
                raise ValueError(f"wrong init in {stamp}")
            raw = source.load()
        while history and history[0].init + raw.lead_time.values.max() + MAX_OFFSET <= t - window:
            history.popleft()
        if version == 2:
            base = v2.base_ensemble(raw, truth)
            stats = v2.History(t, base.isel(init_time=0, drop=True).mean("ens_member"))
            if t in targets:
                result, fallback = v2.forecast(base, history, truth)
        elif t in targets:
            result, stats, fallback = forecast(raw, history, truth)
        else:
            stats = summarize(raw)
        if t in targets:
            parts.append(result)
            for k in UNITS:
                counts[k] += fallback[k]
        history.append(stats)
    publish(parts, counts, data, period, version)


def publish(parts, counts, data, period, version):
    candidate = xr.concat(parts, dim="init_time", coords="minimal", compat="identical", join="exact")
    for var in candidate.variables.values():
        var.encoding.clear()
    directory = MODEL_ID if version == 1 else f"{MODEL_ID}-v{version}"
    dest = data / "candidates" / directory / f"{period}.zarr"
    if dest.exists():
        with xr.open_zarr(dest, chunks=None) as previous:
            xr.testing.assert_identical(candidate, previous)
    else:
        write(candidate, dest)
    with xr.open_zarr(dest, chunks=None) as saved:
        digest = content_sha256(saved)
    print(json.dumps(dict(period=period, path=str(dest), content_sha256=digest, fallback_points=counts)))


def build_v3(periods, data, schedule_dir):
    """Replay once from January, including unrequested months as causal history only."""
    periods = sorted(set(periods))
    year = int(periods[0][:4])
    start, end = v2.year_bounds(year)
    if any(p not in {f"{year:04d}-{m:02d}" for m in range(1, 13)} for p in periods):
        raise ValueError("expected calendar months within one forecast year")
    targets = {}
    for period in periods:
        record = json.loads((schedule_dir / f"gefs_{period}.json").read_text())
        times = evaluation_init_times(
            np.datetime64(record["period_start"]),
            np.datetime64(record["period_end"]),
            record["init_hours_scored"],
        )
        if (
            not len(times)
            or len(times) != record["n_init_times"]
            or sorted(record["init_hours_scored"]) != [0, 12]
            or any(np.datetime_as_string(t, unit="M") != period for t in times)
            or (year == 2024 and times[-1] > np.datetime64("2024-12-17T12"))
            or times[-1] + v2.LEADS[-1] >= end
        ):
            raise ValueError("invalid reference schedule")
        targets[period] = times
    wanted = {t: period for period, times in targets.items() for t in times}
    last = targets[periods[-1]][-1]
    replay = v3.Replay(year)
    parts = []
    counts = dict.fromkeys(UNITS, 0)
    with xr.open_dataset(data / "truth" / f"isd_conus_{year:04d}.nc") as archive:
        truth = archive.sel(valid_time=slice(start, last)).load()
    for t in evaluation_init_times(start, last, [0, 12]):
        path = data / "baselines/gefs" / f"{pd.Timestamp(t):%Y%m%d%H}.zarr"
        with xr.open_zarr(path, chunks=None) as source:
            if not np.array_equal(source.init_time.values, [t]):
                raise ValueError("source initialization differs from its path")
            out, fallback = replay.issue(source.load(), truth)
        if t in wanted:
            period = wanted[t]
            parts.append(out)
            for k in UNITS:
                counts[k] += fallback[k]
            if t == targets[period][-1]:
                publish(parts, counts, data, period, 3)
                parts.clear()
                counts = dict.fromkeys(UNITS, 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--period", action="append", required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("WRFCOMP_DATA_DIR", Path.home() / "wrfcomp-data")),
    )
    parser.add_argument(
        "--schedule-dir", type=Path, help="directory of authorized gefs_YYYY-MM.json schedules (v3)"
    )
    args = parser.parse_args()
    if args.version == 3:
        if args.schedule_dir and any(p.startswith("2024-") for p in args.period):
            parser.error("2024 uses the committed reference schedules")
        build_v3(args.period, args.data_dir.resolve(), args.schedule_dir or CONFIG.parents[1] / "experiments")
        return
    if args.schedule_dir:
        parser.error("--schedule-dir is for v3 only")
    for period in args.period:
        build(period, args.data_dir.resolve(), args.version)


if __name__ == "__main__":
    main()
