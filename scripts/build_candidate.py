"""Build one period using its committed GEFS reference schedule and local 2024 inputs."""

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
from wrfcomp.models.gefs_station import MAX_OFFSET, MODEL_ID, START, WINDOW, forecast, summarize


def build(period, data):
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
    start = max(START, targets[0] - WINDOW - np.timedelta64(max(record["leads_scored"]), "h") - MAX_OFFSET)
    inits = evaluation_init_times(start, targets[-1], record["init_hours_scored"])
    parts = []
    counts = dict.fromkeys(UNITS, 0)
    history = deque()
    with xr.open_dataset(data / "truth/isd_conus_2024.nc") as archive:
        truth = archive.sel(valid_time=slice(start, targets[-1])).load()
    for t in inits:
        stamp = pd.Timestamp(t).strftime("%Y%m%d%H")
        with xr.open_zarr(data / "baselines/gefs" / f"{stamp}.zarr", chunks=None) as source:
            if not np.array_equal(source.init_time.values, [t]):
                raise ValueError(f"wrong init in {stamp}")
            raw = source.load()
        while history and history[0].init + raw.lead_time.values.max() + MAX_OFFSET <= t - WINDOW:
            history.popleft()
        if t in targets:
            result, stats, fallback = forecast(raw, history, truth)
            parts.append(result)
            for k in UNITS:
                counts[k] += fallback[k]
        else:
            stats = summarize(raw)
        history.append(stats)
    candidate = xr.concat(parts, dim="init_time", coords="minimal", compat="identical", join="exact")
    for var in candidate.variables.values():
        var.encoding.clear()
    dest = data / "candidates" / MODEL_ID / f"{period}.zarr"
    if dest.exists():
        with xr.open_zarr(dest, chunks=None) as previous:
            xr.testing.assert_identical(candidate, previous)
    else:
        write(candidate, dest)
    with xr.open_zarr(dest, chunks=None) as saved:
        digest = content_sha256(saved)
    print(json.dumps(dict(period=period, path=str(dest), content_sha256=digest, fallback_points=counts)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--period", action="append", required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("WRFCOMP_DATA_DIR", Path.home() / "wrfcomp-data")),
    )
    args = parser.parse_args()
    for period in args.period:
        build(period, args.data_dir.resolve())


if __name__ == "__main__":
    main()
