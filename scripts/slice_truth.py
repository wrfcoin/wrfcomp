"""Cut a time range out of a truth file, unchanged, into a new truth file.

    uv run python scripts/slice_truth.py --source <truth.nc> --start <UTC> --end <UTC> --out <truth.nc>

Used for the scored truth of a confirmation period: a slice of the train truth, in the shape of the
validation year's truth file. Values, coordinates and dtypes are copied as they are; nothing is
recomputed. Both ends are included. The source is refused, on its coordinates, if it has sealed
times. Exit codes: 0 written, 2 refused.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from wrfcomp.harness.config import DEFAULT_CONFIG_PATH, load_config
from wrfcomp.harness.schemas import SchemaError, validate_truth_dataset
from wrfcomp.harness.truth import write_truth_netcdf

EXIT_REFUSED = 2


def parse_utc(text: str) -> np.datetime64:
    stamp = pd.Timestamp(text)
    if stamp is pd.NaT:
        raise ValueError(f"not a time: {text!r}")
    if stamp.tz is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    return stamp.to_datetime64().astype("datetime64[ns]")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--start", required=True, metavar="UTC")
    parser.add_argument("--end", required=True, metavar="UTC")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args(argv)
    try:
        if args.out.exists():
            raise ValueError(f"output already exists: {args.out}")
        cfg = load_config(args.config)
        start, end = parse_utc(args.start), parse_utc(args.end)
        if not start <= end:
            raise ValueError("--start is after --end")
        with xr.open_dataset(args.source) as source:
            if "valid_time" not in source.coords or source["valid_time"].dims != ("valid_time",):
                raise ValueError(f"{args.source.name}: no valid_time coordinate")
            times = source["valid_time"].values
            if times.size == 0 or not np.issubdtype(times.dtype, np.datetime64):
                raise ValueError(f"{args.source.name}: valid_time is empty or not a time coordinate")
            if np.isnat(times).any() or times.max() >= cfg.splits.sealed_test.start:
                raise ValueError(f"{args.source.name}: invalid or sealed times")
            keep = np.flatnonzero((times >= start) & (times <= end))
            if keep.size == 0:
                raise ValueError(f"{args.source.name}: no time between {start} and {end}")
            cut = source.isel(valid_time=keep).load()
        validate_truth_dataset(cut, cfg=cfg)
        write_truth_netcdf(cut, args.out)
    except (ValueError, FileNotFoundError, SchemaError, OSError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    first, last = cut["valid_time"].values[[0, -1]]
    print(f"wrote {args.out}: {cut.sizes['valid_time']} hours, {first} .. {last}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
