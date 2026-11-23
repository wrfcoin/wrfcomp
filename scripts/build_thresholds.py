"""Build the tail-threshold file from training-split truth.

    uv run python scripts/build_thresholds.py --train-truth <store> [<store> ...] --out <thresholds.nc>

One threshold per station and calendar month for every primary variable: the configured tail quantile
(0.95) of the training truth, computed by `wrfcomp.harness.metrics.compute_tail_thresholds`.

The input is refused, before any weather value is read, if any `valid_time` lies after the train split
end in `configs/splits.yaml` (or is not a valid time at all). Cut the store to the train split first.
A station with no observation in a calendar month gets a NaN threshold; the counts are printed.

Exit codes: 0 written, 2 refused (bad or post-train input, existing output).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import xarray as xr

from wrfcomp.harness.config import DEFAULT_CONFIG_PATH, HarnessConfig, load_config
from wrfcomp.harness.metrics import compute_tail_thresholds
from wrfcomp.harness.schemas import SchemaError, validate_thresholds

EXIT_REFUSED = 2


class Refused(ValueError):
    """Input or output that this script will not work with. Becomes exit code 2."""


def open_store(path: Path, what: str) -> xr.Dataset:
    """Open a Zarr directory or a NetCDF file lazily."""
    if not path.exists():
        raise Refused(f"{what}: no such file or directory: {path}")
    try:
        if path.is_dir():
            try:
                return xr.open_zarr(path, chunks=None, consolidated=True)
            except (KeyError, ValueError, FileNotFoundError):
                return xr.open_zarr(path, chunks=None, consolidated=False)
        return xr.open_dataset(path)
    except Exception as exc:  # any reader failure is a refusal, not a traceback
        raise Refused(f"{what}: cannot open {path}: {exc}") from exc


def check_train_times(ds: xr.Dataset, what: str, train_end: np.datetime64) -> None:
    """Coordinates only: every time must be a valid time at or before the train split end."""
    if "valid_time" not in ds.coords or ds["valid_time"].dims != ("valid_time",):
        raise Refused(f"{what}: no valid_time coordinate")
    times = ds["valid_time"].values
    if times.size == 0 or not np.issubdtype(times.dtype, np.datetime64):
        raise Refused(f"{what}: valid_time is empty or not a time coordinate")
    if np.isnat(times).any():
        raise Refused(f"{what}: valid_time has invalid (NaT) values")
    late = times > train_end
    if late.any():
        raise Refused(
            f"{what}: {int(late.sum())} time(s) after the train split end "
            f"(latest {times[late].max()} > {train_end}). Cut the store to the train split first."
        )


def build(train_paths: list[Path], cfg: HarnessConfig) -> xr.Dataset:
    """Thresholds in the contract shape: dims (station_id, month), one variable per primary variable."""
    variables = list(cfg.evaluation.variables.primary)
    train_end = cfg.splits.train.end
    parts = []
    for path in train_paths:
        ds = open_store(path, "train truth")
        check_train_times(ds, f"train truth {path.name}", train_end)
        missing = [v for v in variables if v not in ds.data_vars]
        if missing:
            raise Refused(f"train truth {path.name}: missing variable(s) {missing}")
        parts.append(ds[variables])
    if len(parts) == 1:
        truth = parts[0]
    else:
        stations = parts[0].indexes["station_id"]
        for path, part in zip(train_paths, parts, strict=True):
            if not part.indexes["station_id"].equals(stations):
                raise Refused(f"train truth {path.name}: station_id differs from {train_paths[0].name}")
        truth = xr.concat(parts, dim="valid_time", data_vars="all", coords="minimal", compat="override")
        if not truth.indexes["valid_time"].is_unique:
            raise Refused("train truth: the stores overlap in valid_time")
        truth = truth.sortby("valid_time")
    try:
        out = xr.Dataset(
            {
                name: compute_tail_thresholds(truth[name], cfg.gate.tail_quantile, cfg=cfg)
                for name in variables
            }
        )
    except ValueError as exc:
        raise Refused(f"train truth: {exc}") from exc
    finally:
        for part in parts:
            part.close()
    out = out.reset_coords(drop=True)
    out = out.assign_coords(
        station_id=out["station_id"].values.astype(str), month=np.arange(1, 13, dtype=np.int64)
    )
    out.attrs = {
        "tail_quantile": float(cfg.gate.tail_quantile),
        "train_end_utc": str(train_end),
        "protocol_version": cfg.protocol_version,
    }
    validate_thresholds(out, variables=variables)
    return out


def write(thresholds: xr.Dataset, out: Path, variables: list[str]) -> None:
    """Write NetCDF4 next to the target, read it back, check it, then move it into place."""
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = out.with_name(f".{out.name}.partial-{os.getpid()}")
    try:
        thresholds.to_netcdf(staging, engine="h5netcdf")
        with xr.open_dataset(staging) as restored:
            validate_thresholds(restored, variables=variables)
        os.replace(staging, out)
    finally:
        staging.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--train-truth", type=Path, nargs="+", required=True, help="truth store(s), train split only"
    )
    parser.add_argument("--out", type=Path, required=True, help="thresholds NetCDF to write")
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG_PATH, help="splits.yaml (default: this checkout's)"
    )
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output file")
    args = parser.parse_args(argv)
    try:
        if args.out.exists() and not args.overwrite:
            raise Refused(f"output already exists: {args.out} (pass --overwrite to replace it)")
        try:
            cfg = load_config(args.config)
        except (ValueError, FileNotFoundError) as exc:
            raise Refused(f"config: {exc}") from exc
        variables = list(cfg.evaluation.variables.primary)
        thresholds = build(list(args.train_truth), cfg)
        write(thresholds, args.out, variables)
    except (Refused, SchemaError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    cells = thresholds.sizes["station_id"] * 12
    print(f"wrote {args.out}: {thresholds.sizes['station_id']} stations, quantile {cfg.gate.tail_quantile}")
    for name in variables:
        missing = int(np.isnan(thresholds[name].values).sum())
        print(f"  {name}: {missing} of {cells} station-months have no threshold (no training data)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
