"""Shared forecast layout and split boundaries."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import xarray as xr

from wrfcomp.harness.config import load_config
from wrfcomp.harness.schemas import validate_forecast_dataset

UNITS = {"t2m": "K", "td2m": "K", "u10": "m/s", "v10": "m/s", "msl": "Pa"}
DIMS = ("init_time", "lead_time", "station_id", "ens_member")
STATION = ("latitude", "longitude", "elevation_m")
CONFIG = Path(__file__).resolve().parents[3] / "configs/splits.yaml"


def boundaries():
    splits = load_config(CONFIG).splits
    return splits.train.end, splits.sealed_test.start


def layout(stations, inits, leads, members, model):
    times = np.asarray(inits, dtype="datetime64[ns]")
    hours = np.asarray(leads, dtype="timedelta64[h]").astype("timedelta64[ns]")
    _, sealed = boundaries()
    if (
        times.ndim != 1
        or not times.size
        or hours.ndim != 1
        or not hours.size
        or np.isnat(times).any()
        or np.isnat(hours).any()
        or (hours <= np.timedelta64(0, "ns")).any()
        or (times[:, None] + hours >= sealed).any()
        or members < 2
    ):
        raise ValueError("invalid or sealed forecast window/member count")
    coords = {name: stations[name] for name in ("station_id", *STATION)}
    coords.update(init_time=times, lead_time=hours, ens_member=np.arange(members))
    ds = xr.Dataset(coords=coords, attrs={"model_id": model, "model_version": "1"})
    for name, unit in UNITS.items():
        ds[name] = (DIMS, np.full(tuple(ds.sizes[d] for d in DIMS), np.nan, dtype="f4"), {"units": unit})
    return ds


def train_truth(truth, forecast):
    end, _ = boundaries()
    # Historical outcomes used in calibration must also precede the earliest forecast.
    end = min(end, forecast.init_time.values.min())
    return truth.sel(valid_time=slice(np.datetime64("2014-01-01"), end))


def validate(ds):
    validate_forecast_dataset(ds, variables=list(UNITS))


def write(ds, path):
    validate(ds)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with TemporaryDirectory(dir=path.parent) as staging:
        store = Path(staging) / "forecast.zarr"
        ds.to_zarr(store, mode="w-", consolidated=True, zarr_format=2)
        with xr.open_zarr(store) as restored:
            validate(restored)
        store.rename(path)
