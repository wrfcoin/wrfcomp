"""Shared forecast layout and split boundaries."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import xarray as xr
import yaml

UNITS = {"t2m": "K", "td2m": "K", "u10": "m/s", "v10": "m/s", "msl": "Pa"}
DIMS = ("init_time", "lead_time", "station_id", "ens_member")
STATION = ("latitude", "longitude", "elevation_m")
CONFIG = Path(__file__).resolve().parents[3] / "configs/splits.yaml"


def boundaries():
    splits = yaml.safe_load(CONFIG.read_text())["splits"]
    end = np.datetime64(splits["train"]["end_utc"].removesuffix("Z"))
    sealed = np.datetime64(splits["sealed_test"]["start_utc"].removesuffix("Z"))
    return end, sealed


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
    if set(ds.dims) != set(DIMS) or ds.sizes["ens_member"] < 2:
        raise ValueError("forecast dimensions or ensemble size")
    if (
        ds.init_time.dtype != "M8[ns]"
        or ds.lead_time.dtype != "m8[ns]"
        or ds.ens_member.dtype.kind not in "iu"
    ):
        raise ValueError("forecast coordinate dtype")
    if any(not isinstance(s, str) or len(s) != 11 for s in ds.station_id.values):
        raise ValueError("station_id must contain 11-character strings")
    if any(ds[k].dims != ("station_id",) for k in STATION) or not all(
        ds.attrs.get(k) for k in ("model_id", "model_version")
    ):
        raise ValueError("missing station coordinates or model identity")
    for name, unit in UNITS.items():
        if ds[name].dims != DIMS or ds[name].dtype != "float32" or ds[name].attrs.get("units") != unit:
            raise ValueError(f"invalid forecast variable {name}")
    try:
        from wrfcomp.harness.schemas import validate_forecast_dataset
    except ModuleNotFoundError as exc:
        if exc.name != "wrfcomp.harness.schemas":
            raise
    else:
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
