"""Synthetic fixtures for the harness tests. Seeded, generated in the test run, never real observations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

# All five primary variables: name -> (mean, day-to-day spread)
VARIABLES = {
    "t2m": (285.0, 6.0),
    "td2m": (278.0, 5.0),
    "u10": (1.0, 3.0),
    "v10": (-0.5, 3.0),
    "msl": (101325.0, 300.0),
}
UNITS = {"t2m": "K", "td2m": "K", "u10": "m/s", "v10": "m/s", "msl": "Pa"}
PRIMARY = set(VARIABLES)


@dataclass
class Synthetic:
    truth: xr.Dataset
    candidate: xr.Dataset
    baseline: xr.Dataset
    thresholds: xr.Dataset


def station_ids(n: int) -> np.ndarray:
    return np.array([f"7{i:05d}99999" for i in range(n)], dtype=object)


def _station_coords(n: int) -> dict:
    return {
        "latitude": ("station_id", np.linspace(30.0, 45.0, n)),
        "longitude": ("station_id", np.linspace(-120.0, -75.0, n)),
        "elevation_m": ("station_id", np.linspace(5.0, 1500.0, n)),
    }


def make_synthetic(
    seed: int = 20241123,
    *,
    n_stations: int = 20,
    n_init: int = 30,
    lead_hours: tuple[int, ...] = (6, 12, 24),
    n_members: int = 8,
    start: str = "2024-03-01T00:00",
    n_truth_nan: int = 25,
) -> Synthetic:
    """Truth = slow signal + unpredictable part. Candidate knows the signal; the baseline knows it worse."""
    rng = np.random.default_rng(seed)
    stations = station_ids(n_stations)
    init = pd.date_range(start, periods=n_init, freq="12h").values.astype("datetime64[ns]")
    lead = np.array([np.timedelta64(h, "h") for h in lead_hours]).astype("timedelta64[ns]")
    valid = pd.date_range(start, init[-1] + lead.max(), freq="1h").values.astype("datetime64[ns]")
    coords = _station_coords(n_stations)
    pos = np.searchsorted(valid, init[:, None] + lead[None, :])  # (init, lead) -> row of truth

    truth_vars, cand_vars, base_vars, thr_vars = {}, {}, {}, {}
    for name, (mean, spread) in VARIABLES.items():
        signal = mean + spread * np.cumsum(rng.normal(0, 0.15, (valid.size, n_stations)), axis=0)
        noise_sd = 1.0
        obs = signal + rng.normal(0, noise_sd, signal.shape)
        missing = rng.choice(obs.size, size=n_truth_nan, replace=False)
        obs.ravel()[missing] = np.nan
        truth_vars[name] = (("valid_time", "station_id"), obs.astype("float32"), {"units": UNITS[name]})

        mu = signal[pos]  # (init, lead, station)
        shape = (*mu.shape, n_members)
        cand = mu[..., None] + rng.normal(0, noise_sd, shape)
        # Baseline: a per-point error in its centre and too little spread.
        centre = mu + rng.normal(0, 1.5, mu.shape)
        base = centre[..., None] + rng.normal(0, 0.5 * noise_sd, shape)
        dims = ("init_time", "lead_time", "station_id", "ens_member")
        cand_vars[name] = (dims, cand.astype("float32"), {"units": UNITS[name]})
        base_vars[name] = (dims, base.astype("float32"), {"units": UNITS[name]})

        q95 = np.nanquantile(obs, 0.95, axis=0)  # synthetic stand-in for the train-split quantile
        thr_vars[name] = (("station_id", "month"), np.repeat(q95[:, None], 12, axis=1).astype("float32"))

    fc_coords = {
        "init_time": init,
        "lead_time": lead,
        "station_id": stations,
        "ens_member": np.arange(n_members, dtype="int64"),
        **coords,
    }
    truth = xr.Dataset(truth_vars, coords={"valid_time": valid, "station_id": stations, **coords})
    candidate = xr.Dataset(cand_vars, coords=fc_coords, attrs={"model_id": "synth", "model_version": "1"})
    baseline = xr.Dataset(base_vars, coords=fc_coords, attrs={"model_id": "synth_base", "model_version": "1"})
    thresholds = xr.Dataset(thr_vars, coords={"station_id": stations, "month": np.arange(1, 13)})
    return Synthetic(truth=truth, candidate=candidate, baseline=baseline, thresholds=thresholds)


def write_zarr(ds: xr.Dataset, path: Path, chunks: dict[str, int] | None = None) -> Path:
    """Write a zarr store. `chunks` sets the on-disk chunking of every data variable."""
    out = ds.copy()
    for name in out.variables:
        out[name].encoding.clear()
    if chunks is not None:
        out = out.chunk({k: v for k, v in chunks.items() if k in out.dims})
    out.to_zarr(path, mode="w", consolidated=False)
    return path


def open_zarr(path: Path) -> xr.Dataset:
    return xr.open_zarr(path, chunks=None, consolidated=False)


def write_netcdf(ds: xr.Dataset, path: Path, *, netcdf3: bool = False) -> Path:
    """NetCDF4 (h5netcdf) by default; `netcdf3` writes the classic format with scipy."""
    if netcdf3:
        ds.to_netcdf(path, engine="scipy")
    else:
        ds.to_netcdf(path, engine="h5netcdf")
    return path


REPO_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "splits.yaml"


def write_config(root: Path, stations: list[str]) -> Path:
    """A copy of the real splits.yaml next to a synthetic stations file. Returns the config path."""
    cfg_dir = root / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "splits.yaml").write_bytes(REPO_CONFIG.read_bytes())
    pd.DataFrame({"station_id": stations, "station_name": "SYNTHETIC"}).to_csv(
        cfg_dir / "stations_conus.csv", index=False
    )
    return cfg_dir / "splits.yaml"


@pytest.fixture
def synthetic() -> Synthetic:
    return make_synthetic()
