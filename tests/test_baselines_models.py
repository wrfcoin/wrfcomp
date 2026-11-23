import numpy as np
import pytest
import xarray as xr

from wrfcomp.baselines import climatology, earth2_baselines, persistence
from wrfcomp.baselines.io import DIMS, UNITS, validate, write


@pytest.fixture
def truth():
    time = np.arange("2023-12-15", "2024-01-02", dtype="datetime64[h]").astype("datetime64[ns]")
    ds = xr.Dataset(
        coords={
            "valid_time": time,
            "station_id": ["00123456789"],
            "latitude": ("station_id", [40.0]),
            "longitude": ("station_id", [-105.0]),
            "elevation_m": ("station_id", [1500.0]),
        }
    )
    for name in UNITS:
        ds[name] = (("valid_time", "station_id"), np.arange(len(time), dtype="f4")[:, None])
        ds[name + "_offset_min"] = xr.zeros_like(ds[name], dtype=np.int16)
    return ds


def test_climatology_poisoned_split(truth):
    init = np.datetime64("2024-12-16")
    extra = truth.isel(valid_time=[0]).assign_coords(valid_time=[np.datetime64("2024-12-16T03", "ns")])
    truth = xr.concat([truth, extra], dim="valid_time")
    out = climatology.forecast(truth, [init], [3])
    assert np.isfinite(out.t2m).all()
    for name in UNITS:
        truth[name].loc[dict(valid_time=slice("2023-12-17T19", None))] = 1e9
    xr.testing.assert_identical(out, climatology.forecast(truth, [init], [3]))
    validate(out)


def test_persistence_observed_cutoff_and_last_slot(truth, tmp_path):
    init = np.datetime64("2024-01-01T00", "ns")
    truth.t2m_offset_min.loc[dict(valid_time=init)] = 1
    truth.t2m.loc[dict(valid_time=init)] = 99999
    truth.t2m.loc[dict(valid_time=init - np.timedelta64(1, "h"))] = 800
    truth.u10.loc[dict(valid_time=init)] = 500
    truth.v10.loc[dict(valid_time=slice(init - np.timedelta64(3, "h"), init))] = np.nan
    out = persistence.forecast(truth, [init], [3])
    np.testing.assert_equal(out.t2m.values, np.full((1, 1, 1, 30), 803))
    assert (out.u10 == 503).all()
    assert out.v10.isnull().all()
    truth.t2m.loc[dict(valid_time=slice(init, None))] = -99999
    xr.testing.assert_identical(out, persistence.forecast(truth, [init], [3]))
    path = tmp_path / "forecast.zarr"
    write(out, path)
    with xr.open_zarr(path) as restored:
        validate(restored)
        xr.testing.assert_identical(restored, out)
    assert out.t2m.dims == DIMS


def test_ai_stub():
    with pytest.raises(RuntimeError, match="ai extra and an NVIDIA GPU"):
        earth2_baselines.load()
