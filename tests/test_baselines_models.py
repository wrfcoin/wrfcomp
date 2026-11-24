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
    expected = climatology.forecast(truth, [init], [3])
    assert np.isfinite(expected.v10).all()
    xr.testing.assert_identical(out.v10, expected.v10)
    truth.t2m.loc[dict(valid_time=slice(init, None))] = -99999
    xr.testing.assert_identical(out, persistence.forecast(truth, [init], [3]))
    path = tmp_path / "forecast.zarr"
    write(out, path)
    with xr.open_zarr(path) as restored:
        validate(restored)
        xr.testing.assert_identical(restored, out)
    assert out.t2m.dims == DIMS


def test_no_training_wind_or_recent_anchor_leaves_members_missing(truth):
    init = np.datetime64("2024-01-01T00", "ns")
    truth.v10.values[:] = np.nan
    truth.v10_offset_min.values[:] = -32768
    # Valid verifying truth exists, but arrives after initialization.
    valid = init + np.timedelta64(3, "h")
    truth.v10.loc[dict(valid_time=valid)] = 5
    truth.v10_offset_min.loc[dict(valid_time=valid)] = 0
    for model in (climatology, persistence):
        out = model.forecast(truth, [init], [3])
        assert out.v10.isnull().all()
        assert np.isfinite(out.u10).all()


def test_ai_stub():
    with pytest.raises(RuntimeError, match="ai extra and an NVIDIA GPU"):
        earth2_baselines.load()


# --------------------------------------------------------------------------- explicit train_end


@pytest.fixture
def long_truth():
    """Hourly synthetic truth from 2022-01-01 to 2023-02-10, one station, seeded."""
    rng = np.random.default_rng(11)
    time = np.arange("2022-01-01", "2023-02-10", dtype="datetime64[h]").astype("datetime64[ns]")
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
        ds[name] = (("valid_time", "station_id"), rng.normal(0, 1, (len(time), 1)).astype("f4"))
        ds[name + "_offset_min"] = xr.zeros_like(ds[name], dtype=np.int16)
    return ds


CUTOFF = np.datetime64("2022-12-17T18", "ns")
INIT_2023 = np.datetime64("2023-02-05T00", "ns")


def _poison(truth, start, end):
    bad = truth.copy(deep=True)
    for name in UNITS:
        bad[name].loc[dict(valid_time=slice(start, end))] = 1e9
    return bad


@pytest.mark.parametrize("model", [climatology, persistence])
def test_default_cutoff_follows_the_request_and_train_end_fixes_it(long_truth, model):
    """The trap: before the train end the default cutoff is the earliest init, so data between a
    fixed cutoff and the init is learned from. `train_end` stops that and nothing else."""
    # After the cutoff, but early enough that persistence's starting value at init is untouched.
    poisoned = _poison(long_truth, CUTOFF + np.timedelta64(1, "h"), INIT_2023 - np.timedelta64(4, "h"))
    default = model.forecast(long_truth, [INIT_2023], [3, 24])
    assert np.isfinite(default.t2m).all()
    assert "train_end_utc" not in default.attrs
    assert not default.t2m.equals(model.forecast(poisoned, [INIT_2023], [3, 24]).t2m)

    fixed = model.forecast(long_truth, [INIT_2023], [3, 24], train_end=CUTOFF)
    assert np.isfinite(fixed.t2m).all()
    assert fixed.attrs["train_end_utc"] == str(CUTOFF)
    xr.testing.assert_identical(fixed, model.forecast(poisoned, [INIT_2023], [3, 24], train_end=CUTOFF))
    assert not fixed.t2m.equals(default.t2m)
    validate(fixed)


@pytest.mark.parametrize("model", [climatology, persistence])
def test_train_end_equal_to_the_default_cutoff_changes_no_value(long_truth, model):
    default = model.forecast(long_truth, [INIT_2023], [3, 24])
    same = model.forecast(long_truth, [INIT_2023], [3, 24], train_end=INIT_2023)
    for name in UNITS:
        np.testing.assert_array_equal(default[name].values, same[name].values)


@pytest.mark.parametrize("model", [climatology, persistence])
def test_train_end_can_only_be_earlier(long_truth, truth, model):
    with pytest.raises(ValueError, match="train_end"):  # after the earliest init of the request
        model.forecast(long_truth, [INIT_2023], [3], train_end=INIT_2023 + np.timedelta64(1, "h"))
    with pytest.raises(ValueError, match="train_end"):  # after the configured train split end
        model.forecast(truth, [np.datetime64("2024-01-01T00", "ns")], [3], train_end="2023-12-17T19")
    with pytest.raises(ValueError, match="train_end"):
        model.forecast(long_truth, [INIT_2023], [3], train_end="NaT")


def test_persistence_with_train_end_still_starts_from_the_latest_observation(long_truth):
    long_truth.t2m.loc[dict(valid_time=INIT_2023)] = 700.0
    out = persistence.forecast(long_truth, [INIT_2023], [3], train_end=CUTOFF)
    assert abs(float(out.t2m.mean()) - 700.0) < 5.0
