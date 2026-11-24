import numpy as np
import pytest
import xarray as xr

from wrfcomp.baselines.io import UNITS, layout, validate, write
from wrfcomp.models.gefs_station import forecast, summarize


@pytest.fixture
def world():
    rng = np.random.default_rng(419)
    stations = xr.Dataset(
        coords={
            "station_id": ["00123456789"],
            "latitude": ("station_id", [40.0]),
            "longitude": ("station_id", [-106.0]),
            "elevation_m": ("station_id", [1100.0]),
        }
    )
    dates = np.arange("2024-01-01", "2024-02-12", dtype="datetime64[h]").astype("datetime64[ns]")
    truth = xr.Dataset(coords={"valid_time": dates, **dict(stations.coords)})
    for k in UNITS:
        truth[k] = (("valid_time", "station_id"), np.zeros((len(dates), 1), dtype="f4"))
        truth[k + "_offset_min"] = xr.zeros_like(truth[k], dtype="int16")

    def raw(day):
        ds = layout(stations, [np.datetime64(day)], [24], 31, "noaa-gefs")
        anomaly = rng.normal(size=31)
        anomaly -= anomaly.mean()
        anomaly /= anomaly.std(ddof=1)
        for k in UNITS:
            ds[k].values[:] = 8 + anomaly
        return ds

    return raw, truth


def test_causal_cutoff_and_ten_pairs(world):
    raw, truth = world
    t = np.datetime64("2024-01-12")
    now = raw(t)
    # At T the 11th past forecast verifies. Only nine older valid reports
    # remain after masking one missing station report.
    history = [summarize(raw(d)) for d in np.arange(np.datetime64("2024-01-01"), t)]
    truth.u10.loc[dict(valid_time=np.datetime64("2024-01-02"))] = np.nan
    truth.u10_offset_min.loc[dict(valid_time=t)] = 1
    initial, _, missing = forecast(now, history, truth)
    assert missing["u10"] == 1
    np.testing.assert_array_equal(initial.u10, now.u10)
    truth.u10.loc[dict(valid_time=t)] = 100000
    poisoned, _, _ = forecast(now, history, truth)
    xr.testing.assert_identical(initial, poisoned)
    truth.u10.loc[dict(valid_time=t)] = 0
    truth.u10_offset_min.loc[dict(valid_time=t)] = 0
    fitted, _, missing = forecast(now, history, truth)
    assert missing["u10"] == 0
    assert abs(fitted.u10.values.mean()) < 1e-5
    np.testing.assert_allclose(fitted.u10.std(), now.u10.std(), atol=1e-5)


def test_spread_window_hour_cap_and_repeatability(world, tmp_path):
    raw, truth = world
    t = np.datetime64("2024-02-03")
    now = raw(t)
    history = [summarize(raw(d)) for d in np.arange(np.datetime64("2024-01-01"), t)]
    # Thirty errors, alternating +/-3 about the constant bias of 8.
    for i, date in enumerate(np.arange(t - np.timedelta64(29, "D"), t + np.timedelta64(1, "D"))):
        truth.u10.loc[dict(valid_time=date)] = (-1) ** i * 3
    truth.td2m[:] = 20
    a, _, missing = forecast(now, history, truth)
    assert not any(missing.values())
    assert abs(a.u10.values.mean()) < 1e-5
    spread = float(a.u10.values.std(ddof=1))
    assert spread / 3 == pytest.approx(1, abs=1e-5)
    assert (a.td2m == a.t2m).all()
    validate(a)
    # Poison the expired left endpoint, and add an opposite-hour forecast.
    truth.u10.loc[dict(valid_time=t - np.timedelta64(30, "D"))] = 9999
    other = raw(t - np.timedelta64(36, "h"))
    other.u10.values[:] = -10000
    b, _, _ = forecast(now, [*history, summarize(other)], truth)
    for name, ds in (("a", a), ("b", b)):
        write(ds, tmp_path / name)

    def files(directory):
        return {p.relative_to(directory): p.read_bytes() for p in directory.rglob("*") if p.is_file()}

    assert files(tmp_path / "a") == files(tmp_path / "b")
    truth.u10[:] *= 100
    clipped, _, _ = forecast(now, history, truth)
    assert clipped.u10.values.std(ddof=1) == pytest.approx(6, abs=1e-5)
