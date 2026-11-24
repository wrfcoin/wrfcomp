import numpy as np
import pytest
import xarray as xr

from wrfcomp.baselines.io import UNITS, layout, validate, write
from wrfcomp.models import gefs_station_v2 as v2


def inputs():
    rng = np.random.default_rng(419)
    stations = xr.Dataset(
        coords={
            "station_id": ["00123456789"],
            "latitude": ("station_id", [40.0]),
            "longitude": ("station_id", [-106.0]),
            "elevation_m": ("station_id", [1100.0]),
        }
    )
    t = np.datetime64("2024-03-10T12", "ns")
    raw = layout(stations, [t], [3, 6, 12, 24, 48], 31, "noaa-gefs")
    truth = xr.Dataset(
        coords={
            "valid_time": np.arange(t - 63 * 24 * v2.HOUR, t + 49 * v2.HOUR, v2.HOUR),
            **dict(stations.coords),
        }
    )
    for k in UNITS:
        truth[k] = (("valid_time", "station_id"), np.full((len(truth.valid_time), 1), 2.0, dtype="f4"))
        truth[k + "_offset_min"] = xr.zeros_like(truth[k], dtype="int16")
        raw[k].values[:] = 8 + rng.normal(size=raw[k].shape)
    return raw, truth, t


def test_one_hour_observation_cutoff_and_warmup():
    raw, truth, t = inputs()
    truth.u10_offset_min.loc[dict(valid_time=t)] = 1
    truth.u10_offset_min.loc[dict(valid_time=t - v2.HOUR)] = -15
    truth.v10[:] = np.nan
    base = v2.base_ensemble(raw, truth)
    np.testing.assert_array_equal(base.lead_time, v2.LEADS)
    g = raw.u10.values[0, 0].astype(float)
    np.testing.assert_allclose(base.u10.values[0, 0], 2 + (2.25 / 4.25) * (g - 2), rtol=1e-14)
    np.testing.assert_array_equal(base.v10.values[0, 0], raw.v10.values[0, 0])
    out, counts = v2.forecast(base, [], truth)
    assert counts["u10"] == 6
    for k in ("u10", "v10"):
        a = base[k].values
        mu = a.mean(axis=-1, keepdims=True)
        np.testing.assert_array_equal(out[k], (mu + 4 * (a - mu)).astype("f4"))
    # Poison the label T with offset +1 minute, and every hourly slot after T.
    truth.u10.loc[dict(valid_time=slice(t, None))] = 99999
    xr.testing.assert_identical(v2.base_ensemble(raw, truth), base)
    validate(out)


def test_fit_reference_and_ties():
    ids = np.arange(31)[::-1]
    members = np.full((1, 31), 8.0)
    x = np.full((1, 40), 8.0)
    # Equal predictor distances select the newest 31 columns. Rare residuals
    # exercise the unmodified empirical shape rather than a Gaussian spread.
    y = np.arange(40.0, dtype=float)[None, :] ** 2
    out, count = v2.fit_members(members, x, y, np.ones_like(x, dtype=bool), ids)
    chosen = y[0, :31]
    r = chosen - chosen.mean()
    target = np.sqrt((r**2).sum() / 29 * (1 + 1 / 31))
    np.testing.assert_allclose(out[0, ::-1], chosen.mean() + r / r.std(ddof=1) * target)
    assert count == 0
    # For 10 pairs cover rank replication and the constrained slope endpoints.
    x = np.arange(10.0)[None, :]
    for slope in (-2.0, 0.5, 2.0):
        y = 3 + slope * x
        b = np.clip(slope, 0, 1)
        residual = (y - y.mean() - b * (x - x.mean())).ravel()
        selected = np.sort(residual)[((np.arange(31) + 0.5) * 10 / 31).astype(int)]
        target = np.sqrt((residual**2).sum() / 8 * 1.1)
        center = y.mean() + b * (8 - x.mean())
        want = center + (
            (selected - selected.mean()) / selected.std(ddof=1) * target if target else np.zeros(31)
        )
        got, count = v2.fit_members(members, x, y, np.ones_like(x, dtype=bool), ids)
        np.testing.assert_allclose(got[0, ::-1], want)
        assert count == 0


def test_history_cutoff_bias_spread_cap_and_bytes(tmp_path):
    raw, truth, t = inputs()
    # Turn every member into an under-dispersed ensemble with the same mean.
    for k in UNITS:
        raw[k].values[:] = 8 + np.linspace(-0.125, 0.125, 31)
    for days in range(10):
        truth.u10.loc[dict(valid_time=t - days * 24 * v2.HOUR)] = 2 + 3 * (-1) ** days
    truth.td2m[:] = 100
    base = v2.base_ensemble(raw, truth)
    history = []
    for days in range(1, 11):
        issue = t - days * 24 * v2.HOUR
        historical = v2.base_ensemble(raw.assign_coords(init_time=[issue]), truth)
        history.append(v2.History(issue, historical.isel(init_time=0, drop=True).mean("ens_member")))
    out, _ = v2.forecast(base, history, truth)
    wind = out.u10.sel(lead_time=24 * v2.HOUR).values
    assert wind.mean() == pytest.approx(2.0)
    expected = np.sqrt(90 / 8 * 1.1)
    assert wind.std(ddof=1) == pytest.approx(expected)
    assert (out.td2m <= out.t2m).all()
    assert (out.msl.sel(lead_time=24 * v2.HOUR) == 2.0).all()
    assert out.attrs["model_version"] == "2"
    for k in UNITS:
        truth[k].loc[dict(valid_time=slice(t + v2.HOUR, None))] = 9999
    again, _ = v2.forecast(v2.base_ensemble(raw, truth), history, truth)
    for path, ds in ((tmp_path / "a", out), (tmp_path / "b", again)):
        write(ds, path)
    for p in (tmp_path / "a").rglob("*"):
        if p.is_file():
            assert p.read_bytes() == (tmp_path / "b" / p.relative_to(tmp_path / "a")).read_bytes()
    truth.u10_offset_min.loc[dict(valid_time=t)] = 1
    delayed, _ = v2.forecast(base, history, truth)
    x = base.u10.sel(lead_time=24 * v2.HOUR).values
    mu = x.mean(axis=-1, keepdims=True)
    np.testing.assert_array_equal(delayed.u10.sel(lead_time=24 * v2.HOUR), (mu + 4 * (x - mu)).astype("f4"))
