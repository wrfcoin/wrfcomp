"""Acceptance checks for the frozen pressure-only registration."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from wrfcomp.baselines.io import UNITS, layout
from wrfcomp.models import gefs_station_v2 as v2
from wrfcomp.models import gefs_station_v3 as v3

H = v2.HOUR
D = np.timedelta64(1, "D")
M = v2.MINUTE


def fixture(year=2024):
    start = np.datetime64(f"{year}-01-01", "ns")
    rng = np.random.default_rng(671)
    stations = xr.Dataset(
        coords={
            "station_id": ["00123456789", "00123456780"],
            "latitude": ("station_id", [41.0, 42.0]),
            "longitude": ("station_id", [-90.0, -91.0]),
            "elevation_m": ("station_id", [100.0, 120.0]),
        }
    )
    raw = layout(stations, [start], [3, 6, 12, 24, 48], 31, "noaa-gefs")
    times = np.arange(start - 3 * H, start + 36 * D, H)
    truth = xr.Dataset(coords={"valid_time": times, **dict(stations.coords)})
    for k in UNITS:
        scale = 100 if k == "msl" else 1
        centre = 100000 if k == "msl" else 280
        raw[k].values[:] = centre + scale * rng.normal(size=raw[k].shape)
        truth[k] = (
            ("valid_time", "station_id"),
            (centre + scale * rng.normal(size=(len(times), 2))).astype("f4"),
        )
        truth[k + "_offset_min"] = xr.zeros_like(truth[k], dtype=np.int16)
    # On the transition date one station fits, the other continues warm.
    truth.msl.values[: 24 * 5, 1] = np.nan
    return start, raw, truth


def assert_bits(a, b):
    xr.testing.assert_identical(a, b)
    assert a.dtype == b.dtype == np.dtype("float32")
    assert a.values.tobytes(order="C") == b.values.tobytes(order="C")


def test_copy_v2_through_cold_start_transition_and_month_boundary():
    start, raw, truth = fixture()
    replay = v3.Replay(2024)
    history = []
    for d in range(33):
        t = start + d * D
        source = raw.assign_coords(init_time=[t])
        base = v2.base_ensemble(source, truth)
        core, counts = v2.forecast(base, history, truth)
        history.append(v2.History(t, base.isel(init_time=0, drop=True).mean("ens_member")))
        out, got = replay.issue(source, truth)
        assert got == counts
        for k in UNITS.keys() - {"msl"}:
            assert_bits(core[k], out[k])
        if d == 0:
            m = core.msl.values.astype(float).mean(-1, keepdims=True)
            np.testing.assert_array_equal(out.msl, (m + 0.5 * (core.msl.values - m)).astype("f4"))
        if d == 10:
            np.testing.assert_array_equal(replay.pressure_history[-1].warm[0], [False, True])
        np.testing.assert_allclose(
            out.msl.values.astype(float).mean(-1), core.msl.values.astype(float).mean(-1), atol=0.004, rtol=0
        )
        np.testing.assert_allclose(
            replay.pressure_history[-1].variance,
            core.msl.values[0].astype(float).var(-1, ddof=1) * 32 / 31,
            rtol=5e-16,
            atol=0,
        )
        assert out.attrs == {**core.attrs, "model_version": "3"}


def test_gain_from_prior_v2_moments_and_both_window_boundaries():
    start, raw, truth = fixture()
    t = start + 32 * D
    core, _ = v2.forecast(v2.base_ensemble(raw.assign_coords(init_time=[t]), truth), [], truth)
    issues = []
    current_branch = np.tile([True, False], (6, 1))
    specs = [
        # init offset in days, lead, measurement offsets, errors, corrected variances
        (-31, 24, [0, 1], [1000, 3], [1, 4]),  # lower bound excluded; one minute inside included
        (-1, 24, [0, 1], [6, 1000], [9, 1]),  # U=T included; future measurement excluded
        (-2, 24, [16, -16], [1000, 1000], [1, 1]),  # invalid offsets
        (-4, 24, [0, 0], [8, 4], [16, 9]),  # historical branch membership is independent of current branch
    ]
    for day, lead, offsets, errors, variances in specs:
        i = t + day * D
        means = np.zeros((6, 2))
        vs = np.zeros_like(means)
        warm = current_branch.copy()
        if day == -4:
            warm[4] = [False, True]
        label = i + lead * H
        truth.msl.loc[dict(valid_time=label)] = 100000
        truth.msl_offset_min.loc[dict(valid_time=label)] = offsets
        means[4] = 100000 + np.array(errors)
        vs[4] = variances
        issues.append(v3.PressureIssue(i, means, vs, warm))
    # Ineligible other-hour, same-issue and future-issued records cannot supply errors.
    for i in (t - 12 * H, t, t + D):
        issues.append(v3.PressureIssue(i, np.zeros((6, 2)), np.ones((6, 2)), current_branch))
    # For the warm group the eligible errors are 6,4 and variances 9,9.
    # For fitted they are 3,8 and variances 4,16. S=2 in both groups.
    gains = np.sqrt([(2 * ((36 + 16) / (9 + 9)) + 2 * 0.25) / 4, (2 * ((9 + 64) / (4 + 16)) + 2) / 4])
    out, _ = v3.adjust_pressure(core, current_branch, list(reversed(issues)), truth, 2024)
    vals = core.msl.values[0, 4].astype(float)
    m = vals.mean(-1, keepdims=True)
    np.testing.assert_array_equal(out.msl.values[0, 4], (m + gains[:, None] * (vals - m)).astype("f4"))
    # Change all future values, including a positive-offset measurement at the current label.
    poison = truth.copy(deep=True)
    poison.msl.values[poison.valid_time.values > t] = -77777
    poison.msl.loc[dict(valid_time=t, station_id=truth.station_id.values[1])] = 77777
    again, _ = v3.adjust_pressure(core, current_branch, issues, poison, 2024)
    assert_bits(out.msl, again.msl)
    # No history and zero variance both use the registered prior, including fitted fallback.
    assert v3.pressure_gain(np.array([]), np.array([]), 2, False) == 1
    assert v3.pressure_gain(np.array([20.0]), np.array([0.0]), 2, True) == 0.5
    assert v3.pressure_gain(np.array([20.0]), np.array([0.0]), 2, False) == 1
    with pytest.raises(ValueError, match="duplicate"):
        v3.adjust_pressure(core, current_branch, [issues[0], issues[0]], truth, 2024)


@pytest.mark.parametrize("year", [2023, 2024])
def test_year_isolation_and_issue_cutoff(year):
    start, raw, truth = fixture(year)
    poisoned = truth.copy(deep=True)
    for k in UNITS:
        truth[k + "_offset_min"].loc[dict(valid_time=start)] = -1
        poisoned[k + "_offset_min"].loc[dict(valid_time=start)] = -1
        poisoned[k].loc[dict(valid_time=slice(None, start))] = -77777
        poisoned[k].loc[dict(valid_time=slice(start + H, None))] = 99999
    r = v3.Replay(year)
    historical = raw.assign_coords(init_time=[start - D])
    r.base_history.append(v2.History(start - D, historical.isel(init_time=0, drop=True).mean("ens_member")))
    r.pressure_history.append(
        v3.PressureIssue(start - D, np.zeros((6, 2)), np.ones((6, 2)), np.ones((6, 2), bool))
    )
    clean, _ = v3.Replay(year).issue(raw, truth)
    got, _ = r.issue(raw, poisoned)
    xr.testing.assert_identical(clean, got)
    np.testing.assert_array_equal(v2.base_ensemble(raw, truth, year=year).msl[:, 0], raw.msl[:, 0])
    # A preceding-year forecast must remain ineligible even when its +24 h verification
    # is present exactly at January 1 with a zero offset.
    truth.msl_offset_min.loc[dict(valid_time=start)] = 0
    stamped, _, _ = v2.forecast_with_branches(v2.base_ensemble(raw, truth, year=year), [], truth, year=year)
    fitted_branch = np.zeros((6, 2), dtype=bool)
    empty, _ = v3.adjust_pressure(stamped, fitted_branch, [], truth, year)
    prior = v3.PressureIssue(start - D, np.zeros((6, 2)), np.ones((6, 2)), fitted_branch)
    with_prior, _ = v3.adjust_pressure(stamped, fitted_branch, [prior], truth, year)
    assert_bits(empty.msl, with_prior.msl)
    with pytest.raises(ValueError, match="in-year"):
        r.issue(historical, truth)


def test_monthwise_build_reconstructs_uninterrupted_year(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location(
        "build_candidate", Path(__file__).parents[1] / "scripts/build_candidate.py"
    )
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    start, raw, truth = fixture()
    # Stage the real truth I/O and published Zarr path; synthesize only the raw cycle reader.
    (tmp_path / "truth").mkdir()
    truth.to_netcdf(tmp_path / "truth/isd_conus_2024.nc", engine="h5netcdf")
    original = xr.open_zarr

    def open_source(path, **kwargs):
        if Path(path).parent.name == "gefs":
            t = np.datetime64(builder.pd.to_datetime(Path(path).stem, format="%Y%m%d%H"), "ns")
            return raw.assign_coords(init_time=[t])
        return original(path, **kwargs)

    monkeypatch.setattr(xr, "open_zarr", open_source)
    for month, last, n in [("01", "31", 62), ("02", "01", 2)]:
        (tmp_path / f"gefs_2024-{month}.json").write_text(
            json.dumps(
                dict(
                    period_start=f"2024-{month}-01T00:00",
                    period_end=f"2024-{month}-{last}T12:00",
                    init_hours_scored=[0, 12],
                    n_init_times=n,
                )
            )
        )
    builder.build_v3(["2024-01", "2024-02"], tmp_path, tmp_path)
    path = tmp_path / "candidates/gefs-station-corrected-v3/2024-02.zarr"
    with original(path, chunks=None) as ds:
        first = ds.load()
    path.rename(tmp_path / "continuous.zarr")
    builder.build_v3(["2024-02"], tmp_path, tmp_path)
    with original(path, chunks=None) as rebuilt:
        for k in UNITS:
            assert_bits(first[k], rebuilt[k])
