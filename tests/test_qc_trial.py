"""Offline trial checks. All observations here are seeded synthetic fixtures."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

SPEC = importlib.util.spec_from_file_location(
    "qc_trial", Path(__file__).parents[1] / "scripts/run_qc_trial.py"
)
trial = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(trial)


def test_official_matching_uses_measurement_time_and_expires():
    hour = np.datetime64("2024-07-11T15", "ns")
    measured = hour + np.array([-12, 6]) * trial.MINUTE
    times = hour + np.array([-15, 0, 5, 6, 21, 22]) * trial.MINUTE
    np.testing.assert_array_equal(trial.latest_indices(measured, times), [-1, 0, -1, 1, 1, -1])
    assert np.all(trial.latest_indices(measured[:0], times) == -1)


def test_forecast_selection_excludes_unissued_and_prefers_shorter_lead():
    base = np.datetime64("2024-07-11T00", "ns")
    issued = base + np.array([0, 12, 12, 12]) * trial.HOUR
    valid = base + np.array([12, 15, 18, 24]) * trial.HOUR
    times = base + np.array([0, 12 * 60, 13 * 60 + 30, 27 * 60 + 1]) * trial.MINUTE
    np.testing.assert_array_equal(trial.forecast_indices(issued, valid, times), [-1, 0, 1, -1])
    # A future-issued forecast with the same valid time can never change the earlier selection.
    assert trial.forecast_indices(np.append(issued, times[1]), np.append(valid, times[1]), times)[1] == 0


def test_official_series_uses_per_variable_offsets_and_derives_synchronous_wind():
    rng = np.random.default_rng(31)
    label = np.datetime64("2024-07-10T12", "ns")
    ds = xr.Dataset(coords={"valid_time": [label], "station_id": [0]})
    for var in trial.DIRECT:
        ds[var] = (("valid_time", "station_id"), rng.normal(size=(1, 1)))
        ds[var + "_offset_min"] = (("valid_time", "station_id"), [[7 if var == "u10" else -4]])
    means = {var: np.zeros((1, 1)) for var in trial.SUPPORTED}
    series = trial.official_series(ds, (np.array([label - 3 * trial.HOUR]), np.array([label]), means))
    assert series["wind_speed"][0][0].size == 0
    obs, _ = trial.align(series["u10"][0], np.array([label, label + 7 * trial.MINUTE]))
    assert np.isnan(obs[0])
    assert obs[1] == ds.u10.item()


def test_station_pressure_and_missing_buddies_are_not_background_failures():
    from collections import Counter, defaultdict

    rng = np.random.default_rng(32)
    time = np.datetime64("2024-07-12T12", "ns")
    times = time + np.arange(7) * 10 * trial.MINUTE
    wide = pd.DataFrame(
        {"station_pressure": np.repeat(100000.0, 7), "t2m": 290 + rng.normal(0, 0.1, 7)}, index=times
    )
    official = xr.Dataset(
        coords={
            "station_id": [0, 1],
            "latitude": ("station_id", [0.0, 1.0]),
            "longitude": ("station_id", [0.0, 1.0]),
            "elevation_m": ("station_id", [0, 0]),
        }
    )
    forecasts = (np.array([time - 3 * trial.HOUR]), np.array([time]), {"t2m": np.array([[290.0, 290.0]])})
    series = {"t2m": [(times, wide.t2m.values, np.zeros(7))] * 2}
    totals, modes, samples = defaultdict(Counter), defaultdict(set), defaultdict(list)
    trial.aggregate_station(wide, (0, 0, 0), official, forecasts, series, totals, modes, samples, 0)
    assert totals["station_pressure"]["total"] == 7
    assert totals["station_pressure"]["background_applied"] == 0
    assert totals["station_pressure"]["FLATLINE"] == 2
    assert totals["t2m"]["FEW_BUDDIES"] == 7
    assert totals["t2m"]["background_applied"] == 7
    assert totals["reports"]["total"] == 7
    assert np.max(samples["t2m"][0]) <= trial.CFG.new_prior


def test_indoor_screen_requires_independent_official_times_and_a_swing():
    rng = np.random.default_rng(42)
    times = trial.START + np.arange(24) * trial.HOUR
    official = np.linspace(280, 288, len(times))
    stable = np.full(len(times), 294.0) + rng.normal(0, 0.01, len(times))
    assert trial.indoor_screen(times, stable, times, official) == (True, True)
    assert trial.indoor_screen(times, official, times, official) == (True, False)
    assert trial.indoor_screen(times, stable, times[::3], official[::3]) == (False, False)
