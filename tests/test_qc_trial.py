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
    series = {
        "t2m": [
            (
                np.r_[times[[0, -1]] - 24 * trial.HOUR, times],
                np.r_[290.0, 290.0, wide.t2m.values],
                np.zeros(9),
            )
        ]
        * 2
    }
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


def test_trial_citizen_join_uses_distinct_stations_and_available_reports(tmp_path):
    from wrfcomp.assimilation.citizen_obs import save_citizen_batch
    from wrfcomp.assimilation.tokens import ObservationBatch

    rng = np.random.default_rng(92)
    start = trial.START
    hours = np.array([start - 3 * trial.HOUR])
    forecasts = (hours, np.array([start]), {"wind_speed": np.array([[3.0, 3.0]])})
    official = xr.Dataset(
        coords={
            "station_id": [0, 1],
            "latitude": ("station_id", [5.0, 6.0]),
            "longitude": ("station_id", [0.0, 0.0]),
            "elevation_m": ("station_id", [0, 0]),
        }
    )
    series = {"wind_speed": [(np.array([start]), np.array([3.0]), np.array([0.0]))] * 2}
    stations = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    times = start + np.array([0, 1, 1, 1, 10, 10, 10, 10]) * trial.MINUTE
    for mixed in (False, True):
        raw = ObservationBatch(
            [f"{i:011}" for i in stations],
            np.zeros(8),
            np.where((stations == 3) & mixed, 0.1, 0),
            np.zeros(8),
            times,
            np.zeros(8),
            ["wind_speed"] * 8,
            rng.normal(3, 0.1, 8),
        )
        path = tmp_path / "synthetic.nc"
        save_citizen_batch(raw, path)
        counts = trial.trial_region(path, official, forecasts, series)["counts"]["wind_speed"]
        assert counts["spatial_applied"] == (7 if mixed else 0)
        assert counts["FEW_BUDDIES"] == (1 if mixed else 8)


def test_official_evidence_counts_each_measurement_once_for_rolling_statistics():
    rng = np.random.default_rng(144)
    measured = trial.START + np.arange(18) * trial.HOUR
    when = np.repeat(measured, 2) + np.tile([0, 10], len(measured)) * trial.MINUTE
    official = rng.normal(8, 2, len(measured))
    first = official + rng.uniform(-0.2, 0.2, len(measured))
    reports = np.repeat(first, 2)
    reports[1::2] += 5
    raw, stamps, r, rms, counts, spans = trial.official_evidence(
        (measured, official, np.zeros(18)), when, reports
    )
    np.testing.assert_array_equal(stamps, np.repeat(measured.astype(np.int64), 2))
    np.testing.assert_array_equal(raw, np.repeat(official, 2))
    assert counts[-1] == len(measured) and spans[-1] == 17
    assert np.isnan(r[: 2 * (trial.CFG.min_pairs - 1)]).all()
    np.testing.assert_allclose(r[-2:], np.corrcoef(first, official)[0, 1])
    np.testing.assert_allclose(rms[-2:], np.sqrt(np.mean((first - official) ** 2)))


def test_hourly_storm_support_uses_only_a_new_measurement_inside_the_step():
    t = trial.START + 8 * trial.HOUR
    times = t + np.array([-15, 0, 15, 30]) * trial.MINUTE
    stamps = t + np.array([-60, 0]) * trial.MINUTE
    series = [(stamps, np.array([298.0, 286.0]), np.zeros(2))]
    mask = trial.official_storm_support(
        times, np.array([298.0, 286.0, 298.0, 286.0]), [0], np.zeros(1), 0, series
    )
    np.testing.assert_array_equal(mask, [False, True, False, False])
