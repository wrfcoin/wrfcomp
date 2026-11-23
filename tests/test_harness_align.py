from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from conftest import PRIMARY, open_zarr, write_zarr

from wrfcomp.harness.align import (
    MAX_LISTED,
    CoverageError,
    ForecastNaNError,
    SealedSplitError,
    align_truth,
    check_baseline_covers,
    check_forecast_not_sealed,
    check_init_times_contiguous,
    check_nan_rule,
    check_stations,
    expected_init_times,
)
from wrfcomp.harness.hashing import content_sha256


def _tiny_forecast(init: list[str], lead_hours: list[int], stations: list[str]) -> xr.Dataset:
    shape = (len(init), len(lead_hours), len(stations), 2)
    return xr.Dataset(
        {"t2m": (("init_time", "lead_time", "station_id", "ens_member"), np.zeros(shape, "float32"))},
        coords={
            "init_time": np.array(init, dtype="datetime64[ns]"),
            "lead_time": np.array(lead_hours, dtype="timedelta64[h]").astype("timedelta64[ns]"),
            "station_id": np.array(stations, dtype=object),
            "ens_member": [0, 1],
        },
    )


def _tiny_truth(times: list[str], stations: list[str], values: np.ndarray) -> xr.Dataset:
    return xr.Dataset(
        {"t2m": (("valid_time", "station_id"), np.asarray(values, "float32"))},
        coords={"valid_time": np.array(times, dtype="datetime64[ns]"), "station_id": np.array(stations, "O")},
    )


# ------------------------------------------------------------------ align_truth


def test_alignment_matches_truth_at_init_plus_lead(synthetic) -> None:
    obs = align_truth(synthetic.candidate, synthetic.truth)
    assert obs["t2m"].dims == ("init_time", "lead_time", "station_id")
    assert set(obs.data_vars) == PRIMARY
    truth = synthetic.truth
    for i in (0, 7, 29):
        for j in range(3):
            when = synthetic.candidate["init_time"].values[i] + synthetic.candidate["lead_time"].values[j]
            assert obs["valid_time"].values[i, j] == when
            expected = truth["t2m"].sel(valid_time=when).values
            np.testing.assert_array_equal(obs["t2m"].values[i, j], expected)  # NaN equals NaN here


def test_missing_valid_time_is_nan_not_the_neighbour() -> None:
    # Truth has 05:00 and 07:00 but not 06:00. The 06:00 forecast point must not borrow either.
    truth = _tiny_truth(["2024-03-01T05", "2024-03-01T07"], ["A", "B"], np.array([[1.0, 2.0], [3.0, 4.0]]))
    forecast = _tiny_forecast(["2024-03-01T00"], [5, 6, 7], ["A", "B"])
    obs = align_truth(forecast, truth)["t2m"].values[0]
    np.testing.assert_array_equal(obs[0], [1.0, 2.0])
    assert np.isnan(obs[1]).all()
    np.testing.assert_array_equal(obs[2], [3.0, 4.0])


def test_valid_time_outside_truth_and_unknown_station_are_nan() -> None:
    truth = _tiny_truth(["2024-03-01T06"], ["A"], np.array([[1.0]]))
    forecast = _tiny_forecast(["2024-03-01T00", "2024-03-02T00"], [6], ["A", "ZZ"])
    obs = align_truth(forecast, truth)["t2m"]
    assert obs.sel(station_id="A").values[0, 0] == 1.0
    assert np.isnan(obs.sel(station_id="A").values[1, 0])  # after the end of truth
    assert np.isnan(obs.sel(station_id="ZZ").values).all()  # station truth does not have


def test_off_hour_valid_time_is_nan() -> None:
    truth = _tiny_truth(["2024-03-01T06", "2024-03-01T07"], ["A"], np.array([[1.0], [2.0]]))
    forecast = _tiny_forecast(["2024-03-01T00:10"], [6], ["A"])  # valid 06:10, inside +-15 min of 06:00
    assert np.isnan(align_truth(forecast, truth)["t2m"].values).all()


def test_alignment_does_not_depend_on_storage(tmp_path, synthetic) -> None:
    direct = align_truth(synthetic.candidate, synthetic.truth)
    path = write_zarr(synthetic.truth, tmp_path / "t.zarr", {"valid_time": 37, "station_id": 3})
    stored = align_truth(synthetic.candidate, open_zarr(path))
    assert content_sha256(direct) == content_sha256(stored.load())


def test_offset_companion_variables_are_ignored(synthetic) -> None:
    # Data contract addendum 1: truth carries X_offset_min (int16) next to each X. Scoring ignores them.
    truth = synthetic.truth.copy()
    for name in ("t2m", "u10"):
        offset = np.where(np.isnan(truth[name].values), -32768, 7).astype("int16")
        truth[f"{name}_offset_min"] = (("valid_time", "station_id"), offset)
    obs = align_truth(synthetic.candidate, truth)
    assert set(obs.data_vars) == PRIMARY
    assert content_sha256(obs) == content_sha256(align_truth(synthetic.candidate, synthetic.truth))
    check_nan_rule(synthetic.candidate, obs)


def test_duplicate_truth_time_is_refused() -> None:
    truth = _tiny_truth(["2024-03-01T06", "2024-03-01T06"], ["A"], np.array([[1.0], [2.0]]))
    with pytest.raises(ValueError, match="duplicate"):
        align_truth(_tiny_forecast(["2024-03-01T00"], [6], ["A"]), truth)


# ------------------------------------------------------------------ sealed split


def test_sealed_init_time_is_refused() -> None:
    forecast = _tiny_forecast(["2024-12-31T12", "2025-01-01T00"], [6], ["A"])
    truth = _tiny_truth(["2024-12-31T18"], ["A"], np.array([[1.0]]))
    with pytest.raises(SealedSplitError, match="init_time"):
        align_truth(forecast, truth)


def test_sealed_valid_time_is_refused_even_with_2024_init() -> None:
    forecast = _tiny_forecast(["2024-12-31T12"], [6, 12], ["A"])  # 12 h lead lands on 2025-01-01T00
    with pytest.raises(SealedSplitError, match="valid_time"):
        check_forecast_not_sealed(forecast)


def test_sealed_truth_is_refused_inside_align_truth() -> None:
    # Review finding 2: a 2024 forecast against a truth file that also holds 2025 rows.
    forecast = _tiny_forecast(["2024-12-30T00"], [6], ["A"])
    truth = _tiny_truth(["2024-12-30T06", "2025-01-01T00"], ["A"], np.array([[1.0], [2.0]]))
    with pytest.raises(SealedSplitError, match="truth valid_time"):
        align_truth(forecast, truth)
    with pytest.warns(UserWarning, match="SEALED SPLIT IN USE"):
        obs = align_truth(forecast, truth, allow_sealed=True)
    assert obs["t2m"].values[0, 0, 0] == 1.0
    assert align_truth(forecast, truth.isel(valid_time=[0]))["t2m"].values[0, 0, 0] == 1.0


def test_last_unsealed_hour_is_allowed() -> None:
    forecast = _tiny_forecast(["2024-12-31T12"], [11], ["A"])  # valid 2024-12-31T23
    assert check_forecast_not_sealed(forecast) is False


def test_allow_sealed_warns_loudly() -> None:
    forecast = _tiny_forecast(["2025-01-01T00"], [6], ["A"])
    truth = _tiny_truth(["2025-01-01T06"], ["A"], np.array([[1.0]]))
    with pytest.warns(UserWarning, match="SEALED SPLIT IN USE"):
        obs = align_truth(forecast, truth, allow_sealed=True)
    assert obs["t2m"].values[0, 0, 0] == 1.0


# ------------------------------------------------------------------ NaN rule


def test_nan_rule_passes_clean_forecast(synthetic) -> None:
    check_nan_rule(synthetic.candidate, align_truth(synthetic.candidate, synthetic.truth))


def test_nan_rule_names_the_point(synthetic) -> None:
    obs = align_truth(synthetic.candidate, synthetic.truth)
    forecast = synthetic.candidate.copy(deep=True)
    assert np.isfinite(obs["u10"].values[4, 1, 9])
    forecast["u10"].values[4, 1, 9, 3] = np.nan  # one member is enough
    with pytest.raises(ForecastNaNError) as info:
        check_nan_rule(forecast, obs)
    init = np.datetime_as_string(forecast["init_time"].values[4], unit="s") + "Z"
    assert info.value.n_points == 1
    assert info.value.offenders == [("u10", init, "12h", str(forecast["station_id"].values[9]))]
    assert init in str(info.value)


def test_nan_rule_treats_inf_as_missing(synthetic) -> None:
    obs = align_truth(synthetic.candidate, synthetic.truth)
    forecast = synthetic.candidate.copy(deep=True)
    i, j, k = np.argwhere(np.isfinite(obs["t2m"].values))[0]
    forecast["t2m"].values[i, j, k, 0] = np.inf
    with pytest.raises(ForecastNaNError):
        check_nan_rule(forecast, obs)


def test_nan_rule_ignores_points_without_truth(synthetic) -> None:
    obs = align_truth(synthetic.candidate, synthetic.truth)
    holes = np.argwhere(np.isnan(obs["t2m"].values))
    assert len(holes) > 0, "fixture should have truth gaps on the forecast grid"
    forecast = synthetic.candidate.copy(deep=True)
    for i, j, k in holes:
        forecast["t2m"].values[i, j, k, :] = np.nan
    check_nan_rule(forecast, obs)


def test_nan_rule_lists_at_most_ten(synthetic) -> None:
    obs = align_truth(synthetic.candidate, synthetic.truth)
    forecast = synthetic.candidate.copy(deep=True)
    forecast["t2m"].values[0] = np.nan
    with pytest.raises(ForecastNaNError) as info:
        check_nan_rule(forecast, obs)
    assert len(info.value.offenders) == MAX_LISTED
    assert info.value.n_points == int(np.isfinite(obs["t2m"].values[0]).sum()) > MAX_LISTED


# ------------------------------------------------------------------ coverage


def test_expected_init_times_follow_the_forecast_cadence() -> None:
    init = np.array(["2024-03-01T06", "2024-03-01T18", "2024-03-03T06"], dtype="datetime64[ns]")
    expected = expected_init_times(init)
    want = ["2024-03-01T06", "2024-03-01T18", "2024-03-02T06", "2024-03-02T18", "2024-03-03T06"]
    np.testing.assert_array_equal(expected, np.array(want, dtype="datetime64[ns]"))


def test_contiguous_init_times_pass_in_any_order(synthetic) -> None:
    check_init_times_contiguous(synthetic.candidate)
    check_init_times_contiguous(synthetic.candidate.isel(init_time=slice(None, None, -1)))
    check_init_times_contiguous(synthetic.candidate.isel(init_time=[4]))


def test_missing_date_is_named(synthetic) -> None:
    gone = synthetic.candidate["init_time"].values[12]
    with pytest.raises(CoverageError) as info:
        check_init_times_contiguous(synthetic.candidate.drop_sel(init_time=[gone]))
    assert info.value.missing == [np.datetime_as_string(gone, unit="s") + "Z"]


def test_missing_station_is_named(synthetic) -> None:
    required = [str(s) for s in synthetic.candidate["station_id"].values]
    check_stations(synthetic.candidate, required)
    check_stations(synthetic.candidate, required[:5])  # extra stations are not this check's business
    with pytest.raises(CoverageError) as info:
        check_stations(synthetic.candidate.isel(station_id=slice(2, None)), required, role="baseline")
    assert info.value.missing == required[:2] and info.value.role == "baseline"


def test_baseline_must_have_every_candidate_init_time(synthetic) -> None:
    check_baseline_covers(synthetic.candidate.isel(init_time=slice(5, 9)), synthetic.baseline)
    with pytest.raises(CoverageError, match="baseline forecast is missing 2 required init time"):
        check_baseline_covers(synthetic.candidate, synthetic.baseline.isel(init_time=slice(2, None)))


def test_coverage_error_lists_at_most_ten() -> None:
    error = CoverageError("candidate", "station(s)", [f"S{i:02d}" for i in range(25)])
    assert str(error).count("\n") == MAX_LISTED and "missing 25 required" in str(error)
