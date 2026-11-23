from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from conftest import open_zarr, write_zarr

from wrfcomp.harness.align import (
    MAX_LISTED,
    ForecastNaNError,
    SealedSplitError,
    align_truth,
    check_forecast_not_sealed,
    check_nan_rule,
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
    assert set(obs.data_vars) == {"t2m", "u10"}
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
