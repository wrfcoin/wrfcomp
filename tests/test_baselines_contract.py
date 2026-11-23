"""Seeded properties of the public historical forecast entry points."""

import numpy as np
import pytest
import xarray as xr

from wrfcomp.baselines import climatology, persistence
from wrfcomp.baselines.io import UNITS
from wrfcomp.harness.config import load_config
from wrfcomp.harness.schemas import validate_forecast_dataset

MODELS = (climatology, persistence)
CFG = load_config()
FLOOR = CFG.gate.min_members_for_gate
END = CFG.splits.train.end
HOUR = np.timedelta64(1, "h")
STATION = ("valid_time", "station_id")


def observations(start, stop, seed):
    rng = np.random.default_rng(seed)
    times = np.arange(start, stop, dtype="datetime64[h]").astype("datetime64[ns]")
    truth = xr.Dataset(
        coords={
            "valid_time": times,
            "station_id": ["00000000001", "00000000002"],
            "latitude": ("station_id", [40.0, 41.0]),
            "longitude": ("station_id", [-90.0, -91.0]),
            "elevation_m": ("station_id", [100.0, 110.0]),
        }
    )
    for name, unit in UNITS.items():
        values = rng.normal({"t2m": 285, "td2m": 275, "msl": 101325}.get(name, 0), 2, (len(times), 2)).astype(
            "float32"
        )
        offsets = rng.integers(-15, 16, size=values.shape, dtype=np.int16)
        missing = rng.random(values.shape) < 0.15
        values[missing] = np.nan
        offsets[missing] = -32768
        truth[name] = (STATION, values, {"units": unit})
        truth[name + "_offset_min"] = (STATION, offsets, {"_FillValue": np.int16(-32768)})
    return truth


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("seed", [17, 29])
def test_eligibility_cutoffs_and_no_imputation(model, seed):
    truth = observations("2023-12-01", "2024-12-19", seed)
    inits = np.array(["2024-12-14T00", "2024-12-14T06", "2024-12-14T12"], dtype="datetime64[ns]")
    leads = [1, 6, 12, 48]
    for name in UNITS:
        truth[name].values[:, 1] = np.nan
        truth[name + "_offset_min"].values[:, 1] = -32768
        # A measured-after-cutoff calibration target, labelled at the training boundary.
        truth[name].loc[dict(valid_time=END, station_id="00000000001")] = 300
        truth[name + "_offset_min"].loc[dict(valid_time=END, station_id="00000000001")] = 1
        for init in inits:
            # An unavailable init-slot report and a guaranteed eligible fallback.
            rows = dict(valid_time=[init - HOUR, init], station_id="00000000001")
            truth[name].loc[rows] = [280, 281]
            truth[name + "_offset_min"].loc[rows] = [0, 1]

    clean = model.forecast(truth, inits, leads)
    validate_forecast_dataset(clean, variables=list(UNITS))
    assert clean.sizes["ens_member"] >= FLOOR
    for name in UNITS:
        assert np.isfinite(clean[name].isel(station_id=0)).all()
        assert clean[name].isel(station_id=1).isnull().all()

    for i, init in enumerate(inits):
        poisoned = truth.copy(deep=True)
        for name in UNITS:
            offsets = truth[name + "_offset_min"].values
            labels = truth.valid_time.values[:, None]
            measured = labels + offsets.astype("timedelta64[m]")
            excluded = measured > END
            if model is persistence:
                # Persistence legitimately needs recent post-training measurements as anchors.
                anchor = (labels >= init - 3 * HOUR) & (labels <= init) & (measured <= init)
                excluded &= ~anchor
            mask = excluded & np.isfinite(truth[name])
            poisoned[name].values[mask] += 1e5
        xr.testing.assert_identical(clean.isel(init_time=[i]), model.forecast(poisoned, [init], leads))


@pytest.mark.parametrize("model", MODELS)
def test_in_training_issue_cutoff(model):
    truth = observations("2023-12-01", "2023-12-18", 61)
    for init in np.arange("2023-12-14T06", "2023-12-15", 6, dtype="datetime64[h]"):
        poisoned = truth.copy(deep=True)
        for name in UNITS:
            truth[name + "_offset_min"].loc[dict(valid_time=init)] = 1
            poisoned[name + "_offset_min"].loc[dict(valid_time=init)] = 1
            offset = truth[name + "_offset_min"].values.astype("timedelta64[m]")
            future = truth.valid_time.values[:, None] + offset > init
            poisoned[name].values[future] += 1e5
        clean = model.forecast(truth, [init], [6, 12, 24], members=FLOOR)
        assert np.isfinite(clean.t2m).any()
        xr.testing.assert_identical(clean, model.forecast(poisoned, [init], [6, 12, 24], members=FLOOR))
    with pytest.raises(ValueError, match="ensemble members"):
        model.forecast(truth, [init], [6], members=FLOOR - 1)


def test_climatology_represents_empirical_frequency_without_date_aliasing():
    truth = observations("2022-12-01", "2022-12-16", 13)
    rows = truth.valid_time.dt.hour.values == 0
    # Fifteen consecutive days alternate regimes. Eight chronological indices all pick the cold one.
    sample = np.resize([280.0, 290.0], rows.sum())
    truth.t2m.values[rows, 0] = sample
    truth.t2m_offset_min.values[rows, 0] = 0
    out = climatology.forecast(truth, [np.datetime64("2023-12-07T18")], [6], members=8)
    old = sample[np.linspace(0, len(sample) - 1, 8).round().astype(int)]
    assert np.count_nonzero(old == 290) == 0
    np.testing.assert_array_equal(out.t2m.values[0, 0, 0], [280] * 4 + [290] * 4)


def test_persistence_preserves_observed_error_mass_and_falls_back_without_calibration():
    truth = observations("2023-12-01", "2023-12-06", 72)
    truth.t2m.values[:] = np.nan
    truth.t2m_offset_min.values[:] = -32768
    for day, delta in [(1, -5), (2, 5), (5, None)]:
        t = np.datetime64(f"2023-12-{day:02d}T00", "ns")
        truth.t2m.loc[dict(valid_time=t)] = 280
        truth.t2m_offset_min.loc[dict(valid_time=t)] = 0
        if delta is not None:
            # Only the first station has any historical verifying reports.
            row = dict(valid_time=t + HOUR, station_id="00000000001")
            truth.t2m.loc[row] = 280 + delta
            truth.t2m_offset_min.loc[row] = 0
    out = persistence.forecast(truth, [t], [1], members=8)
    np.testing.assert_array_equal(out.t2m.values[0, 0, 0], [275] * 4 + [285] * 4)
    np.testing.assert_array_equal(out.t2m.values[0, 0, 1], [280] * 8)
    old = np.quantile([-5, 5], (np.arange(8) + 0.5) / 8)
    assert np.ptp(old) == 8.75


def test_staged_schema_failure_keeps_destination_unpublished(tmp_path, monkeypatch):
    from wrfcomp.baselines.io import write
    from wrfcomp.harness.schemas import SchemaError

    truth = observations("2023-12-01", "2023-12-06", 22)
    forecast = climatology.forecast(truth, [np.datetime64("2023-12-05")], [1])
    serialize = xr.Dataset.to_zarr

    def lose_model_id(ds, *args, **kwargs):
        damaged = ds.copy(deep=False)
        del damaged.attrs["model_id"]
        return serialize(damaged, *args, **kwargs)

    monkeypatch.setattr(xr.Dataset, "to_zarr", lose_model_id)
    with pytest.raises(SchemaError, match="model_id"):
        write(forecast, tmp_path / "forecast.zarr")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("init", ["2023-12-16T18", "2024-12-16T18"])
def test_ordered_pools_and_fallback_measurement_cutoffs(model, init):
    truth = observations("2022-11-01", "2024-12-19", 91)
    init = np.datetime64(init, "ns")
    cutoff = min(init, END)
    truth.t2m.values[:] = np.nan
    truth.t2m_offset_min.values[:] = -32768

    def report(time, station, value, offset=0):
        row = dict(valid_time=np.datetime64(time, "ns"), station_id=truth.station_id.values[station])
        truth.t2m.loc[row] = value
        truth.t2m_offset_min.loc[row] = offset

    # Target December 17 at 18Z. Each wider pool has a distinct empirical distribution.
    for time, value in [
        ("2022-12-10T18", 270),
        ("2022-12-11T18", 272),
        ("2022-11-17T18", 280),
        ("2022-11-18T18", 282),
        ("2022-12-01T03", 290),
        ("2022-12-02T04", 292),
    ]:
        report(time, 0, value)
    # A different variable/station cannot fill this station's December pool.
    report("2022-11-16T18", 1, 299)
    report("2022-11-30T04", 1, 299)

    for stage, expected in enumerate(([270, 272], [280, 282], [290, 292])):
        clean = model.forecast(truth, [init], [24], members=8)
        assert clean.attrs["model_version"] == "3"
        np.testing.assert_array_equal(clean.t2m.values[0, 0, 0], np.repeat(expected, 4))
        assert clean.t2m.isel(station_id=1).isnull().all()

        poisoned = truth.copy(deep=True)
        for late, offset in [(cutoff, 1), (cutoff + 24 * HOUR, 0), (init, 1), (init + HOUR, 0)]:
            poisoned.t2m.loc[dict(valid_time=late)] = 330
            poisoned.t2m_offset_min.loc[dict(valid_time=late)] = offset
        xr.testing.assert_identical(clean, model.forecast(poisoned, [init], [24], members=8))
        if stage < 2:
            removed = ["2022-12-10T18", "2022-12-11T18"] if stage == 0 else ["2022-11-17T18", "2022-11-18T18"]
            for time in removed:
                report(time, 0, np.nan, -32768)


def test_wider_pool_wraps_across_new_year():
    truth = observations("2022-12-01", "2023-01-05", 92)
    truth.t2m.values[:] = np.nan
    truth.t2m_offset_min.values[:] = -32768
    truth.t2m.loc[dict(valid_time="2022-12-15T06")] = 277
    truth.t2m_offset_min.loc[dict(valid_time="2022-12-15T06")] = 0
    init = np.datetime64("2023-01-01T00")
    for model in MODELS:
        out = model.forecast(truth, [init], [6])
        assert (out.t2m == 277).all()
