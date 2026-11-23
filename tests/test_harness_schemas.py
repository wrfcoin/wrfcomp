"""Config loader, data-contract validators and the Scorecard model. Synthetic data, fixed seeds."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml
from pydantic import ValidationError

from wrfcomp.harness.config import DEFAULT_CONFIG_PATH, HarnessConfig, load_config
from wrfcomp.harness.schemas import (
    GATE_NAMES,
    SchemaError,
    Scorecard,
    current_harness_version,
    validate_forecast_dataset,
    validate_thresholds,
    validate_truth_dataset,
)

STATIONS = ["72011053983", "72015103049", "72406093721"]
VARIABLES = ["t2m", "td2m", "u10", "v10", "msl"]


def _station_coords() -> dict:
    return {
        "station_id": STATIONS,
        "latitude": ("station_id", np.array([30.784, 30.383, 39.173])),
        "longitude": ("station_id", np.array([-98.662, -103.683, -76.684])),
        "elevation_m": ("station_id", np.array([335.9, 1375.6, 47.5])),
    }


def make_forecast(n_members: int = 4) -> xr.Dataset:
    rng = np.random.default_rng(0)
    init = pd.date_range("2024-03-01", periods=4, freq="6h").values.astype("datetime64[ns]")
    lead = np.array([1, 6, 24], dtype="timedelta64[h]").astype("timedelta64[ns]")
    shape = (len(init), len(lead), len(STATIONS), n_members)
    dims = ("init_time", "lead_time", "station_id", "ens_member")
    data = {v: (dims, rng.normal(size=shape).astype(np.float32)) for v in VARIABLES}
    coords = {"init_time": init, "lead_time": lead, "ens_member": np.arange(n_members), **_station_coords()}
    return xr.Dataset(data, coords=coords, attrs={"model_id": "synthetic", "model_version": "t0"})


def make_truth() -> xr.Dataset:
    rng = np.random.default_rng(1)
    times = pd.date_range("2024-03-01", periods=48, freq="1h").values.astype("datetime64[ns]")
    shape = (len(times), len(STATIONS))
    data = {v: (("valid_time", "station_id"), rng.normal(size=shape).astype(np.float32)) for v in VARIABLES}
    ds = xr.Dataset(data, coords={"valid_time": times, **_station_coords()})
    ds["t2m"][3, 1] = np.nan  # missing observations are legal
    return ds


def make_thresholds() -> xr.Dataset:
    rng = np.random.default_rng(2)
    shape = (len(STATIONS), 12)
    data = {v: (("station_id", "month"), rng.normal(size=shape).astype(np.float32)) for v in VARIABLES}
    return xr.Dataset(data, coords={"station_id": STATIONS, "month": np.arange(1, 13)})


# --- config ------------------------------------------------------------------------------------


def test_load_config_matches_yaml() -> None:
    cfg = load_config()
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    gate = raw["evaluation"]["promotion_gate"]
    assert cfg.protocol_version == raw["protocol_version"]
    assert cfg.splits.train.end == np.datetime64("2023-12-17T18:00:00", "ns")
    assert cfg.splits.validation.start == np.datetime64("2024-01-01T00:00:00", "ns")
    assert cfg.splits.sealed_test.start == np.datetime64("2025-01-01T00:00:00", "ns")
    assert cfg.splits.sealed_test.end is None
    assert cfg.evaluation.init_hours_utc == tuple(raw["evaluation"]["init_hours_utc"])
    assert cfg.evaluation.lead_hours == tuple(raw["evaluation"]["lead_hours"])
    assert cfg.evaluation.variables.primary == tuple(raw["evaluation"]["variables"]["primary"])
    assert cfg.gate.alpha_dm == gate["alpha_dm"]
    assert cfg.gate.spread_skill_min == gate["spread_skill_min"]
    assert cfg.gate.spread_skill_max == gate["spread_skill_max"]
    assert cfg.gate.tail_quantile == gate["tail_quantile"]
    assert cfg.gate.loss_weights.crps_fair == gate["loss_weights"]["crps_fair"]
    assert cfg.gate.loss_weights.twcrps_p95 == gate["loss_weights"]["twcrps_p95"]


def test_config_is_frozen() -> None:
    cfg = load_config()
    with pytest.raises(ValidationError):
        cfg.protocol_version = "9.9.9"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        cfg.gate.alpha_dm = 0.5  # type: ignore[misc]
    assert isinstance(cfg.evaluation.lead_hours, tuple)


def _mutated(path: tuple[str, ...], value: object) -> dict:
    raw = copy.deepcopy(yaml.safe_load(DEFAULT_CONFIG_PATH.read_text()))
    node = raw
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return raw


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("evaluation", "promotion_gate", "loss_weights", "crps_fair"), 0.9),  # weights no longer sum to 1
        (("evaluation", "promotion_gate", "spread_skill_min"), 1.2),  # band inverted
        (("evaluation", "promotion_gate", "tail_threshold_source"), "validation"),
        (("splits", "train", "end_utc"), "2024-06-01T00:00:00Z"),  # train overlaps validation
        (("splits", "validation", "end_utc"), "2025-06-01T00:00:00Z"),  # validation overlaps sealed
        (("splits", "train", "end_utc"), "2023-12-17T18:00:00"),  # no UTC offset
        (("evaluation", "unexpected_key"), 1),
    ],
)
def test_config_rejects_bad_values(path: tuple[str, ...], value: object) -> None:
    with pytest.raises(ValidationError):
        HarnessConfig.model_validate(_mutated(path, value))


def test_load_config_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


# --- forecast ----------------------------------------------------------------------------------


def test_valid_forecast_passes() -> None:
    validate_forecast_dataset(make_forecast(), variables=VARIABLES)


def test_forecast_with_units_attrs() -> None:
    ds = make_forecast()
    ds["t2m"].attrs["units"] = "K"
    ds["u10"].attrs["units"] = "m s-1"
    validate_forecast_dataset(ds)
    ds["msl"].attrs["units"] = "hPa"
    with pytest.raises(SchemaError, match="units"):
        validate_forecast_dataset(ds)


def _break_forecast(case: str) -> xr.Dataset:
    ds = make_forecast()
    if case == "one_member":
        return ds.isel(ens_member=[0])
    if case == "float64":
        return ds.astype(np.float64)
    if case == "missing_dim":
        return ds.isel(ens_member=0, drop=True)
    if case == "lead_as_int":
        return ds.assign_coords(lead_time=[1, 6, 24])
    if case == "init_seconds":
        return ds.assign_coords(init_time=ds["init_time"].values.astype("datetime64[s]"))
    if case == "member_float":
        return ds.assign_coords(ens_member=ds["ens_member"].values.astype(np.float64))
    if case == "short_station_id":
        return ds.assign_coords(station_id=["7201105398", *STATIONS[1:]])
    if case == "int_station_id":
        return ds.assign_coords(station_id=[72011053983, 72015103049, 72406093721])
    if case == "duplicate_station":
        return ds.assign_coords(station_id=[STATIONS[0], STATIONS[0], STATIONS[2]])
    if case == "no_elevation":
        return ds.drop_vars("elevation_m")
    if case == "bad_latitude":
        return ds.assign_coords(latitude=("station_id", np.array([130.0, 30.0, 39.0])))
    if case == "no_model_id":
        ds.attrs.pop("model_id")
        return ds
    if case == "unknown_variable":
        return ds.assign(rh2m=ds["t2m"])
    if case == "missing_required":
        return ds.drop_vars("msl")
    if case == "zero_lead":
        leads = np.array([0, 6, 24], dtype="timedelta64[h]").astype("timedelta64[ns]")
        return ds.assign_coords(lead_time=leads)
    raise AssertionError(case)


@pytest.mark.parametrize(
    "case",
    [
        "one_member",
        "float64",
        "missing_dim",
        "lead_as_int",
        "init_seconds",
        "member_float",
        "short_station_id",
        "int_station_id",
        "duplicate_station",
        "no_elevation",
        "bad_latitude",
        "no_model_id",
        "unknown_variable",
        "missing_required",
        "zero_lead",
    ],
)
def test_forecast_violations_are_refused(case: str) -> None:
    with pytest.raises(SchemaError):
        validate_forecast_dataset(_break_forecast(case), variables=VARIABLES)


def test_forecast_reaching_sealed_period_is_refused() -> None:
    ds = make_forecast()
    # last init 2024-12-31T18 + 24 h lead = 2025-01-01T18, inside the sealed period
    init = pd.date_range("2024-12-31", periods=4, freq="6h").values.astype("datetime64[ns]")
    sealed = ds.assign_coords(init_time=init)
    with pytest.raises(SchemaError, match="sealed"):
        validate_forecast_dataset(sealed)
    validate_forecast_dataset(sealed, allow_sealed=True)
    # one step earlier: last valid time 2024-12-31T18 is still allowed
    validate_forecast_dataset(ds.assign_coords(init_time=init - np.timedelta64(1, "D")))


def test_not_a_dataset() -> None:
    with pytest.raises(SchemaError):
        validate_forecast_dataset(make_forecast()["t2m"])  # type: ignore[arg-type]


# --- truth -------------------------------------------------------------------------------------


def test_valid_truth_passes() -> None:
    validate_truth_dataset(make_truth(), variables=VARIABLES)


def _break_truth(case: str) -> xr.Dataset:
    ds = make_truth()
    if case == "gap":
        return ds.isel(valid_time=[*range(10), *range(11, 48)])
    if case == "half_hour":
        return ds.assign_coords(valid_time=ds["valid_time"].values + np.timedelta64(30, "m"))
    if case == "three_hourly":
        return ds.isel(valid_time=slice(None, None, 3))
    if case == "descending":
        return ds.isel(valid_time=slice(None, None, -1))
    if case == "float64":
        return ds.astype(np.float64)
    if case == "ens_dim":
        return ds.expand_dims(ens_member=[0, 1])
    if case == "no_latitude":
        return ds.drop_vars("latitude")
    if case == "sealed":
        times = pd.date_range("2024-12-31", periods=48, freq="1h").values.astype("datetime64[ns]")
        return ds.assign_coords(valid_time=times)
    raise AssertionError(case)


@pytest.mark.parametrize(
    "case", ["gap", "half_hour", "three_hourly", "descending", "float64", "ens_dim", "no_latitude", "sealed"]
)
def test_truth_violations_are_refused(case: str) -> None:
    with pytest.raises(SchemaError):
        validate_truth_dataset(_break_truth(case))


# --- thresholds --------------------------------------------------------------------------------


def test_valid_thresholds_pass() -> None:
    validate_thresholds(make_thresholds(), variables=VARIABLES)


def _break_thresholds(case: str) -> xr.Dataset:
    ds = make_thresholds()
    if case == "eleven_months":
        return ds.isel(month=slice(0, 11))
    if case == "months_from_zero":
        return ds.assign_coords(month=np.arange(12))
    if case == "missing_variable":
        return ds.drop_vars("u10")
    if case == "extra_dim":
        return ds.expand_dims(quantile=[0.95])
    if case == "integer_values":
        return ds.astype(np.int32)
    raise AssertionError(case)


@pytest.mark.parametrize(
    "case", ["eleven_months", "months_from_zero", "missing_variable", "extra_dim", "integer_values"]
)
def test_threshold_violations_are_refused(case: str) -> None:
    with pytest.raises(SchemaError):
        validate_thresholds(_break_thresholds(case), variables=VARIABLES)


# --- scorecard ---------------------------------------------------------------------------------

SHA_A, SHA_B, SHA_C, SHA_D = ("a" * 64, "b" * 64, "c" * 64, "d" * 64)


def _lead_metrics(n_pairs: int) -> dict:
    return {
        "crps": 0.8,
        "twcrps_p95": 0.1,
        "loss": 0.59,
        "spread_skill_ratio": 0.97,
        "n_pairs": n_pairs,
        "crpss_vs_baseline": 0.12,
        "baseline_twcrps_p95": 0.11,
        "dm_pvalue": 0.001,
    }


def scorecard_payload() -> dict:
    return {
        "harness_version": current_harness_version().model_dump(),
        "protocol_version": "0.1.0",
        "model_id": "synthetic",
        "model_version": "t0",
        "baseline_id": "persistence",
        "truth_sha256": SHA_A,
        "forecast_sha256": SHA_B,
        "baseline_sha256": SHA_C,
        "thresholds_sha256": SHA_D,
        "n_stations": 3,
        "n_init_times": 4,
        "init_hours_scored": [0, 6, 12, 18],
        "leads_scored": [1, 24],
        "n_pairs": {"t2m": 24},
        "metrics": {"t2m": {"all": _lead_metrics(24), "1": _lead_metrics(12), "24": _lead_metrics(12)}},
        "gates": {"t2m": dict.fromkeys(GATE_NAMES, True)},
        "passed_gate": True,
    }


def test_scorecard_round_trips_through_json() -> None:
    card = Scorecard.model_validate(scorecard_payload())
    assert card.commit_sha256 is None
    assert card.harness_version.scores
    again = Scorecard.model_validate_json(card.model_dump_json())
    assert again == card
    assert again.metrics["t2m"]["24"].crps == 0.8


def test_scorecard_without_baseline() -> None:
    payload = scorecard_payload()
    payload["baseline_id"] = None
    payload["baseline_sha256"] = None
    for entry in payload["metrics"]["t2m"].values():
        for key in ("crpss_vs_baseline", "baseline_twcrps_p95", "dm_pvalue"):
            entry.pop(key)
    with pytest.raises(ValidationError, match="baseline"):
        Scorecard.model_validate(payload)  # still claims passed_gate
    payload["gates"] = {"t2m": {"skill": False, "tail": False, "spread_skill": True, "dm": False}}
    payload["passed_gate"] = False
    card = Scorecard.model_validate(payload)
    assert card.metrics["t2m"]["all"].dm_pvalue is None


def _break_scorecard(case: str) -> dict:
    p = scorecard_payload()
    if case == "short_sha":
        p["truth_sha256"] = "abc123"
    elif case == "uppercase_sha":
        p["forecast_sha256"] = "A" * 64
    elif case == "bad_lead_key":
        p["metrics"]["t2m"]["24h"] = p["metrics"]["t2m"].pop("24")
    elif case == "missing_all":
        p["metrics"]["t2m"].pop("all")
    elif case == "lead_not_listed":
        p["leads_scored"] = [1]
    elif case == "passed_but_gate_failed":
        p["gates"]["t2m"]["dm"] = False
    elif case == "failed_but_gates_passed":
        p["passed_gate"] = False
    elif case == "missing_gate":
        p["gates"]["t2m"].pop("tail")
    elif case == "n_pairs_mismatch":
        p["n_pairs"]["t2m"] = 23
    elif case == "variable_without_gates":
        p["gates"] = {}
    elif case == "nan_metric":
        p["metrics"]["t2m"]["1"]["crps"] = float("nan")
    elif case == "pvalue_above_one":
        p["metrics"]["t2m"]["1"]["dm_pvalue"] = 1.5
    elif case == "baseline_id_without_sha":
        p["baseline_sha256"] = None
    elif case == "extra_field":
        p["passed_significance_gate"] = True
    elif case == "unsorted_leads":
        p["leads_scored"] = [24, 1]
    elif case == "empty_model_id":
        p["model_id"] = ""
    else:
        raise AssertionError(case)
    return p


@pytest.mark.parametrize(
    "case",
    [
        "short_sha",
        "uppercase_sha",
        "bad_lead_key",
        "missing_all",
        "lead_not_listed",
        "passed_but_gate_failed",
        "failed_but_gates_passed",
        "missing_gate",
        "n_pairs_mismatch",
        "variable_without_gates",
        "nan_metric",
        "pvalue_above_one",
        "baseline_id_without_sha",
        "extra_field",
        "unsorted_leads",
        "empty_model_id",
    ],
)
def test_scorecard_violations_are_refused(case: str) -> None:
    with pytest.raises(ValidationError):
        Scorecard.model_validate(_break_scorecard(case))
