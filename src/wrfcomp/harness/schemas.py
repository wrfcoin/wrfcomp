"""Data-contract validators and the scorecard model.

The validators raise `SchemaError` (a `ValueError`) on the first violation and return `None` otherwise.
They read coordinates and metadata only, never the full data arrays, so they are cheap on lazy datasets.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from importlib.metadata import version as _pkg_version
from typing import Annotated

import numpy as np
import xarray as xr
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

import wrfcomp
from wrfcomp.harness.config import HarnessConfig, load_config

# Variable -> units, from the sprint data contract.
VARIABLE_UNITS: dict[str, str] = {"t2m": "K", "td2m": "K", "u10": "m/s", "v10": "m/s", "msl": "Pa"}
# Other spellings of the same unit that are accepted in a `units` attribute.
_UNIT_ALIASES: dict[str, frozenset[str]] = {
    "K": frozenset({"K", "kelvin"}),
    "m/s": frozenset({"m/s", "m s-1", "m s**-1", "m.s-1"}),
    "Pa": frozenset({"Pa", "pascal"}),
}

FORECAST_DIMS = ("init_time", "lead_time", "station_id", "ens_member")
TRUTH_DIMS = ("valid_time", "station_id")
THRESHOLD_DIMS = ("station_id", "month")
STATION_COORDS = ("latitude", "longitude", "elevation_m")
STATION_ID_LENGTH = 11
MIN_ENS_MEMBERS = 2
GATE_NAMES = ("skill", "tail", "spread_skill", "dm")


class SchemaError(ValueError):
    """A dataset does not follow the wrfcomp data contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SchemaError(message)


def _check_dataset(ds: object, kind: str) -> xr.Dataset:
    _require(isinstance(ds, xr.Dataset), f"{kind}: expected an xarray.Dataset, got {type(ds).__name__}")
    return ds  # type: ignore[return-value]


def _check_index(ds: xr.Dataset, kind: str, dim: str) -> np.ndarray:
    _require(dim in ds.dims, f"{kind}: missing dimension '{dim}'")
    _require(dim in ds.coords, f"{kind}: dimension '{dim}' has no coordinate values")
    _require(ds.sizes[dim] > 0, f"{kind}: dimension '{dim}' is empty")
    values = np.asarray(ds[dim].values)
    _require(len(set(values.tolist())) == values.size, f"{kind}: '{dim}' has duplicate values")
    return values


def _check_station_ids(ds: xr.Dataset, kind: str) -> None:
    values = _check_index(ds, kind, "station_id")
    _require(
        values.dtype.kind in "UOT",
        f"{kind}: station_id must be a string coordinate, got dtype {values.dtype}",
    )
    for sid in values.tolist():
        _require(
            isinstance(sid, str) and len(sid) == STATION_ID_LENGTH and sid.isascii() and sid.isalnum(),
            f"{kind}: station_id {sid!r} is not an {STATION_ID_LENGTH}-character USAF+WBAN id",
        )


def _check_station_coords(ds: xr.Dataset, kind: str) -> None:
    for name in STATION_COORDS:
        _require(name in ds.coords, f"{kind}: missing station coordinate '{name}'")
        coord = ds.coords[name]
        _require(
            coord.dims == ("station_id",), f"{kind}: coordinate '{name}' must have dims (station_id,)"
        )
        _require(coord.dtype.kind == "f", f"{kind}: coordinate '{name}' must be floating point")
        _require(bool(np.isfinite(coord.values).all()), f"{kind}: coordinate '{name}' has non-finite values")
    lat, lon = ds.coords["latitude"].values, ds.coords["longitude"].values
    _require(bool(((lat >= -90) & (lat <= 90)).all()), f"{kind}: latitude outside -90..90")
    _require(bool(((lon >= -180) & (lon <= 360)).all()), f"{kind}: longitude outside -180..360")


def _check_variables(
    ds: xr.Dataset, kind: str, dims: tuple[str, ...], required: Iterable[str] | None, dtype: str | None
) -> None:
    names = [str(n) for n in ds.data_vars]
    _require(len(names) > 0, f"{kind}: no data variables")
    for name in names:
        _require(
            name in VARIABLE_UNITS,
            f"{kind}: unknown variable '{name}' (allowed: {', '.join(VARIABLE_UNITS)})",
        )
        var = ds[name]
        _require(
            set(var.dims) == set(dims) and len(var.dims) == len(dims),
            f"{kind}: variable '{name}' has dims {tuple(var.dims)}, expected {dims}",
        )
        if dtype is None:
            _require(var.dtype.kind == "f", f"{kind}: variable '{name}' must be floating point")
        else:
            _require(
                var.dtype == np.dtype(dtype), f"{kind}: variable '{name}' is {var.dtype}, expected {dtype}"
            )
        units = var.attrs.get("units")
        if units is not None:
            _require(
                units in _UNIT_ALIASES[VARIABLE_UNITS[name]],
                f"{kind}: variable '{name}' has units {units!r}, expected {VARIABLE_UNITS[name]!r}",
            )
    for name in sorted(set(required or ())):
        _require(name in names, f"{kind}: required variable '{name}' is missing")


def _check_not_sealed(
    latest: np.datetime64, kind: str, cfg: HarnessConfig | None, allow_sealed: bool
) -> None:
    if allow_sealed:
        return
    sealed_start = (cfg or load_config()).splits.sealed_test.start
    _require(
        latest < sealed_start,
        f"{kind}: contains times at or after the sealed test start {sealed_start} (latest {latest})",
    )


def validate_forecast_dataset(
    ds: xr.Dataset,
    *,
    variables: Iterable[str] | None = None,
    cfg: HarnessConfig | None = None,
    allow_sealed: bool = False,
) -> None:
    """Check a forecast dataset: dims `(init_time, lead_time, station_id, ens_member)`.

    `variables`, if given, must all be present. Unless `allow_sealed` is set, any valid time
    (`init_time + lead_time`) at or after the sealed-test start in `cfg` is refused.
    Finite values at every valid truth point (the zero-cherry-picking rule) need the truth and are
    enforced by `wrfcomp.harness.metrics.score_variable`, not here.
    """
    kind = "forecast"
    ds = _check_dataset(ds, kind)
    for dim in FORECAST_DIMS:
        _check_index(ds, kind, dim)
    _require(ds["init_time"].dtype == np.dtype("datetime64[ns]"), f"{kind}: init_time must be datetime64[ns]")
    _require(
        ds["lead_time"].dtype == np.dtype("timedelta64[ns]"), f"{kind}: lead_time must be timedelta64[ns]"
    )
    _require(not bool(np.isnat(ds["init_time"].values).any()), f"{kind}: init_time has NaT")
    _require(not bool(np.isnat(ds["lead_time"].values).any()), f"{kind}: lead_time has NaT")
    _require(
        bool((ds["lead_time"].values > np.timedelta64(0, "ns")).all()), f"{kind}: lead_time must be positive"
    )
    _require(ds["ens_member"].dtype.kind in "iu", f"{kind}: ens_member must be an integer coordinate")
    _require(
        ds.sizes["ens_member"] >= MIN_ENS_MEMBERS,
        f"{kind}: needs at least {MIN_ENS_MEMBERS} ensemble members, got {ds.sizes['ens_member']}",
    )
    _check_station_ids(ds, kind)
    _check_station_coords(ds, kind)
    _check_variables(ds, kind, FORECAST_DIMS, variables, "float32")
    for attr in ("model_id", "model_version"):
        value = ds.attrs.get(attr)
        _require(isinstance(value, str) and value.strip() != "", f"{kind}: attribute '{attr}' is missing")
    latest = ds["init_time"].values.max() + ds["lead_time"].values.max()
    _check_not_sealed(latest, kind, cfg, allow_sealed)


def validate_truth_dataset(
    ds: xr.Dataset,
    *,
    variables: Iterable[str] | None = None,
    cfg: HarnessConfig | None = None,
    allow_sealed: bool = False,
) -> None:
    """Check a truth dataset: dims `(valid_time, station_id)`, hourly on the full hour, NaN where missing."""
    kind = "truth"
    ds = _check_dataset(ds, kind)
    times = _check_index(ds, kind, "valid_time")
    _require(times.dtype == np.dtype("datetime64[ns]"), f"{kind}: valid_time must be datetime64[ns]")
    _require(not bool(np.isnat(times).any()), f"{kind}: valid_time has NaT")
    hour = np.timedelta64(1, "h")
    _require(
        bool((times == times.astype("datetime64[h]")).all()), f"{kind}: valid_time must be on the full hour"
    )
    _require(
        bool((np.diff(times) == hour).all()),
        f"{kind}: valid_time must be every full hour, increasing, with no gaps (missing hours are NaN rows)",
    )
    _check_station_ids(ds, kind)
    _check_station_coords(ds, kind)
    _check_variables(ds, kind, TRUTH_DIMS, variables, "float32")
    _check_not_sealed(times.max(), kind, cfg, allow_sealed)


def validate_thresholds(ds: xr.Dataset, *, variables: Iterable[str] | None = None) -> None:
    """Check tail thresholds: dims `(station_id, month)`, month exactly 1..12.

    `variables` should be the forecast variables being scored; each needs a threshold variable.
    NaN thresholds (a station with no training data in a month) are legal here; scoring refuses to
    use one at a point that has a valid observation.
    """
    kind = "thresholds"
    ds = _check_dataset(ds, kind)
    months = _check_index(ds, kind, "month")
    _require(months.dtype.kind in "iu", f"{kind}: month must be an integer coordinate")
    _require(months.tolist() == list(range(1, 13)), f"{kind}: month must be exactly 1..12 in order")
    _check_station_ids(ds, kind)
    _check_variables(ds, kind, THRESHOLD_DIMS, variables, None)


# --- Scorecard ---------------------------------------------------------------------------------

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
NonEmptyStr = Annotated[str, StringConstraints(min_length=1)]


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class HarnessVersion(_Strict):
    wrfcomp: NonEmptyStr
    scores: NonEmptyStr


def current_harness_version() -> HarnessVersion:
    """Versions of the installed `wrfcomp` and `scores` packages."""
    return HarnessVersion(wrfcomp=wrfcomp.__version__, scores=_pkg_version("scores"))


class LeadMetrics(_Strict):
    """Metrics for one variable at one lead (or `"all"`). `None` means not defined, never zero."""

    crps: float | None
    twcrps_p95: float | None
    loss: float | None
    spread_skill_ratio: float | None
    n_pairs: int = Field(ge=0)
    crpss_vs_baseline: float | None = None
    baseline_twcrps_p95: float | None = None
    dm_pvalue: float | None = Field(default=None, ge=0.0, le=1.0)


def _check_lead_key(key: str) -> None:
    if key != "all" and not (key.isascii() and key.isdigit() and str(int(key)) == key and int(key) > 0):
        raise ValueError(f"lead key {key!r} must be 'all' or a positive whole number of hours")


class Scorecard(_Strict):
    harness_version: HarnessVersion
    protocol_version: NonEmptyStr
    model_id: NonEmptyStr
    model_version: NonEmptyStr
    baseline_id: str | None = None
    truth_sha256: Sha256
    forecast_sha256: Sha256
    baseline_sha256: Sha256 | None = None
    thresholds_sha256: Sha256
    commit_sha256: Sha256 | None = None
    # First and last init time scored, UTC, timezone-naive (ISO strings without offset in JSON).
    period_start: datetime
    period_end: datetime
    # True when any scored time lies in the sealed-test split. Set by the evaluation CLI.
    sealed_data_used: bool = False
    n_stations: int = Field(ge=0)
    n_init_times: int = Field(ge=0)
    init_hours_scored: list[Annotated[int, Field(ge=0, le=23)]]
    leads_scored: list[Annotated[int, Field(gt=0)]]
    n_pairs: dict[str, Annotated[int, Field(ge=0)]]
    metrics: dict[str, dict[str, LeadMetrics]]
    gates: dict[str, dict[str, bool]]
    passed_gate: bool

    @model_validator(mode="after")
    def _check(self) -> Scorecard:
        if (self.baseline_id is None) != (self.baseline_sha256 is None):
            raise ValueError("baseline_id and baseline_sha256 must be given together")
        for name in ("period_start", "period_end"):
            if getattr(self, name).tzinfo is not None:
                raise ValueError(f"{name} must be timezone-naive UTC (no offset)")
        if self.period_end < self.period_start:
            raise ValueError("period_end is before period_start")
        for name in ("init_hours_scored", "leads_scored"):
            values = getattr(self, name)
            if values != sorted(set(values)):
                raise ValueError(f"{name} must be sorted and unique")
        if set(self.metrics) != set(self.n_pairs) or set(self.metrics) != set(self.gates):
            raise ValueError("metrics, n_pairs and gates must cover the same variables")
        expected_leads = {"all", *(str(h) for h in self.leads_scored)}
        for variable, per_lead in self.metrics.items():
            for key in per_lead:
                _check_lead_key(key)
            if set(per_lead) != expected_leads:
                raise ValueError(f"metrics[{variable!r}] must have exactly 'all' and every scored lead")
            if per_lead["all"].n_pairs != self.n_pairs[variable]:
                raise ValueError(f"n_pairs[{variable!r}] differs from metrics[{variable!r}]['all'].n_pairs")
            if set(self.gates[variable]) != set(GATE_NAMES):
                raise ValueError(f"gates[{variable!r}] must have exactly the gates {GATE_NAMES}")
        expected = len(self.gates) > 0 and all(all(g.values()) for g in self.gates.values())
        if self.passed_gate != expected:
            raise ValueError("passed_gate must be true exactly when every gate of every variable passed")
        if self.passed_gate and self.baseline_id is None:
            raise ValueError("a scorecard without a baseline cannot pass the gate")
        return self
