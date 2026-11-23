"""Align station truth to a forecast, and the two refusals that protect the score.

- :func:`align_truth` puts truth on ``(init_time, lead_time, station_id)`` by exact match on
  ``valid_time = init_time + lead_time``. The ±15 minute tolerance is applied when truth is built, not here.
  A valid time (or station) that truth does not have gives NaN: never an error, never a neighbour's value.
- :func:`check_not_sealed` refuses anything that touches the sealed split.
- :func:`check_nan_rule` refuses a forecast with a non-finite member where truth is finite.
- :func:`check_stations`, :func:`check_init_times_contiguous` and :func:`check_baseline_covers` refuse a
  forecast that leaves out whole stations or dates, which the NaN rule alone cannot see.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import xarray as xr

POINT_DIMS = ("init_time", "lead_time", "station_id")
ENS_DIM = "ens_member"
SEALED_START = np.datetime64("2025-01-01T00:00:00", "ns")
MAX_LISTED = 10


class SealedSplitError(ValueError):
    """A dataset touches the sealed test split and ``allow_sealed`` was not given."""


class ForecastNaNError(ValueError):
    """A forecast has a non-finite member at a point where truth is finite."""

    def __init__(self, role: str, n_points: int, offenders: list[tuple[str, str, str, str]]) -> None:
        self.role = role
        self.n_points = n_points
        self.offenders = offenders
        lines = [
            f"{role} forecast is not finite at {n_points} point(s) where truth is valid; "
            f"first {len(offenders)} as (variable, init_time, lead_time, station_id):"
        ]
        lines += [f"  ({v}, {i}, {lead}, {s})" for v, i, lead, s in offenders]
        super().__init__("\n".join(lines))


def valid_times(forecast: xr.Dataset | xr.DataArray) -> xr.DataArray:
    """``init_time + lead_time`` on dims ``(init_time, lead_time)``, as ``datetime64[ns]``."""
    init = forecast["init_time"].astype("datetime64[ns]")
    lead = forecast["lead_time"].astype("timedelta64[ns]")
    valid = (init + lead).transpose("init_time", "lead_time")
    return valid.astype("datetime64[ns]").rename("valid_time")


def check_not_sealed(
    times: xr.DataArray | np.ndarray,
    *,
    what: str,
    sealed_start: np.datetime64 = SEALED_START,
    allow_sealed: bool = False,
) -> bool:
    """Refuse times at or after ``sealed_start``. Returns True if sealed times are present and allowed."""
    values = np.asarray(times).astype("datetime64[ns]").ravel()
    values = values[~np.isnat(values)]
    if values.size == 0 or values.max() < sealed_start:
        return False
    n = int((values >= sealed_start).sum())
    start, latest = (np.datetime_as_string(t, unit="s") for t in (sealed_start, values.max()))
    message = f"{what}: {n} time(s) fall in the sealed split (>= {start}Z), latest {latest}Z"
    if not allow_sealed:
        raise SealedSplitError(message + ". Refused. Sealed data needs --allow-sealed and a human decision.")
    warnings.warn("SEALED SPLIT IN USE. " + message, stacklevel=2)
    return True


def check_forecast_not_sealed(
    forecast: xr.Dataset,
    *,
    what: str = "forecast",
    sealed_start: np.datetime64 = SEALED_START,
    allow_sealed: bool = False,
) -> bool:
    """Apply :func:`check_not_sealed` to the forecast's init times and valid times."""
    a = check_not_sealed(
        forecast["init_time"], what=f"{what} init_time", sealed_start=sealed_start, allow_sealed=allow_sealed
    )
    b = check_not_sealed(
        valid_times(forecast), what=f"{what} valid_time", sealed_start=sealed_start, allow_sealed=allow_sealed
    )
    return a or b


def align_truth(
    forecast: xr.Dataset,
    truth: xr.Dataset,
    *,
    sealed_start: np.datetime64 = SEALED_START,
    allow_sealed: bool = False,
) -> xr.Dataset:
    """Truth on the forecast's ``(init_time, lead_time, station_id)`` points.

    Returns every truth data variable that the forecast also has, with a ``valid_time`` coordinate on
    ``(init_time, lead_time)``. Only the truth rows at the forecast's valid times are read.
    """
    check_forecast_not_sealed(forecast, sealed_start=sealed_start, allow_sealed=allow_sealed)
    # The truth file as a whole, not only the rows used: nothing sealed is opened by accident.
    check_not_sealed(
        truth["valid_time"], what="truth valid_time", sealed_start=sealed_start, allow_sealed=allow_sealed
    )
    for dim in ("valid_time", "station_id"):
        if not truth.indexes[dim].is_unique:
            raise ValueError(f"truth {dim} has duplicate values")
    names = [name for name in truth.data_vars if name in forecast.data_vars]
    if not names:
        raise ValueError("truth and forecast share no variables")

    valid = valid_times(forecast)
    wanted_times = pd.DatetimeIndex(np.unique(valid.values))
    wanted_stations = forecast.indexes["station_id"]
    # reindex is an exact label match; labels truth does not have become NaN rows.
    sub = truth[names].reindex(valid_time=wanted_times, station_id=wanted_stations)
    obs = sub.sel(valid_time=valid.drop_vars(["valid_time"], errors="ignore"))
    obs = obs.assign_coords(
        init_time=forecast["init_time"].values,
        lead_time=forecast["lead_time"].values,
        valid_time=(("init_time", "lead_time"), valid.values),
    )
    return obs.transpose(*POINT_DIMS)


def check_nan_rule(forecast: xr.Dataset, obs: xr.Dataset, *, role: str = "candidate") -> None:
    """Raise :class:`ForecastNaNError` if truth is finite somewhere and a forecast member there is not.

    ``obs`` is truth already aligned to the forecast. Offenders are listed in variable name order, then
    in ``(init_time, lead_time, station_id)`` index order, up to ``MAX_LISTED``.
    """
    offenders: list[tuple[str, str, str, str]] = []
    total = 0
    for name in sorted(str(n) for n in obs.data_vars):
        if name not in forecast.data_vars:
            continue
        fc_ok = np.isfinite(forecast[name]).all(ENS_DIM).transpose(*POINT_DIMS)
        truth_ok = np.isfinite(obs[name]).transpose(*POINT_DIMS)
        bad = np.asarray((truth_ok & ~fc_ok).values)
        count = int(bad.sum())
        if count == 0:
            continue
        total += count
        room = MAX_LISTED - len(offenders)
        for i, j, k in np.argwhere(bad)[: max(room, 0)]:
            offenders.append(
                (
                    name,
                    np.datetime_as_string(obs["init_time"].values[i].astype("datetime64[s]")) + "Z",
                    f"{int(obs['lead_time'].values[j] / np.timedelta64(1, 'h'))}h",
                    str(obs["station_id"].values[k]),
                )
            )
    if total:
        raise ForecastNaNError(role, total, offenders)


class CoverageError(ValueError):
    """A forecast leaves out stations or init times it is required to have."""

    def __init__(self, role: str, what: str, missing: list[str]) -> None:
        self.role = role
        self.what = what
        self.missing = missing
        shown = missing[:MAX_LISTED]
        lines = [f"{role} forecast is missing {len(missing)} required {what}; first {len(shown)}:"]
        lines += [f"  {item}" for item in shown]
        super().__init__("\n".join(lines))


def _stamp(values: np.ndarray) -> list[str]:
    return [np.datetime_as_string(v, unit="s") + "Z" for v in np.asarray(values).astype("datetime64[s]")]


def expected_init_times(init_times: np.ndarray) -> np.ndarray:
    """Every time from the first to the last init time whose hour of day is one the forecast uses.

    The forecast's cadence is the set of hours of day its init times fall on (for example 00 and 12).
    Init times must be on whole hours; the caller checks that.
    """
    init = np.sort(np.asarray(init_times).astype("datetime64[ns]"))
    hours = pd.DatetimeIndex(init).hour
    every_hour = pd.date_range(init[0], init[-1], freq="1h")
    return every_hour[every_hour.hour.isin(np.unique(hours))].values.astype("datetime64[ns]")


def check_init_times_contiguous(forecast: xr.Dataset, *, role: str = "candidate") -> None:
    """Raise :class:`CoverageError` if a date is missing between the first and the last init time."""
    have = forecast["init_time"].values.astype("datetime64[ns]")
    expected = expected_init_times(have)
    missing = expected[~np.isin(expected, have)]
    if missing.size:
        raise CoverageError(role, "init time(s)", _stamp(missing))


def check_stations(forecast: xr.Dataset, required: list[str], *, role: str = "candidate") -> None:
    """Raise :class:`CoverageError` if the forecast lacks any required station."""
    have = {str(s) for s in forecast["station_id"].values}
    missing = sorted(s for s in set(required) if s not in have)
    if missing:
        raise CoverageError(role, "station(s)", missing)


def check_baseline_covers(candidate: xr.Dataset, baseline: xr.Dataset) -> None:
    """Raise :class:`CoverageError` if the baseline lacks an init time of the candidate.

    Stations are checked separately against the required list, for both forecasts.
    """
    want = candidate["init_time"].values.astype("datetime64[ns]")
    have = baseline["init_time"].values.astype("datetime64[ns]")
    missing = np.sort(want[~np.isin(want, have)])
    if missing.size:
        raise CoverageError("baseline", "init time(s) of the candidate", _stamp(missing))
