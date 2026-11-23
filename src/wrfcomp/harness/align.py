"""Align station truth to a forecast, and the two refusals that protect the score.

- :func:`align_truth` puts truth on ``(init_time, lead_time, station_id)`` by exact match on
  ``valid_time = init_time + lead_time``. The ±15 minute tolerance is applied when truth is built, not here.
  A valid time (or station) that truth does not have gives NaN: never an error, never a neighbour's value.
- :func:`check_not_sealed` refuses anything that touches the sealed split.
- :func:`check_nan_rule` refuses a forecast with a non-finite member where truth is finite.
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
