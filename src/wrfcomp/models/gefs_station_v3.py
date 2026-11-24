"""Registered C3 pressure-only calibration of as-issued float32 v2 ensembles."""

from collections import deque
from dataclasses import dataclass

import numpy as np
import xarray as xr

from wrfcomp.models import gefs_station_v2 as v2
from wrfcomp.models.gefs_station import MAX_OFFSET

WINDOW = np.timedelta64(30, "D")


@dataclass
class PressureIssue:
    init: np.datetime64
    mean: np.ndarray
    variance: np.ndarray
    warm: np.ndarray


def pressure_gain(errors, variances, station_count, warm):
    """Inputs in ascending issue and station-ID order; each row is one eligible pair."""
    prior = 0.5 if warm else 1.0
    n = errors.size
    if not n:
        return prior
    b = np.cumsum(variances, dtype=np.float64)[-1]
    if b == 0:
        return prior
    a = np.cumsum(np.square(errors), dtype=np.float64)[-1]
    return np.sqrt((n * (a / b) + station_count * prior**2) / (n + station_count))


def adjust_pressure(core, pressure_warm, past, truth, year):
    """Return v3 and the unmodified v2 moments to use when this issue verifies."""
    start, end = v2.year_bounds(year)
    t = core.init_time.values[0]
    if not start <= t or t + v2.LEADS[-1] >= end or core.attrs["model_version"] != "2":
        raise ValueError("expected an in-year v2 issue")
    if not core.indexes["station_id"].equals(truth.indexes["station_id"]):
        raise ValueError("truth and forecast station grids differ")
    msl = core.msl.values[0].astype(np.float64)
    mean = msl.mean(axis=-1)
    issue = PressureIssue(t, mean, msl.var(axis=-1, ddof=1) * (32 / 31), pressure_warm)
    indices = np.argsort(core.station_id.values)
    stations = indices.size
    history = sorted(
        (
            item
            for item in past
            if start <= item.init < t
            and (t - item.init) % (24 * v2.HOUR) == 0 * v2.HOUR
            and item.init + v2.LEADS[-1] + MAX_OFFSET > t - WINDOW
        ),
        key=lambda item: item.init,
    )
    dates = np.array([item.init for item in history], dtype="datetime64[ns]")
    if len(np.unique(dates)) != len(dates):
        raise ValueError("duplicate historical pressure issue")

    adjusted = np.empty_like(msl, dtype=np.float32)
    for j, lead in enumerate(v2.LEADS):
        labels = dates + lead
        select = np.flatnonzero((labels <= t) & (labels + MAX_OFFSET > t - WINDOW))
        gains = np.where(pressure_warm[j], 0.5, 1.0)
        if select.size:
            slots = truth.sel(valid_time=labels[select])
            y = slots.msl.values[:, indices].astype(np.float64)
            offsets = slots.msl_offset_min.values[:, indices].astype(np.float64)
            age_min = ((t - labels[select]) / v2.MINUTE)[:, None] - offsets
            ok = (
                np.isfinite(y)
                & (abs(offsets) <= 15)
                & (age_min >= 0)
                & (age_min < WINDOW / v2.MINUTE)
                & (age_min <= (t - start) / v2.MINUTE)
            )
            error = np.stack([history[i].mean[j, indices] for i in select]) - y
            variance = np.stack([history[i].variance[j, indices] for i in select])
            warm = np.stack([history[i].warm[j, indices] for i in select])
            for branch in (False, True):
                keep = ok & (warm == branch)
                gains[pressure_warm[j] == branch] = pressure_gain(
                    error[keep], variance[keep], stations, branch
                )
        adjusted[j] = mean[j, :, None] + gains[:, None] * (msl[j] - mean[j, :, None])
    out = core.copy(deep=False)
    out.attrs = {**core.attrs, "model_version": "3"}
    out["msl"] = core.msl.copy(data=adjusted[None])
    return out, issue


class Replay:
    """One ordered same-year replay. Monthly publication never resets either history."""

    def __init__(self, year):
        self.year = year
        self.base_history = deque()
        self.pressure_history = deque()

    def issue(self, raw: xr.Dataset, truth: xr.Dataset):
        base = v2.base_ensemble(raw, truth, year=self.year)
        t = base.init_time.values[0]
        if self.base_history and t <= self.base_history[-1].init:
            raise ValueError("initializations must increase")
        for history, window in ((self.base_history, v2.WINDOW), (self.pressure_history, WINDOW)):
            while history and history[0].init + v2.LEADS[-1] + MAX_OFFSET <= t - window:
                history.popleft()
        core, counts, pressure_warm = v2.forecast_with_branches(
            base, self.base_history, truth, year=self.year
        )
        out, moments = adjust_pressure(core, pressure_warm, self.pressure_history, truth, self.year)
        self.pressure_history.append(moments)
        self.base_history.append(v2.History(t, base.isel(init_time=0, drop=True).mean("ens_member")))
        return out, counts
