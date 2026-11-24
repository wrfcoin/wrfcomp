"""Frozen C2 local-regression and empirical-residual ensemble."""

from dataclasses import dataclass

import numpy as np
import xarray as xr

from wrfcomp.baselines.io import UNITS, validate
from wrfcomp.models.gefs_station import END, MAX_OFFSET, MODEL_ID

MIN_PAIRS = 10
WINDOW = np.timedelta64(60, "D")
HOUR = np.timedelta64(1, "h")
MINUTE = np.timedelta64(1, "m")
LEADS = np.array([1, 3, 6, 12, 24, 48], dtype="timedelta64[h]")


@dataclass
class History:
    init: np.datetime64
    mean: xr.Dataset


def year_bounds(year):
    start = np.datetime64(f"{year:04d}-01-01", "D")
    end = np.datetime64(f"{year + 1:04d}-01-01", "D")
    if end > END:
        raise ValueError("sealed forecast year")
    return start, end


def base_ensemble(raw, truth, *, year=2024):
    """Build all six leads at the original issue time, also for historical cases."""
    start, end = year_bounds(year)
    validate(raw)
    t = raw.init_time.values[0]
    if (
        raw.attrs["model_id"] != "noaa-gefs"
        or raw.sizes["init_time"] != 1
        or raw.sizes["ens_member"] != 31
        or not np.array_equal(raw.lead_time.values, LEADS[1:])
        or not start <= t < end
        or t + LEADS[-1] >= end
    ):
        raise ValueError("expected one in-year GEFS cycle with 31 members and five leads")
    if not raw.indexes["station_id"].equals(truth.indexes["station_id"]):
        raise ValueError("truth and forecast station grids differ")
    base = raw[list(UNITS)].astype("float64").reindex(lead_time=LEADS.astype("timedelta64[ns]"))
    labels = t - np.arange(4) * HOUR
    slots = truth.reindex(valid_time=labels[labels >= start])
    for k in UNITS:
        if not np.isfinite(raw[k].values).all():
            raise ValueError(f"incomplete GEFS variable {k}")
        three = base[k].values[0, 1]
        one = three.copy()
        missing = np.ones(raw.sizes["station_id"], dtype=bool)
        for lag, (obs, offset) in enumerate(
            zip(slots[k].values, slots[k + "_offset_min"].values, strict=True)
        ):
            age = lag - offset.astype("float64") / 60
            take = (
                missing
                & np.isfinite(obs)
                & (np.abs(offset.astype("float64")) <= 15)
                & (age >= 0)
                & (age <= (t - start) / HOUR)
            )
            o = obs[take, None].astype("float64")
            a = age[take, None]
            one[take] = o + ((a + 1) / (a + 3)) * (three[take] - o)
            missing[take] = False
        base[k].values[0, 0] = one
    return base


def fit_members(members, x, y, eligible, member_ids):
    """Arrays are point x member or point x history (newest issue first)."""
    x0 = members.mean(axis=1, keepdims=True)
    out = x0 + 4 * (members - x0)
    fitted = eligible.sum(axis=1) >= MIN_PAIRS
    if not fitted.any():
        return out, int((~fitted).sum())

    x, y, eligible = x[fitted], y[fitted], eligible[fitted]
    distance = np.where(eligible, abs(x - x0[fitted]), np.inf)
    choice = np.argsort(distance, axis=1, kind="stable")[:, :31]
    # Restore chronological order before the stable sort of equal residuals.
    choice = np.sort(choice, axis=1)[:, ::-1]
    keep = np.take_along_axis(eligible, choice, axis=1)
    x = np.take_along_axis(x, choice, axis=1)
    y = np.take_along_axis(y, choice, axis=1)
    n = keep.sum(axis=1, keepdims=True)
    xbar = np.where(keep, x, 0).sum(axis=1, keepdims=True) / n
    ybar = np.where(keep, y, 0).sum(axis=1, keepdims=True) / n
    dx = np.where(keep, x - xbar, 0)
    dy = np.where(keep, y - ybar, 0)
    denom = (dx * dx).sum(axis=1, keepdims=True)
    varying = np.where(keep, x, -np.inf).max(axis=1) != np.where(keep, x, np.inf).min(axis=1)
    slope = np.divide(
        (dx * dy).sum(axis=1, keepdims=True),
        denom,
        out=np.ones_like(denom),
        where=(denom > 0) & varying[:, None],
    ).clip(0, 1)
    residual = dy - slope * dx
    ordered = np.sort(np.where(keep, residual, np.inf), axis=1, kind="stable")
    ranks = ((np.arange(31) + 0.5) * n / 31).astype(int)
    selected = np.take_along_axis(ordered, ranks, axis=1)
    anomalies = selected - selected.mean(axis=1, keepdims=True)
    sd = selected.std(axis=1, ddof=1, keepdims=True)
    target = np.sqrt((residual**2).sum(axis=1, keepdims=True) / (n - 2) * (1 + 1 / n))
    scale = np.divide(
        target, sd, out=np.zeros_like(sd), where=(sd > 0) & (selected[:, 0:1] != selected[:, -1:])
    )
    ordered_values = ybar + slope * (x0[fitted] - xbar) + anomalies * scale
    member_order = np.lexsort((np.broadcast_to(member_ids, members[fitted].shape), members[fitted]), axis=1)
    assigned = np.empty_like(ordered_values)
    np.put_along_axis(assigned, member_order, ordered_values, axis=1)
    out[fitted] = assigned
    return out, int((~fitted).sum())


def forecast(base, past, truth):
    out, warm, _ = forecast_with_branches(base, past, truth, year=2024)
    return out, warm


def forecast_with_branches(base, past, truth, *, year):
    """Calibrate an issue using causal raw-base means from previous issues."""
    start, end = year_bounds(year)
    t = base.init_time.values[0]
    if not start <= t < end or t + LEADS[-1] >= end:
        raise ValueError("forecast outside requested year")
    history = sorted(
        (
            item
            for item in past
            if start <= item.init < t
            and (t - item.init) % (24 * HOUR) == 0 * HOUR
            and item.init + LEADS[-1] + MAX_OFFSET > t - WINDOW
        ),
        key=lambda item: item.init,
        reverse=True,
    )
    issues = np.array([item.init for item in history], dtype="datetime64[ns]")
    if len(issues) != len(np.unique(issues)):
        raise ValueError("duplicate historical issue")
    for item in history:
        if any(not item.mean.indexes[d].equals(base.indexes[d]) for d in ("lead_time", "station_id")):
            raise ValueError("historical and current forecast grids differ")
    out = base.copy(deep=True)
    out.attrs.update(model_id=MODEL_ID, model_version="2")
    warm = dict.fromkeys(UNITS, 0)
    pressure_warm = np.empty((len(LEADS), base.sizes["station_id"]), dtype=bool)
    for j, lead in enumerate(LEADS):
        valid = issues + lead
        picks = np.flatnonzero((valid <= t) & (valid + MAX_OFFSET > t - WINDOW))
        slots = truth.sel(valid_time=valid[picks])
        minutes = (t - valid[picks]) / MINUTE
        for k in UNITS:
            members = base[k].values[0, j]
            x = (
                np.stack([history[i].mean[k].values[j] for i in picks], axis=1)
                if picks.size
                else np.empty((members.shape[0], 0))
            )
            y = slots[k].values.T.astype("float64")
            offset = slots[k + "_offset_min"].values.T.astype("float64")
            age = minutes - offset
            eligible = (
                np.isfinite(y)
                & (abs(offset) <= 15)
                & (age >= 0)
                & (age < WINDOW / MINUTE)
                & (age <= (t - start) / MINUTE)
            )
            if k == "msl":
                pressure_warm[j] = eligible.sum(axis=1) < MIN_PAIRS
            result, n_warm = fit_members(members, x, y, eligible, base.ens_member.values)
            out[k].values[0, j] = result
            warm[k] += n_warm
    np.minimum(out.td2m.values, out.t2m.values, out=out.td2m.values)
    return out.astype("float32"), warm, pressure_warm
