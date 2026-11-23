"""Offset-aware latest observations and empirical lead-specific error quantiles."""

import numpy as np

from .io import UNITS, boundaries, layout, train_truth


def latest(truth, name, times):
    """Latest eligible hourly slot, independently for each variable and station."""
    slots = truth.valid_time.values
    out = np.full((len(times), truth.sizes["station_id"]), np.nan, dtype="f4")
    for lag in range(4):
        labels = times - np.timedelta64(lag, "h")
        ix = np.searchsorted(slots, labels)
        rows = np.flatnonzero(ix < len(slots))
        rows = rows[slots[ix[rows]] == labels[rows]]
        if not rows.size:
            continue
        sel = dict(valid_time=ix[rows])
        offsets = truth[name + "_offset_min"].isel(sel).values
        values = truth[name].isel(sel).values
        usable = (abs(offsets.astype(float)) <= 15) & (offsets <= 60 * lag) & np.isfinite(values)
        out[rows] = np.where(np.isnan(out[rows]) & usable, values, out[rows])
    return out


def forecast(truth, inits, leads, members=30):
    out = layout(truth, inits, leads, members, "persistence")
    archive = train_truth(truth, out)
    dates = archive.valid_time.to_index()
    times = out.init_time.values
    end, _ = boundaries()
    cutoff = min(times.min(), end)
    qs = (np.arange(members) + 0.5) / members
    for name in UNITS:
        now = latest(truth, name, times)
        for j, lead in enumerate(out.lead_time.values):
            for hour in np.unique(out.init_time.dt.hour):
                rows = np.flatnonzero(out.init_time.dt.hour.values == hour)
                starts = dates.values[(dates.hour == hour) & (dates.values + lead <= cutoff)]
                if not starts.size:
                    continue
                targets = starts + lead
                observed = archive[name].reindex(valid_time=targets).values
                offset = archive[name + "_offset_min"].reindex(valid_time=targets).values
                usable = (abs(offset.astype(float)) <= 15) & (
                    np.nan_to_num(offset) <= (cutoff - targets[:, None]) / np.timedelta64(1, "m")
                )
                errors = np.where(usable, observed, np.nan) - latest(archive, name, starts)
                for k, series in enumerate(errors.T):
                    finite = series[np.isfinite(series)]
                    if finite.size:
                        out[name].values[rows, j, k] = now[rows, k, None] + np.quantile(finite, qs)
    return out
