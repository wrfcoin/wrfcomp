"""Deterministic seasonal empirical ensemble, including the New Year wrap."""

import numpy as np
import pandas as pd

from .io import UNITS, calibration_cutoff, historical_layout, train_truth


def calendar_day(times):
    # A common leap calendar preserves month/day alignment after February.
    return np.array([t.replace(year=2000).day_of_year for t in times])


def forecast(truth, inits, leads, members=None, train_end=None):
    """Use the narrowest eligible seasonal pool at each point.

    `train_end` fixes the calibration cutoff (see `io.calibration_cutoff`) and is recorded in the
    output attributes. Without it the output is exactly what it was before the argument existed.
    """
    out = historical_layout(truth, inits, leads, members, "climatology")
    cutoff = calibration_cutoff(out, train_end)
    archive = train_truth(truth, out, train_end)
    fill_missing_members(out, archive, cutoff)
    if train_end is not None:
        out.attrs["train_end_utc"] = str(cutoff)
    return out


def fill_missing_members(out, archive, cutoff):
    """Fill empty station ensembles in place using the shared training archive."""
    probabilities = (out.ens_member.values + 0.5) / out.sizes["ens_member"]
    dates = archive.valid_time.to_index()
    day = calendar_day(dates)
    for i, init in enumerate(out.init_time.values):
        for j, lead in enumerate(out.lead_time.values):
            target = pd.Timestamp(init + lead)
            distance = abs(day - target.replace(year=2000).day_of_year)
            distance = np.minimum(distance, 366 - distance)
            same_hour = dates.hour == target.hour
            pools = (same_hour & (distance <= 7), same_hour & (distance <= 30), dates.month == target.month)
            for name in UNITS:
                members = out[name].values[i, j]
                for pool in pools:
                    missing = np.flatnonzero(np.isnan(members[:, 0]))
                    if not missing.size:
                        break
                    slots = np.flatnonzero(pool)
                    if not slots.size:
                        continue
                    selection = dict(valid_time=slots, station_id=missing)
                    sample = archive[name].isel(selection).values
                    offsets = archive[name + "_offset_min"].isel(selection).values
                    minutes_left = (cutoff - dates.values[slots]) / np.timedelta64(1, "m")
                    eligible = (abs(offsets.astype(float)) <= 15) & (offsets <= minutes_left[:, None])
                    for column, station in enumerate(missing):
                        values = sample[:, column]
                        finite = values[eligible[:, column] & np.isfinite(values)]
                        if finite.size:
                            members[station] = np.quantile(finite, probabilities, method="inverted_cdf")
