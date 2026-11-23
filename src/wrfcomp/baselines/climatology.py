"""Deterministic seasonal empirical ensemble, including the New Year wrap."""

import numpy as np
import pandas as pd

from .io import UNITS, boundaries, historical_layout, train_truth


def calendar_day(times):
    # A common leap calendar preserves month/day alignment after February.
    return np.array([t.replace(year=2000).day_of_year for t in times])


def forecast(truth, inits, leads, members=None):
    """Use empirical midpoint quantiles of finite reports; an empty pool yields all NaNs."""
    out = historical_layout(truth, inits, leads, members, "climatology")
    archive = train_truth(truth, out)
    cutoff = min(boundaries()[0], out.init_time.values.min())
    probabilities = (out.ens_member.values + 0.5) / out.sizes["ens_member"]
    dates = archive.valid_time.to_index()
    day = calendar_day(dates)
    for i, init in enumerate(out.init_time.values):
        for j, lead in enumerate(out.lead_time.values):
            target = pd.Timestamp(init + lead)
            distance = abs(day - target.replace(year=2000).day_of_year)
            slots = np.flatnonzero((np.minimum(distance, 366 - distance) <= 7) & (dates.hour == target.hour))
            for name in UNITS:
                sample = archive[name].isel(valid_time=slots).values
                offsets = archive[name + "_offset_min"].isel(valid_time=slots).values
                minutes_left = (cutoff - dates.values[slots]) / np.timedelta64(1, "m")
                eligible = (abs(offsets.astype(float)) <= 15) & (offsets <= minutes_left[:, None])
                sample = np.where(eligible, sample, np.nan)
                for k, column in enumerate(sample.T):
                    observations = column[np.isfinite(column)]
                    if observations.size:
                        out[name].values[i, j, k] = np.quantile(
                            observations, probabilities, method="inverted_cdf"
                        )
    return out
