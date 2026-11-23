"""Deterministic seasonal empirical ensemble, including the New Year wrap."""

import numpy as np
import pandas as pd

from .io import UNITS, layout, train_truth


def calendar_day(times):
    # A common leap calendar preserves month/day alignment after February.
    return np.array([t.replace(year=2000).day_of_year for t in times])


def forecast(truth, inits, leads, members=30):
    out = layout(truth, inits, leads, members, "climatology")
    archive = train_truth(truth, out)
    dates = archive.valid_time.to_index()
    day = calendar_day(dates)
    for i, init in enumerate(out.init_time.values):
        for j, lead in enumerate(out.lead_time.values):
            target = pd.Timestamp(init + lead)
            distance = abs(day - target.replace(year=2000).day_of_year)
            slots = np.flatnonzero((np.minimum(distance, 366 - distance) <= 7) & (dates.hour == target.hour))
            for name in UNITS:
                sample = archive[name].isel(valid_time=slots).values
                for k, column in enumerate(sample.T):
                    observations = column[np.isfinite(column)]
                    if observations.size:
                        idx = np.linspace(0, observations.size - 1, members).round().astype(int)
                        out[name].values[i, j, k] = observations[idx]
    return out
