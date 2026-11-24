"""Pre-registered rolling GEFS station correction, version 1."""

from dataclasses import dataclass

import numpy as np
import xarray as xr

from wrfcomp.baselines.io import UNITS, validate

MODEL_ID = "gefs-station-corrected"
WINDOW = np.timedelta64(30, "D")
MAX_OFFSET = np.timedelta64(15, "m")
START = np.datetime64("2024-01-01")
END = np.datetime64("2025-01-01")


@dataclass
class History:
    init: np.datetime64
    mean: xr.Dataset
    variance: xr.Dataset


def summarize(cycle):
    validate(cycle)
    if (
        cycle.attrs["model_id"] != "noaa-gefs"
        or cycle.sizes["ens_member"] != 31
        or cycle.sizes["init_time"] != 1
    ):
        raise ValueError("expected a single 31-member NOAA GEFS cycle")
    init = cycle.init_time.values[0]
    if not START <= init < END or (init + cycle.lead_time.values >= END).any():
        raise ValueError("only unsealed 2024 forecasts are eligible")
    values = cycle[list(UNITS)].isel(init_time=0, drop=True).astype("float64")
    if any(not np.isfinite(values[name].values).all() for name in UNITS):
        raise ValueError("incomplete GEFS cycle")
    return History(init, values.mean("ens_member"), values.var("ens_member", ddof=1))


def forecast(cycle, past, truth):
    """Correct one issue. The caller retains summaries of prior raw forecast cycles."""
    current = summarize(cycle)
    t = current.init
    leads = cycle.lead_time.values
    shape = (len(leads), cycle.sizes["station_id"])
    if not cycle.indexes["station_id"].equals(truth.indexes["station_id"]):
        raise ValueError("truth station order differs from GEFS")
    out = cycle.copy(deep=True)
    out.attrs.update(model_id=MODEL_ID, model_version="1")
    # Accumulate in issue order. Welford's sum of squares is the squared error
    # about the fitted bias, without subtracting nearly equal second moments.
    count = {k: np.zeros(shape, dtype=int) for k in UNITS}
    bias = {k: np.zeros(shape) for k in UNITS}
    squared = {k: np.zeros(shape) for k in UNITS}
    variance = {k: np.zeros(shape) for k in UNITS}
    for item in past:
        age = t - item.init
        if age <= np.timedelta64(0, "h") or age % np.timedelta64(1, "D") != np.timedelta64(0, "h"):
            continue
        valid = item.init + leads
        rows = np.flatnonzero((valid <= t) & (valid + MAX_OFFSET > t - WINDOW))
        if not rows.size:
            continue
        if any(not item.mean.indexes[d].equals(cycle.indexes[d]) for d in ("lead_time", "station_id")):
            raise ValueError("history grid differs from current GEFS")
        observed = truth.sel(valid_time=valid[rows])
        for k in UNITS:
            y = observed[k].values
            offset = observed[k + "_offset_min"].values.astype("float64")
            minutes = (t - valid[rows, None]) / np.timedelta64(1, "m")
            eligible = (
                np.isfinite(y)
                & (abs(offset) <= 15)
                & (offset <= minutes)
                & (offset > minutes - WINDOW / np.timedelta64(1, "m"))
            )
            error = item.mean[k].values[rows] - y
            delta = np.where(eligible, error - bias[k][rows], 0.0)
            count[k][rows] += eligible
            bias[k][rows] += delta / np.maximum(count[k][rows], 1)
            squared[k][rows] += delta * np.where(eligible, error - bias[k][rows], 0.0)
            variance[k][rows] += np.where(eligible, item.variance[k].values[rows], 0.0)

    fallback = {}
    for k in UNITS:
        use = count[k] >= 10
        fallback[k] = int(np.count_nonzero(~use))
        # The two means have the same denominator (eligible pair count).
        ratio = np.where(squared[k] > 0, np.inf, 1.0)
        np.divide(squared[k], variance[k], out=ratio, where=variance[k] > 0)
        scale = np.sqrt(np.maximum(ratio, 0.0)).clip(1.0, 6.0)
        mean = current.mean[k].values
        members = cycle[k].values[0]
        corrected = (mean - bias[k])[..., None] + scale[..., None] * (members - mean[..., None])
        out[k].values[0] = np.where(use[..., None], corrected, members)
    np.minimum(out.td2m.values, out.t2m.values, out=out.td2m.values)
    return out, current, fallback
