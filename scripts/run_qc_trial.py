"""Offline July 2024 QC trial. Output consists exclusively of region-level aggregates."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from wrfcomp.assimilation.citizen_obs import load_citizen_batch
from wrfcomp.assimilation.qc import (
    Flag,
    background_gate,
    distances_km,
    solar_elevation,
    spatial_gate,
    static_gate,
    temporal_gate,
)
from wrfcomp.assimilation.tokens import QCConfig
from wrfcomp.assimilation.trust import StationState, update_state

START = np.datetime64("2024-07-08", "ns")
STOP = np.datetime64("2024-07-22", "ns")
MINUTE = np.timedelta64(1, "m")
HOUR = np.timedelta64(1, "h")
CFG = QCConfig()
DIRECT = ("t2m", "td2m", "u10", "v10")
SUPPORTED = (*DIRECT, "wind_speed")
# Trial input uncertainty, in K or m/s. These are explicit assumptions, not fitted on July.
SIGMA = 2.0
REGIONS = {"Amarillo": "amarillo_tx", "Seattle": "seattle_wa"}


def latest_indices(measured, times):
    """Index of most recent available report at most 15 minutes old, or -1."""
    i = np.searchsorted(measured, times, side="right") - 1
    if not len(measured):
        return np.full(len(times), -1, dtype=int)
    return np.where((i >= 0) & (times - measured[i] <= 15 * MINUTE), i, -1)


def forecast_indices(issued, valid, times):
    """Nearest available valid time; ties prefer the shortest lead, then earlier valid time."""
    best = np.full(len(times), -1, dtype=int)
    gap = np.full(len(times), np.inf)
    for i in np.lexsort((valid, valid - issued)):
        delta = abs(times - valid[i]) / MINUTE
        use = (issued[i] < times) & (delta < gap) & (delta <= 180)
        best[use] = i
        gap[use] = delta[use]
    return best


def forecast_archive(root, ids):
    issued, valid = [], []
    values = defaultdict(list)
    for path in sorted((root / "baselines/gefs").glob("202407*.zarr")):
        cycle = pd.to_datetime(path.stem, format="%Y%m%d%H").to_datetime64()
        if not START - 48 * HOUR <= cycle < STOP:
            continue
        with xr.open_zarr(path, chunks=None, consolidated=True) as ds:
            if ds.sizes["init_time"] != 1 or ds.init_time.values[0] != cycle:
                raise ValueError("unexpected GEFS cycle")
            block = ds.sel(station_id=ids).isel(init_time=0)
            for var in DIRECT:
                raw = block[var].transpose("lead_time", "station_id", "ens_member").values
                values[var].append(raw.mean(axis=2, dtype=float))
                if var == "u10":
                    u = raw
                if var == "v10":
                    values["wind_speed"].append(np.hypot(u, raw).mean(axis=2, dtype=float))
            issued.extend([cycle] * block.sizes["lead_time"])
            valid.extend(cycle + block.lead_time.values)
    if not issued:
        raise ValueError("missing GEFS cycles for trial")
    return np.array(issued), np.array(valid), {k: np.concatenate(v) for k, v in values.items()}


def official_series(ds, forecasts):
    """Each official series carries actual measurement time, value and prior-forecast innovation z."""
    issued, valid, bg = forecasts
    # Only a few distinct official measurement times per hour; share the forecast-choice lookup.
    labels = ds.valid_time.values
    lookup_times = np.arange(labels.min() - 15 * MINUTE, labels.max() + 16 * MINUTE, MINUTE)
    lookup = forecast_indices(issued, valid, lookup_times)
    result = {}
    for var in SUPPORTED:
        if var == "wind_speed":
            values = np.hypot(ds.u10.values, ds.v10.values)
            offsets = np.where(ds.u10_offset_min.values == ds.v10_offset_min.values,
                               ds.u10_offset_min.values, np.nan)
        else:
            values, offsets = ds[var].values, ds[var + "_offset_min"].values
        series = []
        for j in range(ds.sizes["station_id"]):
            keep = np.isfinite(values[:, j]) & (abs(offsets[:, j]) <= 15)
            measured = labels[keep] + offsets[keep, j].astype("timedelta64[m]")
            y = values[keep, j].astype(float)
            ix = lookup[((measured - lookup_times[0]) / MINUTE).astype(int)]
            z = np.where(ix >= 0, (y - bg[var][ix, j]) / SIGMA, np.nan)
            series.append((measured, y, z))
        result[var] = series
    return result


def align(series, times):
    measured, value, z = series
    out = np.full(len(times), np.nan)
    votes = out.copy()
    ix = latest_indices(measured, times)
    ok = ix >= 0
    out[ok], votes[ok] = value[ix[ok]], z[ix[ok]]
    return out, votes
