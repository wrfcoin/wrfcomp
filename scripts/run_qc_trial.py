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
    cells,
    citizen_trust_median,
    distances_km,
    official_hour_bias,
    official_step_agrees,
    solar_elevation,
    spatial_series,
    static_gate,
    temporal_gate,
)
from wrfcomp.assimilation.tokens import QCConfig
from wrfcomp.assimilation.trust import StationState, advance_state, suspected_copy, veto_recent_copy

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
            offsets = np.where(
                ds.u10_offset_min.values == ds.v10_offset_min.values, ds.u10_offset_min.values, np.nan
            )
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


def station_points(batch):
    codes, ids = pd.factorize(batch.station_ids)
    variables = pd.Categorical(batch.variable)
    frame = pd.DataFrame(dict(station=codes, time=batch.obs_time, variable=variables, value=batch.values))
    geo = pd.DataFrame(dict(station=codes, lat=batch.lat, lon=batch.lon, height=batch.elevation_m))
    geo = geo.drop_duplicates().set_index("station").sort_index()
    if len(geo) != len(ids):
        raise ValueError("station position changed within trial")
    return frame, geo


def indoor_screen(when, values, measured, reference):
    """Count distinct hourly matches in a trailing day, with at least 18 hours of coverage."""
    ix = latest_indices(measured, when)
    matched = np.flatnonzero(ix >= 0)
    if not len(matched):
        return False, False
    rows = matched[np.r_[True, np.diff(ix[matched]) != 0]]
    windows = pd.DataFrame(
        {"citizen": values[rows], "official": reference[ix[rows]], "hours": (when[rows] - START) / HOUR},
        index=when[rows],
    ).rolling("24h", min_periods=12)
    ranges = windows.max() - windows.min()
    eligible = (ranges.hours >= 18) & (ranges.official >= 5)
    return bool(eligible.any()), bool((eligible & (ranges.citizen <= 1)).any())


def trailing_correlation(times, values, comparison):
    """Vectorized equivalent of trust.correlation on the trailing paired observations."""
    index = pd.DatetimeIndex(times)
    x = pd.Series(np.where(np.isfinite(comparison), values, np.nan), index=index)
    y = pd.Series(comparison, index=index)
    window = f"{CFG.window_days}D"
    xx = x.rolling(window, min_periods=CFG.min_pairs)
    yy = y.rolling(window, min_periods=CFG.min_pairs)
    corr = xx.corr(y).where(yy.max() > yy.min())
    corr = corr.mask((xx.max() == xx.min()) & (yy.max() > yy.min()), 0.0)
    rms = ((x - y) ** 2).rolling(window, min_periods=CFG.min_pairs).mean().pow(0.5)
    return corr.to_numpy(), rms.to_numpy()


def official_evidence(series, times, values):
    """Align the reference for instantaneous gates; keep one pair per official time for rolling trust."""
    measured, observations, _ = series
    indices = latest_indices(measured, times)
    matched = np.flatnonzero(indices >= 0)
    reference = np.full(len(times), np.nan)
    stamps = np.full(len(times), -1, dtype=np.int64)
    reference[matched] = observations[indices[matched]]
    stamps[matched] = measured[indices[matched]].astype(np.int64)
    distinct = np.full(len(times), np.nan)
    first = matched[np.r_[True, np.diff(indices[matched]) > 0]] if matched.size else matched
    distinct[first] = reference[first]
    r, rms = trailing_correlation(times, values, distinct)
    actual = pd.Series(
        np.where(np.isfinite(distinct), (stamps - int(START.astype(np.int64))) / 3.6e12, np.nan),
        index=pd.DatetimeIndex(times),
    )
    window = actual.rolling(f"{CFG.window_days}D", min_periods=1)
    return reference, stamps, r, rms, window.count().to_numpy(), (window.max() - window.min()).to_numpy()


def official_storm_support(times, values, locations, elevations, height, station_series):
    """Use only official measurements inside the citizen step, with a preceding hourly observation."""
    supported = np.zeros(len(times), dtype=bool)
    for site in locations:
        measured, official, _ = station_series[site]
        ix = latest_indices(measured, times)
        rows = np.flatnonzero((ix[1:] > 0) & (np.diff(times) <= CFG.step_minutes * MINUTE)) + 1
        k = ix[rows]
        recent = (
            (measured[k] > times[rows - 1])
            & (measured[k - 1] <= times[rows - 1])
            & (measured[k] - measured[k - 1] <= CFG.official_storm_gap_minutes * MINUTE)
        )
        shift = CFG.lapse_k_per_km * (elevations[site] - height) / 1000
        supported[rows] |= recent & official_step_agrees(
            values[rows - 1], values[rows], official[k - 1] + shift, official[k] + shift, SIGMA, CFG
        )
    return supported


def citizen_sources(wides, geo, official, forecasts, series):
    """Prepare each neighbour's own time-ordered series. Failed reports supersede stale good ones."""
    out = {}
    fault_times = defaultdict(list)
    issued, valid, means = forecasts
    for station, wide in wides.items():
        lat, lon, height = geo.loc[station, ["lat", "lon", "height"]]
        distances = distances_km(lat, lon, official.latitude.values, official.longitude.values)
        nearest = int(np.argmin(distances))
        times = wide.index.values.astype("datetime64[ns]")
        choice = forecast_indices(issued, valid, times)
        sun = np.array([solar_elevation(t, lat, lon) > CFG.day_degrees for t in times])
        peers = wide.to_dict("records")
        for var in wide:
            rows = np.flatnonzero(np.isfinite(wide[var].values))
            when = times[rows]
            values = wide[var].values[rows].astype(float)
            z = np.full(len(rows), np.nan)
            reference = z.copy()
            copycat = np.zeros(len(rows), bool)
            if var in SUPPORTED:
                reference, _, r, rms, count, span = official_evidence(series[var][nearest], when, values)
                copycat = suspected_copy(
                    r, rms, SIGMA, CFG, thermal=var in ("t2m", "td2m"), count=count, hours=span
                )
                if copycat.any():
                    fault_times[station].extend(when[copycat].astype(np.int64).tolist())
                bg = np.where(choice[rows] >= 0, means[var][choice[rows], nearest], np.nan)
                if var in ("t2m", "td2m"):
                    measured, _, official_z = series[var][nearest]
                    correction = official_hour_bias(measured, official_z * SIGMA, when)
                    if distances[nearest] > CFG.official_bias_radius_km:
                        correction[:] = np.nan
                    bg += (
                        correction
                        + CFG.lapse_k_per_km * (official.elevation_m.values[nearest] - height) / 1000
                    )
                z = (values - bg) / SIGMA
            biases = [0.0, 0.0]
            storm = (
                official_storm_support(
                    when,
                    values,
                    np.flatnonzero(distances <= CFG.radius_km),
                    official.elevation_m.values,
                    height,
                    series[var],
                )
                if var in ("t2m", "td2m")
                else np.zeros(len(rows), bool)
            )
            eligible = ~copycat
            state = StationState()
            for k, row in enumerate(rows):
                ns = int(when[k].astype(np.int64))
                v = float(values[k])
                flag = static_gate(var, v, height, peers[row], CFG)
                flag |= temporal_gate(
                    var,
                    v,
                    ns,
                    state,
                    CFG,
                    reference=reference[k] if distances[nearest] <= CFG.radius_km else np.nan,
                )
                if k and eligible[k - 1] and flag == Flag.TEMPORAL and storm[k]:
                    flag = Flag(0)
                if flag:
                    eligible[k] = False
                if np.isfinite(z[k]):
                    regime = int(sun[row])
                    z[k] -= biases[regime] / SIGMA
                    if eligible[k] and abs(z[k]) <= CFG.gross_cut:
                        biases[regime] += CFG.ema_alpha * z[k] * SIGMA
                record_recent(state, ns, v)
            out[station, var] = (when, np.where(eligible, values, np.nan), np.where(eligible, z, np.nan))
    return out, {site: np.unique(stamps) for site, stamps in fault_times.items()}


def record_recent(state, ns, value):
    if not state.recent or state.recent[-1] != value or (ns - state.last_ns) / 60e9 > CFG.step_minutes:
        state.constant_since_ns = ns
    state.recent = (state.recent + [value])[-CFG.flat_reports :]
    state.last_ns = ns


def aggregate_station(
    wide,
    position,
    official,
    forecasts,
    series,
    totals,
    modes,
    samples,
    station,
    *,
    geo=None,
    sources=None,
    copy_times=(),
):
    times = wide.index.values.astype("datetime64[ns]")
    lat, lon, height = position
    dist = distances_km(lat, lon, official.latitude.values, official.longitude.values)
    nearest = int(np.argmin(dist))
    buddies = np.flatnonzero(dist <= CFG.radius_km)
    official_dist = distances_km(
        official.latitude.values[nearest],
        official.longitude.values[nearest],
        official.latitude.values,
        official.longitude.values,
    )
    official_dist[nearest] = np.inf
    other = int(np.argmin(official_dist))
    issued, valid, means = forecasts
    fi = forecast_indices(issued, valid, times)
    sun = np.array([solar_elevation(t, lat, lon) > CFG.day_degrees for t in times])
    peers = wide.to_dict("records")
    copy_times = np.asarray(copy_times, dtype=np.int64)
    last_copy = np.searchsorted(copy_times, times.astype(np.int64), side="right") - 1
    report_flags = np.zeros(len(times), dtype=int)
    reports = totals["reports"]
    reports["total"] += len(times)
    citizen_near = []
    external = np.ones(len(buddies), dtype=bool)
    if geo is not None:
        distances = distances_km(lat, lon, geo.lat.values, geo.lon.values)
        citizen_near = geo.index[(distances <= CFG.radius_km) & (geo.index != station)].tolist()
        external = np.r_[
            external, cells(geo.loc[citizen_near].lat, geo.loc[citizen_near].lon) != cells(lat, lon)
        ]
    is_official = np.arange(len(external)) < len(buddies)
    any_spatial = np.zeros(len(times), bool)
    for var in wide:
        rows = np.flatnonzero(np.isfinite(wide[var].values))
        values = wide[var].values[rows]
        when = times[rows]
        counts = totals[var]
        counts["total"] += len(rows)
        state = StationState()
        end_errors = Counter()
        day_count = night_count = 0
        flags = np.zeros(len(rows), dtype=int)
        weights = []
        thermal = var in ("t2m", "td2m")
        adjusted = np.full((len(when), len(external)), np.nan)
        votes = adjusted.copy()
        reference = np.full(len(when), np.nan)
        for col, source in enumerate(citizen_near, start=len(buddies)):
            stream = sources.get((source, var))
            if stream is None:
                continue
            adjusted[:, col], votes[:, col] = align(stream, when)
            dh = geo.loc[source, "height"] - height
            if thermal:
                adjusted[:, col] += CFG.lapse_k_per_km * dh / 1000
            if var == "station_pressure":
                adjusted[:, col] *= np.exp(dh / CFG.scale_height_m)
        if var in SUPPORTED:
            if var == "t2m" and dist[nearest] <= CFG.radius_km:
                eligible, indoor = indoor_screen(when, values, *series[var][nearest][:2])
                if eligible:
                    modes["indoor_evaluable"].add(station)
                if indoor:
                    modes["indoor_screen"].add(station)
            reference, ref_times, ref_r, ref_rmse, ref_count, ref_hours = official_evidence(
                series[var][nearest], when, values
            )
            comparison, _ = align(series[var][other], when)
            official_pair = np.isfinite(reference) & np.isfinite(comparison)
            counts["official_pairs"] += int(official_pair.sum())
            counts["official_sq"] += float(
                np.sum((reference[official_pair] - comparison[official_pair]) ** 2)
            )
            both = np.isfinite(reference)
            counts["reference_pairs"] += int(both.sum())
            counts["reference_sq"] += float(np.sum((values[both] - reference[both]) ** 2))
            for col, buddy in enumerate(buddies):
                adjusted[:, col], votes[:, col] = align(series[var][buddy], when)
                if thermal:
                    adjusted[:, col] += (
                        CFG.lapse_k_per_km * (official.elevation_m.values[buddy] - height) / 1000
                    )
            background = np.where(fi[rows] >= 0, means[var][fi[rows], nearest], np.nan)
            if thermal:
                background += CFG.lapse_k_per_km * (official.elevation_m.values[nearest] - height) / 1000
                measured, _, official_z = series[var][nearest]
                correction = official_hour_bias(measured, official_z * SIGMA, when)
                if dist[nearest] > CFG.official_bias_radius_km:
                    correction[:] = np.nan
                counts["corrected"] += int(np.isfinite(correction).sum())
                background += correction
        spatial_sigma = {"precip": 1.0, "station_pressure": 200.0}.get(var, SIGMA)
        spatial_flags, _ = spatial_series(values, adjusted, is_official, external, spatial_sigma, CFG)
        decided = spatial_flags != int(Flag.FEW_BUDDIES)
        counts["spatial_applied"] += int(decided.sum())
        any_spatial[rows] |= decided
        medians = citizen_trust_median(adjusted, is_official, external, CFG)
        buddy_r, _ = trailing_correlation(when, values, medians)
        storm = (
            official_storm_support(when, values, buddies, official.elevation_m.values, height, series[var])
            if thermal
            else np.zeros(len(rows), bool)
        )
        for j, row in enumerate(rows):
            value = float(values[j])
            ns = int(times[row].astype(np.int64))
            flag = static_gate(var, value, height, peers[row], CFG)
            flag |= temporal_gate(
                var,
                value,
                ns,
                state,
                CFG,
                reference=reference[j] if dist[nearest] <= CFG.radius_km else np.nan,
            )
            rescued = flag == Flag.TEMPORAL and state.last_usable and storm[j]
            if rescued:
                flag = Flag(0)
            flag |= Flag(int(spatial_flags[j]))
            if var in SUPPORTED:
                median = medians[j]
                innovation = value - background[j]
                day = bool(sun[row])
                bias = state.bias_day if day else state.bias_night
                if np.isfinite(innovation):
                    bg_flag, huber = background_gate(
                        (innovation - (bias or 0)) / SIGMA, votes[j], CFG, independent=external
                    )
                    if rescued and huber == 0:
                        huber = CFG.huber_cut / abs((innovation - (bias or 0)) / SIGMA)
                    flag |= bg_flag
                    counts["background_applied"] += 1
                else:
                    huber = 0
                hard = bool(flag & (Flag.STATIC | Flag.TEMPORAL))
                state = advance_state(
                    state,
                    time_ns=ns,
                    value=value,
                    buddy_r=buddy_r[j],
                    buddy_fresh=np.isfinite(median),
                    ref_r=ref_r[j] if np.isfinite(ref_r[j]) else None,
                    ref_rmse=ref_rmse[j],
                    ref_count=ref_count[j],
                    ref_hours=ref_hours[j],
                    wind_drift=var in ("u10", "v10", "wind_speed"),
                    verify_temperature=var == "t2m" and dist[nearest] <= CFG.radius_km,
                    reference_time_ns=int(ref_times[j]) if ref_times[j] >= 0 else None,
                    reference=reference[j],
                    innovation=innovation,
                    day=day,
                    solar_sensitive=thermal,
                    huber=huber,
                    sigma_bg=SIGMA,
                    official=False,
                    hard_fail=hard,
                    cfg=CFG,
                )
                if last_copy[row] >= 0:
                    state = veto_recent_copy(state, ns, int(copy_times[last_copy[row]]), CFG)
                if state.earned:
                    modes["ever_earned"].add(station)
                    if var == "t2m" and np.isfinite(reference[j]) and np.isfinite(comparison[j]):
                        end_errors["end_pairs"] += 1
                        end_errors["end_sq"] += (value - reference[j]) ** 2
                        end_errors["end_official_sq"] += (reference[j] - comparison[j]) ** 2
                        counts["earned_pairs"] += 1
                        counts["earned_sq"] += (value - reference[j]) ** 2
                        counts["earned_official_sq"] += (reference[j] - comparison[j]) ** 2
                if state.copycat:
                    flag |= Flag.COPYCAT
                    modes["copycat"].add(station)
                if np.isfinite(innovation):
                    weights.append(state.weight)
                    if state.weight > 0.8 and np.isfinite(reference[j]):
                        counts["high_pairs"] += 1
                        counts["high_sq"] += (value - reference[j]) ** 2
                    if huber and not hard and not state.copycat:
                        day_count += int(day)
                        night_count += int(not day)
                if var == "t2m":
                    if min(day_count, night_count) >= CFG.min_pairs:
                        if state.bias_day - state.bias_night > CFG.solar_allowance:
                            modes["daytime_bias"].add(station)
                    if state.correlation_guard < 0.1 and not state.copycat and np.isfinite(median):
                        modes["low_temperature_correlation"].add(station)
            else:
                # No comparable official/GEFS field. Maintain the true temporal history only.
                record_recent(state, ns, value)
            if flag & Flag.FLATLINE:
                modes["flatline"].add(station)
                if var == "wind_speed":
                    modes["wind_speed_flatline"].add(station)
                    counts["calm_flatline"] += int(value == 0)
            flags[j] = int(flag)
        if len(copy_times):
            state = veto_recent_copy(state, int(STOP.astype(np.int64)), int(copy_times[-1]), CFG)
        if var == "t2m" and state.earned and when[-1] >= STOP - 24 * HOUR:
            modes["end_earned"].add(station)
            counts.update(end_errors)
        report_flags[rows] |= flags
        for bit in Flag:
            counts[bit.name] += int(np.count_nonzero(flags & bit))
        if weights:
            samples[var].append(np.array(weights))
    reports["spatial_applied"] += int(any_spatial.sum())
    for bit in Flag:
        reports[bit.name] += int(np.count_nonzero(report_flags & bit))


def buddy_geometry(geo, official):
    """Station counts only. Eight km bounds the displacement from rounding both axes to a tenth."""
    counts = Counter()
    for lat, lon in geo[["lat", "lon"]].itertuples(index=False, name=None):
        distances = distances_km(lat, lon, official.latitude.values, official.longitude.values)
        for label, radius in (
            ("guaranteed", CFG.radius_km - 8),
            ("nominal", CFG.radius_km),
            ("possible", CFG.radius_km + 8),
        ):
            counts[label] += int(np.count_nonzero(distances <= radius) >= CFG.minimum_buddies)
    return counts


def consistent_temperature(wide, position, official, series):
    """Fixed descriptive cohort for comparing wind weights; independent of QC version and weights."""
    if "t2m" not in wide:
        return False
    lat, lon, _ = position
    km = distances_km(lat, lon, official.latitude.values, official.longitude.values)
    nearest = int(km.argmin())
    records = wide.t2m.dropna()
    times = records.index.values.astype("datetime64[ns]")
    ref, _ = align(series["t2m"][nearest], times)
    matched = np.isfinite(ref)
    if matched.sum() < CFG.min_pairs:
        return False
    x, y = records.values[matched], ref[matched]
    daylight = np.array([solar_elevation(t, lat, lon) > CFG.day_degrees for t in times[matched]])
    if min(daylight.sum(), (~daylight).sum()) < CFG.min_pairs or np.ptp(x) == 0 or np.ptp(y) == 0:
        return False
    residual = x - y
    return (
        np.sqrt(np.mean(residual**2)) < 2.0
        and abs(residual[daylight].mean() - residual[~daylight].mean()) <= 1.0
        and np.corrcoef(x, y)[0, 1] >= 0.85
    )


def trial_region(path, official, forecasts, series):
    batch = load_citizen_batch(path)
    if not len(batch.values) or np.any((batch.obs_time < START) | (batch.obs_time >= STOP)):
        raise ValueError("citizen observations outside the trial window or empty")
    frame, geo = station_points(batch)
    del batch
    totals = defaultdict(Counter)
    modes = {
        name: set()
        for name in (
            "ever_earned",
            "end_earned",
            "indoor_screen",
            "indoor_evaluable",
            "daytime_bias",
            "low_temperature_correlation",
            "flatline",
            "wind_speed_flatline",
            "copycat",
        )
    }
    station_wind_medians = []
    samples = defaultdict(list)
    wides = {}
    for station, group in frame.groupby("station", sort=True):
        wide = group.pivot(index="time", columns="variable", values="value").sort_index()
        wide = wide.dropna(axis=1, how="all")
        wides[station] = wide
    sources, faults = citizen_sources(wides, geo, official, forecasts, series)
    for station, wide in wides.items():
        position = geo.iloc[station][["lat", "lon", "height"]].to_numpy()
        previous_wind = len(samples.get("wind_speed", ()))
        aggregate_station(
            wide,
            position,
            official,
            forecasts,
            series,
            totals,
            modes,
            samples,
            station,
            geo=geo,
            sources=sources,
            copy_times=faults.get(station, ()),
        )

        if len(samples.get("wind_speed", ())) > previous_wind and consistent_temperature(
            wide, position, official, series
        ):
            station_wind_medians.append(float(np.median(samples["wind_speed"][-1])))

    def distribution(parts):
        w = np.concatenate(parts)
        return dict(
            n=len(w),
            zero=int(np.count_nonzero(w == 0)),
            above08=int(np.count_nonzero(w > 0.8)),
            quantiles=np.quantile(w, [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]).tolist(),
        )

    weights = {var: distribution(parts) for var, parts in samples.items()}
    weights["all"] = distribution([p for parts in samples.values() for p in parts])
    return dict(
        stations=len(geo),
        consistent_temp_wind_stations=len(station_wind_medians),
        consistent_temp_wind_median=float(np.median(station_wind_medians)) if station_wind_medians else None,
        geometry=buddy_geometry(geo, official),
        counts=totals,
        weights=weights,
        signatures={k: len(v) for k, v in modes.items()},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path.home() / "wrfcomp-data")
    parser.add_argument("--region", choices=REGIONS, required=True)
    args = parser.parse_args()
    root = args.data_dir
    path = root / "citizen" / f"{REGIONS[args.region]}_20240708_20240721.nc"
    with xr.open_dataset(root / "truth/isd_conus_2024.nc") as ds:
        official = ds.sel(valid_time=slice(START - 48 * HOUR, STOP)).load()
    forecasts = forecast_archive(root, official.station_id.values)
    print(
        json.dumps(
            trial_region(path, official, forecasts, official_series(official, forecasts)), allow_nan=False
        )
    )


if __name__ == "__main__":
    main()
