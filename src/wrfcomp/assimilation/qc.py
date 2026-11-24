"""Pure QC gates and their chronological, copy-on-update batch driver."""

from dataclasses import dataclass
from enum import IntFlag

import numpy as np

from .tokens import ObservationBatch, QCConfig
from .trust import StationState, update_state, veto_recent_copy


class Flag(IntFlag):
    STATIC = 1
    TEMPORAL = 2
    SPATIAL = 4
    BACKGROUND = 8
    FEW_BUDDIES = 16
    COPYCAT = 32
    FLATLINE = 64


def static_gate(var, value, elevation, peers, cfg):
    bounds = {key: (lo, hi) for key, lo, hi in cfg.bounds}
    if var not in bounds:
        raise ValueError("unsupported observation variable")
    lo, hi = bounds[var]
    bad = not lo <= value <= hi
    if var == "td2m":
        bad |= value > peers.get("t2m", np.inf)
    if var == "wind_gust":
        speed = peers.get("wind_speed", np.hypot(peers.get("u10", np.nan), peers.get("v10", np.nan)))
        bad |= value < speed
    if var == "station_pressure":
        expected = cfg.sea_pressure_pa * np.exp(-elevation / cfg.scale_height_m)
        bad |= abs(value - expected) > cfg.pressure_tolerance_pa
    return Flag.STATIC if bad else Flag(0)


def temporal_gate(var, value, time_ns, state, cfg, *, reference=np.nan):
    if not state.recent:
        return Flag(0)
    minutes = (time_ns - state.last_ns) / 60e9
    if minutes <= 0:
        raise ValueError("station-variable timestamps must increase")
    limit = dict(cfg.steps)[var] * minutes / cfg.step_minutes
    spike = minutes <= cfg.step_minutes and abs(value - state.recent[-1]) > limit
    window = (state.recent + [value])[-cfg.flat_reports :]
    flat = len(window) == cfg.flat_reports and min(window) == max(window)
    if flat and var in ("wind_speed", "wind_gust", "u10", "v10"):
        start = state.constant_since_ns
        flat = (
            (abs(value) > cfg.calm_wind_ms or abs(reference - value) >= cfg.calm_contradiction_ms)
            and minutes <= cfg.wind_flat_max_gap_minutes
            and start is not None
            and (time_ns - start) / 60e9 >= cfg.wind_flat_minutes
            and not abs(value - reference) <= cfg.wind_bin_ms / 2
        )
    if flat and not (var == "precip" and value == 0):
        return Flag.TEMPORAL | Flag.FLATLINE
    return Flag.TEMPORAL if spike else Flag(0)


def spatial_gate(value, adjusted_buddies, sigma_total, cfg, *, official=None, independent=None):
    """One entry per distinct non-self station. Independence is relative to the target cell."""
    n = len(adjusted_buddies)
    flags, medians = spatial_series(
        np.array([value]),
        np.asarray(adjusted_buddies)[None, :],
        np.zeros(n, bool) if official is None else official,
        np.ones(n, bool) if independent is None else independent,
        sigma_total,
        cfg,
    )
    return Flag(int(flags[0])), medians[0]


def spatial_series(values, neighbours, official, independent, sigma, cfg):
    """Vectorized weighted median/MAD for the trial's per-report as-of joins."""
    present = np.isfinite(neighbours)
    ok = (present.sum(axis=1) >= cfg.minimum_buddies) & present[:, independent].any(axis=1)
    medians = np.full(len(values), np.nan)
    flags = np.full(len(values), int(Flag.FEW_BUDDIES))
    if ok.any():
        pool = neighbours[ok].copy()
        n_official = present[ok][:, official].sum(axis=1)
        n_citizen = present[ok].sum(axis=1) - n_official
        # Anchor the distribution (including its MAD) when citizens would overwhelm officials.
        anchor = (n_official >= 2) | ((n_official > 0) & (n_citizen > cfg.official_buddy_weight * n_official))
        pool[np.ix_(anchor, ~official)] = np.nan
        weighted = np.repeat(pool, np.where(official, cfg.official_buddy_weight, 1), axis=1)
        median = np.nanmedian(weighted, axis=1)
        mad = np.nanmedian(abs(weighted - median[:, None]), axis=1)
        scale = np.maximum(cfg.mad_scale * mad, cfg.scale_floor_fraction * sigma)
        medians[ok] = median
        flags[ok] = np.where(abs(values[ok] - median) > cfg.spatial_z * scale, int(Flag.SPATIAL), 0)
    return flags, medians


def _has_separated_pair(lats, lons, cfg):
    for j in range(len(lats) - 1):
        if np.any(distances_km(lats[j], lons[j], lats[j + 1 :], lons[j + 1 :]) >= cfg.independence_km):
            return True
    return False


def background_gate(z, buddy_z, cfg, *, independent=None, positions=None, official=None):
    magnitude = abs(z)
    if not np.isfinite(z):
        return Flag.BACKGROUND, 0.0
    if magnitude <= cfg.huber_cut:
        return Flag(0), 1.0
    if magnitude > cfg.gross_cut:
        votes = np.isfinite(buddy_z) & (np.sign(z) * buddy_z > cfg.corroboration_z)
        if votes.sum() < cfg.corroborating_stations or (
            independent is not None and not np.any(votes & independent)
        ):
            return Flag.BACKGROUND, 0.0
        if positions is not None and not np.any(official[votes]):
            lat, lon = positions
            if not _has_separated_pair(lat[votes], lon[votes], cfg):
                return Flag.BACKGROUND, 0.0
    return Flag.BACKGROUND, cfg.huber_cut / magnitude


def cells(lat, lon):
    """Integer labels for the supplied tenth-degree grid."""
    return np.rint(np.asarray(lat) * 10).astype(int) * 4000 + np.rint(np.asarray(lon) * 10).astype(int)


def official_hour_bias(measured, innovations, targets):
    """Expanding same-UTC-hour mean. Availability is strictly before each target."""
    measured = np.asarray(measured, dtype="datetime64[ns]")
    targets = np.asarray(targets, dtype="datetime64[ns]")
    innovations = np.asarray(innovations, dtype=float)
    result = np.full(len(targets), np.nan)
    mh = measured.astype("datetime64[h]").astype(np.int64) % 24
    th = targets.astype("datetime64[h]").astype(np.int64) % 24
    for hour in np.unique(th):
        valid = (mh == hour) & np.isfinite(innovations)
        ts, samples = measured[valid], innovations[valid]
        order = np.argsort(ts, kind="stable")
        at = np.flatnonzero(th == hour)
        count = np.searchsorted(ts[order], targets[at], side="left")
        keep = count > 0
        result[at[keep]] = np.cumsum(samples[order])[count[keep] - 1] / count[keep]
    return result


def solar_elevation(time, lat, lon):
    """Fractional-year approximation using UTC equation of time and solar declination."""
    year = time.astype("datetime64[Y]")
    days = (time.astype("datetime64[D]") - year).astype(int)
    hours = float((time - time.astype("datetime64[D]")) / np.timedelta64(1, "h"))
    next_year = year + np.timedelta64(1, "Y")
    year_days = float((next_year.astype("datetime64[D]") - year) / np.timedelta64(1, "D"))
    angle = 2 * np.pi * (days + (hours - 12) / 24) / year_days
    eq = 229.18 * (
        0.000075
        + 0.001868 * np.cos(angle)
        - 0.032077 * np.sin(angle)
        - 0.014615 * np.cos(2 * angle)
        - 0.040849 * np.sin(2 * angle)
    )
    decl = (
        0.006918
        - 0.399912 * np.cos(angle)
        + 0.070257 * np.sin(angle)
        - 0.006758 * np.cos(2 * angle)
        + 0.000907 * np.sin(2 * angle)
        - 0.002697 * np.cos(3 * angle)
        + 0.00148 * np.sin(3 * angle)
    )
    hour_angle = np.radians(15 * (hours - 12) + eq / 4 + lon)
    phi = np.radians(lat)
    sine = np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(hour_angle)
    return np.degrees(np.arcsin(np.clip(sine, -1, 1)))


def distances_km(lat, lon, lats, lons):
    a = (
        np.sin(np.radians(lats - lat) / 2) ** 2
        + np.cos(np.radians(lat)) * np.cos(np.radians(lats)) * np.sin(np.radians(lons - lon) / 2) ** 2
    )
    return 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _corroborated_steps(batch, pending, candidates, changes, official, cfg):
    """Resolve raw temperature steps from the same time block before excluding temporal outliers."""
    confirmed = set()
    for i in pending:
        near = candidates[(candidates != i) & (batch.variable[candidates] == batch.variable[i])]
        km = distances_km(batch.lat[i], batch.lon[i], batch.lat[near], batch.lon[near])
        agrees = np.sign(changes[i]) * changes[near] >= cfg.storm_step_fraction * abs(changes[i])
        voters = near[(km <= cfg.radius_km) & agrees]
        if np.any(official[voters]):
            confirmed.add(i)
            continue
        witnesses = near[(km >= cfg.independence_km) & (km <= cfg.radius_km) & agrees]
        if _has_separated_pair(batch.lat[witnesses], batch.lon[witnesses], cfg):
            confirmed.add(i)
    return confirmed


@dataclass(repr=False)
class QCResult:
    qc_flags: np.ndarray
    weights: np.ndarray
    huber: np.ndarray
    buddy_median: np.ndarray
    states: dict[tuple[str, str], StationState]


def quality_control(
    batch: ObservationBatch,
    background,
    sigma_total,
    sigma_bg,
    *,
    official=False,
    reference=np.nan,
    official_background_bias=None,
    reference_time_ns=None,
    states=None,
    cfg=QCConfig(),
):
    """Process simultaneous reports together, retaining the original input row order."""
    size = len(batch.values)

    def column(value, dtype=float):
        return np.broadcast_to(np.asarray(value, dtype=dtype), (size,))

    background, total, bg = map(column, (background, sigma_total, sigma_bg))
    reference, official = column(reference), column(official, bool)
    if not np.all(np.isfinite(total) & (total > 0) & np.isfinite(bg) & (bg > 0)):
        raise ValueError("innovation scales must be finite and positive")
    states = dict(states or {})
    flags = batch.qc_flags.copy()
    weights, huber = np.zeros(size), np.zeros(size)
    medians = np.full(size, np.nan)
    corrections = None if official_background_bias is None else column(official_background_bias)
    raw = batch.values - background
    innovations = raw.copy()
    bias_available = np.ones(size, dtype=bool)
    grid = cells(batch.lat, batch.lon)
    reference_times = None if reference_time_ns is None else column(reference_time_ns, np.int64)
    if reference_times is not None and np.any(
        np.isfinite(reference) & (reference_times > batch.obs_time.astype(np.int64))
    ):
        raise ValueError("reference measurement cannot be later than the observation")
    changes = np.full(size, np.nan)
    z = np.full(size, np.nan)
    times, groups = np.unique(batch.obs_time, return_inverse=True)
    order = np.argsort(groups, kind="stable")
    blocks = np.split(order, np.cumsum(np.bincount(groups))[:-1]) if size else []
    for time, rows in zip(times, blocks, strict=True):
        time_ns = int(time.astype(np.int64))
        day = solar_elevation(time, batch.lat[rows], batch.lon[rows]) > cfg.day_degrees
        peers = {}
        for i in rows:
            readings = peers.setdefault(batch.station_ids[i], {})
            var = batch.variable[i]
            if var in readings:
                raise ValueError("duplicate station-variable report at one timestamp")
            readings[var] = batch.values[i]
        hour = str(int(time.astype("datetime64[h]").astype(np.int64) % 24))
        history = {}
        if corrections is None:
            for (_, var), previous in states.items():
                if (
                    previous.official_site is not None
                    and previous.last_ns < time_ns
                    and hour in previous.hourly_innovations
                ):
                    count, total_error = previous.hourly_innovations[hour]
                    history.setdefault(var, []).append([*previous.official_site, total_error / count])
        witnesses = []
        pending = set()
        eligible = {}
        for i, daytime in zip(rows, day, strict=True):
            station, var = batch.station_ids[i], batch.variable[i]
            value = float(batch.values[i])
            old = states.get((station, var), StationState())
            if not official[i] and var in ("t2m", "td2m"):
                correction = np.nan
                if corrections is not None:
                    correction = corrections[i]
                elif var in history:
                    lats, lons, errors = np.asarray(history[var]).T
                    km = distances_km(batch.lat[i], batch.lon[i], lats, lons)
                    nearest = int(np.argmin(km))
                    if km[nearest] <= cfg.official_bias_radius_km:
                        correction = errors[nearest]
                bias_available[i] = np.isfinite(correction)
                if bias_available[i]:
                    innovations[i] -= correction
            bias = old.bias_day if daytime else old.bias_night
            z[i] = (innovations[i] - ((bias or 0) if bias_available[i] else 0)) / total[i]
            flags[i] |= int(static_gate(var, value, batch.elevation_m[i], peers[station], cfg))
            temporal = temporal_gate(var, value, time_ns, old, cfg, reference=reference[i])
            if (
                var in ("t2m", "td2m")
                and not flags[i]
                and old.last_usable
                and old.recent
                and not temporal & Flag.FLATLINE
                and time_ns - old.last_ns <= cfg.step_minutes * 60e9
            ):
                changes[i] = value - old.recent[-1]
                witnesses.append(i)
                if temporal == Flag.TEMPORAL:
                    pending.add(i)
            flags[i] |= int(temporal)
            if not (int(flags[i]) & (Flag.STATIC | Flag.TEMPORAL | Flag.COPYCAT)) and not old.copycat:
                eligible.setdefault(var, []).append(i)
        confirmed = (
            _corroborated_steps(batch, pending, np.asarray(witnesses, dtype=int), changes, official, cfg)
            if pending
            else set()
        )
        for i in confirmed:
            flags[i] = int(flags[i]) & ~int(Flag.TEMPORAL)
            eligible.setdefault(batch.variable[i], []).append(i)
        for i, daytime in zip(rows, day, strict=True):
            station, var = batch.station_ids[i], batch.variable[i]
            indices = np.asarray(eligible.get(var, []), dtype=int)
            distances = distances_km(batch.lat[i], batch.lon[i], batch.lat[indices], batch.lon[indices])
            nearby = (indices != i) & (distances <= cfg.radius_km)
            near = indices[nearby]
            adjusted = batch.values[near].astype(float)
            if var in ("t2m", "td2m"):
                adjusted += cfg.lapse_k_per_km * (batch.elevation_m[near] - batch.elevation_m[i]) / 1000
            if var == "station_pressure":
                adjusted *= np.exp((batch.elevation_m[near] - batch.elevation_m[i]) / cfg.scale_height_m)
            separated = distances[nearby]
            independent = official[near] | ((grid[near] != grid[i]) & (separated >= cfg.independence_km))
            flag, medians[i] = spatial_gate(
                batch.values[i], adjusted, total[i], cfg, official=official[near], independent=independent
            )
            flags[i] |= int(flag)
            flag, huber[i] = background_gate(
                z[i],
                z[near],
                cfg,
                independent=independent,
                positions=(batch.lat[near], batch.lon[near]),
                official=official[near],
            )
            if i in confirmed and huber[i] == 0 and np.isfinite(z[i]):
                huber[i] = cfg.huber_cut / abs(z[i])
            flags[i] |= int(flag)
            key = (station, var)
            new = update_state(
                states.get(key, StationState()),
                time_ns=time_ns,
                value=float(batch.values[i]),
                buddy=medians[i],
                reference=reference[i],
                innovation=float(innovations[i]),
                day=bool(daytime),
                solar_sensitive=var in ("t2m", "td2m"),
                huber=huber[i],
                sigma_bg=bg[i],
                verify_temperature=var == "t2m",
                learn_bias=bool(bias_available[i]),
                reference_time_ns=None if reference_times is None else int(reference_times[i]),
                official=official[i],
                hard_fail=bool(int(flags[i]) & (Flag.STATIC | Flag.TEMPORAL | Flag.COPYCAT)),
                cfg=cfg,
            )
            if official[i] and var in ("t2m", "td2m") and np.isfinite(raw[i]) and not flags[i] & Flag.STATIC:
                new.official_site = [float(batch.lat[i]), float(batch.lon[i])]
                count, error = new.hourly_innovations.get(hour, [0, 0.0])
                new.hourly_innovations = {**new.hourly_innovations, hour: [count + 1, error + float(raw[i])]}
            states[key] = new
            if new.copycat:
                flags[i] |= int(Flag.COPYCAT)
            weights[i] = new.weight
        # Resolve simultaneous channels together so row order cannot change eligibility.
        faults = {}
        for (site, _), state in states.items():
            if state.last_copycat_ns is not None:
                faults[site] = max(faults.get(site, state.last_copycat_ns), state.last_copycat_ns)
        for site in peers.keys() & faults.keys():
            key = (site, "t2m")
            if key in states:
                states[key] = veto_recent_copy(states[key], time_ns, faults[site], cfg)
        for i in rows:
            weights[i] = states[(batch.station_ids[i], batch.variable[i])].weight
    return QCResult(flags, weights, huber, medians, states)
