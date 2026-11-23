"""Pure QC gates and their chronological, copy-on-update batch driver."""

from dataclasses import dataclass
from enum import IntFlag

import numpy as np

from .tokens import ObservationBatch, QCConfig
from .trust import StationState, update_state


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


def temporal_gate(var, value, time_ns, state, cfg):
    if not state.recent:
        return Flag(0)
    minutes = (time_ns - state.last_ns) / 60e9
    if minutes <= 0:
        raise ValueError("station-variable timestamps must increase")
    limit = dict(cfg.steps)[var] * minutes / cfg.step_minutes
    spike = minutes <= cfg.step_minutes and abs(value - state.recent[-1]) > limit
    window = (state.recent + [value])[-cfg.flat_reports :]
    flat = len(window) == cfg.flat_reports and min(window) == max(window)
    if flat and not (var == "precip" and value == 0):
        return Flag.TEMPORAL | Flag.FLATLINE
    return Flag.TEMPORAL if spike else Flag(0)


def spatial_gate(value, adjusted_buddies, sigma_total, cfg):
    if len(adjusted_buddies) < cfg.minimum_buddies:
        return Flag.FEW_BUDDIES, np.nan
    median = float(np.median(adjusted_buddies))
    mad = np.median(abs(adjusted_buddies - median))
    scale = max(cfg.mad_scale * mad, cfg.scale_floor_fraction * sigma_total)
    return (Flag.SPATIAL if abs(value - median) > cfg.spatial_z * scale else Flag(0)), median


def background_gate(z, buddy_z, cfg):
    magnitude = abs(z)
    if not np.isfinite(z):
        return Flag.BACKGROUND, 0.0
    if magnitude <= cfg.huber_cut:
        return Flag(0), 1.0
    votes = np.count_nonzero(np.isfinite(buddy_z) & (np.sign(z) * buddy_z > cfg.corroboration_z))
    if magnitude > cfg.gross_cut and votes < cfg.corroborating_stations:
        return Flag.BACKGROUND, 0.0
    return Flag.BACKGROUND, cfg.huber_cut / magnitude


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
    innovations = batch.values - background
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
        eligible = {}
        for i, daytime in zip(rows, day, strict=True):
            station, var = batch.station_ids[i], batch.variable[i]
            value = float(batch.values[i])
            old = states.get((station, var), StationState())
            bias = old.bias_day if daytime else old.bias_night
            z[i] = (innovations[i] - (bias or 0)) / total[i]
            flags[i] |= int(static_gate(var, value, batch.elevation_m[i], peers[station], cfg))
            flags[i] |= int(temporal_gate(var, value, time_ns, old, cfg))
            if not (int(flags[i]) & (Flag.STATIC | Flag.TEMPORAL | Flag.COPYCAT)) and not old.copycat:
                eligible.setdefault(var, []).append(i)
        for i, daytime in zip(rows, day, strict=True):
            station, var = batch.station_ids[i], batch.variable[i]
            indices = np.asarray(eligible.get(var, []), dtype=int)
            distances = distances_km(batch.lat[i], batch.lon[i], batch.lat[indices], batch.lon[indices])
            near = indices[(indices != i) & (distances <= cfg.radius_km)]
            adjusted = batch.values[near].astype(float)
            if var in ("t2m", "td2m"):
                adjusted += cfg.lapse_k_per_km * (batch.elevation_m[near] - batch.elevation_m[i]) / 1000
            flag, medians[i] = spatial_gate(batch.values[i], adjusted, total[i], cfg)
            flags[i] |= int(flag)
            flag, huber[i] = background_gate(z[i], z[near], cfg)
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
                official=official[i],
                hard_fail=bool(int(flags[i]) & (Flag.STATIC | Flag.TEMPORAL | Flag.COPYCAT)),
                cfg=cfg,
            )
            states[key] = new
            if new.copycat:
                flags[i] |= int(Flag.COPYCAT)
            weights[i] = new.weight
    return QCResult(flags, weights, huber, medians, states)
