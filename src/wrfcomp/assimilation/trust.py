"""Causal, JSON-native state for one station and variable."""

from dataclasses import dataclass, field, replace
from math import exp, sqrt

import numpy as np

from .tokens import QCConfig


@dataclass
class StationState:
    last_ns: int | None = None
    recent: list[float] = field(default_factory=list)
    bias_day: float | None = None
    bias_night: float | None = None
    mse_ema: float = 0
    # UTC nanoseconds, observation, buddy median, independent reference or null
    pairs: list[list] = field(default_factory=list)
    correlation_guard: float = 1
    copycat: bool = False
    weight: float = 0


def qod_score(station_state: StationState) -> float:
    return station_state.weight


def correlation(pairs):
    x, y = np.asarray(pairs, dtype=float).T
    if np.ptp(y) == 0:
        return None
    if np.ptp(x) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def update_state(
    old,
    *,
    time_ns,
    value,
    buddy,
    reference,
    innovation,
    day,
    solar_sensitive,
    huber,
    sigma_bg,
    official,
    hard_fail,
    cfg: QCConfig,
):
    if old.last_ns is not None and time_ns <= old.last_ns:
        raise ValueError("station-variable timestamps must increase")
    cutoff = time_ns - cfg.window_days * 86400 * 10**9
    pairs = [p for p in old.pairs if p[0] > cutoff]
    if np.isfinite(value) and (np.isfinite(buddy) or np.isfinite(reference)):
        pairs.append(
            [
                time_ns,
                value,
                float(buddy) if np.isfinite(buddy) else None,
                float(reference) if np.isfinite(reference) else None,
            ]
        )
    guard = 1.0
    copycat = False
    buddies = [(p[1], p[2]) for p in pairs if p[2] is not None]
    if len(buddies) >= cfg.min_pairs:
        r = correlation(buddies)
        if r is not None and r < cfg.low_correlation:
            guard = old.correlation_guard * cfg.guard_decay
    refs = [(p[1], p[3]) for p in pairs if p[3] is not None]
    if len(refs) >= cfg.min_pairs:
        r = correlation(refs)
        residual = np.subtract(*np.asarray(refs).T)
        copycat = bool(
            r is not None
            and r > cfg.copy_correlation
            and np.sqrt(np.mean(residual**2)) <= cfg.copy_residual_sigma * sigma_bg
        )
    recent = (old.recent + [value])[-cfg.flat_reports :] if np.isfinite(value) else []
    new = replace(
        old,
        last_ns=time_ns,
        recent=recent,
        pairs=pairs,
        correlation_guard=0.0 if copycat else guard,
        copycat=copycat,
    )
    bias = old.bias_day if day else old.bias_night
    reliability = exp(-sqrt(old.mse_ema) / sigma_bg)
    solar = 1.0
    if solar_sensitive and day and old.bias_day is not None and old.bias_night is not None:
        excess = abs(old.bias_day - old.bias_night) - cfg.solar_allowance
        solar = exp(-max(0, excess) / cfg.solar_scale)
    prior = cfg.official_prior if official else cfg.new_prior
    new.weight = (
        0.0
        if hard_fail
        else float(np.clip(prior * huber * reliability * solar * new.correlation_guard, 0, 1))
    )
    if not hard_fail and huber > 0 and not copycat:
        residual = innovation - (bias or 0)
        new.mse_ema += cfg.ema_alpha * (residual**2 - new.mse_ema)
        learned = (bias or 0) + cfg.ema_alpha * residual
        if day:
            new.bias_day = learned
        else:
            new.bias_night = learned
    return new
