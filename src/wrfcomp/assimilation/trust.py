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
    constant_since_ns: int | None = None
    # Completed and current UTC days: [day, count, sum of signed reference errors, adverse evidence].
    verification_days: list[list] = field(default_factory=list)
    last_verified_ns: int | None = None
    last_copycat_ns: int | None = None
    earned: bool = False
    official_site: list[float] | None = None
    hourly_innovations: dict[str, list[float]] = field(default_factory=dict)


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
    verify_temperature=False,
    reference_time_ns=None,
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
    buddies = [(p[1], p[2]) for p in pairs if p[2] is not None]
    buddy_r = correlation(buddies) if len(buddies) >= cfg.min_pairs else None
    refs = [(p[1], p[3]) for p in pairs if p[3] is not None]
    ref_r, ref_rmse = None, np.nan
    if len(refs) >= cfg.min_pairs:
        ref_r = correlation(refs)
        ref_rmse = np.sqrt(np.mean(np.subtract(*np.asarray(refs).T) ** 2))
    new = advance_state(
        old,
        time_ns=time_ns,
        value=value,
        reference=reference,
        innovation=innovation,
        day=day,
        solar_sensitive=solar_sensitive,
        huber=huber,
        sigma_bg=sigma_bg,
        official=official,
        hard_fail=hard_fail,
        cfg=cfg,
        buddy_r=buddy_r,
        ref_r=ref_r,
        ref_rmse=ref_rmse,
        verify_temperature=verify_temperature,
        reference_time_ns=reference_time_ns,
    )
    new.pairs = pairs
    return new


def advance_state(
    old,
    *,
    time_ns,
    value,
    reference,
    innovation,
    day,
    solar_sensitive,
    huber,
    sigma_bg,
    official,
    hard_fail,
    cfg,
    buddy_r,
    ref_r,
    ref_rmse,
    verify_temperature=False,
    reference_time_ns=None,
):
    """Shared transition; evidence statistics include only the trailing seven days up to this report."""
    low = buddy_r is not None and buddy_r < cfg.low_correlation
    guard = old.correlation_guard * cfg.guard_decay if low else 1.0
    copycat = bool(
        ref_r is not None and ref_r > cfg.copy_correlation and ref_rmse <= cfg.copy_residual_sigma * sigma_bg
    )
    recent = (old.recent + [value])[-cfg.flat_reports :] if np.isfinite(value) else []
    continued = old.recent and value == old.recent[-1] and (time_ns - old.last_ns) / 60e9 <= cfg.step_minutes
    new = replace(
        old,
        last_ns=time_ns,
        recent=recent,
        copycat=copycat,
        last_copycat_ns=time_ns if copycat else old.last_copycat_ns,
        correlation_guard=0.0 if copycat else guard,
        constant_since_ns=(old.constant_since_ns if old.constant_since_ns is not None else old.last_ns)
        if continued
        else time_ns,
    )
    if verify_temperature and not official:
        update_verification(
            new,
            time_ns,
            value,
            reference,
            reference_time_ns,
            sigma_bg,
            bad=copycat or low or (ref_r is not None and ref_r < cfg.low_correlation),
            accepted=not hard_fail and huber > 0 and ref_r is not None,
            cfg=cfg,
        )
    bias = old.bias_day if day else old.bias_night
    reliability = exp(-sqrt(old.mse_ema) / sigma_bg)
    solar = 1.0
    if solar_sensitive and day and old.bias_day is not None and old.bias_night is not None:
        excess = abs(old.bias_day - old.bias_night) - cfg.solar_allowance
        solar = exp(-max(0, excess) / cfg.solar_scale)
    prior = cfg.official_prior if official else (cfg.verified_prior if new.earned else cfg.new_prior)
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


def update_verification(state, time_ns, value, reference, reference_time_ns, sigma, *, bad, accepted, cfg):
    """Only temperature earns a tier; evidence never transfers to unrelated sensor channels."""
    day = time_ns // 86_400_000_000_000
    days = state.verification_days
    if not days or days[-1][0] != day:
        history = [record for record in days if day - cfg.verified_days <= record[0] < day]
        biases = [record[2] / record[1] for record in history if record[1] >= cfg.verified_reports_per_day]
        state.earned = bool(
            len(biases) == cfg.verified_days
            and sum(d[1] for d in history) >= cfg.verified_reports
            and not any(d[3] for d in history)
            and max(map(abs, biases)) <= cfg.verified_bias_sigma * sigma
            and max(biases) - min(biases) <= cfg.verified_bias_sigma * sigma
        )
        days = [*history, [day, 0, 0.0, False]]
    record = days[-1].copy()
    if bad:
        state.earned = False
        record[3] = True
    if (
        accepted
        and not bad
        and np.isfinite(reference)
        and reference_time_ns is not None
        and reference_time_ns <= time_ns
        and (state.last_verified_ns is None or reference_time_ns > state.last_verified_ns)
    ):
        record[1] += 1
        record[2] += value - reference
        state.last_verified_ns = reference_time_ns
    state.verification_days = [*days[:-1], record]


def veto_recent_copy(state, time_ns, last_copycat_ns, cfg):
    """An active copycat history on any channel withdraws the temperature credential."""
    if (
        state.earned
        and last_copycat_ns is not None
        and 0 <= time_ns - last_copycat_ns < cfg.window_days * 86400 * 10**9
    ):
        return replace(state, earned=False, weight=state.weight * cfg.new_prior / cfg.verified_prior)
    return state
