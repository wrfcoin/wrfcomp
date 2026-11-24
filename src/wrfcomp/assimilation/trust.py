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
    raw_bias_day: float | None = None
    raw_bias_night: float | None = None
    mse_ema: float = 0
    # UTC nanoseconds, observation, buddy median, independent reference or null
    pairs: list[list] = field(default_factory=list)
    correlation_guard: float = 1
    copycat: bool = False
    weight: float = 0
    last_usable: bool = False
    constant_since_ns: int | None = None
    # Completed and current UTC days: [day, count, sum of signed reference errors, adverse evidence].
    verification_days: list[list] = field(default_factory=list)
    last_verified_ns: int | None = None
    last_reference_ns: int | None = None
    distinct_reference_history: bool = False
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


def fresh_reference(old, value, stamp):
    """Use the first available pair for each advancing official measurement timestamp."""
    return np.isfinite(value) and (old.last_reference_ns is None or stamp > old.last_reference_ns)


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
    learn_bias=True,
    verify_temperature=False,
    reference_time_ns=None,
):
    if old.last_ns is not None and time_ns <= old.last_ns:
        raise ValueError("station-variable timestamps must increase")
    if not old.distinct_reference_history and old.pairs:
        # Older checkpoints interleaved official anchors with citizen medians and lack the
        # measurement stamps needed to deduplicate them. Warm up only the correlation series.
        old = replace(old, pairs=[], correlation_guard=1.0, last_reference_ns=None)
    cutoff = time_ns - cfg.window_days * 86400 * 10**9
    pairs = [p for p in old.pairs if p[0] > cutoff]
    stamp = time_ns if reference_time_ns is None else reference_time_ns
    fresh = fresh_reference(old, reference, stamp)
    if np.isfinite(value) and (np.isfinite(buddy) or fresh):
        pairs.append(
            [
                time_ns,
                value,
                float(buddy) if np.isfinite(buddy) else None,
                float(reference) if fresh else None,
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
        buddy_fresh=np.isfinite(buddy),
        ref_r=ref_r,
        ref_rmse=ref_rmse,
        learn_bias=learn_bias,
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
    buddy_fresh=True,
    ref_r,
    ref_rmse,
    learn_bias=True,
    verify_temperature=False,
    reference_time_ns=None,
):
    """Shared transition; evidence statistics include only the trailing seven days up to this report."""
    stamp = time_ns if reference_time_ns is None else reference_time_ns
    ref_fresh = np.isfinite(value) and fresh_reference(old, reference, stamp)
    evidence_fresh = buddy_fresh
    guard_r = buddy_r
    if solar_sensitive and (guard_r is None or not np.isfinite(guard_r)):
        guard_r = ref_r
        evidence_fresh = ref_fresh
    low = guard_r is not None and guard_r < cfg.low_correlation
    guard = 1.0
    if low:
        guard = old.correlation_guard * cfg.guard_decay if evidence_fresh else old.correlation_guard
    copycat = bool(
        ref_r is not None and ref_r > cfg.copy_correlation and ref_rmse <= cfg.copy_residual_sigma * sigma_bg
    )
    recent = (old.recent + [value])[-cfg.flat_reports :] if np.isfinite(value) else []
    continued = (
        old.recent
        and value == old.recent[-1]
        and (time_ns - old.last_ns) / 60e9 <= cfg.wind_flat_max_gap_minutes
    )
    new = replace(
        old,
        last_ns=time_ns,
        distinct_reference_history=True,
        last_reference_ns=stamp if ref_fresh else old.last_reference_ns,
        last_usable=bool(not hard_fail and huber > 0 and not copycat),
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
            day,
            bad=copycat or low or (ref_r is not None and ref_r < cfg.low_correlation),
            accepted=learn_bias and not hard_fail and huber > 0 and ref_r is not None,
            cfg=cfg,
        )
    bias = old.bias_day if day else old.bias_night
    reliability = exp(-sqrt(old.mse_ema) / sigma_bg)
    solar = 1.0
    day_bias, night_bias = (
        (old.bias_day, old.bias_night) if learn_bias else (old.raw_bias_day, old.raw_bias_night)
    )
    if solar_sensitive and not official:
        # Separate common-mode drift from a purely daytime solar excess. Bias correction alone
        # cannot establish reliability. Absolute reference errors remain independent evidence.
        common = 0.0
        if day_bias is not None and night_bias is not None and day_bias * night_bias > 0:
            common = min(abs(day_bias), abs(night_bias))
        persistent_error = max(common, ref_rmse) if np.isfinite(ref_rmse) else common
        reliability *= exp(-max(0.0, persistent_error / sigma_bg - cfg.verified_bias_sigma))
    if solar_sensitive and day and day_bias is not None and night_bias is not None:
        excess = abs(day_bias - night_bias) - cfg.solar_allowance
        solar = exp(-max(0, excess) / cfg.solar_scale)
    prior = cfg.official_prior if official else (cfg.verified_prior if new.earned else cfg.new_prior)
    new.weight = (
        0.0
        if hard_fail
        else float(np.clip(prior * huber * reliability * solar * new.correlation_guard, 0, 1))
    )
    if learn_bias and not hard_fail and huber > 0 and not copycat:
        residual = innovation - (bias or 0)
        new.mse_ema += cfg.ema_alpha * (residual**2 - new.mse_ema)
        learned = (bias or 0) + cfg.ema_alpha * residual
        if day:
            new.bias_day = learned
        else:
            new.bias_night = learned
    elif solar_sensitive and not learn_bias and not hard_fail and huber > 0 and not copycat:
        # A raw day/night contrast may down-weight a report, but never grants verification credit.
        previous = (day_bias if day else night_bias) or 0.0
        learned = previous + cfg.ema_alpha * (innovation - previous)
        if day:
            new.raw_bias_day = learned
        else:
            new.raw_bias_night = learned
    return new


def update_verification(
    state, time_ns, value, reference, reference_time_ns, sigma, daytime, *, bad, accepted, cfg
):
    """Only temperature earns a tier; evidence never transfers to unrelated sensor channels."""
    day = time_ns // 86_400_000_000_000
    days = state.verification_days
    if not days or days[-1][0] != day:
        history = [record for record in days if day - cfg.verified_days <= record[0] < day]
        biases = [record[2] / record[1] for record in history if record[1] >= cfg.verified_reports_per_day]
        # Pre-Q4 records lack day/night evidence and expire naturally from the window.
        diurnal = [record for record in history if len(record) == 8]
        day_n = sum(record[4] for record in diurnal)
        night_n = sum(record[6] for record in diurnal)
        stable_diurnal = (
            len(diurnal) == len(history)
            and day_n >= cfg.min_pairs
            and night_n >= cfg.min_pairs
            and abs(
                sum(record[5] for record in diurnal) / day_n - sum(record[7] for record in diurnal) / night_n
            )
            <= cfg.solar_allowance
        )
        state.earned = bool(
            stable_diurnal
            and len(biases) == cfg.verified_days
            and sum(d[1] for d in history) >= cfg.verified_reports
            and not any(d[3] for d in history)
            and max(map(abs, biases)) <= cfg.verified_bias_sigma * sigma
            and max(biases) - min(biases) <= cfg.verified_bias_sigma * sigma
        )
        days = [*history, [day, 0, 0.0, False, 0, 0.0, 0, 0.0]]
    record = days[-1].copy()
    if len(record) == 4:
        state.earned = False
        record.extend([0, 0.0, 0, 0.0])
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
        offset = 4 if daytime else 6
        record[offset] += 1
        record[offset + 1] += value - reference
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
