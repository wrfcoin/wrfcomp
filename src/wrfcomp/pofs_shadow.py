"""Off-chain Proof-of-Forecast-Skill (PoFS) shadow payout ledger.

Paper-only implementation of the compute pool (Section 4) and data provider pool
(Section 5) formulas in ``docs/TOKENOMICS.md``. No chain, no wallet, no network.
All calculations are deterministic pure functions over NumPy arrays.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ShadowLedgerParams:
    """Frozen tokenomics parameters matching ``docs/TOKENOMICS.md`` defaults.

    ``alpha_slash`` is not fixed in ``docs/TOKENOMICS.md`` and is therefore a
    required parameter with no default.
    """

    alpha_slash: float
    p_comp_share: float = 0.55
    p_data_share: float = 0.35
    p_val_share: float = 0.10
    w_delta: float = 0.90
    twcrps_weight: float = 0.30
    max_slash_cap: float = 0.10
    stake_exponent: float = 0.75
    protocol_fault_slash: float = 0.05
    h3_cell_cap: int = 1

    def __post_init__(self) -> None:
        if not math.isfinite(self.alpha_slash) or self.alpha_slash < 0.0:
            raise ValueError(f"alpha_slash must be finite and >= 0, got {self.alpha_slash}")
        shares = (self.p_comp_share, self.p_data_share, self.p_val_share)
        if any(not math.isfinite(s) or s < 0.0 for s in shares):
            raise ValueError("pool shares must be finite and >= 0")
        if abs(sum(shares) - 1.0) > 1e-12:
            raise ValueError(f"pool shares must sum to 1.0, got {sum(shares)}")
        for name, val in (
            ("w_delta", self.w_delta),
            ("twcrps_weight", self.twcrps_weight),
            ("max_slash_cap", self.max_slash_cap),
            ("protocol_fault_slash", self.protocol_fault_slash),
        ):
            if not math.isfinite(val) or not (0.0 <= val <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {val}")
        if not math.isfinite(self.stake_exponent) or self.stake_exponent <= 0.0:
            raise ValueError(f"stake_exponent must be finite and > 0, got {self.stake_exponent}")
        if (
            isinstance(self.h3_cell_cap, bool)
            or not isinstance(self.h3_cell_cap, int)
            or self.h3_cell_cap < 1
        ):
            raise ValueError(f"h3_cell_cap must be an integer >= 1, got {self.h3_cell_cap}")


@dataclass(frozen=True)
class MinerEntry:
    """Single compute miner input for an epoch."""

    miner_id: str
    stake: float
    forecast: np.ndarray | None = None
    pointwise_loss: np.ndarray | None = None
    protocol_fault: bool = False


@dataclass(frozen=True)
class MinerPayout:
    """Deterministic per-miner compute pool settlement record."""

    miner_id: str
    stake: float
    protocol_fault: bool
    loss: float | None
    crpss: float | None
    delta_crps: float | None
    composite_score: float | None
    reward: float
    slash: float


@dataclass(frozen=True)
class ComputePoolSettlement:
    """Aggregated compute pool settlement for an epoch."""

    pool_amount: float
    baseline_loss: float
    meta_ensemble_loss: float | None
    payouts: tuple[MinerPayout, ...]
    total_paid: float
    unpaid_remainder: float
    total_slashed: float


@dataclass(frozen=True)
class StationEntry:
    """Single weather station input for the data provider pool."""

    station_id: str
    cell_id: str
    qod: float
    pol: float
    hw: float = 1.0
    impact: float = 1.0
    cell_cap: int | None = None


@dataclass(frozen=True)
class StationPayout:
    """Deterministic per-station data pool settlement record."""

    station_id: str
    cell_id: str
    cell_count: int
    cell_cap_limit: int
    cell_cap_factor: float
    weight: float
    reward: float


@dataclass(frozen=True)
class DataPoolSettlement:
    """Aggregated data provider pool settlement for an epoch."""

    pool_amount: float
    payouts: tuple[StationPayout, ...]
    total_weight: float
    total_paid: float
    unpaid_remainder: float


@dataclass(frozen=True)
class EpochSettlement:
    """Combined paper-only settlement across all three emission pools."""

    epoch_emission: float
    compute: ComputePoolSettlement
    data: DataPoolSettlement
    validator_pool: float


# --------------------------------------------------------------------------- CRPS primitives


def _as_2d_forecast(forecast: np.ndarray, n_obs: int) -> np.ndarray:
    arr = np.asarray(forecast, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    if arr.ndim != 2 or arr.shape[0] != n_obs or arr.shape[1] < 1:
        raise ValueError(
            f"forecast must have shape ({n_obs},) or ({n_obs}, M>=1), got {arr.shape}"
        )
    if not np.isfinite(arr).all():
        raise ValueError("forecast contains non-finite values")
    return arr


def fair_ensemble_crps(forecast: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Pointwise fair ensemble CRPS over last axis (reduces to MAE for M=1).

    For M >= 2 members x_1..x_M and observation y:
        CRPS_fair = (1/M) sum_m |x_m - y| - 1/(2 M (M-1)) sum_{m,m'} |x_m - x_{m'}|
    Computed in O(M log M) via sorting.
    """
    y = np.asarray(obs, dtype=np.float64)
    if y.ndim != 1 or y.size == 0 or not np.isfinite(y).all():
        raise ValueError("obs must be a non-empty 1-D finite array")
    x = _as_2d_forecast(forecast, y.size)
    m = x.shape[1]
    term1 = np.mean(np.abs(x - y[:, np.newaxis]), axis=1)
    if m == 1:
        return term1
    xs = np.sort(x, axis=1)
    # sum_{a<b} (x_(b) - x_(a)) = sum_k (2k - M + 1) * x_(k) for k=0..M-1
    coeffs = 2.0 * np.arange(m, dtype=np.float64) - (m - 1.0)
    pair_sum = np.sum(xs * coeffs[np.newaxis, :], axis=1)  # equals 0.5 * sum_{a,b} |x_a - x_b|
    return term1 - pair_sum / (m * (m - 1.0))


def weighted_mixture_crps(
    forecasts: list[np.ndarray],
    weights: np.ndarray,
    obs: np.ndarray,
) -> np.ndarray:
    """Exact pointwise CRPS of a stake-weighted probability mixture of ensembles.

    Each miner j contributes weight p_j = weights[j] / sum(weights) split evenly
    across its M_j ensemble members. Uses the exact O(K log K) identity:
        CRPS(F, y) = sum_k w_k |x_k - y| - sum_{k=0}^{K-2} W_k (1 - W_k) (x_(k+1) - x_(k))
    """
    y = np.asarray(obs, dtype=np.float64)
    if y.ndim != 1 or y.size == 0 or not np.isfinite(y).all():
        raise ValueError("obs must be a non-empty 1-D finite array")
    w_raw = np.asarray(weights, dtype=np.float64)
    if w_raw.ndim != 1 or w_raw.size != len(forecasts) or w_raw.size == 0:
        raise ValueError("weights must be a non-empty 1-D array matching forecasts")
    if (w_raw < 0.0).any() or not np.isfinite(w_raw).all() or float(np.sum(w_raw)) <= 0.0:
        raise ValueError("weights must be finite, non-negative, and sum to > 0")

    probs = w_raw / float(np.sum(w_raw))
    x_parts: list[np.ndarray] = []
    w_parts: list[np.ndarray] = []
    for fc, p in zip(forecasts, probs, strict=True):
        arr = _as_2d_forecast(fc, y.size)
        if p == 0.0:
            continue
        m = arr.shape[1]
        x_parts.append(arr)
        w_parts.append(np.full(m, p / m, dtype=np.float64))

    x_all = np.concatenate(x_parts, axis=1)  # (N, K)
    w_all = np.concatenate(w_parts, axis=0)  # (K,)
    w_all = w_all / float(np.sum(w_all))

    term1 = np.sum(np.abs(x_all - y[:, np.newaxis]) * w_all[np.newaxis, :], axis=1)
    if x_all.shape[1] == 1:
        return term1

    order = np.argsort(x_all, axis=1, kind="mergesort")
    xs = np.take_along_axis(x_all, order, axis=1)
    ws = w_all[order]
    cdf = np.cumsum(ws, axis=1)[:, :-1]
    dx = np.diff(xs, axis=1)
    spread = np.sum(cdf * (1.0 - cdf) * dx, axis=1)
    return term1 - spread


def composite_pointwise_loss(
    crps: np.ndarray,
    twcrps: np.ndarray,
    params: ShadowLedgerParams,
) -> np.ndarray:
    """Blend pointwise CRPS and tail-weighted CRPS per Section 4.1."""
    c = np.asarray(crps, dtype=np.float64)
    t = np.asarray(twcrps, dtype=np.float64)
    if c.shape != t.shape or not np.isfinite(c).all() or not np.isfinite(t).all():
        raise ValueError("crps and twcrps must be finite arrays of identical shape")
    return (1.0 - params.twcrps_weight) * c + params.twcrps_weight * t


def _clip_tail(values: np.ndarray, threshold: np.ndarray | float) -> np.ndarray:
    thr = np.asarray(threshold, dtype=np.float64)
    if thr.ndim == 1 and values.ndim == 2:
        thr = thr[:, np.newaxis]
    return np.maximum(values, thr)


def composite_forecast_loss(
    forecast: np.ndarray,
    obs: np.ndarray,
    params: ShadowLedgerParams,
    tail_threshold: np.ndarray | float | None = None,
) -> float:
    """Scalar composite loss L_i for a single forecast array."""
    crps = fair_ensemble_crps(forecast, obs)
    if tail_threshold is None:
        return float(np.mean(crps))
    y = np.asarray(obs, dtype=np.float64)
    x = _as_2d_forecast(forecast, y.size)
    twcrps = fair_ensemble_crps(_clip_tail(x, tail_threshold), _clip_tail(y, tail_threshold))
    return float(np.mean(composite_pointwise_loss(crps, twcrps, params)))


def meta_ensemble_loss(
    forecasts: list[np.ndarray],
    stakes: np.ndarray,
    obs: np.ndarray,
    params: ShadowLedgerParams,
    tail_threshold: np.ndarray | float | None = None,
) -> float:
    """Scalar composite loss L(M_S) for a stake-weighted mixture of miner forecasts."""
    crps = weighted_mixture_crps(forecasts, stakes, obs)
    if tail_threshold is None:
        return float(np.mean(crps))
    y = np.asarray(obs, dtype=np.float64)
    clipped_fc = [_clip_tail(_as_2d_forecast(fc, y.size), tail_threshold) for fc in forecasts]
    clipped_obs = _clip_tail(y, tail_threshold)
    twcrps = weighted_mixture_crps(clipped_fc, stakes, clipped_obs)
    return float(np.mean(composite_pointwise_loss(crps, twcrps, params)))


# --------------------------------------------------------------------------- Compute Pool


def settle_compute_pool(
    miners: list[MinerEntry],
    *,
    pool_amount: float,
    params: ShadowLedgerParams,
    baseline_loss: float | None = None,
    baseline_forecast: np.ndarray | None = None,
    obs: np.ndarray | None = None,
    tail_threshold: np.ndarray | float | None = None,
) -> ComputePoolSettlement:
    """Settle the Compute Pool (Section 4 of ``docs/TOKENOMICS.md``).

    Accepts either:
      - Forecast-array mode: every non-faulty miner provides ``forecast`` plus ``obs``
        (and either ``baseline_loss`` or ``baseline_forecast``).
      - Pointwise-loss mode: every non-faulty miner provides ``pointwise_loss`` plus
        scalar ``baseline_loss``.
    """
    if not math.isfinite(pool_amount) or pool_amount < 0.0:
        raise ValueError(f"pool_amount must be finite and >= 0, got {pool_amount}")

    seen_ids: set[str] = set()
    for m in miners:
        if not m.miner_id:
            raise ValueError("miner_id must be non-empty")
        if m.miner_id in seen_ids:
            raise ValueError(f"duplicate miner_id: {m.miner_id!r}")
        seen_ids.add(m.miner_id)
        if not math.isfinite(m.stake) or m.stake < 0.0:
            raise ValueError(f"miner {m.miner_id!r}: stake must be finite and >= 0")

    if baseline_loss is None:
        if baseline_forecast is None or obs is None:
            raise ValueError("provide either baseline_loss or (baseline_forecast and obs)")
        baseline_loss = composite_forecast_loss(
            baseline_forecast, obs, params, tail_threshold=tail_threshold
        )
    if not math.isfinite(baseline_loss) or baseline_loss <= 0.0:
        raise ValueError(f"baseline_loss must be finite and > 0, got {baseline_loss}")

    # Deterministic ordering by miner_id ensures bitwise-identical float reductions.
    ordered = sorted(miners, key=lambda m: m.miner_id)
    valid = [m for m in ordered if not m.protocol_fault]

    if valid:
        has_fc = [m.forecast is not None for m in valid]
        has_pl = [m.pointwise_loss is not None for m in valid]
        if any(fc and pl for fc, pl in zip(has_fc, has_pl, strict=True)):
            raise ValueError("a miner entry must set either forecast or pointwise_loss, not both")
        if all(has_fc):
            mode = "forecast"
            if obs is None:
                raise ValueError("obs is required in forecast mode")
        elif all(has_pl):
            mode = "pointwise_loss"
        else:
            raise ValueError(
                "all non-faulty miners must use the same input mode (forecast or pointwise_loss)"
            )
    else:
        mode = "empty"

    staked_valid = [m for m in valid if m.stake > 0.0]

    meta_loss: float | None = None
    indiv_losses: dict[str, float] = {}
    delta_crps_map: dict[str, float] = {}

    if mode == "forecast":
        assert obs is not None
        y = np.asarray(obs, dtype=np.float64)
        fc_by_id = {m.miner_id: _as_2d_forecast(m.forecast, y.size) for m in valid}  # type: ignore[arg-type]
        for m in valid:
            indiv_losses[m.miner_id] = composite_forecast_loss(
                fc_by_id[m.miner_id], y, params, tail_threshold=tail_threshold
            )

        if staked_valid:
            all_fc = [fc_by_id[m.miner_id] for m in staked_valid]
            all_stakes = np.array([m.stake for m in staked_valid], dtype=np.float64)
            meta_loss = meta_ensemble_loss(
                all_fc, all_stakes, y, params, tail_threshold=tail_threshold
            )
            for idx, m in enumerate(staked_valid):
                if len(staked_valid) == 1:
                    # Single-miner fallback: without the sole miner, only the baseline remains.
                    loo_loss = baseline_loss
                else:
                    other_fc = [fc for j, fc in enumerate(all_fc) if j != idx]
                    other_stakes = np.delete(all_stakes, idx)
                    loo_loss = meta_ensemble_loss(
                        other_fc, other_stakes, y, params, tail_threshold=tail_threshold
                    )
                delta_crps_map[m.miner_id] = (loo_loss - meta_loss) / baseline_loss

        for m in valid:
            if m.miner_id not in delta_crps_map:
                delta_crps_map[m.miner_id] = 0.0

    elif mode == "pointwise_loss":
        pl_by_id: dict[str, np.ndarray] = {}
        expected_shape: tuple[int, ...] | None = None
        for m in valid:
            arr = np.asarray(m.pointwise_loss, dtype=np.float64)
            if arr.size == 0 or not np.isfinite(arr).all():
                raise ValueError(f"miner {m.miner_id!r}: pointwise_loss must be non-empty and finite")
            if expected_shape is None:
                expected_shape = arr.shape
            elif arr.shape != expected_shape:
                raise ValueError("all miners' pointwise_loss arrays must have identical shape")
            pl_by_id[m.miner_id] = arr
            indiv_losses[m.miner_id] = float(np.mean(arr))

        if staked_valid:
            stakes = np.array([m.stake for m in staked_valid], dtype=np.float64)
            losses = np.array([indiv_losses[m.miner_id] for m in staked_valid], dtype=np.float64)
            total_stake = float(np.sum(stakes))
            meta_loss = float(np.sum(stakes * losses) / total_stake)
            for idx, m in enumerate(staked_valid):
                if len(staked_valid) == 1:
                    loo_loss = baseline_loss
                else:
                    other_stakes = np.delete(stakes, idx)
                    other_losses = np.delete(losses, idx)
                    loo_loss = float(np.sum(other_stakes * other_losses) / float(np.sum(other_stakes)))
                delta_crps_map[m.miner_id] = (loo_loss - meta_loss) / baseline_loss

        for m in valid:
            if m.miner_id not in delta_crps_map:
                delta_crps_map[m.miner_id] = 0.0

    # Compute scores, raw positive payout weights, and slashes.
    scores: dict[str, float] = {}
    crpss_map: dict[str, float] = {}
    pos_weights: dict[str, float] = {}
    slashes: dict[str, float] = {}

    for m in ordered:
        if m.protocol_fault:
            slashes[m.miner_id] = params.protocol_fault_slash * m.stake
            pos_weights[m.miner_id] = 0.0
            continue
        crpss = 1.0 - indiv_losses[m.miner_id] / baseline_loss
        delta = delta_crps_map[m.miner_id]
        s_i = (1.0 - params.w_delta) * crpss + params.w_delta * delta
        crpss_map[m.miner_id] = crpss
        scores[m.miner_id] = s_i
        if s_i > 0.0 and m.stake > 0.0:
            pos_weights[m.miner_id] = (m.stake**params.stake_exponent) * s_i
            slashes[m.miner_id] = 0.0
        elif s_i < 0.0:
            pos_weights[m.miner_id] = 0.0
            slash_frac = min(params.max_slash_cap, params.alpha_slash * abs(s_i))
            slashes[m.miner_id] = slash_frac * m.stake
        else:
            pos_weights[m.miner_id] = 0.0
            slashes[m.miner_id] = 0.0

    total_pos_weight = float(np.sum([pos_weights[m.miner_id] for m in ordered]))
    payouts: list[MinerPayout] = []
    for m in ordered:
        reward = (
            pool_amount * (pos_weights[m.miner_id] / total_pos_weight)
            if total_pos_weight > 0.0 and pos_weights[m.miner_id] > 0.0
            else 0.0
        )
        payouts.append(
            MinerPayout(
                miner_id=m.miner_id,
                stake=float(m.stake),
                protocol_fault=m.protocol_fault,
                loss=None if m.protocol_fault else indiv_losses[m.miner_id],
                crpss=None if m.protocol_fault else crpss_map[m.miner_id],
                delta_crps=None if m.protocol_fault else delta_crps_map[m.miner_id],
                composite_score=None if m.protocol_fault else scores[m.miner_id],
                reward=float(reward),
                slash=float(slashes[m.miner_id]),
            )
        )

    total_paid = float(np.sum([p.reward for p in payouts]))
    unpaid_remainder = float(pool_amount) if total_pos_weight == 0.0 else 0.0
    total_slashed = float(np.sum([p.slash for p in payouts]))

    return ComputePoolSettlement(
        pool_amount=float(pool_amount),
        baseline_loss=float(baseline_loss),
        meta_ensemble_loss=meta_loss,
        payouts=tuple(payouts),
        total_paid=total_paid,
        unpaid_remainder=unpaid_remainder,
        total_slashed=total_slashed,
    )


# --------------------------------------------------------------------------- Data Provider Pool


def settle_data_pool(
    stations: list[StationEntry],
    *,
    pool_amount: float,
    params: ShadowLedgerParams,
) -> DataPoolSettlement:
    """Settle the Data Provider Pool (Section 5 of ``docs/TOKENOMICS.md``).

    ``W_s = QoD_s * PoL_s * CellCap_s * HW_s * Impact_s`` where
    ``CellCap_s = min(1.0, C_max / N_cell)`` for all stations sharing ``cell_id``.
    """
    if not math.isfinite(pool_amount) or pool_amount < 0.0:
        raise ValueError(f"pool_amount must be finite and >= 0, got {pool_amount}")

    seen_ids: set[str] = set()
    cell_counts: dict[str, int] = {}
    cell_caps: dict[str, int] = {}

    for st in stations:
        if not st.station_id:
            raise ValueError("station_id must be non-empty")
        if not st.cell_id:
            raise ValueError(f"station {st.station_id!r}: cell_id must be non-empty")
        if st.station_id in seen_ids:
            raise ValueError(f"duplicate station_id: {st.station_id!r}")
        seen_ids.add(st.station_id)
        if not math.isfinite(st.qod) or not (0.0 <= st.qod <= 1.0):
            raise ValueError(f"station {st.station_id!r}: qod must be in [0, 1], got {st.qod}")
        if not math.isfinite(st.pol) or not (0.0 <= st.pol <= 1.0):
            raise ValueError(f"station {st.station_id!r}: pol must be in [0, 1], got {st.pol}")
        if not math.isfinite(st.hw) or st.hw < 0.0:
            raise ValueError(f"station {st.station_id!r}: hw must be finite and >= 0, got {st.hw}")
        if not math.isfinite(st.impact) or not (1.0 <= st.impact <= 1.5):
            raise ValueError(f"station {st.station_id!r}: impact must be in [1.0, 1.5], got {st.impact}")
        cap = params.h3_cell_cap if st.cell_cap is None else st.cell_cap
        if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
            raise ValueError(f"station {st.station_id!r}: cell_cap must be int >= 1, got {cap}")
        if st.cell_id in cell_caps and cell_caps[st.cell_id] != cap:
            raise ValueError(f"conflicting cell_cap values for cell {st.cell_id!r}")
        cell_caps[st.cell_id] = cap
        cell_counts[st.cell_id] = cell_counts.get(st.cell_id, 0) + 1

    ordered = sorted(stations, key=lambda s: s.station_id)
    weights: list[tuple[StationEntry, int, int, float, float]] = []
    for st in ordered:
        n_cell = cell_counts[st.cell_id]
        c_max = cell_caps[st.cell_id]
        cell_factor = min(1.0, c_max / n_cell)
        w_s = st.qod * st.pol * cell_factor * st.hw * st.impact
        weights.append((st, n_cell, c_max, cell_factor, float(w_s)))

    total_weight = float(np.sum([w for _, _, _, _, w in weights]))
    payouts: list[StationPayout] = []
    for st, n_cell, c_max, cell_factor, w_s in weights:
        reward = pool_amount * (w_s / total_weight) if total_weight > 0.0 and w_s > 0.0 else 0.0
        payouts.append(
            StationPayout(
                station_id=st.station_id,
                cell_id=st.cell_id,
                cell_count=n_cell,
                cell_cap_limit=c_max,
                cell_cap_factor=float(cell_factor),
                weight=float(w_s),
                reward=float(reward),
            )
        )

    total_paid = float(np.sum([p.reward for p in payouts]))
    unpaid_remainder = float(pool_amount) if total_weight == 0.0 else 0.0

    return DataPoolSettlement(
        pool_amount=float(pool_amount),
        payouts=tuple(payouts),
        total_weight=total_weight,
        total_paid=total_paid,
        unpaid_remainder=unpaid_remainder,
    )


# --------------------------------------------------------------------------- Full Epoch


def settle_epoch(
    *,
    epoch_emission: float,
    miners: list[MinerEntry],
    stations: list[StationEntry],
    params: ShadowLedgerParams,
    baseline_loss: float | None = None,
    baseline_forecast: np.ndarray | None = None,
    obs: np.ndarray | None = None,
    tail_threshold: np.ndarray | float | None = None,
) -> EpochSettlement:
    """Split ``epoch_emission`` into 55/35/10 pools and settle compute and data payouts."""
    if not math.isfinite(epoch_emission) or epoch_emission < 0.0:
        raise ValueError(f"epoch_emission must be finite and >= 0, got {epoch_emission}")

    p_comp = epoch_emission * params.p_comp_share
    p_data = epoch_emission * params.p_data_share
    p_val = epoch_emission - p_comp - p_data

    compute = settle_compute_pool(
        miners,
        pool_amount=p_comp,
        params=params,
        baseline_loss=baseline_loss,
        baseline_forecast=baseline_forecast,
        obs=obs,
        tail_threshold=tail_threshold,
    )
    data = settle_data_pool(stations, pool_amount=p_data, params=params)
    return EpochSettlement(
        epoch_emission=float(epoch_emission),
        compute=compute,
        data=data,
        validator_pool=float(p_val),
    )
