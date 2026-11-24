"""Off-chain Proof-of-Forecast-Skill (PoFS) shadow payout ledger.

Paper-only implementation of the compute pool (Section 4) and data provider pool
(Section 5) formulas in ``docs/TOKENOMICS.md``. No chain, no wallet, no network.
All calculations are deterministic pure functions over NumPy arrays.
"""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType

import numpy as np
import xarray as xr

from wrfcomp.harness.config import load_config
from wrfcomp.harness.metrics import diebold_mariano_pvalue, evaluate_gates
from wrfcomp.harness.schemas import Scorecard

DEFAULT_REPLAY_STAKE = 100.0


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
    stake_exponent: float = 1.0
    k_min: float = 10.0
    clone_relative_tolerance: float = 0.05
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
        ):
            if not math.isfinite(val) or not (0.0 <= val <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {val}")
        if not math.isfinite(self.k_min) or self.k_min <= 0:
            raise ValueError("k_min must be finite and positive")
        if not math.isfinite(self.clone_relative_tolerance) or self.clone_relative_tolerance < 0:
            raise ValueError("clone_relative_tolerance must be finite and nonnegative")
        if not math.isfinite(self.stake_exponent) or self.stake_exponent < 1.0:
            raise ValueError(
                f"stake_exponent must be at least 1 for split resistance, got {self.stake_exponent}"
            )
        if (
            isinstance(self.h3_cell_cap, bool)
            or not isinstance(self.h3_cell_cap, int)
            or self.h3_cell_cap < 1
        ):
            raise ValueError(f"h3_cell_cap must be an integer >= 1, got {self.h3_cell_cap}")


@dataclass(frozen=True)
class Commitment:
    """Receipt from the epoch recorder, separate from the revealed payload."""

    digest: str
    received_at: np.datetime64


@dataclass(frozen=True)
class EpochEvidence:
    """Trusted epoch facts. init_time is the first init covered by a batch."""

    init_time: np.datetime64
    gefs_digest: str
    commitments: Mapping[str, Commitment]

    def __post_init__(self):
        object.__setattr__(self, "commitments", MappingProxyType(dict(self.commitments)))
        if not isinstance(self.init_time, np.datetime64) or np.isnat(self.init_time):
            raise ValueError("invalid epoch init")
        for digest in [self.gefs_digest, *(r.digest for r in self.commitments.values())]:
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("invalid SHA-256 digest")
        for receipt in self.commitments.values():
            if not isinstance(receipt.received_at, np.datetime64) or np.isnat(receipt.received_at):
                raise ValueError("invalid commitment time")


def forecast_digest(forecast: np.ndarray) -> str:
    """Raw-array commitment to the ordered forecast rows, dimensions and members."""
    arr = np.asarray(forecast, dtype="<f8")
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError("expected a 2-D ensemble")
    h = hashlib.sha256(b"wrfcomp-shadow-forecast-v1" + struct.pack("<QQ", *arr.shape))
    canonical = np.where(arr == 0, 0.0, arr).astype("<f8", copy=False)
    h.update(canonical.tobytes(order="C"))
    return h.hexdigest()


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
    status: str


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
        raise ValueError(f"forecast must have shape ({n_obs},) or ({n_obs}, M>=1), got {arr.shape}")
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


def _validate_tail_threshold(
    threshold: np.ndarray | float | int,
    n_obs: int | None = None,
) -> np.ndarray:
    """Validate that tail_threshold is finite, real, float64-representable, and shape-compatible."""
    if isinstance(threshold, bool):
        raise ValueError("tail_threshold must be numeric, not bool")
    if isinstance(threshold, complex):
        raise ValueError("tail_threshold must be real, not complex")
    try:
        raw = np.asarray(threshold)
        if raw.dtype.kind == "b":
            raise ValueError("tail_threshold must be numeric, not bool")
        if raw.dtype.kind == "c":
            raise ValueError("tail_threshold must be real, not complex")
        with np.errstate(over="ignore"):
            thr = np.asarray(raw, dtype=np.float64)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"tail_threshold must be finite and representable in float64: {exc}") from exc
    if thr.ndim > 1:
        raise ValueError(f"tail_threshold must be a scalar or 1-D array, got shape {thr.shape}")
    if thr.ndim == 1:
        if thr.size == 0:
            raise ValueError("tail_threshold must be non-empty")
        if n_obs is not None and thr.shape != (n_obs,):
            raise ValueError(f"tail_threshold length {thr.shape[0]} does not match obs length {n_obs}")
    if not np.isfinite(thr).all():
        raise ValueError(
            "tail_threshold must be finite (NaN, +/-infinity, and float64-overflowing values are refused)"
        )
    return thr


def _clip_tail(values: np.ndarray, threshold: np.ndarray | float) -> np.ndarray:
    n_obs = values.shape[0] if values.ndim >= 1 else None
    thr = _validate_tail_threshold(threshold, n_obs=n_obs)
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
    thr = _validate_tail_threshold(tail_threshold, n_obs=y.size)
    x = _as_2d_forecast(forecast, y.size)
    twcrps = fair_ensemble_crps(_clip_tail(x, thr), _clip_tail(y, thr))
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
    thr = _validate_tail_threshold(tail_threshold, n_obs=y.size)
    clipped_fc = [_clip_tail(_as_2d_forecast(fc, y.size), thr) for fc in forecasts]
    clipped_obs = _clip_tail(y, thr)
    twcrps = weighted_mixture_crps(clipped_fc, stakes, clipped_obs)
    return float(np.mean(composite_pointwise_loss(crps, twcrps, params)))


# --------------------------------------------------------------------------- Compute Pool


def _array_gate(fc, gefs, obs, threshold, issue_times, lead_hours, cfg):
    """Single variable, single lead, one verifying observation per issue."""
    if threshold is None or issue_times is None or lead_hours is None:
        return False
    times = np.asarray(issue_times)
    if (
        times.shape != obs.shape
        or times.dtype.kind != "M"
        or np.isnat(times).any()
        or not np.isfinite(lead_hours)
        or lead_hours <= 0
    ):
        raise ValueError("invalid DM initialization grid or lead")

    def losses(x):
        def scores(values, truth):
            fair = fair_ensemble_crps(values, truth)
            mae = np.abs(values - truth[:, None]).mean(axis=1)
            return fair, fair + (mae - fair) / values.shape[1]

        c, ce = scores(x, obs)
        tail, te = scores(_clip_tail(x, threshold), _clip_tail(obs, threshold))
        return (
            cfg.gate.loss_weights.crps_fair * c + cfg.gate.loss_weights.twcrps_p95 * tail,
            cfg.gate.loss_weights.crps_fair * ce + cfg.gate.loss_weights.twcrps_p95 * te,
            tail.mean(),
        )

    base, be, bt = losses(gefs)
    cand, ce, ct = losses(fc)
    rmse = np.sqrt(np.mean((fc.mean(axis=1) - obs) ** 2))
    n = fc.shape[1]
    if n < 2 or rmse == 0 or base.mean() <= 0 or be.mean() <= 0:
        return False
    spread = np.sqrt((n + 1) / n * fc.var(axis=1, ddof=1).mean()) / rmse
    diff = xr.DataArray(base - cand, dims="init_time", coords={"init_time": times})
    metrics = dict(
        crpss_vs_baseline=1 - cand.mean() / base.mean(),
        crpss_vs_baseline_ecdf=1 - ce.mean() / be.mean(),
        twcrps_p95=ct,
        baseline_twcrps_p95=bt,
        spread_skill_ratio=spread,
        n_members=n,
        dm_pvalue=diebold_mariano_pvalue(diff, lead_hours),
    )
    return all(evaluate_gates({"all": metrics}, cfg).values())


def _mean_wasserstein(left: np.ndarray, right: np.ndarray) -> float:
    """Mean row-wise W1 of sorted, equally weighted empirical ensembles."""
    m, n = left.shape[1], right.shape[1]
    if m == n:
        return float(np.abs(left - right).mean())
    # On each interval between empirical quantile jumps both quantiles
    # are constant. Integer coordinates on [0, m*n] keep the jumps exact.
    edges = np.union1d(np.arange(m + 1) * n, np.arange(n + 1) * m)
    lower = edges[:-1]
    distances = np.abs(left[:, lower // n] - right[:, lower // m]).mean(axis=0)
    return float(np.dot(np.diff(edges) / (m * n), distances))


def settle_compute_pool(
    miners: list[MinerEntry],
    *,
    pool_amount: float,
    params: ShadowLedgerParams,
    baseline_loss: float | None = None,
    baseline_forecast: np.ndarray | None = None,
    obs: np.ndarray | None = None,
    tail_threshold: np.ndarray | float | None = None,
    evidence: EpochEvidence | None = None,
    issue_times: np.ndarray | None = None,
    lead_hours: float | None = None,
    gefs_forecast: np.ndarray | None = None,
) -> ComputePoolSettlement:
    """The raw-array adapter settles a single variable/lead forecast batch.

    A scalar baseline or supplied pointwise losses alone cannot prove eligibility.
    For multi-variable epochs use the evaluator scorecard adapter.
    """
    y = None if obs is None else np.asarray(obs, dtype=float)
    if tail_threshold is not None:
        _validate_tail_threshold(tail_threshold, None if y is None else y.size)
    gefs = None
    if gefs_forecast is not None:
        baseline_forecast = gefs_forecast
    if baseline_forecast is not None:
        if y is None:
            raise ValueError("baseline_forecast needs observations")
        gefs = _as_2d_forecast(baseline_forecast, y.size)
        if evidence and forecast_digest(gefs) != evidence.gefs_digest:
            raise ValueError("baseline does not match epoch GEFS digest")
        actual = composite_forecast_loss(gefs, y, params, tail_threshold)
        if baseline_loss is not None and baseline_loss != actual:
            raise ValueError("baseline loss differs from forecast")
        baseline_loss = actual
    if issue_times is not None and evidence and np.min(issue_times) != evidence.init_time:
        raise ValueError("commit cutoff differs from first initialization")
    losses, digests, gates, arrays = {}, {}, {}, {}
    cfg = load_config()
    gate_cache = {}
    loss_shape = None
    for m in miners:
        if m.protocol_fault:
            continue
        if m.forecast is not None and m.pointwise_loss is not None:
            raise ValueError("provide only one reveal representation")
        if m.forecast is not None:
            if y is None:
                raise ValueError("forecast needs observations")
            try:
                fc = _as_2d_forecast(m.forecast, y.size)
            except (TypeError, ValueError, OverflowError):
                if evidence and m.miner_id in evidence.commitments:
                    continue  # malformed committed reveal: maximum slash
                raise
            arrays[m.miner_id] = np.sort(fc, axis=1)
            digests[m.miner_id] = forecast_digest(fc)
            losses[m.miner_id] = composite_forecast_loss(fc, y, params, tail_threshold)
            digest = digests[m.miner_id]
            if digest not in gate_cache:
                gate_cache[digest] = bool(
                    evidence
                    and gefs is not None
                    and _array_gate(fc, gefs, y, tail_threshold, issue_times, lead_hours, cfg)
                )
            gates[m.miner_id] = gate_cache[digest]
        elif m.pointwise_loss is not None:
            arr = np.asarray(m.pointwise_loss, dtype=float)
            if not arr.size or not np.isfinite(arr).all():
                raise ValueError("invalid pointwise loss")
            if loss_shape is not None and arr.shape != loss_shape:
                raise ValueError("all pointwise_loss arrays must have identical shape")
            loss_shape = arr.shape
            losses[m.miner_id] = float(arr.mean())

    if arrays and loss_shape is not None:
        raise ValueError("cannot mix forecast and pointwise-loss inputs")

    def mixture(ids):
        return meta_ensemble_loss([arrays[k] for k in ids], np.ones(len(ids)), y, params, tail_threshold)

    clone_distance = 0.0
    if gefs is not None:
        field_scale = max(float(np.sqrt(gefs.var(axis=1).mean())), float(np.std(y)))
        clone_distance = params.clone_relative_tolerance * field_scale
    return _settle(
        miners,
        pool_amount,
        params,
        baseline_loss,
        evidence,
        losses,
        digests,
        gates,
        arrays,
        mixture,
        clone_distance,
    )


def _settle(
    miners, pool, params, baseline, evidence, losses, digests, gates, arrays, mixture, clone_distance=0.0
):
    if (
        not math.isfinite(pool)
        or pool < 0
        or baseline is None
        or not math.isfinite(baseline)
        or baseline <= 0
    ):
        raise ValueError("invalid pool or baseline loss")
    ordered = sorted(miners, key=lambda m: m.miner_id)
    wallets = {m.miner_id for m in ordered}
    if len(wallets) != len(miners) or any(
        not m.miner_id or not math.isfinite(m.stake) or m.stake < 0 for m in miners
    ):
        raise ValueError("invalid or duplicate miner or stake")
    receipts = evidence.commitments if evidence else {}
    if receipts.keys() - wallets:
        raise ValueError("committed wallet missing from epoch roster")
    status = {}
    revealed = []
    for m in ordered:
        k = m.miner_id
        receipt = receipts.get(k)
        if m.protocol_fault:
            status[k] = "protocol_fault"
        elif receipt is None:
            status[k] = "uncommitted"
        elif digests.get(k) != receipt.digest:
            status[k] = "non_reveal_or_mismatch"
        elif receipt.received_at >= evidence.init_time:
            status[k] = "late"
        elif m.stake < params.k_min:
            status[k] = "below_minimum"
        elif digests[k] == evidence.gefs_digest or not gates.get(k, False) or losses[k] >= baseline:
            status[k] = "gefs_gate"
        else:
            revealed.append(m)
    winners = []
    for m in sorted(revealed, key=lambda m: (receipts[m.miner_id].received_at, m.miner_id)):
        k = m.miner_id
        is_clone = any(
            digests[k] == digests[earlier]
            or (k in arrays and _mean_wasserstein(arrays[k], arrays[earlier]) <= clone_distance)
            for earlier in winners
        )
        if is_clone:
            status[k] = "clone"
        else:
            status[k] = "eligible"
            winners.append(k)
    winners.sort()
    meta = mixture(winners) if winners else None
    payouts = []
    weights = []
    for m in ordered:
        k = m.miner_id
        loss = losses.get(k)
        crpss = None if loss is None else 1 - loss / baseline
        delta = score = None
        slash = weight = 0.0
        if status[k] in ("non_reveal_or_mismatch", "protocol_fault"):
            slash = m.stake * params.max_slash_cap
        elif status[k] == "eligible":
            others = [j for j in winners if j != k]
            delta = ((mixture(others) if others else baseline) - meta) / baseline
            score = (1 - params.w_delta) * crpss + params.w_delta * delta
            weight = m.stake**params.stake_exponent * max(0.0, score)
        elif status[k] == "gefs_gate" and loss is not None:
            # Measure harm against the already-fixed eligible pool only for the
            # excluded miner's own downside. It never affects a winner's Delta.
            delta = min(0.0, (meta - mixture(sorted([*winners, k]))) / baseline) if winners else 0.0
            score = min(0.0, (1 - params.w_delta) * crpss + params.w_delta * delta)
        if score is not None and score < 0:
            slash = m.stake * min(params.max_slash_cap, params.alpha_slash * abs(score))
        weights.append(weight)
        payouts.append(
            MinerPayout(
                k,
                m.stake,
                m.protocol_fault or status[k] == "non_reveal_or_mismatch",
                loss,
                crpss,
                delta,
                score,
                0.0,
                slash,
                status[k],
            )
        )
    total = math.fsum(weights)
    if total:
        payouts = [replace(p, reward=pool * (w / total)) for p, w in zip(payouts, weights, strict=True)]
    return ComputePoolSettlement(
        pool,
        baseline,
        meta,
        tuple(payouts),
        math.fsum(p.reward for p in payouts),
        0.0 if total else pool,
        math.fsum(p.slash for p in payouts),
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
    evidence: EpochEvidence | None = None,
    issue_times: np.ndarray | None = None,
    lead_hours: float | None = None,
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
        evidence=evidence,
        issue_times=issue_times,
        lead_hours=lead_hours,
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


# --------------------------------------------------------------------------- Scorecard Loader


def load_scorecard(path: Path | str) -> Scorecard:
    """Read and validate a ``wrfcomp-eval`` JSON scorecard via ``Scorecard``."""
    return Scorecard.model_validate_json(Path(path).read_text(encoding="utf-8"))


def scorecard_variable_loss(
    scorecard: Scorecard,
    variable: str,
    leads: Sequence[int] | None = None,
) -> float:
    """Return the pair-weighted composite loss for ``variable`` over ``leads`` (or ``"all"``)."""
    if variable not in scorecard.metrics:
        raise ValueError(f"scorecard {scorecard.model_id!r} has no metrics for variable {variable!r}")
    per_lead = scorecard.metrics[variable]
    if leads is None:
        entry = per_lead["all"]
        if entry.loss is None or not math.isfinite(entry.loss):
            raise ValueError(f"scorecard {scorecard.model_id!r} has undefined 'all' loss for {variable!r}")
        return float(entry.loss)

    total_weighted = 0.0
    total_pairs = 0
    for lead in leads:
        key = str(int(lead))
        if key not in per_lead:
            raise ValueError(f"scorecard {scorecard.model_id!r} lacks lead {lead}h for {variable!r}")
        entry = per_lead[key]
        if entry.n_pairs == 0:
            continue
        if entry.loss is None or not math.isfinite(entry.loss):
            raise ValueError(f"scorecard {scorecard.model_id!r} has undefined loss at {variable!r} {lead}h")
        total_weighted += float(entry.loss) * entry.n_pairs
        total_pairs += entry.n_pairs
    if total_pairs == 0:
        raise ValueError(f"scorecard {scorecard.model_id!r} has zero scored pairs for {variable!r}")
    return total_weighted / total_pairs


def miners_from_scorecards(
    scorecards: Mapping[str, Scorecard] | Sequence[tuple[str, Scorecard]],
    *,
    baseline_key: str,
    stakes: Mapping[str, float] | float = DEFAULT_REPLAY_STAKE,
    variables: Sequence[str] | None = None,
    leads: Sequence[int] | None = None,
    fault_ids: Collection[str] = (),
) -> tuple[list[MinerEntry], float, tuple[str, ...], tuple[int, ...]]:
    """Build compute-pool ``MinerEntry`` inputs from validated ``wrfcomp-eval`` scorecards.

    Aligns all non-faulty scorecards onto the same truth/thresholds hashes, evaluation period,
    common variables, and common scored leads. For a single variable, returns physical-unit
    losses with ``baseline_loss = L_base(v)``; for multiple variables, returns dimensionless
    baseline-normalized loss ratios ``L_i(v) / L_base(v)`` with ``baseline_loss = 1.0`` so
    variables with different units (Pa vs K vs m/s) receive equal weight.
    """
    items = list(scorecards.items()) if isinstance(scorecards, Mapping) else list(scorecards)
    if not items:
        raise ValueError("scorecards must be non-empty")
    card_map = dict(items)
    if len(card_map) != len(items):
        raise ValueError("duplicate miner keys in scorecards")
    if baseline_key not in card_map:
        raise ValueError(f"baseline_key {baseline_key!r} not found in scorecards")

    fault_set = set(fault_ids)
    if baseline_key in fault_set:
        raise ValueError(f"baseline_key {baseline_key!r} cannot be marked as a protocol fault")

    active_keys = [k for k in sorted(card_map) if k not in fault_set]
    ref = card_map[baseline_key]
    for key in active_keys:
        sc = card_map[key]
        if (
            sc.truth_sha256 != ref.truth_sha256
            or sc.thresholds_sha256 != ref.thresholds_sha256
            or sc.period_start != ref.period_start
            or sc.period_end != ref.period_end
            or sc.init_hours_scored != ref.init_hours_scored
        ):
            raise ValueError(
                f"scorecard {key!r} does not share truth, thresholds, period, and init hours "
                f"with baseline {baseline_key!r}"
            )

    shared_vars = set.intersection(*(set(card_map[k].metrics) for k in active_keys))
    if variables is None:
        chosen_vars = tuple(sorted(shared_vars))
    else:
        chosen_vars = tuple(variables)
        missing_vars = [v for v in chosen_vars if v not in shared_vars]
        if missing_vars:
            raise ValueError(f"requested variable(s) not present in all scorecards: {missing_vars}")
    if not chosen_vars:
        raise ValueError("scorecards share no common scored variables")

    shared_leads = set.intersection(*(set(card_map[k].leads_scored) for k in active_keys))
    if leads is None:
        chosen_leads = tuple(sorted(shared_leads))
    else:
        chosen_leads = tuple(sorted(int(h) for h in leads))
        missing_leads = [h for h in chosen_leads if h not in shared_leads]
        if missing_leads:
            raise ValueError(f"requested lead(s) not present in all scorecards: {missing_leads}")
    if not chosen_leads:
        raise ValueError("scorecards share no common scored leads")

    base_losses = {v: scorecard_variable_loss(ref, v, chosen_leads) for v in chosen_vars}
    for v, loss in base_losses.items():
        if loss <= 0.0:
            raise ValueError(f"baseline loss for {v!r} must be > 0, got {loss}")

    def _stake_for(key: str) -> float:
        if isinstance(stakes, Mapping):
            if key not in stakes:
                raise ValueError(f"missing stake for miner {key!r}")
            return float(stakes[key])
        return float(stakes)

    miners: list[MinerEntry] = []
    for key in sorted(card_map):
        stake = _stake_for(key)
        if key in fault_set:
            miners.append(MinerEntry(miner_id=key, stake=stake, protocol_fault=True))
            continue
        sc = card_map[key]
        if len(chosen_vars) == 1:
            v = chosen_vars[0]
            loss_arr = np.array([scorecard_variable_loss(sc, v, chosen_leads)], dtype=np.float64)
        else:
            loss_arr = np.array(
                [scorecard_variable_loss(sc, v, chosen_leads) / base_losses[v] for v in chosen_vars],
                dtype=np.float64,
            )
        miners.append(MinerEntry(miner_id=key, stake=stake, pointwise_loss=loss_arr))

    baseline_loss = base_losses[chosen_vars[0]] if len(chosen_vars) == 1 else 1.0
    return miners, float(baseline_loss), chosen_vars, chosen_leads


def settle_from_scorecards(
    scorecards: Mapping[str, Scorecard] | Sequence[tuple[str, Scorecard]],
    *,
    baseline_key: str,
    pool_amount: float,
    params: ShadowLedgerParams,
    stakes: Mapping[str, float] | float = DEFAULT_REPLAY_STAKE,
    variables: Sequence[str] | None = None,
    leads: Sequence[int] | None = None,
    fault_ids: Collection[str] = (),
    evidence: EpochEvidence | None = None,
) -> ComputePoolSettlement:
    """Validate ``wrfcomp-eval`` scorecards and settle the compute pool in pointwise-loss mode."""
    cards = dict(scorecards)
    if len(cards) != len(scorecards) or baseline_key not in cards:
        raise ValueError("duplicate scorecard wallet or missing GEFS reference")
    ref = cards[baseline_key]
    if ref.model_id != "noaa-gefs" or baseline_key in fault_ids or ref.sealed_data_used:
        raise ValueError("scorecard settlement requires a valid NOAA GEFS baseline")
    if evidence and (
        evidence.gefs_digest != ref.forecast_sha256 or evidence.init_time != np.datetime64(ref.period_start)
    ):
        raise ValueError("epoch GEFS digest or init differs from reference scorecard")
    cfg = load_config()
    primary = tuple(cfg.evaluation.variables.primary)
    chosen_vars = tuple(variables) if variables is not None else primary
    chosen_leads = tuple(leads) if leads is not None else tuple(ref.leads_scored)
    if (
        not chosen_vars
        or not chosen_leads
        or not set(chosen_vars) <= set(ref.metrics)
        or not set(chosen_leads) <= set(ref.leads_scored)
    ):
        raise ValueError("invalid diagnostic variables or leads")
    base = {v: scorecard_variable_loss(ref, v, chosen_leads) for v in chosen_vars}
    if min(base.values()) <= 0:
        raise ValueError("nonpositive GEFS loss")
    single = len(chosen_vars) == 1
    baseline_loss = base[chosen_vars[0]] if single else 1.0
    losses, digests, gates = {}, {}, {}
    miners = []
    roster = set(cards) | set(fault_ids)
    if evidence:
        roster.update(evidence.commitments)
    for key in sorted(roster):
        stake = float(stakes[key] if isinstance(stakes, Mapping) else stakes)
        miners.append(MinerEntry(key, stake, protocol_fault=key in fault_ids))
        if key not in cards or key in fault_ids:
            continue
        sc = cards[key]
        if (
            any(
                getattr(sc, k) != getattr(ref, k)
                for k in (
                    "harness_version",
                    "protocol_version",
                    "truth_sha256",
                    "thresholds_sha256",
                    "period_start",
                    "period_end",
                    "init_hours_scored",
                    "n_init_times",
                    "n_stations",
                )
            )
            or sc.sealed_data_used
        ):
            continue
        if not set(chosen_vars) <= set(sc.metrics) or not set(chosen_leads) <= set(sc.leads_scored):
            continue
        try:
            per_var = {v: scorecard_variable_loss(sc, v, chosen_leads) for v in chosen_vars}
        except ValueError:
            continue  # Undefined loss cannot substantiate a committed reveal.
        losses[key] = (
            per_var[chosen_vars[0]] if single else float(np.mean([per_var[v] / base[v] for v in base]))
        )
        # Preserve historical loss diagnostics, but recognize a reveal only
        # after checking integrity on the entire epoch, even for a diagnostic slice.
        if sc == ref:
            digests[key] = sc.forecast_sha256  # authentic GEFS stays ineligible
            continue
        if (
            sc.baseline_id != "noaa-gefs"
            or sc.baseline_sha256 != ref.forecast_sha256
            or sc.variables_missing
            or set(sc.metrics) != set(primary)
            or sc.leads_scored != ref.leads_scored
        ):
            continue
        if any(
            sc.metrics[v][h].n_pairs != ref.metrics[v][h].n_pairs
            or sc.metrics[v][h].n_masked_no_threshold != ref.metrics[v][h].n_masked_no_threshold
            for v in primary
            for h in ref.metrics[v]
        ):
            continue
        derived = {v: evaluate_gates({"all": sc.metrics[v]["all"].model_dump()}, cfg) for v in primary}
        if derived != sc.gates:
            continue
        digests[key] = sc.forecast_sha256
        gates[key] = (
            sc.passed_gate and set(chosen_vars) == set(primary) and list(chosen_leads) == ref.leads_scored
        )

    def mixture(ids):
        return float(np.mean([losses[k] for k in ids]))

    return _settle(miners, pool_amount, params, baseline_loss, evidence, losses, digests, gates, {}, mixture)
