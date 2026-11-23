"""Scoring kernel: fair CRPS, tail-weighted CRPS, spread-skill, Diebold-Mariano and the promotion gates.

Rules this module follows (see docs/VERIFICATION_PROTOCOL.md and configs/splits.yaml):

* **Evaluation grid.** The caller names the init times and stations to be scored
  (`expected_init_times`, `expected_station_ids`). They must come from the protocol (configured init
  hours over the scored period, the stations file), never from the forecast being scored.
  Observations must sit on exactly that grid;
  the candidate and the baseline must cover all of it, so a forecast cannot drop a station or a date.
  Whole leads are the one exception: the leads scored are the candidate's leads.
* **Mask.** A grid point `(init_time, lead_time, station_id)` is scored when its observation is finite
  AND the tail threshold of its station, variable and calendar month of the valid time is finite
  (`scoring_mask`). A threshold is missing exactly where the station has no training observation in
  that month, so the mask is fixed by the truth and the training data alone: no forecast can change
  it, and it is the same for every model. Points with a valid observation but no threshold are
  counted per entry as `n_masked_no_threshold`. Only NaN marks a missing threshold: plus or minus
  infinity anywhere in the thresholds is refused. That one mask is used for the candidate and for the
  baseline. A forecast (candidate or baseline) that is not finite at a scored point is an error,
  never a silently skipped point.
* **Members are a sample.** Fair CRPS and the spread-skill correction assume the members are
  exchangeable random draws from the forecast distribution. Fair CRPS is not a proper score for
  members placed deliberately: two fixed members score exactly 0 whenever the observation falls
  between them, and members fixed at inflated quantiles beat the true distribution while staying
  inside the spread-skill band (by about 45 % with 2 members, 21 % with 5, 10 % with 10, 5 % with 20,
  3 % with 32). The kernel cannot tell sampled members from placed ones, so it has two defences.
  Every CRPS is also computed with the plain ensemble ("ecdf") estimator, which has no such hole
  (it is biased high for small sampled ensembles instead), both are reported, and the skill gate
  needs skill above the minimum under BOTH estimators. And an ensemble with fewer than
  `min_members_for_gate` members (a required config value) is scored but can never pass the gate.
  The tail gate and the Diebold-Mariano test use the fair estimator.
* **Averages.** `crps`, `twcrps_p95` and the spread-skill terms are plain means over the scored points
  of a lead, or of every lead for `"all"`. Nothing is weighted.
* **Undefined is `None`.** A metric that cannot be computed (no scored points, a zero baseline loss,
  a Diebold-Mariano series that is too short or all zero) is `None`, and a `None` never passes a gate.
* **Diebold-Mariano.** The series is, per init time, the mean over the scored stations of
  `baseline loss - candidate loss`, with the composite loss taken point by point. Init times with no
  scored station are NaN. The test needs an evenly spaced series, because the autocovariance is
  estimated by position: leading and trailing NaNs are trimmed, and if what remains has a NaN inside
  it, or the init times are not evenly spaced, the p-value is undefined (`None`). Nothing is dropped
  from the middle or interpolated. The lag is `h = max(1, ceil(lead / spacing))` with spacing the
  step between init times. `"all"` averages the leads per init time and
  uses the lag of the longest lead. The test statistic is the Hering-Genton ("HG") form from
  `scores`, compared against Student's t with `n - 1` degrees of freedom; the p-value is one-sided
  (small when the candidate is better). HG fits an exponential model to the autocovariances of the
  series itself, over `max(floor((n - 1) / 2), h)` lags, so `h` is only a floor on how much
  autocorrelation is modelled: on a long series the result does not change with `h`. HG was chosen
  over the Harvey-Leybourne-Newbold ("HLN") form, which truncates at exactly `h - 1` lags and rejects
  a true null far too often when the differences stay correlated beyond that (as they do at short
  leads, where forecast errors follow multi-day weather regimes). Measured false-rejection rates at a
  nominal 5 %, 400 seeded mean-zero series each, Student's t, series lengths 350 and 1400:

      differences        h    HG (n=350, 1400)    HLN (n=350, 1400)
      AR(1) phi=0.5      1    5.0 %,  5.3 %       15.5 %, 17.0 %
      AR(1) phi=0.8      1    4.8 %,  6.3 %       27.3 %, 30.8 %
      AR(1) phi=0.8      8    4.8 %,  6.3 %        6.3 %,  7.5 %
      MA(7)              8    5.0 %,  5.0 %        5.5 %,  5.5 %
      MA(1)              2    5.3 %,  4.3 %        6.0 %,  5.3 %

  HG stays near 5 % in every case; HLN is only sound when the correlation really ends at `h - 1`.
* **Spread-skill.** `sqrt((M + 1) / M) * sqrt(mean(ens_var)) / sqrt(mean((ens_mean - obs)^2))` with
  the unbiased (ddof=1) variance and M the number of members. The `(M + 1) / M` factor is the standard
  finite-ensemble correction: the mean of M members has its own sampling error, so without it a
  perfectly calibrated ensemble would score `sqrt(M / (M + 1))` (0.82 for 2 members) instead of 1.
* **Determinism.** Inputs are sorted by coordinate, cast to float64, and the ensemble dimension is put
  in a single dask chunk. Pointwise scores are brought into memory and reduced with numpy in a fixed
  order, so results do not depend on dask chunking, input ordering or dict order. Nothing is random.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
import scipy.stats
import xarray as xr
from scores.probability import crps_for_ensemble, tw_crps_for_ensemble
from scores.stats.statistical_tests import diebold_mariano

from wrfcomp.harness.config import HarnessConfig, load_config

ENS_DIM = "ens_member"
POINT_DIMS = ("init_time", "lead_time", "station_id")
DM_METHOD = "HG"

# One metrics entry, as stored under a lead key. `None` means "not defined" (never 0).
MetricValues = dict[str, float | int | None]

_HOUR_NS = 3_600_000_000_000


def _as_float64(da: xr.DataArray) -> xr.DataArray:
    return da.astype(np.float64)


def _prepare_ensemble(fcst: xr.DataArray) -> xr.DataArray:
    if ENS_DIM not in fcst.dims:
        raise ValueError(f"forecast has no '{ENS_DIM}' dimension")
    if fcst.sizes[ENS_DIM] < 2:
        raise ValueError("fair CRPS and ensemble spread need at least 2 ensemble members")
    fcst = _as_float64(fcst)
    if fcst.chunks is not None:
        fcst = fcst.chunk({ENS_DIM: -1})
    return fcst


def fair_crps(fcst: xr.DataArray, obs: xr.DataArray) -> xr.DataArray:
    """Pointwise fair ensemble CRPS (no averaging). NaN where the observation is NaN."""
    return crps_for_ensemble(
        _prepare_ensemble(fcst), _as_float64(obs), ENS_DIM, method="fair", preserve_dims="all"
    )


def ecdf_crps(fcst: xr.DataArray, obs: xr.DataArray) -> xr.DataArray:
    """Pointwise CRPS of the ensemble's empirical distribution (no averaging). NaN where obs is NaN.

    Proper for any member set, sampled or placed, but biased high for a small sampled ensemble.
    """
    return crps_for_ensemble(
        _prepare_ensemble(fcst), _as_float64(obs), ENS_DIM, method="ecdf", preserve_dims="all"
    )


def broadcast_threshold(
    threshold: xr.DataArray, init_time: xr.DataArray, lead_time: xr.DataArray
) -> xr.DataArray:
    """Thresholds `(station_id, month)` -> `(init_time, lead_time, station_id)` by month of the VALID time."""
    if set(threshold.dims) != {"station_id", "month"}:
        raise ValueError(f"threshold must have dims (station_id, month), got {tuple(threshold.dims)}")
    months = threshold["month"].values.tolist()
    if months != list(range(1, 13)):
        raise ValueError("threshold month coordinate must be exactly 1..12")
    init = xr.DataArray(init_time.values, dims="init_time", coords={"init_time": init_time.values})
    lead = xr.DataArray(lead_time.values, dims="lead_time", coords={"lead_time": lead_time.values})
    valid_month = (init + lead).dt.month
    out = threshold.sel(month=valid_month).drop_vars("month")
    return out.transpose("init_time", "lead_time", "station_id")


def _threshold_float64(threshold: xr.DataArray | float) -> xr.DataArray | float:
    """The threshold as float64, the type scoring computes in, refused if any value is infinite.

    A threshold is a number, or NaN for "no training data". Plus or minus infinity is neither: it
    would pass for a missing threshold in the mask, or make the tail score meaningless (clipping at
    minus infinity gives back the plain CRPS, clipping at plus infinity gives NaN).

    The check is made on the float64 values, AFTER the cast, because those are the values that are
    used: a wider type can hold a finite number (long double 1e400) that becomes infinity in float64.
    Every public function that takes a threshold gets it through here, before any mask or clipping.
    """
    with warnings.catch_warnings(), np.errstate(over="ignore"):
        warnings.simplefilter("ignore", category=RuntimeWarning)  # the overflow is reported below
        if isinstance(threshold, xr.DataArray):
            cast: xr.DataArray | float = _as_float64(threshold)
            infinite = int(np.isinf(cast).sum())
        else:
            cast = float(threshold)
            infinite = int(math.isinf(cast))
    if infinite:
        raise ValueError(
            f"tail threshold has {infinite} infinite value(s) as float64; "
            "only NaN may mark a missing threshold"
        )
    return cast


def tail_crps(
    fcst: xr.DataArray,
    obs: xr.DataArray,
    threshold: xr.DataArray | float,
    *,
    method: Literal["fair", "ecdf"] = "fair",
) -> xr.DataArray:
    """Pointwise threshold-weighted CRPS for the upper tail (weight 1 at or above the threshold).

    `method` picks the estimator: "fair" (default, the headline score) or "ecdf".

    Uses `scores.probability.tw_crps_for_ensemble` with the chaining function `v(x) = max(x, t)`,
    which is the same as clipping forecast and observation from below at `t`. A `threshold` with a
    `month` dimension is broadcast per station and calendar month of the valid time; any other
    threshold (a number, or an array already on the forecast's dims) is used as given. The result is
    NaN where the threshold is NaN. An infinite threshold raises `ValueError`.
    """
    fcst = _prepare_ensemble(fcst)
    threshold = _threshold_float64(threshold)
    if isinstance(threshold, xr.DataArray) and "month" in threshold.dims:
        threshold = threshold.sel(station_id=fcst["station_id"])
        threshold = broadcast_threshold(threshold, fcst["init_time"], fcst["lead_time"])

    def _clip_below(values: xr.DataArray) -> xr.DataArray:
        return np.maximum(values, threshold)  # NaN stays NaN

    return tw_crps_for_ensemble(
        fcst, _as_float64(obs), ENS_DIM, _clip_below, method=method, preserve_dims="all"
    )


def _thresholds_one(da: xr.DataArray, quantile: float, train_end: np.datetime64) -> xr.DataArray:
    if set(da.dims) != {"valid_time", "station_id"}:
        raise ValueError(f"train truth must have dims (valid_time, station_id), got {tuple(da.dims)}")
    times = da["valid_time"].values
    if times.size == 0:
        raise ValueError("train truth is empty")
    if not np.issubdtype(times.dtype, np.datetime64):
        raise ValueError("train truth valid_time must be datetime64")
    # Checked on every timestamp, before any value is read. NaT compares false to everything and
    # poisons max(), so it is refused on its own.
    if np.isnat(times).any():
        raise ValueError("refusing to compute tail thresholds: truth has invalid (NaT) times")
    late = times > train_end
    if late.any():
        raise ValueError(
            f"refusing to compute tail thresholds: truth has {int(late.sum())} time(s) after the train "
            f"split end (latest {times[late].max()} > {train_end})"
        )
    data = _as_float64(da).load()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN station-months give NaN
        out = data.groupby("valid_time.month").quantile(quantile, dim="valid_time", skipna=True)
    out = out.drop_vars("quantile").reindex(month=np.arange(1, 13)).transpose("station_id", "month")
    out = out.astype(np.float32)
    out.attrs = {k: v for k, v in da.attrs.items() if k == "units"}
    out.attrs["tail_quantile"] = float(quantile)
    out.attrs["train_end_utc"] = str(train_end)
    return out


def compute_tail_thresholds(
    train_truth: xr.Dataset | xr.DataArray, quantile: float, *, cfg: HarnessConfig | None = None
) -> xr.Dataset | xr.DataArray:
    """Per station and calendar month quantile of training truth, dims `(station_id, month)`.

    Raises `ValueError` if `train_truth` has any `valid_time` after the train split end in `cfg`
    (default: the repository's `configs/splits.yaml`). NaN observations are ignored; a station with no
    observation in a calendar month gets a NaN threshold. Output is float32 with months 1..12.
    """
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be strictly between 0 and 1")
    train_end = (cfg or load_config()).splits.train.end
    if isinstance(train_truth, xr.Dataset):
        if not train_truth.data_vars:
            raise ValueError("train truth has no data variables")
        out = xr.Dataset({str(n): _thresholds_one(train_truth[n], quantile, train_end) for n in train_truth})
        return out
    return _thresholds_one(train_truth, quantile, train_end)


def _spread_terms(fcst: xr.DataArray, obs: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """Pointwise finite-ensemble-corrected variance and squared error of the ensemble mean.

    The variance is the unbiased one (ddof=1) times `(M + 1) / M`, so that for a calibrated ensemble
    its mean equals the mean squared error of the ensemble mean.
    """
    fcst = _prepare_ensemble(fcst)
    n_members = fcst.sizes[ENS_DIM]
    variance = fcst.var(ENS_DIM, ddof=1) * ((n_members + 1) / n_members)
    sq_error = (fcst.mean(ENS_DIM) - _as_float64(obs)) ** 2
    return variance, sq_error


def spread_skill_ratio(
    fcst: xr.DataArray, obs: xr.DataArray, *, preserve_dims: tuple[str, ...] = ()
) -> xr.DataArray:
    """`sqrt((M + 1) / M) * sqrt(mean(ens_var)) / sqrt(mean((ens_mean - obs)^2))`, obs finite only.

    The ensemble variance is the unbiased one (ddof=1) and M is the number of members. With the
    finite-ensemble factor a perfectly calibrated ensemble scores 1.0 whatever its size.
    """
    variance, sq_error = _spread_terms(fcst, obs)
    mask = np.isfinite(_as_float64(obs))
    reduce = [d for d in sq_error.dims if d not in preserve_dims]
    mean_var = variance.where(mask).mean(reduce, skipna=True)
    mean_sq_error = sq_error.where(mask).mean(reduce, skipna=True)
    return np.sqrt(mean_var) / np.sqrt(mean_sq_error)


def composite_loss(crps: float, twcrps_p95: float, cfg: HarnessConfig) -> float:
    """`w_crps * crps + w_tail * twcrps_p95` with the weights from the config (0.70 / 0.30)."""
    weights = cfg.gate.loss_weights
    return weights.crps_fair * crps + weights.twcrps_p95 * twcrps_p95


def dm_lag(lead_hours: float, init_spacing_hours: float) -> int:
    """The `h` of an h-step-ahead forecast: `max(1, ceil(lead / init spacing))`.

    It is the number of forecasts issued before the first one can be verified.
    """
    if not lead_hours > 0 or not init_spacing_hours > 0:
        raise ValueError("lead_hours and init_spacing_hours must be positive")
    return max(1, math.ceil(lead_hours / init_spacing_hours))


def diebold_mariano_pvalue(
    loss_diff: xr.DataArray,
    lead_hours: float,
    init_spacing_hours: float | None = None,
    *,
    method: str = DM_METHOD,
) -> float | None:
    """One-sided p-value that the candidate is better than the baseline.

    `loss_diff` is the per-`init_time` series of station-mean `baseline loss - candidate loss`
    (positive = candidate better). The spacing is the step of the `init_time` coordinate;
    `init_spacing_hours`, if given, must equal it. Leading and trailing NaNs are trimmed. Returns
    `None` when the test is not defined: a NaN inside the series, unevenly spaced init times, fewer
    than two init times, a series not longer than the lag, or all differences exactly zero.
    """
    if loss_diff.dims != ("init_time",):
        raise ValueError(f"loss_diff must have dims (init_time,), got {tuple(loss_diff.dims)}")
    series = loss_diff.sortby("init_time")
    times = series["init_time"].values.astype("datetime64[ns]")
    if np.isnat(times).any() or np.unique(times).size != times.size:
        raise ValueError("loss_diff init_time has NaT or duplicate values")
    values = np.asarray(series.values, dtype=np.float64)
    if np.isinf(values).any():
        raise ValueError("loss_diff has infinite values")
    present = np.flatnonzero(~np.isnan(values))
    if present.size == 0:
        return None
    values, times = values[present[0] : present[-1] + 1], times[present[0] : present[-1] + 1]
    if np.isnan(values).any() or times.size < 2:
        return None  # a gap inside the series, or a single init time
    steps = np.diff(times)
    if (steps != steps[0]).any():
        return None  # unevenly spaced init times
    spacing = float(steps[0] / np.timedelta64(1, "ns")) / _HOUR_NS
    if init_spacing_hours is not None and float(init_spacing_hours) != spacing:
        raise ValueError(f"init_spacing_hours={init_spacing_hours} but the init times are {spacing} h apart")
    lag = dm_lag(lead_hours, spacing)
    n = values.size
    if n <= lag or n < 3 or not np.any(values != 0.0):
        return None
    da = xr.DataArray(
        values[np.newaxis, :],
        dims=("series", "step"),
        coords={"series": [0], "step": np.arange(n), "h": ("series", [lag])},
    )
    with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
        warnings.simplefilter("ignore", category=RuntimeWarning)
        result = diebold_mariano(da, "series", "h", method=method, statistic_distribution="t")
    statistic = float(result["dm_test_stat"].values[0])
    if math.isnan(statistic):
        return None
    return float(scipy.stats.t.sf(statistic, n - 1))


def evaluation_init_times(
    period_start: np.datetime64 | str, period_end: np.datetime64 | str, init_hours: Sequence[int]
) -> np.ndarray:
    """Every init time from `period_start` to `period_end` (both included) at the given UTC hours.

    This is the date half of the evaluation grid for `score_variable`. `init_hours` is the set a run
    covers (all of, or a subset of, the configured `init_hours_utc`).
    """
    start = np.datetime64(period_start, "ns")
    end = np.datetime64(period_end, "ns")
    hours = sorted({int(h) for h in init_hours})
    if np.isnat(start) or np.isnat(end) or end < start:
        raise ValueError("period_start and period_end must be valid times with start <= end")
    if not hours or any(not 0 <= h <= 23 for h in hours):
        raise ValueError("init_hours must be a non-empty set of hours within 0..23")
    first_hour = start.astype("datetime64[h]")
    if first_hour < start:
        first_hour += np.timedelta64(1, "h")
    every_hour = np.arange(first_hour, end.astype("datetime64[h]") + np.timedelta64(1, "h"))
    hour_of_day = (every_hour - every_hour.astype("datetime64[D]")).astype(np.int64)
    return every_hour[np.isin(hour_of_day, hours)].astype("datetime64[ns]")


def _canonical(da: xr.DataArray, name: str, *, ensemble: bool) -> xr.DataArray:
    expected = (*POINT_DIMS, ENS_DIM) if ensemble else POINT_DIMS
    if set(da.dims) != set(expected) or len(da.dims) != len(expected):
        raise ValueError(f"{name} must have dims {expected}, got {tuple(da.dims)}")
    for dim in POINT_DIMS:
        if dim not in da.coords:
            raise ValueError(f"{name} has no coordinate for '{dim}'")
        if not da.indexes[dim].is_unique:
            raise ValueError(f"{name} has duplicate '{dim}' values")
        if da.indexes[dim].hasnans:
            raise ValueError(f"{name} has missing (NaT/NaN) '{dim}' values")
    sort_dims = [*POINT_DIMS, ENS_DIM] if ensemble and ENS_DIM in da.coords else list(POINT_DIMS)
    return da.sortby(sort_dims).transpose(*expected)


def _lead_hours(lead_time: np.ndarray) -> list[int]:
    nanos = lead_time.astype("timedelta64[ns]").astype(np.int64)
    if (nanos <= 0).any() or (nanos % _HOUR_NS != 0).any():
        raise ValueError("lead_time must be a positive whole number of hours")
    return [int(v) for v in nanos // _HOUR_NS]


def _require_finite_at(fcst: xr.DataArray, mask: np.ndarray, name: str) -> None:
    finite = np.isfinite(fcst).all(ENS_DIM).transpose(*POINT_DIMS).values
    bad = int(np.count_nonzero(mask & ~finite))
    if bad:
        raise ValueError(
            f"{name} has missing or non-finite values at {bad} point(s) where the observation is valid"
        )


def _pointwise(fcst: xr.DataArray, obs: xr.DataArray, threshold: xr.DataArray) -> dict[str, np.ndarray]:
    # Values at points that are not scored (no truth, no threshold) may be NaN or infinite; they are
    # computed and then left out by the mask, so their arithmetic warnings mean nothing.
    with warnings.catch_warnings(), np.errstate(invalid="ignore", over="ignore"):
        warnings.simplefilter("ignore", category=RuntimeWarning)
        variance, sq_error = _spread_terms(fcst, obs)
        arrays = {
            "crps": fair_crps(fcst, obs),
            "twcrps": tail_crps(fcst, obs, threshold),
            "crps_ecdf": ecdf_crps(fcst, obs),
            "twcrps_ecdf": tail_crps(fcst, obs, threshold, method="ecdf"),
            "variance": variance,
            "sq_error": sq_error,
        }
        return {k: np.asarray(a.transpose(*POINT_DIMS).values, dtype=np.float64) for k, a in arrays.items()}


def _finite_or_none(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    return float(np.mean(values[mask]))


_SUMMARY_KEYS = (
    "crps", "twcrps_p95", "loss", "crps_ecdf", "twcrps_p95_ecdf", "loss_ecdf", "spread_skill_ratio"
)  # fmt: skip


def _summary(
    arrays: dict[str, np.ndarray], index: slice, mask: np.ndarray, cfg: HarnessConfig
) -> dict[str, float | None]:
    """Means over the scored points of the leads in `index`. `mask` is already cut to `index`."""
    mean = {key: _masked_mean(values[:, index, :], mask) for key, values in arrays.items()}
    ratio = math.sqrt(mean["variance"] / mean["sq_error"]) if mean["sq_error"] > 0 else math.nan
    values = {
        "crps": mean["crps"],
        "twcrps_p95": mean["twcrps"],
        "loss": composite_loss(mean["crps"], mean["twcrps"], cfg),
        "crps_ecdf": mean["crps_ecdf"],
        "twcrps_p95_ecdf": mean["twcrps_ecdf"],
        "loss_ecdf": composite_loss(mean["crps_ecdf"], mean["twcrps_ecdf"], cfg),
        "spread_skill_ratio": ratio,
    }
    return {key: _finite_or_none(values[key]) for key in _SUMMARY_KEYS}


_BASELINE_KEYS = (
    "crpss_vs_baseline", "crpss_vs_baseline_ecdf", "baseline_twcrps_p95", "baseline_twcrps_p95_ecdf",
    "dm_pvalue",
)  # fmt: skip


def _skill(loss: float | None, base_loss: float | None) -> float | None:
    if loss is None or base_loss is None or not base_loss > 0:
        return None
    return _finite_or_none(1.0 - loss / base_loss)


def _per_init_mean(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean over the scored points of each init time; NaN where an init time has none."""
    flat_values = values.reshape(values.shape[0], -1)
    flat_mask = mask.reshape(mask.shape[0], -1)
    out = np.full(values.shape[0], np.nan)
    for i in range(values.shape[0]):
        if flat_mask[i].any():
            out[i] = np.mean(flat_values[i][flat_mask[i]])
    return out


def _point_thresholds(obs: xr.DataArray, tail_threshold: xr.DataArray) -> xr.DataArray:
    """Thresholds `(station_id, month)` on the points of `obs`, float64, dims `POINT_DIMS`.

    The one place `scoring_mask` and `score_variable` take their thresholds from. Refuses a value
    anywhere in the table that is infinite as float64, so NaN is the only thing the mask can read as
    "missing".
    """
    if set(tail_threshold.dims) != {"station_id", "month"}:
        raise ValueError(
            f"tail_threshold must have dims (station_id, month), got {tuple(tail_threshold.dims)}"
        )
    tail_threshold = _threshold_float64(tail_threshold)
    if not obs.indexes["station_id"].isin(tail_threshold.indexes["station_id"]).all():
        raise ValueError("tail_threshold does not cover every station of the forecast")
    selected = tail_threshold.sel(station_id=obs["station_id"].values)
    return broadcast_threshold(selected, obs["init_time"], obs["lead_time"])


def scoring_mask(obs: xr.DataArray, tail_threshold: xr.DataArray) -> xr.DataArray:
    """True where a point is scored: the observation is finite and so is its tail threshold.

    `obs` has dims `(init_time, lead_time, station_id)`, `tail_threshold` has `(station_id, month)`
    and must list every station of `obs`. The threshold used is the one for the station and the
    calendar month of the valid time. A NaN threshold masks the point; an infinite one raises
    `ValueError`. The result depends on truth and training data only, so it is
    the same for every forecast scored against them. Dims `(init_time, lead_time, station_id)`.
    """
    if set(obs.dims) != set(POINT_DIMS):
        raise ValueError(f"obs must have dims {POINT_DIMS}, got {tuple(obs.dims)}")
    obs = obs.transpose(*POINT_DIMS)
    threshold = _point_thresholds(obs, tail_threshold)
    mask = np.isfinite(np.asarray(obs.values, dtype=np.float64)) & np.isfinite(threshold.values)
    return xr.DataArray(mask, dims=POINT_DIMS, coords={dim: obs[dim].values for dim in POINT_DIMS})


def score_variable(
    fcst: xr.DataArray,
    obs: xr.DataArray,
    tail_threshold: xr.DataArray,
    baseline: xr.DataArray | None = None,
    *,
    cfg: HarnessConfig,
    expected_init_times: Sequence[np.datetime64] | np.ndarray,
    expected_station_ids: Sequence[str] | np.ndarray,
) -> dict[str, MetricValues]:
    """Score one variable. Keys are `"all"` and each lead in whole hours as a string.

    `expected_init_times` and `expected_station_ids` are the evaluation grid. Build them from the
    protocol (`evaluation_init_times`, the configured stations file), NOT from the forecast: they are
    what stops a forecast from leaving out a station or a date.

    `obs` has dims `(init_time, lead_time, station_id)`, exactly the grid's init times and stations,
    and at least the forecast's leads. `fcst` and `baseline` have dims
    `(init_time, lead_time, station_id, ens_member)` and must cover every grid init time and station
    (anything outside the grid is ignored). The leads scored are the forecast's leads; the baseline
    must have all of them. `tail_threshold` has dims `(station_id, month)` and must list every
    grid station; a point whose threshold is NaN is not scored for any model (see `scoring_mask`).

    Each entry has `crps`, `twcrps_p95`, `loss` (fair estimator), `crps_ecdf`, `twcrps_p95_ecdf`,
    `loss_ecdf` (ecdf estimator), `spread_skill_ratio`, `n_pairs`, `n_masked_no_threshold` (points
    with a valid observation but no tail threshold, not scored), `n_members`, and with a baseline
    also `crpss_vs_baseline`, `crpss_vs_baseline_ecdf`, `baseline_twcrps_p95`,
    `baseline_twcrps_p95_ecdf`, `dm_pvalue`. See the module docstring for the rules.
    """
    grid = {
        "init_time": pd.Index(np.asarray(expected_init_times).astype("datetime64[ns]")),
        "station_id": pd.Index(np.asarray(expected_station_ids, dtype=object)),
    }
    for dim, index in grid.items():
        if index.size == 0 or index.hasnans or not index.is_unique:
            raise ValueError(f"evaluation grid '{dim}' is empty or has missing or duplicate values")
        grid[dim] = index.sort_values()

    fcst = _canonical(fcst, "forecast", ensemble=True)
    obs = _canonical(obs, "obs", ensemble=False)
    for dim, index in grid.items():
        if not obs.indexes[dim].equals(index):
            raise ValueError(f"obs '{dim}' values are not exactly the evaluation grid")
    grid["lead_time"] = fcst.indexes["lead_time"]
    if not grid["lead_time"].isin(obs.indexes["lead_time"]).all():
        raise ValueError("obs does not cover every lead_time of the forecast")
    obs = obs.sel(lead_time=grid["lead_time"])

    def on_grid(da: xr.DataArray, name: str) -> xr.DataArray:
        for dim, index in grid.items():
            missing = int((~index.isin(da.indexes[dim])).sum())
            if missing:
                raise ValueError(
                    f"{name} does not cover the evaluation grid: {missing} '{dim}' value(s) missing"
                )
        return da.sel({dim: index.values for dim, index in grid.items()})

    fcst = on_grid(fcst, "forecast")
    if baseline is not None:
        baseline = on_grid(_canonical(baseline, "baseline", ensemble=True), "baseline")

    hours = _lead_hours(fcst["lead_time"].values)
    threshold = _point_thresholds(obs, tail_threshold)
    obs_finite = np.isfinite(np.asarray(obs.values, dtype=np.float64))
    mask = obs_finite & np.isfinite(threshold.values)
    no_threshold = obs_finite & ~mask

    _require_finite_at(fcst, mask, "forecast")
    cand = _pointwise(fcst, obs, threshold)
    base = None
    loss_diff = None
    if baseline is not None:
        _require_finite_at(baseline, mask, "baseline")
        base = _pointwise(baseline, obs, threshold)
        weights = cfg.gate.loss_weights
        loss_diff = weights.crps_fair * (base["crps"] - cand["crps"]) + weights.twcrps_p95 * (
            base["twcrps"] - cand["twcrps"]
        )
    scored_init_times = fcst["init_time"].values
    n_members = int(fcst.sizes[ENS_DIM])

    def entry(index: slice, lead_for_lag: int) -> MetricValues:
        sub_mask = mask[:, index, :]
        n_pairs = int(np.count_nonzero(sub_mask))
        out: MetricValues = dict.fromkeys(_SUMMARY_KEYS)
        if n_pairs:
            out.update(_summary(cand, index, sub_mask, cfg))
        out["n_pairs"] = n_pairs
        out["n_masked_no_threshold"] = int(np.count_nonzero(no_threshold[:, index, :]))
        out["n_members"] = n_members
        if base is None or loss_diff is None:
            return out
        out.update(dict.fromkeys(_BASELINE_KEYS))
        if not n_pairs:
            return out
        base_summary = _summary(base, index, sub_mask, cfg)
        out["crpss_vs_baseline"] = _skill(out["loss"], base_summary["loss"])
        out["crpss_vs_baseline_ecdf"] = _skill(out["loss_ecdf"], base_summary["loss_ecdf"])
        out["baseline_twcrps_p95"] = base_summary["twcrps_p95"]
        out["baseline_twcrps_p95_ecdf"] = base_summary["twcrps_p95_ecdf"]
        series = xr.DataArray(
            _per_init_mean(loss_diff[:, index, :], sub_mask),
            dims="init_time",
            coords={"init_time": scored_init_times},
        )
        out["dm_pvalue"] = diebold_mariano_pvalue(series, lead_for_lag)
        return out

    result: dict[str, MetricValues] = {"all": entry(slice(None), max(hours))}
    for position, hour in enumerate(hours):
        result[str(hour)] = entry(slice(position, position + 1), hour)
    return result


def evaluate_gates(metrics: dict[str, MetricValues], cfg: HarnessConfig) -> dict[str, bool]:
    """The promotion gates for one variable, judged on its `"all"` entry.

    Keys: `skill` (`crpss_vs_baseline` AND `crpss_vs_baseline_ecdf` above the configured minimum),
    `tail` (`twcrps_p95` not above the baseline's), `spread_skill` (ratio inside the configured band),
    `dm` (p-value below alpha) and `min_members` (at least `min_members_for_gate` members).
    A missing or `None` value fails its gate, so without a baseline `skill`, `tail` and `dm` are False.
    """
    if "all" not in metrics:
        raise ValueError("metrics has no 'all' entry")
    entry = metrics["all"]
    gate = cfg.gate
    crpss, crpss_ecdf = entry.get("crpss_vs_baseline"), entry.get("crpss_vs_baseline_ecdf")
    twcrps, base_twcrps = entry.get("twcrps_p95"), entry.get("baseline_twcrps_p95")
    ratio = entry.get("spread_skill_ratio")
    pvalue = entry.get("dm_pvalue")
    n_members = entry.get("n_members")
    return {
        "skill": (
            crpss is not None
            and crpss_ecdf is not None
            and crpss > gate.min_crpss_vs_baseline
            and crpss_ecdf > gate.min_crpss_vs_baseline
        ),
        "tail": twcrps is not None and base_twcrps is not None and twcrps <= base_twcrps,
        "spread_skill": ratio is not None and gate.spread_skill_min <= ratio <= gate.spread_skill_max,
        "dm": pvalue is not None and pvalue < gate.alpha_dm,
        "min_members": n_members is not None and n_members >= gate.min_members_for_gate,
    }
