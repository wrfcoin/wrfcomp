"""Scoring kernel: fair CRPS, tail-weighted CRPS, spread-skill, Diebold-Mariano and the promotion gates.

Rules this module follows (see docs/VERIFICATION_PROTOCOL.md and configs/splits.yaml):

* **Mask.** A point `(init_time, lead_time, station_id)` is scored when its observation is finite.
  That one mask is used for the candidate and for the baseline. A forecast (candidate or baseline)
  that is not finite at a scored point is an error, never a silently skipped point.
* **Averages.** `crps`, `twcrps_p95` and the spread-skill terms are plain means over the scored points
  of a lead, or of every lead for `"all"`. Nothing is weighted.
* **Undefined is `None`.** A metric that cannot be computed (no scored points, a zero baseline loss,
  a Diebold-Mariano series that is too short or all zero) is `None`, and a `None` never passes a gate.
* **Diebold-Mariano.** The series is, per init time, the mean over the scored stations of
  `baseline loss - candidate loss`, with the composite loss taken point by point. Init times with no
  scored station are NaN. NaNs are dropped and the remaining values are treated as consecutive (this
  is also what `scores` does); a gap therefore shortens the series and is not interpolated. The lag is
  `h = max(1, ceil(lead / spacing))`, where spacing is the smallest gap between init times, so an
  uneven series gets the larger (more cautious) lag. `"all"` averages the leads per init time and
  uses the lag of the longest lead. The test statistic is the Hering-Genton ("HG") form from
  `scores`, compared against Student's t with `n - 1` degrees of freedom; the p-value is one-sided
  (small when the candidate is better). HG fits an exponential model to the autocovariances of the
  series itself, over `max(floor((n - 1) / 2), h)` lags, so `h` is only a floor on how much
  autocorrelation is modelled: on a long series the result does not change with `h`. HG was chosen
  over the Harvey-Leybourne-Newbold ("HLN") form, which truncates at exactly `h - 1` lags and rejects
  a true null far too often when the differences stay correlated beyond that (as they do at short
  leads, where forecast errors follow multi-day weather regimes).
* **Determinism.** Inputs are sorted by coordinate, cast to float64, and the ensemble dimension is put
  in a single dask chunk. Pointwise scores are brought into memory and reduced with numpy in a fixed
  order, so results do not depend on dask chunking, input ordering or dict order. Nothing is random.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
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


def tail_crps(fcst: xr.DataArray, obs: xr.DataArray, threshold: xr.DataArray | float) -> xr.DataArray:
    """Pointwise threshold-weighted fair CRPS for the upper tail (weight 1 at or above the threshold).

    Uses `scores.probability.tw_crps_for_ensemble` with the chaining function `v(x) = max(x, t)`,
    which is the same as clipping forecast and observation from below at `t`. A `threshold` with a
    `month` dimension is broadcast per station and calendar month of the valid time; any other
    threshold (a number, or an array already on the forecast's dims) is used as given.
    """
    fcst = _prepare_ensemble(fcst)
    if isinstance(threshold, xr.DataArray):
        if "month" in threshold.dims:
            threshold = threshold.sel(station_id=fcst["station_id"])
            threshold = broadcast_threshold(threshold, fcst["init_time"], fcst["lead_time"])
        threshold = _as_float64(threshold)
    else:
        threshold = float(threshold)

    def _clip_below(values: xr.DataArray) -> xr.DataArray:
        return np.maximum(values, threshold)  # NaN stays NaN

    return tw_crps_for_ensemble(
        fcst, _as_float64(obs), ENS_DIM, _clip_below, method="fair", preserve_dims="all"
    )


def _thresholds_one(da: xr.DataArray, quantile: float, train_end: np.datetime64) -> xr.DataArray:
    if set(da.dims) != {"valid_time", "station_id"}:
        raise ValueError(f"train truth must have dims (valid_time, station_id), got {tuple(da.dims)}")
    times = da["valid_time"].values
    if times.size == 0:
        raise ValueError("train truth is empty")
    if not np.issubdtype(times.dtype, np.datetime64):
        raise ValueError("train truth valid_time must be datetime64")
    latest = times.max()
    if latest > train_end:
        raise ValueError(
            f"refusing to compute tail thresholds: truth has times after the train split end "
            f"({latest} > {train_end})"
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
    """Pointwise ensemble variance (ddof=1) and squared error of the ensemble mean."""
    fcst = _prepare_ensemble(fcst)
    variance = fcst.var(ENS_DIM, ddof=1)
    sq_error = (fcst.mean(ENS_DIM) - _as_float64(obs)) ** 2
    return variance, sq_error


def spread_skill_ratio(
    fcst: xr.DataArray, obs: xr.DataArray, *, preserve_dims: tuple[str, ...] = ()
) -> xr.DataArray:
    """`sqrt(mean(ens_var)) / sqrt(mean((ens_mean - obs)^2))` over points where obs is finite.

    The ensemble variance is the unbiased one (ddof=1). This is the protocol formula as written: a
    perfectly calibrated M-member ensemble scores `sqrt(M / (M + 1))`, not exactly 1.
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


def _init_spacing_hours(init_times: np.ndarray) -> float | None:
    """Smallest gap between distinct init times in hours, or None with fewer than two."""
    unique = np.unique(init_times.astype("datetime64[ns]"))
    if unique.size < 2:
        return None
    return float(np.diff(unique).min() / np.timedelta64(1, "ns")) / _HOUR_NS


def diebold_mariano_pvalue(
    loss_diff: xr.DataArray,
    lead_hours: float,
    init_spacing_hours: float | None = None,
    *,
    method: str = DM_METHOD,
) -> float | None:
    """One-sided p-value that the candidate is better than the baseline.

    `loss_diff` is the per-`init_time` series of station-mean `baseline loss - candidate loss`
    (positive = candidate better). `init_spacing_hours` defaults to the smallest gap in the
    `init_time` coordinate. NaNs are dropped (see the module docstring). Returns `None` when the test
    is not defined: fewer than two init times, a series not longer than the lag, or all differences
    exactly zero.
    """
    if loss_diff.dims != ("init_time",):
        raise ValueError(f"loss_diff must have dims (init_time,), got {tuple(loss_diff.dims)}")
    series = loss_diff.sortby("init_time")
    if init_spacing_hours is None:
        init_spacing_hours = _init_spacing_hours(series["init_time"].values)
        if init_spacing_hours is None:
            return None
    lag = dm_lag(lead_hours, init_spacing_hours)
    values = np.asarray(series.values, dtype=np.float64)
    if np.isinf(values).any():
        raise ValueError("loss_diff has infinite values")
    values = values[~np.isnan(values)]
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


def _canonical(da: xr.DataArray, name: str, *, ensemble: bool) -> xr.DataArray:
    expected = (*POINT_DIMS, ENS_DIM) if ensemble else POINT_DIMS
    if set(da.dims) != set(expected) or len(da.dims) != len(expected):
        raise ValueError(f"{name} must have dims {expected}, got {tuple(da.dims)}")
    for dim in POINT_DIMS:
        if dim not in da.coords:
            raise ValueError(f"{name} has no coordinate for '{dim}'")
        if not da.indexes[dim].is_unique:
            raise ValueError(f"{name} has duplicate '{dim}' values")
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


def _pointwise(
    fcst: xr.DataArray, obs: xr.DataArray, threshold: xr.DataArray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    variance, sq_error = _spread_terms(fcst, obs)
    arrays = (fair_crps(fcst, obs), tail_crps(fcst, obs, threshold), variance, sq_error)
    return tuple(np.asarray(a.transpose(*POINT_DIMS).values, dtype=np.float64) for a in arrays)  # type: ignore[return-value]


def _finite_or_none(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    return float(np.mean(values[mask]))


def _summary(
    crps: np.ndarray, twcrps: np.ndarray, variance: np.ndarray, sq_error: np.ndarray, mask: np.ndarray,
    cfg: HarnessConfig,
) -> dict[str, float | None]:
    crps_mean, tw_mean = _masked_mean(crps, mask), _masked_mean(twcrps, mask)
    mean_sq_error = _masked_mean(sq_error, mask)
    ratio = math.sqrt(_masked_mean(variance, mask) / mean_sq_error) if mean_sq_error > 0 else math.nan
    return {
        "crps": _finite_or_none(crps_mean),
        "twcrps_p95": _finite_or_none(tw_mean),
        "loss": _finite_or_none(composite_loss(crps_mean, tw_mean, cfg)),
        "spread_skill_ratio": _finite_or_none(ratio),
    }


def _per_init_mean(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean over the scored points of each init time; NaN where an init time has none."""
    flat_values = values.reshape(values.shape[0], -1)
    flat_mask = mask.reshape(mask.shape[0], -1)
    out = np.full(values.shape[0], np.nan)
    for i in range(values.shape[0]):
        if flat_mask[i].any():
            out[i] = np.mean(flat_values[i][flat_mask[i]])
    return out


def score_variable(
    fcst: xr.DataArray,
    obs: xr.DataArray,
    tail_threshold: xr.DataArray,
    baseline: xr.DataArray | None = None,
    *,
    cfg: HarnessConfig,
) -> dict[str, MetricValues]:
    """Score one variable. Keys are `"all"` and each lead in whole hours as a string.

    `fcst` and `baseline` have dims `(init_time, lead_time, station_id, ens_member)`; `obs` is already
    aligned to `(init_time, lead_time, station_id)` with exactly the forecast's coordinates.
    `tail_threshold` has dims `(station_id, month)` and must be finite wherever it is needed.
    The baseline must cover every init time, lead and station of the forecast (extra ones are ignored).

    Each entry has `crps`, `twcrps_p95`, `loss`, `spread_skill_ratio`, `n_pairs`, and with a baseline
    also `crpss_vs_baseline`, `baseline_twcrps_p95`, `dm_pvalue`. See the module docstring for the rules.
    """
    fcst = _canonical(fcst, "forecast", ensemble=True)
    obs = _canonical(obs, "obs", ensemble=False)
    for dim in POINT_DIMS:
        if not fcst.indexes[dim].equals(obs.indexes[dim]):
            raise ValueError(f"obs and forecast differ on '{dim}'; align obs to the forecast first")
    if baseline is not None:
        baseline = _canonical(baseline, "baseline", ensemble=True)
        for dim in POINT_DIMS:
            if not fcst.indexes[dim].isin(baseline.indexes[dim]).all():
                raise ValueError(f"baseline does not cover every '{dim}' value of the forecast")
        baseline = baseline.sel({dim: fcst[dim] for dim in POINT_DIMS})

    hours = _lead_hours(fcst["lead_time"].values)
    obs_values = np.asarray(obs.values, dtype=np.float64)
    mask = np.isfinite(obs_values)

    if set(tail_threshold.dims) != {"station_id", "month"}:
        raise ValueError(
            f"tail_threshold must have dims (station_id, month), got {tuple(tail_threshold.dims)}"
        )
    if not fcst.indexes["station_id"].isin(tail_threshold.indexes["station_id"]).all():
        raise ValueError("tail_threshold does not cover every station of the forecast")
    threshold = broadcast_threshold(
        _as_float64(tail_threshold).sel(station_id=fcst["station_id"]), fcst["init_time"], fcst["lead_time"]
    )
    missing = int(np.count_nonzero(mask & ~np.isfinite(threshold.values)))
    if missing:
        raise ValueError(
            f"tail_threshold is not finite at {missing} point(s) where the observation is valid"
        )

    _require_finite_at(fcst, mask, "forecast")
    cand = _pointwise(fcst, obs, threshold)
    base = None
    loss_diff = None
    if baseline is not None:
        _require_finite_at(baseline, mask, "baseline")
        base = _pointwise(baseline, obs, threshold)
        weights = cfg.gate.loss_weights
        loss_diff = weights.crps_fair * (base[0] - cand[0]) + weights.twcrps_p95 * (base[1] - cand[1])
    init_times = fcst["init_time"].values
    spacing = _init_spacing_hours(init_times)

    def entry(index: slice, lead_for_lag: int) -> MetricValues:
        sub_mask = mask[:, index, :]
        n_pairs = int(np.count_nonzero(sub_mask))
        out: MetricValues = {"crps": None, "twcrps_p95": None, "loss": None, "spread_skill_ratio": None}
        if n_pairs:
            out.update(_summary(*(a[:, index, :] for a in cand), sub_mask, cfg))
        out["n_pairs"] = n_pairs
        if base is None or loss_diff is None:
            return out
        out.update({"crpss_vs_baseline": None, "baseline_twcrps_p95": None, "dm_pvalue": None})
        if not n_pairs:
            return out
        base_summary = _summary(*(a[:, index, :] for a in base), sub_mask, cfg)
        out["baseline_twcrps_p95"] = base_summary["twcrps_p95"]
        loss, base_loss = out["loss"], base_summary["loss"]
        if loss is not None and base_loss is not None and base_loss > 0:
            out["crpss_vs_baseline"] = _finite_or_none(1.0 - loss / base_loss)
        if spacing is not None:
            series = xr.DataArray(
                _per_init_mean(loss_diff[:, index, :], sub_mask),
                dims="init_time",
                coords={"init_time": init_times},
            )
            out["dm_pvalue"] = diebold_mariano_pvalue(series, lead_for_lag, spacing)
        return out

    result: dict[str, MetricValues] = {"all": entry(slice(None), max(hours))}
    for position, hour in enumerate(hours):
        result[str(hour)] = entry(slice(position, position + 1), hour)
    return result


def evaluate_gates(metrics: dict[str, MetricValues], cfg: HarnessConfig) -> dict[str, bool]:
    """The four protocol gates for one variable, judged on its `"all"` entry.

    Keys: `skill` (`crpss_vs_baseline` above the configured minimum), `tail` (`twcrps_p95` not above
    the baseline's), `spread_skill` (ratio inside the configured band) and `dm` (p-value below alpha).
    A missing or `None` value fails its gate, so without a baseline `skill`, `tail` and `dm` are False.
    """
    if "all" not in metrics:
        raise ValueError("metrics has no 'all' entry")
    entry = metrics["all"]
    gate = cfg.gate
    crpss = entry.get("crpss_vs_baseline")
    twcrps, base_twcrps = entry.get("twcrps_p95"), entry.get("baseline_twcrps_p95")
    ratio = entry.get("spread_skill_ratio")
    pvalue = entry.get("dm_pvalue")
    return {
        "skill": crpss is not None and crpss > gate.min_crpss_vs_baseline,
        "tail": twcrps is not None and base_twcrps is not None and twcrps <= base_twcrps,
        "spread_skill": ratio is not None and gate.spread_skill_min <= ratio <= gate.spread_skill_max,
        "dm": pvalue is not None and pvalue < gate.alpha_dm,
    }
