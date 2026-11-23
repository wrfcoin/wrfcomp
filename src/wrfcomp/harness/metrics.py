"""Scoring kernel: fair CRPS, tail-weighted CRPS, spread-skill, Diebold-Mariano and the promotion gates.

SIGNATURES ONLY in this commit (lane H2 builds on them); the bodies land in the next commit.
"""

from __future__ import annotations

import xarray as xr

from wrfcomp.harness.config import HarnessConfig

ENS_DIM = "ens_member"
POINT_DIMS = ("init_time", "lead_time", "station_id")
DM_METHOD = "HG"

# One metrics entry, as stored under a lead key. `None` means "not defined" (never 0).
MetricValues = dict[str, float | int | None]


def fair_crps(fcst: xr.DataArray, obs: xr.DataArray) -> xr.DataArray:
    """Pointwise fair ensemble CRPS (no averaging). Result has the dims of `obs`."""
    raise NotImplementedError


def broadcast_threshold(
    threshold: xr.DataArray, init_time: xr.DataArray, lead_time: xr.DataArray
) -> xr.DataArray:
    """Thresholds `(station_id, month)` -> `(init_time, lead_time, station_id)` by month of the VALID time."""
    raise NotImplementedError


def tail_crps(fcst: xr.DataArray, obs: xr.DataArray, threshold: xr.DataArray | float) -> xr.DataArray:
    """Pointwise threshold-weighted fair CRPS for the upper tail (weight 1 at or above the threshold)."""
    raise NotImplementedError


def compute_tail_thresholds(
    train_truth: xr.Dataset | xr.DataArray, quantile: float, *, cfg: HarnessConfig | None = None
) -> xr.Dataset | xr.DataArray:
    """Per station and calendar month quantile of training truth, dims `(station_id, month)`.

    Raises `ValueError` if `train_truth` has any `valid_time` after the train split end in `cfg`.
    """
    raise NotImplementedError


def spread_skill_ratio(
    fcst: xr.DataArray, obs: xr.DataArray, *, preserve_dims: tuple[str, ...] = ()
) -> xr.DataArray:
    """`sqrt(mean(ens_var)) / sqrt(mean((ens_mean - obs)^2))` over points where obs is finite."""
    raise NotImplementedError


def composite_loss(crps: float, twcrps_p95: float, cfg: HarnessConfig) -> float:
    """`w_crps * crps + w_tail * twcrps_p95` with the weights from the config (0.70 / 0.30)."""
    raise NotImplementedError


def dm_lag(lead_hours: float, init_spacing_hours: float) -> int:
    """The `h` of an h-step-ahead forecast: `max(1, ceil(lead / init spacing))`."""
    raise NotImplementedError


def diebold_mariano_pvalue(
    loss_diff: xr.DataArray,
    lead_hours: float,
    init_spacing_hours: float | None = None,
    *,
    method: str = DM_METHOD,
) -> float | None:
    """One-sided p-value that the candidate is better than the baseline.

    `loss_diff` is the per-`init_time` series of station-mean `baseline loss - candidate loss`
    (positive = candidate better). Returns `None` when the test is not defined.
    """
    raise NotImplementedError


def score_variable(
    fcst: xr.DataArray,
    obs: xr.DataArray,
    tail_threshold: xr.DataArray,
    baseline: xr.DataArray | None = None,
    *,
    cfg: HarnessConfig,
) -> dict[str, MetricValues]:
    """Score one variable. Keys are `"all"` and each lead in whole hours as a string.

    Each entry has `crps`, `twcrps_p95`, `loss`, `spread_skill_ratio`, `n_pairs`, and with a baseline
    also `crpss_vs_baseline`, `baseline_twcrps_p95`, `dm_pvalue`.
    """
    raise NotImplementedError


def evaluate_gates(metrics: dict[str, MetricValues], cfg: HarnessConfig) -> dict[str, bool]:
    """The four protocol gates for one variable, judged on its `"all"` entry.

    Keys: `skill`, `tail`, `spread_skill`, `dm`. Without a baseline `skill`, `tail` and `dm` are False.
    """
    raise NotImplementedError
