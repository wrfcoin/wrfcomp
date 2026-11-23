# `wrfcoin/wrfcomp` — Starter Repository Blueprint

A ready-to-initialize architecture and starter code blueprint for [`wrfcoin/wrfcomp`](https://github.com/wrfcoin), the weather research, verification, and forecasting computation stack for the WRFCoin ecosystem.

This blueprint puts the **verification harness first** (your immutable "Lean kernel" for both AI-assisted research and future Proof-of-Forecast-Skill scoring) and wires it to an **observation-native assimilation layer** built around [NVIDIA Earth2Studio](https://github.com/NVIDIA/earth2studio) and the open [HealDA](https://huggingface.co/nvidia/healda) checkpoint.

---

## 1. Repository Structure

Keep `harness/` strictly isolated from `models/` and `experiments/` so neither coding agents nor external contributors can accidentally alter the grading logic while iterating on models.

```text
wrfcoin/wrfcomp/
├── README.md
├── pyproject.toml
├── .gitignore
├── src/
│   └── wrfcomp/
│       ├── __init__.py
│       ├── harness/                  # Immutable verification engine (version-pinned)
│       │   ├── __init__.py
│       │   ├── schemas.py            # Strict Zarr/Parquet forecast & obs contracts
│       │   ├── truth.py              # NOAA ISD/ASOS station + ARCO-ERA5 loaders
│       │   ├── metrics.py            # Fair CRPS, tail twCRPS, Spread-Skill, DM test
│       │   └── evaluate.py           # Deterministic CLI: wrfcomp-eval
│       ├── assimilation/             # Observation-native AI data assimilation
│       │   ├── __init__.py
│       │   ├── tokens.py             # 4D point-cloud observation token format
│       │   ├── qc.py                 # Background checks & rolling IoT trust scores
│       │   ├── healda_adapter.py     # Global HEALPix x0 adapter (NVIDIA HealDA)
│       │   └── sda_hook.py           # Local Score-Based Data Assimilation interface
│       ├── baselines/                # Reference models for skill normalization
│       │   ├── __init__.py
│       │   ├── climatology.py        # Historical climatology & persistence floor
│       │   └── earth2_baselines.py   # AIFS / FCN3 / GraphCast via Earth2Studio
│       └── models/                   # Candidate WRFComp forecasting architectures
│           └── __init__.py
├── configs/
│   ├── splits.yaml                   # Locked temporal splits (train <=2023, val 2024, test 2025-2026)
│   └── stations_conus.csv            # Canonical reference station IDs (ASOS/ISD)
├── experiments/                      # Reproducible run configs & scorecard JSONs
│   └── .gitkeep
├── tests/
│   ├── test_harness_metrics.py
│   └── test_qc_filters.py
└── docs/
    ├── VERIFICATION_PROTOCOL.md      # Formal scoring rules & anti-leakage spec
    └── ASSIMILATION_SPEC.md          # Point-cloud schema & sensor onboarding guide
```

---

## 2. Starter `README.md`

```markdown
# wrfcomp — Weather Research & Forecasting Computations

`wrfcomp` is the open forecasting research, verification, and observation-native
data assimilation engine for the [WRFCoin](https://github.com/wrfcoin) ecosystem.

## Core Design Principles

1. **Verification First:** Every model change is graded by an immutable,
   proper-scoring verification harness (`wrfcomp.harness`) using fair CRPS,
   threshold-weighted tail CRPS, spread-skill calibration, and Diebold-Mariano
   significance testing against real surface observations ([NOAA ISD](https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database) /
   [ASOS](https://www.ncei.noaa.gov/products/land-based-station/automated-surface-weather-observing-systems))
   and [ARCO-ERA5](https://github.com/google-research/arco-era5).
2. **Observation-Native Assimilation:** Ingests irregular point-cloud streams
   (official stations, citizen networks like [NOAA MADIS/CWOP](https://madis.ncep.noaa.gov/madis_cwop.shtml),
   geostationary [GOES-R](https://www.noaa.gov/nodd/datasets) imagery, and future
   WRFCoin IoT sensors) through background QC and global/local AI assimilation.
3. **Distributed-Ready Forecasts:** Targets embarrassingly parallel probabilistic
   ensembles and kilometer-scale regional downscaling so decentralized GPUs can
   contribute verifiable forecast skill without supercomputer interconnects.

## Quickstart

Requires Python 3.13 and [`uv`](https://docs.astral.sh/uv/) (recommended by
[Earth2Studio](https://nvidia.github.io/earth2studio/0.20.0/userguide/about/install/)).

```bash
git clone https://github.com/wrfcoin/wrfcomp.git
cd wrfcomp
uv sync --extra ai

# Score a forecast ensemble Zarr store against held-out station observations
uv run wrfcomp-eval \
  --forecast outputs/aifs_ens_2024.zarr \
  --truth data/truth/isd_2024.nc \
  --baseline outputs/climatology_2024.zarr \
  --out experiments/aifs_2024_scorecard.json
```

## Temporal Split Policy (No Leakage)

| Split | Date Window | Purpose |
| :--- | :--- | :--- |
| Train | <= 2023-12-31 | Model training & QC threshold calibration |
| Validation | 2024-01-01 to 2024-12-31 | Architecture search & agentic experiment loop |
| Sealed Test | 2025-01-01 to present | Final release verification & leaderboard checks |
```

---

## 3. Starter `pyproject.toml`

Uses `uv` with Python 3.13, the Bureau of Meteorology's [`scores[fast]`](https://scores.readthedocs.io/en/stable/release_notes.html) (Numba-accelerated CRPS), [`WeatherBench-X`](https://github.com/google-research/weatherbenchX)-compatible `xarray`/`zarr`/`dask` dependencies, and an optional `[ai]` group for [Earth2Studio](https://github.com/NVIDIA/earth2studio).

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "wrfcomp"
version = "0.1.0"
description = "Verification harness, observation-native AI assimilation, and next-gen weather forecasting for WRFCoin."
readme = "README.md"
requires-python = ">=3.11,<3.14"
license = { text = "Apache-2.0" }
dependencies = [
    "dask>=2024.10.0",
    "numpy>=2.0.0",
    "pandas>=2.2.0",
    "pydantic>=2.9.0",
    "scores[fast]>=2.7.0",
    "scipy>=1.14.0",
    "typer>=0.12.0",
    "xarray>=2024.10.0",
    "zarr>=2.18.0",
]

[project.optional-dependencies]
ai = [
    "earth2studio>=0.20.0",
    "torch>=2.5.0",
]
dev = [
    "pytest>=8.3.0",
    "ruff>=0.7.0",
]

[project.scripts]
wrfcomp-eval = "wrfcomp.harness.evaluate:app"

[tool.hatch.build.targets.wheel]
packages = ["src/wrfcomp"]
```

---

## 4. Verification Harness Interfaces

### `src/wrfcomp/harness/schemas.py`

Enforces a single strict contract for every forecast submission. Notice `commit_sha256` and `init_time_utc`—these let the exact same schema support both offline research today and commit-reveal Proof-of-Forecast-Skill verification on WRFCoin later.

```python
from __future__ import annotations

from datetime import datetime
from typing import Literal

import xarray as xr
from pydantic import BaseModel, Field

REQUIRED_FORECAST_DIMS = ("init_time", "lead_time", "station_id", "ens_member")
REQUIRED_TRUTH_DIMS = ("valid_time", "station_id")


class ForecastMetadata(BaseModel):
    model_id: str = Field(..., description="Unique model or miner identifier")
    model_version: str = Field(..., description="Git SHA or checkpoint tag")
    init_time_utc: datetime
    commit_sha256: str | None = Field(
        default=None,
        description="Optional pre-window cryptographic hash for WRFCoin commit-reveal",
    )
    domain: Literal["global", "conus", "regional"] = "conus"
    variables: list[str] = Field(default_factory=lambda: ["t2m", "u10", "v10", "tp6h"])


class Scorecard(BaseModel):
    model_id: str
    model_version: str
    truth_source: str
    n_stations: int
    n_init_times: int
    metrics: dict[str, dict[str, float]]  # variable -> {crps, twcrps, spread_skill, crpss, dm_pvalue}
    passed_significance_gate: bool


def validate_forecast_dataset(ds: xr.Dataset) -> None:
    """Validate that an ensemble forecast Dataset conforms to the WRFComp contract."""
    missing_dims = [d for d in REQUIRED_FORECAST_DIMS if d not in ds.dims]
    if missing_dims:
        raise ValueError(f"Forecast dataset missing required dims: {missing_dims}")
    if ds.sizes["ens_member"] < 2:
        raise ValueError("Probabilistic verification requires at least 2 ensemble members.")
    if "latitude" not in ds.coords or "longitude" not in ds.coords:
        raise ValueError("Forecast dataset must include latitude and longitude coordinates.")


def validate_truth_dataset(ds: xr.Dataset) -> None:
    """Validate that a ground-truth observation Dataset conforms to the WRFComp contract."""
    missing_dims = [d for d in REQUIRED_TRUTH_DIMS if d not in ds.dims]
    if missing_dims:
        raise ValueError(f"Truth dataset missing required dims: {missing_dims}")
```

### `src/wrfcomp/harness/metrics.py`

Wraps [`scores`](https://scores.readthedocs.io/en/stable/tutorials/Rank_Histogram.html) for fair ensemble CRPS, extreme-tail weighting, spread-skill calibration, and Diebold-Mariano significance testing against a baseline.

```python
from __future__ import annotations

import numpy as np
import xarray as xr
from scores.probability import crps_for_ensemble
from scores.statistical_tests import diebold_mariano


def compute_fair_crps(
    fcst: xr.DataArray,
    obs: xr.DataArray,
    member_dim: str = "ens_member",
) -> xr.DataArray:
    """Unbiased ('fair') ensemble CRPS so small and large ensembles compare cleanly."""
    return crps_for_ensemble(fcst, obs, ensemble_member_dim=member_dim, method="fair")


def compute_tail_crps(
    fcst: xr.DataArray,
    obs: xr.DataArray,
    threshold: float,
    member_dim: str = "ens_member",
) -> xr.DataArray:
    """Threshold-weighted CRPS focused strictly on upper-tail extremes (>= threshold)."""
    fcst_clipped = xr.where(fcst > threshold, fcst, threshold)
    obs_clipped = xr.where(obs > threshold, obs, threshold)
    return crps_for_ensemble(fcst_clipped, obs_clipped, ensemble_member_dim=member_dim, method="fair")


def compute_spread_skill_ratio(
    fcst: xr.DataArray,
    obs: xr.DataArray,
    member_dim: str = "ens_member",
) -> float:
    """Ratio of RMS ensemble spread to ensemble-mean RMSE (ideal calibrated value ~ 1.0)."""
    ens_mean = fcst.mean(dim=member_dim)
    ens_var = fcst.var(dim=member_dim, ddof=1)
    rmse = float(np.sqrt(((ens_mean - obs) ** 2).mean()))
    spread = float(np.sqrt(ens_var.mean()))
    return spread / rmse if rmse > 0 else float("nan")


def evaluate_variable(
    fcst: xr.DataArray,
    obs: xr.DataArray,
    baseline_fcst: xr.DataArray | None = None,
    tail_quantile: float = 0.95,
    member_dim: str = "ens_member",
) -> dict[str, float]:
    """Compute the canonical WRFComp metric bundle for a single meteorological variable."""
    # Pointwise CRPS (preserving time/station for Diebold-Mariano significance testing)
    crps_series = compute_fair_crps(fcst, obs, member_dim=member_dim)
    mean_crps = float(crps_series.mean())

    threshold = float(obs.quantile(tail_quantile))
    mean_twcrps = float(compute_tail_crps(fcst, obs, threshold=threshold, member_dim=member_dim).mean())
    spread_skill = compute_spread_skill_ratio(fcst, obs, member_dim=member_dim)

    result = {
        "crps": mean_crps,
        "twcrps_p95": mean_twcrps,
        "spread_skill_ratio": spread_skill,
    }

    if baseline_fcst is not None:
        base_crps_series = compute_fair_crps(baseline_fcst, obs, member_dim=member_dim)
        base_mean = float(base_crps_series.mean())
        result["crpss_vs_baseline"] = 1.0 - (mean_crps / base_mean) if base_mean > 0 else float("nan")

        # Diebold-Mariano test on time-averaged score differences (negative = candidate beats baseline)
        diff = (crps_series - base_crps_series).mean(dim="station_id")
        dm = diebold_mariano(diff, ts_dim="init_time", h=1)
        result["dm_pvalue"] = float(dm["pvalue"].item())

    return result
```

### `src/wrfcomp/harness/evaluate.py`

Deterministic CLI that grades any forecast Zarr store and outputs a versioned JSON scorecard.

```python
from __future__ import annotations

from pathlib import Path

import typer
import xarray as xr

from wrfcomp.harness.metrics import evaluate_variable
from wrfcomp.harness.schemas import Scorecard, validate_forecast_dataset, validate_truth_dataset

app = typer.Typer(help="WRFComp deterministic forecast verification CLI.")


@app.command()
def evaluate(
    forecast: Path = typer.Option(..., help="Path to candidate ensemble forecast Zarr store."),
    truth: Path = typer.Option(..., help="Path to ground-truth observations NetCDF/Zarr store."),
    baseline: Path | None = typer.Option(None, help="Optional reference baseline forecast Zarr."),
    out: Path = typer.Option(Path("scorecard.json"), help="Output JSON scorecard path."),
    model_id: str = typer.Option("candidate", help="Model or miner identifier."),
    model_version: str = typer.Option("dev", help="Git SHA or checkpoint version."),
    alpha: float = typer.Option(0.05, help="Diebold-Mariano p-value threshold for promotion."),
) -> None:
    fcst_ds = xr.open_zarr(forecast)
    truth_ds = xr.open_dataset(truth)
    validate_forecast_dataset(fcst_ds)
    validate_truth_dataset(truth_ds)

    base_ds = xr.open_zarr(baseline) if baseline else None

    # Align valid_time = init_time + lead_time against observations
    valid_times = fcst_ds["init_time"] + fcst_ds["lead_time"]
    obs_aligned = truth_ds.sel(valid_time=valid_times, station_id=fcst_ds["station_id"])

    metric_results: dict[str, dict[str, float]] = {}
    all_significant = True

    for var in fcst_ds.data_vars:
        if var not in obs_aligned.data_vars:
            continue
        base_var = base_ds[var] if (base_ds is not None and var in base_ds.data_vars) else None
        scores_dict = evaluate_variable(fcst_ds[var], obs_aligned[var], baseline_fcst=base_var)
        metric_results[var] = scores_dict

        if base_var is not None:
            if scores_dict.get("crpss_vs_baseline", 0.0) <= 0.0 or scores_dict.get("dm_pvalue", 1.0) >= alpha:
                all_significant = False

    scorecard = Scorecard(
        model_id=model_id,
        model_version=model_version,
        truth_source=str(truth.name),
        n_stations=int(fcst_ds.sizes["station_id"]),
        n_init_times=int(fcst_ds.sizes["init_time"]),
        metrics=metric_results,
        passed_significance_gate=all_significant if base_ds is not None else False,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(scorecard.model_dump_json(indent=2))
    typer.echo(f"Wrote scorecard to {out} (passed_gate={scorecard.passed_significance_gate})")
```

---

## 5. Observation-Native Assimilation Interfaces

### `src/wrfcomp/assimilation/qc.py`

Implements the 4D point-cloud observation batch, automated background checks (mirroring [ECMWF's real-time AIFS-DOP QC workflow](https://www.ecmwf.int/en/about/media-centre/aifs-blog/2026/observations-forecasts-bringing-ai-dop-real-time)), and rolling trust weighting for citizen/IoT stations ([NOAA CWOP](https://madis.ncep.noaa.gov/madis_cwop.shtml) and future WRFCoin hardware).

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ObservationBatch:
    """Unified 4D point-cloud batch for heterogeneous surface & satellite streams."""

    station_ids: np.ndarray       # shape (N,)
    lat: np.ndarray               # degrees [-90, 90]
    lon: np.ndarray               # degrees [-180, 180]
    elevation_m: np.ndarray       # meters above sea level (or pressure level)
    delta_t_sec: np.ndarray       # seconds offset from analysis window center
    sensor_type_id: np.ndarray    # 0=ASOS/ISD, 1=Mesonet, 2=CWOP/IoT, 3=Satellite
    variable: str
    values: np.ndarray            # raw scalar measurements
    trust_weights: np.ndarray     # [0.0, 1.0] reliability weight per observation


class BackgroundQC:
    """Rejects gross outliers and updates per-station trust weights from obs - background."""

    def __init__(
        self,
        max_innovation_sigma: float = 4.0,
        ema_alpha: float = 0.05,
        physical_bounds: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        self.max_innovation_sigma = max_innovation_sigma
        self.ema_alpha = ema_alpha
        self.physical_bounds = physical_bounds or {
            "t2m": (180.0, 340.0),   # Kelvin
            "u10": (-100.0, 100.0),  # m/s
            "v10": (-100.0, 100.0),  # m/s
            "msl": (85000.0, 108500.0),  # Pa
        }
        self._station_rmse_ema: dict[str, float] = {}

    def filter_and_weight(
        self,
        batch: ObservationBatch,
        background_at_obs: np.ndarray,
        background_std: float,
    ) -> ObservationBatch:
        lo, hi = self.physical_bounds.get(batch.variable, (-np.inf, np.inf))
        in_bounds = (batch.values >= lo) & (batch.values <= hi)

        innovation = np.abs(batch.values - background_at_obs)
        passes_bg = innovation <= (self.max_innovation_sigma * max(background_std, 1e-6))
        keep = in_bounds & passes_bg

        # Update rolling trust score for surviving observations (lower residual => higher trust)
        updated_weights = np.zeros_like(batch.trust_weights, dtype=np.float32)
        for idx in np.where(keep)[0]:
            sid = str(batch.station_ids[idx])
            err = float(innovation[idx])
            prev = self._station_rmse_ema.get(sid, background_std)
            ema = (1.0 - self.ema_alpha) * prev + self.ema_alpha * err
            self._station_rmse_ema[sid] = ema
            # Official ASOS stations start with a higher floor than uncalibrated IoT nodes
            tier_prior = 1.0 if batch.sensor_type_id[idx] == 0 else 0.75
            updated_weights[idx] = np.clip(tier_prior * np.exp(-ema / max(background_std, 1e-6)), 0.05, 1.0)

        return ObservationBatch(
            station_ids=batch.station_ids[keep],
            lat=batch.lat[keep],
            lon=batch.lon[keep],
            elevation_m=batch.elevation_m[keep],
            delta_t_sec=batch.delta_t_sec[keep],
            sensor_type_id=batch.sensor_type_id[keep],
            variable=batch.variable,
            values=batch.values[keep],
            trust_weights=updated_weights[keep],
        )


def dataframe_to_batch(df: pd.DataFrame, variable: str) -> ObservationBatch:
    """Convert a normalized MADIS/CWOP/WRFCoin station DataFrame into an ObservationBatch."""
    return ObservationBatch(
        station_ids=df["station_id"].to_numpy(),
        lat=df["lat"].to_numpy(dtype=np.float32),
        lon=df["lon"].to_numpy(dtype=np.float32),
        elevation_m=df["elevation_m"].to_numpy(dtype=np.float32),
        delta_t_sec=df["delta_t_sec"].to_numpy(dtype=np.float32),
        sensor_type_id=df["sensor_type_id"].to_numpy(dtype=np.int32),
        variable=variable,
        values=df[variable].to_numpy(dtype=np.float32),
        trust_weights=np.ones(len(df), dtype=np.float32),
    )
```

### `src/wrfcomp/assimilation/healda_adapter.py`

Provides a clean interface around [NVIDIA's open `nvidia/healda` checkpoint](https://huggingface.co/nvidia/healda) (via [Earth2Studio](https://github.com/NVIDIA/earth2studio)) for global 1° HEALPix state initialization, paired with a pluggable regional **Score-Based Data Assimilation (SDA)** refiner hook ([NVIDIA Earth-2 SDA](https://developer.nvidia.com/blog/turn-your-latest-observations-into-timely-weather-decisions-with-nvidia-earth-2/)) for local citizen/IoT stations.

```python
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Protocol

import xarray as xr

from wrfcomp.assimilation.qc import ObservationBatch


class GlobalAssimilator(Protocol):
    """Protocol for any observation-to-global-state model (HealDA, Aardvark, AIFS-DOP)."""

    def assimilate(
        self,
        analysis_time: datetime,
        observations: list[ObservationBatch],
    ) -> xr.Dataset:
        ...


class HealDAAdapter:
    """
    Adapter for NVIDIA Earth-2's pretrained HealDA global DA model (1-deg HEALPix grid).

    Checkpoint: https://huggingface.co/nvidia/healda
    Runtime:    https://github.com/NVIDIA/earth2studio
    """

    def __init__(self, device: str = "cuda", package_uri: str | None = None) -> None:
        self.device = device
        self.package_uri = package_uri
        self._model = None

    def load(self) -> None:
        """Lazy-load the pretrained HealDA checkpoint via Earth2Studio."""
        # Deferred import so the verification harness runs cleanly on CPU-only nodes
        try:
            import torch  # noqa: F401
            import earth2studio  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Install the AI extra (`uv sync --extra ai`) to run HealDAAdapter."
            ) from exc
        # Wire to Earth2Studio's HealDA model loader once checkpoint cache is configured:
        #   package = HealDA.load_default_package()  # or self.package_uri
        #   self._model = HealDA.load_model(package).to(self.device).eval()
        raise NotImplementedError(
            "Connect Earth2Studio's HealDA loader and UFS/custom observation mapper here."
        )

    def assimilate(
        self,
        analysis_time: datetime,
        observations: list[ObservationBatch],
    ) -> xr.Dataset:
        if self._model is None:
            self.load()
        # 1. Convert QC-filtered ObservationBatches into HealDA point-cloud tensors
        # 2. Run single-GPU forward pass -> 1-deg HEALPix (HPX64) 3D atmospheric state
        # 3. Return xarray.Dataset with standard CF variable names for downstream models
        raise NotImplementedError


class LocalSDARefiner(ABC):
    """
    Base interface for inference-time Score-Based Data Assimilation (SDA).
    Nudges a regional diffusion model (e.g., CorrDiff / StormCast) toward live
    local station observations during denoising—zero retraining when new stations join.
    """

    @abstractmethod
    def refine_ensemble(
        self,
        coarse_state: xr.Dataset,
        local_obs: list[ObservationBatch],
        n_members: int = 32,
        guidance_scale: float = 1.0,
    ) -> xr.Dataset:
        """Return a calibrated, observation-nudged regional ensemble Dataset."""
```

---

## 6. First 4 Weeks Bootstrap Checklist

| Week | Milestone | Acceptance Gate |
| :--- | :--- | :--- |
| **Week 1** | Initialize `wrfcoin/wrfcomp`, lock `configs/splits.yaml`, and implement `harness/truth.py` for [NOAA ISD/ASOS](https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database) (e.g., 500 CONUS stations for 2024). | `wrfcomp-eval` runs end-to-end on synthetic/climatology forecasts and outputs a valid JSON scorecard. |
| **Week 2** | Wrap two reference baselines in `baselines/` using [Earth2Studio](https://github.com/NVIDIA/earth2studio) (e.g., `AIFS` and `FCN3` initialized from GFS/IFS). | Recorded 2024 validation CRPS, `twcrps_p95`, and spread-skill baselines committed to `experiments/`. |
| **Week 3** | Wire [NOAA MADIS/CWOP](https://madis.ncep.noaa.gov/madis_cwop.shtml) ingestion into `assimilation/qc.py` and connect the open [HealDA](https://huggingface.co/nvidia/healda) checkpoint in `healda_adapter.py`. | Observation-only initialization produces a valid global state without NWP 4D-Var inputs. |
| **Week 4** | Prototype a regional `LocalSDARefiner` (or station-nudged downscaler) and test whether adding QC-weighted CWOP citizen stations beats the baseline on held-out ASOS stations. | Positive `crpss_vs_baseline` with `dm_pvalue < 0.05` on the 2024 validation split. |
