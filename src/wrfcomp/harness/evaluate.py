"""`wrfcomp-eval`: score one forecast against station truth and write a deterministic scorecard.

Exit codes
    0  scored. The gate result is in the JSON (``passed_gate``), not in the exit code.
    2  bad input or schema (also: sealed-split data without ``--allow-sealed``).
    3  NaN rule: a forecast is not finite at a point where truth is valid.

Scorecard bytes
    JSON with sorted keys, two-space indent, ASCII only, one trailing newline. Every float is rounded to
    10 significant digits (``float(f"{x:.10g}")``) and written with Python's shortest ``repr``; a
    non-finite float is written as ``null``. No timestamps and no file paths are written, so the same
    inputs give the same bytes.

What is scored
    Every primary variable (``configs/splits.yaml``) that the forecast has, at every v1 lead
    (``evaluation.lead_hours``) that the forecast has. Leads outside that list are not scored and are
    named on stderr. With a baseline, both are scored on the intersection of their
    ``(init_time, lead_time, station_id)`` labels, and the NaN rule applies to the baseline on those points.
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import typer
import xarray as xr

from wrfcomp.harness.align import (
    ENS_DIM,
    POINT_DIMS,
    ForecastNaNError,
    SealedSplitError,
    align_truth,
    check_forecast_not_sealed,
    check_nan_rule,
    check_not_sealed,
)
from wrfcomp.harness.config import load_config
from wrfcomp.harness.hashing import content_sha256
from wrfcomp.harness.metrics import evaluate_gates, score_variable
from wrfcomp.harness.schemas import (
    SchemaError,
    Scorecard,
    current_harness_version,
    validate_forecast_dataset,
    validate_thresholds,
    validate_truth_dataset,
)

EXIT_BAD_INPUT = 2
EXIT_NAN_RULE = 3
FLOAT_SIG_DIGITS = 10

app = typer.Typer(help="WRFComp deterministic forecast verification CLI.", add_completion=False)


class BadInputError(ValueError):
    """Input that cannot be scored. Becomes exit code 2."""


# --------------------------------------------------------------------------- deterministic JSON


def _canonical(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _canonical(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_canonical(v) for v in value]
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if isinstance(value, int | np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        number = float(value)
        if not math.isfinite(number):
            return None
        return float(f"{number:.{FLOAT_SIG_DIGITS}g}")
    if value is None or isinstance(value, str):
        return value
    raise TypeError(f"cannot write {type(value).__name__} to a scorecard")


def canonical_json(payload: dict[str, Any]) -> str:
    """The scorecard text. See the module docstring for the formatting rule."""
    data = _canonical(payload)
    return json.dumps(data, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n"


# --------------------------------------------------------------------------- input helpers


def _open(path: Path, what: str) -> xr.Dataset:
    if not path.exists():
        raise BadInputError(f"{what}: no such file or directory: {path.name}")
    try:
        if path.is_dir():
            return xr.open_zarr(path, chunks=None)
        return xr.open_dataset(path)
    except Exception as exc:
        raise BadInputError(f"{what}: cannot open {path.name}: {exc}") from exc


def _require_unique(ds: xr.Dataset, what: str, dims: tuple[str, ...]) -> None:
    for dim in dims:
        if dim not in ds.indexes:
            raise BadInputError(f"{what}: missing coordinate {dim}")
        if not ds.indexes[dim].is_unique:
            raise BadInputError(f"{what}: duplicate values in {dim}")


def _lead_hours(ds: xr.Dataset, what: str) -> list[int]:
    lead_ns = ds["lead_time"].values.astype("timedelta64[ns]").astype("int64")
    hour_ns = 3_600_000_000_000
    if (lead_ns % hour_ns).any():
        raise BadInputError(f"{what}: lead_time must be whole hours")
    return [int(v) for v in lead_ns // hour_ns]


def _init_hours(ds: xr.Dataset, allowed: list[int], what: str) -> list[int]:
    init = ds["init_time"].values.astype("datetime64[ns]")
    hours = (init - init.astype("datetime64[D]")) / np.timedelta64(1, "h")
    bad = sorted({float(h) for h in hours if h not in allowed})
    if bad:
        raise BadInputError(f"{what}: init_time hour(s) {bad} are not in init_hours_utc {sorted(allowed)}")
    return sorted({int(h) for h in hours})


# --------------------------------------------------------------------------- the run


def run_evaluation(
    *,
    forecast_path: Path,
    truth_path: Path,
    thresholds_path: Path,
    baseline_path: Path | None,
    model_id: str,
    model_version: str,
    baseline_id: str | None,
    config_path: Path,
    allow_sealed: bool,
) -> dict[str, Any]:
    """Score and return the scorecard as plain data. Raises BadInputError or ForecastNaNError."""
    if not config_path.is_file():
        raise BadInputError(f"config: no such file: {config_path.name}")
    try:
        cfg = load_config(config_path)
    except ValueError as exc:
        raise BadInputError(f"config: {exc}") from exc
    sealed_start = cfg.splits.sealed_test.start

    forecast = _open(forecast_path, "forecast")
    truth = _open(truth_path, "truth")
    thresholds = _open(thresholds_path, "thresholds")
    baseline = _open(baseline_path, "baseline") if baseline_path is not None else None
    if baseline is None and baseline_id is not None:
        raise BadInputError("--baseline-id was given without --baseline")

    # Sealed split first: coordinates only, before any value is read or hashed.
    _require_unique(forecast, "forecast", POINT_DIMS)
    _require_unique(truth, "truth", ("valid_time", "station_id"))
    seal = {"sealed_start": sealed_start, "allow_sealed": allow_sealed}
    check_forecast_not_sealed(forecast, what="forecast", **seal)
    check_not_sealed(truth["valid_time"], what="truth valid_time", **seal)
    if baseline is not None:
        _require_unique(baseline, "baseline", POINT_DIMS)
        check_forecast_not_sealed(baseline, what="baseline", **seal)

    validate_forecast_dataset(forecast, cfg=cfg, allow_sealed=allow_sealed)
    validate_truth_dataset(truth, cfg=cfg, allow_sealed=allow_sealed)
    validate_thresholds(thresholds)
    if baseline is not None:
        validate_forecast_dataset(baseline, cfg=cfg, allow_sealed=allow_sealed)
        if baseline_id is None:
            baseline_id = baseline.attrs.get("model_id")
        if not baseline_id:
            raise BadInputError("baseline has no model_id attribute; pass --baseline-id")
    attr_id = forecast.attrs.get("model_id")
    if attr_id is not None and str(attr_id) != model_id:
        raise BadInputError(f"--model-id {model_id!r} does not match the forecast's model_id {attr_id!r}")

    hashes = {
        "forecast_sha256": content_sha256(forecast),
        "truth_sha256": content_sha256(truth),
        "thresholds_sha256": content_sha256(thresholds),
        "baseline_sha256": content_sha256(baseline) if baseline is not None else None,
    }

    # Variables: every primary variable the forecast has. Each must exist everywhere else.
    variables = [v for v in cfg.evaluation.variables.primary if v in forecast.data_vars]
    if not variables:
        raise BadInputError("forecast has none of the primary variables")
    others = {"truth": truth, "thresholds": thresholds}
    if baseline is not None:
        others["baseline"] = baseline
    for what, ds in others.items():
        missing = [v for v in variables if v not in ds.data_vars]
        if missing:
            raise BadInputError(f"{what} is missing variable(s) the forecast has: {missing}")

    # Leads: the v1 leads the forecast has. A whole lead may be absent; nothing finer may be.
    hours = _lead_hours(forecast, "forecast")
    keep = [h in cfg.evaluation.lead_hours for h in hours]
    skipped = [h for h, k in zip(hours, keep, strict=True) if not k]
    if skipped:
        typer.echo(f"note: lead hour(s) {skipped} are not v1 leads and are not scored", err=True)
    if not any(keep):
        raise BadInputError(f"forecast has none of the v1 lead hours {list(cfg.evaluation.lead_hours)}")
    forecast = forecast[variables].isel(lead_time=np.flatnonzero(keep))
    _init_hours(forecast, list(cfg.evaluation.init_hours_utc), "forecast")

    missing_stations = forecast.indexes["station_id"].difference(thresholds.indexes["station_id"])
    if len(missing_stations):
        raise BadInputError(f"thresholds lack {len(missing_stations)} forecast station(s)")

    obs = align_truth(forecast, truth, **seal).load()
    forecast = forecast.load()
    check_nan_rule(forecast, obs, role="candidate")

    if baseline is not None:
        common = {d: forecast.indexes[d][forecast.indexes[d].isin(baseline.indexes[d])] for d in POINT_DIMS}
        for dim, index in common.items():
            if len(index) == 0:
                raise BadInputError(f"candidate and baseline share no {dim}")
            dropped = forecast.sizes[dim] - len(index)
            if dropped:
                typer.echo(f"note: {dropped} candidate {dim} value(s) are not in the baseline", err=True)
        forecast = forecast.sel(common)
        obs = obs.sel(common)
        baseline = baseline[variables].sel(common).load()
        check_nan_rule(baseline, obs, role="baseline")

    metrics: dict[str, Any] = {}
    gates: dict[str, Any] = {}
    n_pairs: dict[str, int] = {}
    for name in variables:
        fc = forecast[name].transpose(*POINT_DIMS, ENS_DIM)
        bl = baseline[name].transpose(*POINT_DIMS, ENS_DIM) if baseline is not None else None
        tail = thresholds[name].sel(station_id=forecast["station_id"].values).load()
        metrics[name] = score_variable(fc, obs[name], tail, bl, cfg=cfg)
        gates[name] = evaluate_gates(metrics[name], cfg)
        n_pairs[name] = int(metrics[name]["all"]["n_pairs"])

    scorecard = Scorecard(
        harness_version=current_harness_version(),
        protocol_version=cfg.protocol_version,
        model_id=model_id,
        model_version=model_version,
        baseline_id=baseline_id,
        commit_sha256=None,
        n_stations=int(forecast.sizes["station_id"]),
        n_init_times=int(forecast.sizes["init_time"]),
        init_hours_scored=_init_hours(forecast, list(cfg.evaluation.init_hours_utc), "forecast"),
        leads_scored=_lead_hours(forecast, "forecast"),
        n_pairs=n_pairs,
        metrics=metrics,
        gates=gates,
        passed_gate=all(bool(ok) for per_var in gates.values() for ok in per_var.values()),
        **hashes,
    )
    return scorecard.model_dump(mode="json")


@app.command()
def evaluate(
    forecast: Path = typer.Option(..., "--forecast", help="Candidate forecast (Zarr)."),
    truth: Path = typer.Option(..., "--truth", help="Station truth (NetCDF or Zarr)."),
    thresholds: Path = typer.Option(..., "--thresholds", help="Tail thresholds (NetCDF)."),
    out: Path = typer.Option(..., "--out", help="Scorecard JSON to write."),
    model_id: str = typer.Option(..., "--model-id"),
    model_version: str = typer.Option(..., "--model-version"),
    baseline: Path | None = typer.Option(None, "--baseline", help="Baseline forecast (Zarr)."),
    baseline_id: str | None = typer.Option(None, "--baseline-id"),
    config: Path = typer.Option(Path("configs/splits.yaml"), "--config"),
    allow_sealed: bool = typer.Option(
        False, "--allow-sealed", help="Permit sealed-split data. Human-approved release verification only."
    ),
) -> None:
    """Score a forecast against truth and write the scorecard."""
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            payload = run_evaluation(
                forecast_path=forecast,
                truth_path=truth,
                thresholds_path=thresholds,
                baseline_path=baseline,
                model_id=model_id,
                model_version=model_version,
                baseline_id=baseline_id,
                config_path=config,
                allow_sealed=allow_sealed,
            )
        sealed = [str(w.message) for w in caught if "SEALED SPLIT IN USE" in str(w.message)]
        if sealed:
            bar = "!" * 78
            typer.echo(f"{bar}\nWARNING: --allow-sealed is set and sealed-split data was scored.", err=True)
            for line in sorted(set(sealed)):
                typer.echo(f"  {line}", err=True)
            typer.echo(f"This scorecard must not feed any automated loop.\n{bar}", err=True)
    except ForecastNaNError as exc:
        typer.echo(f"NaN rule: {exc}", err=True)
        raise typer.Exit(code=EXIT_NAN_RULE) from exc
    except (BadInputError, SchemaError, SealedSplitError) as exc:
        typer.echo(f"bad input: {exc}", err=True)
        raise typer.Exit(code=EXIT_BAD_INPUT) from exc

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(canonical_json(payload).encode("ascii"))
    typer.echo(f"scored {model_id} {model_version}: passed_gate={payload['passed_gate']}")
