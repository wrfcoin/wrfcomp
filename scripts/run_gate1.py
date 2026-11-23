"""Gate 1 runner: score the three reference forecasts for whole calendar months.

    uv run python scripts/run_gate1.py --data-dir <dir> --period 2024-01 --period 2024-07 \\
        --init-hours 0 12 --out experiments/

For every period (a calendar month, `YYYY-MM`) it

  a. builds the climatology and persistence forecasts for every init time of the month at the given
     hours, through `wrfcomp.baselines`, into `<data dir>/gate1/<period>_h<hours>/<model>.zarr`
     (skipped when the store is already there and covers the period);
  b. assembles the per-cycle GEFS point stores `<data dir>/baselines/gefs/<YYYYMMDDHH>.zarr` into
     one forecast store next to them (same skip rule);
  c. runs `wrfcomp-eval` three times: climatology (no baseline), persistence against climatology,
     GEFS against climatology, each on the leads the model has;
  d. writes `<out>/<model>_<period>.json` (`climatology`, `persistence`, `gefs`).

Rules
  * No network. A missing GEFS cycle, truth file or thresholds file is reported, never fetched.
  * Nothing dated in the sealed split is opened. A period must lie inside the validation split and
    its last valid time must be before the sealed start; truth that contains sealed times is refused
    on its coordinates, before any value is read. GEFS cycles are opened by name for the period only.
  * Deterministic. The same inputs give the same scorecard bytes. An existing scorecard that would
    change is NOT overwritten (the run reports it and exits 1) unless `--overwrite` is given.
  * Resumable. Forecast stores that exist are reused, so delete `<data dir>/gate1/` after the truth,
    the station list or the baseline code changes.
  * One model failing (for example GEFS cycles still downloading) does not stop the others.

Exit codes: 0 every requested scorecard is in place; 1 at least one model failed or a scorecard
would change; 2 the request or an input was refused.
"""

from __future__ import annotations

import argparse
import calendar
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import click
import numpy as np
import pandas as pd
import xarray as xr

from wrfcomp.baselines import climatology, persistence
from wrfcomp.baselines import io as baseline_io
from wrfcomp.harness.config import DEFAULT_CONFIG_PATH, HarnessConfig, load_config
from wrfcomp.harness.evaluate import app as eval_app
from wrfcomp.harness.metrics import evaluation_init_times

MODELS = ("climatology", "persistence", "gefs")
MODEL_IDS = {"climatology": "climatology", "persistence": "persistence", "gefs": "noaa-gefs"}
BASELINE = "climatology"
EXIT_FAILED = 1
EXIT_REFUSED = 2
REPO_ROOT = Path(__file__).resolve().parents[1]


class Refused(ValueError):
    """The request or an input that the whole run depends on is not acceptable. Exit code 2."""


class ModelFailed(RuntimeError):
    """One model of one period could not be built or scored. The run goes on; exit code 1."""


@dataclass(frozen=True)
class Period:
    label: str  # "2024-01"
    init_times: np.ndarray  # datetime64[ns], every init of the month at the requested hours

    @property
    def cycles(self) -> list[str]:
        return [pd.Timestamp(t).strftime("%Y%m%d%H") for t in self.init_times]


# --------------------------------------------------------------------------- request checks


def parse_period(label: str, init_hours: list[int], cfg: HarnessConfig) -> Period:
    try:
        year, month = (int(part) for part in label.split("-"))
        if len(label) != 7 or not 1 <= month <= 12:
            raise ValueError
    except ValueError:
        raise Refused(f"period {label!r}: expected YYYY-MM") from None
    start = np.datetime64(f"{year:04d}-{month:02d}-01T00", "ns")
    last_day = calendar.monthrange(year, month)[1]
    end = np.datetime64(f"{year:04d}-{month:02d}-{last_day:02d}T23", "ns")
    inits = evaluation_init_times(start, end, init_hours)
    validation = cfg.splits.validation
    if inits.min() < validation.start or inits.max() > validation.end:
        raise Refused(
            f"period {label}: init times {inits.min()} .. {inits.max()} are not all inside the validation "
            f"split ({validation.start} .. {validation.end})"
        )
    last_valid = inits.max() + np.timedelta64(max(cfg.evaluation.lead_hours), "h")
    if last_valid >= cfg.splits.sealed_test.start:
        raise Refused(f"period {label}: its last valid time {last_valid} reaches the sealed split")
    return Period(label=label, init_times=inits)


def check_data_dir(data_dir: Path, allow_tmp: bool) -> Path:
    data_dir = data_dir.resolve()
    if not data_dir.is_dir():
        raise Refused(f"data dir does not exist: {data_dir}")
    if data_dir.is_relative_to(REPO_ROOT):
        raise Refused("the data dir must be outside the repository")
    if data_dir.is_relative_to("/tmp") and not allow_tmp:
        raise Refused("the data dir must not be under /tmp")
    return data_dir


def required_stations(config_path: Path, cfg: HarnessConfig) -> list[str]:
    """Station ids of the config's stations file, resolved the way `wrfcomp-eval` resolves it."""
    path = Path(cfg.evaluation.stations_file)
    if not path.is_absolute():
        path = config_path.resolve().parent.parent / path
    if not path.is_file():
        raise Refused(f"stations file not found: {path}")
    stations = [
        s.strip() for s in pd.read_csv(path, dtype=str, usecols=["station_id"])["station_id"].dropna()
    ]
    if not stations or len(set(stations)) != len(stations):
        raise Refused(f"stations file {path} is empty or has duplicate station_id")
    return stations


# --------------------------------------------------------------------------- stores


def open_store(path: Path, what: str) -> xr.Dataset:
    if not path.exists():
        raise Refused(f"{what}: no such file or directory: {path}")
    try:
        if path.is_dir():
            try:
                return xr.open_zarr(path, chunks=None, consolidated=True)
            except (KeyError, ValueError, FileNotFoundError):
                return xr.open_zarr(path, chunks=None, consolidated=False)
        return xr.open_dataset(path)
    except Exception as exc:  # any reader failure is a refusal, not a traceback
        raise Refused(f"{what}: cannot open {path}: {exc}") from exc


def check_times(ds: xr.Dataset, what: str, *, latest_allowed: np.datetime64, inclusive: bool) -> np.ndarray:
    """Coordinates only. Returns the times; refuses NaT and anything past the limit."""
    if "valid_time" not in ds.coords or ds["valid_time"].dims != ("valid_time",):
        raise Refused(f"{what}: no valid_time coordinate")
    times = ds["valid_time"].values
    if times.size == 0 or not np.issubdtype(times.dtype, np.datetime64) or np.isnat(times).any():
        raise Refused(f"{what}: valid_time is empty, not a time coordinate, or has invalid (NaT) values")
    over = times > latest_allowed if inclusive else times >= latest_allowed
    if over.any():
        raise Refused(f"{what}: {int(over.sum())} time(s) past the allowed end ({times[over].max()})")
    return times


def load_archive(
    train_paths: list[Path], truth_path: Path, stations: list[str], cfg: HarnessConfig
) -> xr.Dataset:
    """Training truth plus the scored year's truth, in memory, for the baseline models.

    The baselines need the primary variables and their `X_offset_min` companions (data contract
    addendum 1). Stations are cut to the stations file, in its order.
    """
    names = [v for name in cfg.evaluation.variables.primary for v in (name, f"{name}_offset_min")]
    parts = []
    sources = [(p, "train truth", cfg.splits.train.end, True) for p in train_paths]
    sources.append((truth_path, "truth", cfg.splits.sealed_test.start, False))
    for path, what, limit, inclusive in sources:
        ds = open_store(path, what)
        check_times(ds, f"{what} {path.name}", latest_allowed=limit, inclusive=inclusive)
        missing = [n for n in names if n not in ds.data_vars]
        if missing:
            raise Refused(f"{what} {path.name}: missing variable(s) {missing}")
        absent = sorted(set(stations) - set(ds["station_id"].values.astype(str).tolist()))
        if absent:
            raise Refused(f"{what} {path.name}: {len(absent)} station(s) of the stations file are missing")
        ds = ds.assign_coords(station_id=ds["station_id"].values.astype(str))
        parts.append(ds[names].sel(station_id=stations).load())
        ds.close()
    archive = xr.concat(parts, dim="valid_time", data_vars="all", coords="minimal", compat="override")
    if not archive.indexes["valid_time"].is_unique:
        raise Refused("train truth and truth overlap in valid_time")
    return archive.sortby("valid_time")


def existing_store_ok(path: Path, period: Period, stations: list[str], model: str) -> bool:
    """True if a finished store is there and covers the period. A store that does not is refused."""
    if not path.exists():
        return False
    with open_store(path, f"{model} forecast") as ds:
        same_inits = ds.sizes.get("init_time") == period.init_times.size and bool(
            (np.sort(ds["init_time"].values) == period.init_times).all()
        )
        has_stations = set(stations) <= set(ds["station_id"].values.astype(str).tolist())
        same_model = ds.attrs.get("model_id") == MODEL_IDS[model]
    if not (same_inits and has_stations and same_model):
        raise ModelFailed(
            f"{path} exists but does not match this request (init times, stations or model id). "
            "Delete it to rebuild."
        )
    return True


def build_baseline(model: str, path: Path, period: Period, archive: xr.Dataset, cfg: HarnessConfig) -> None:
    module = climatology if model == "climatology" else persistence
    forecast = module.forecast(archive, list(period.init_times), tuple(cfg.evaluation.lead_hours))
    baseline_io.write(forecast, path)


def assemble_gefs(path: Path, period: Period, gefs_dir: Path, stations: list[str]) -> None:
    """Join the per-cycle stores of the period along init_time. Cycles are opened by name only."""
    stores = [gefs_dir / f"{cycle}.zarr" for cycle in period.cycles]
    missing = [s.name for s in stores if not s.is_dir()]
    if missing:
        shown = ", ".join(missing[:10])
        raise ModelFailed(
            f"{len(missing)} of {len(stores)} GEFS cycle store(s) missing in {gefs_dir}: {shown}"
        )
    parts = []
    for store, init in zip(stores, period.init_times, strict=True):
        with open_store(store, "GEFS cycle") as ds:
            if ds.sizes.get("init_time") != 1 or ds["init_time"].values[0] != init:
                raise ModelFailed(f"{store.name}: does not hold exactly the init time {init}")
            parts.append(ds.assign_coords(station_id=ds["station_id"].values.astype(str)).load())
    first = parts[0]
    for store, part in zip(stores, parts, strict=True):
        for dim in ("lead_time", "station_id", "ens_member"):
            if not part.indexes[dim].equals(first.indexes[dim]):
                raise ModelFailed(f"{store.name}: {dim} differs from {stores[0].name}")
        if part.attrs != first.attrs or set(part.data_vars) != set(first.data_vars):
            raise ModelFailed(f"{store.name}: attributes or variables differ from {stores[0].name}")
    absent = sorted(set(stations) - set(first["station_id"].values.tolist()))
    if absent:
        raise ModelFailed(f"GEFS stores lack {len(absent)} station(s) of the stations file")
    joined = xr.concat(parts, dim="init_time", data_vars="all", coords="minimal", compat="override")
    joined.attrs = dict(first.attrs)
    for name in joined.variables:
        joined[name].encoding.clear()
    baseline_io.write(joined, path)


# --------------------------------------------------------------------------- scoring


def run_eval(args: list[str]) -> int:
    """Run the `wrfcomp-eval` command in this process and return its exit code."""
    try:
        code = eval_app(args=args, standalone_mode=False)
    except click.exceptions.Exit as exc:
        return int(exc.exit_code)
    except click.ClickException as exc:
        print(f"wrfcomp-eval: {exc.format_message()}", file=sys.stderr)
        return int(exc.exit_code)
    return int(code or 0)


def score(
    *,
    model: str,
    period: Period,
    forecast: Path,
    baseline: Path | None,
    truth: Path,
    thresholds: Path,
    config: Path,
    out_dir: Path,
    overwrite: bool,
) -> str:
    """Score one model and put its scorecard in place. Returns "written", "unchanged" or raises."""
    target = out_dir / f"{model}_{period.label}.json"
    staging = out_dir / f".{target.name}.partial-{os.getpid()}"
    with open_store(forecast, f"{model} forecast") as ds:
        version = str(ds.attrs.get("model_version", ""))
    args = ["--forecast", str(forecast), "--truth", str(truth), "--thresholds", str(thresholds)]
    args += ["--out", str(staging), "--model-id", MODEL_IDS[model], "--model-version", version]
    args += ["--config", str(config)]
    if baseline is not None:
        args += ["--baseline", str(baseline), "--baseline-id", MODEL_IDS[BASELINE]]
    try:
        code = run_eval(args)
        if code != 0 or not staging.is_file():
            raise ModelFailed(f"wrfcomp-eval exit code {code} (2 = bad input, 3 = NaN or coverage rule)")
        new = staging.read_bytes()
        if target.exists():
            if target.read_bytes() == new:
                return "unchanged"
            if not overwrite:
                raise ModelFailed(
                    f"{target} exists and the new scorecard differs: inputs or code changed. "
                    "Not overwritten (pass --overwrite to replace it)."
                )
        os.replace(staging, target)
        return "written"
    finally:
        staging.unlink(missing_ok=True)


def check_truth_covers(truth: Path, periods: list[Period], cfg: HarnessConfig) -> None:
    """Coordinates only: the scoring truth is not sealed and spans every valid time to be scored."""
    with open_store(truth, "truth") as ds:
        times = check_times(
            ds, f"truth {truth.name}", latest_allowed=cfg.splits.sealed_test.start, inclusive=False
        )
    leads = cfg.evaluation.lead_hours
    for period in periods:
        first = period.init_times.min() + np.timedelta64(min(leads), "h")
        last = period.init_times.max() + np.timedelta64(max(leads), "h")
        if times.min() > first or times.max() < last:
            raise Refused(
                f"truth {truth.name} covers {times.min()} .. {times.max()} but period {period.label} "
                f"needs {first} .. {last}"
            )


# --------------------------------------------------------------------------- the run


def run(args: argparse.Namespace) -> int:
    try:
        cfg = load_config(args.config)
    except (ValueError, FileNotFoundError) as exc:
        raise Refused(f"config: {exc}") from exc
    hours = sorted(set(args.init_hours))
    unknown = [h for h in hours if h not in cfg.evaluation.init_hours_utc]
    if unknown:
        raise Refused(f"init hour(s) {unknown} are not in the config's init_hours_utc")
    if len(set(args.period)) != len(args.period):
        raise Refused("a period is listed twice")
    periods = [parse_period(label, hours, cfg) for label in args.period]
    models = [m for m in MODELS if m in args.models]
    data_dir = check_data_dir(args.data_dir, args.allow_tmp_data_dir)
    stations = required_stations(args.config, cfg)
    truth = args.truth or data_dir / "truth" / "isd_conus_2024.nc"
    train_truth = args.train_truth or [data_dir / "truth" / "isd_conus_train.nc"]
    thresholds = args.thresholds or data_dir / "thresholds" / "tail_p95.nc"
    for path, what in ((truth, "truth"), (thresholds, "thresholds")):
        if not path.exists():
            raise Refused(f"{what}: no such file or directory: {path}")
    check_truth_covers(truth, periods, cfg)
    args.out.mkdir(parents=True, exist_ok=True)
    hour_tag = "-".join(f"{h:02d}" for h in hours)

    archive: xr.Dataset | None = None
    results: list[tuple[str, str, str, str]] = []  # period, model, status, detail
    for period in periods:
        work = data_dir / "gate1" / f"{period.label}_h{hour_tag}"
        stores = {model: work / f"{model}.zarr" for model in MODELS}
        ready: dict[str, bool] = {}
        wanted = [m for m in MODELS if m in models or m == BASELINE]
        for model in wanted:
            try:
                if existing_store_ok(stores[model], period, stations, model):
                    print(f"[{period.label}] {model}: forecast store present, reused")
                elif model == "gefs":
                    assemble_gefs(stores[model], period, data_dir / "baselines" / "gefs", stations)
                    print(f"[{period.label}] gefs: assembled {len(period.cycles)} cycles")
                else:
                    if archive is None:
                        print("loading truth for the baseline models")
                        archive = load_archive(train_truth, truth, stations, cfg)
                    build_baseline(model, stores[model], period, archive, cfg)
                    print(f"[{period.label}] {model}: forecast built for {period.init_times.size} init times")
                ready[model] = True
            except ModelFailed as exc:
                ready[model] = False
                if model in models:
                    results.append((period.label, model, "FAILED", str(exc)))
                else:
                    print(f"[{period.label}] {model}: {exc}", file=sys.stderr)
        for model in models:
            if not ready.get(model):
                continue
            baseline = None if model == BASELINE else stores[BASELINE]
            if baseline is not None and not ready.get(BASELINE):
                results.append((period.label, model, "FAILED", "the climatology baseline is not available"))
                continue
            try:
                status = score(
                    model=model,
                    period=period,
                    forecast=stores[model],
                    baseline=baseline,
                    truth=truth,
                    thresholds=thresholds,
                    config=args.config,
                    out_dir=args.out,
                    overwrite=args.overwrite,
                )
                results.append((period.label, model, status, f"{model}_{period.label}.json"))
            except ModelFailed as exc:
                results.append((period.label, model, "FAILED", str(exc)))

    print("\nGate 1 runner summary")
    for label, model, status, detail in results:
        print(f"  {label}  {model:<12} {status:<9} {detail}")
    return EXIT_FAILED if any(status == "FAILED" for _, _, status, _ in results) else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score climatology, persistence and GEFS for whole months (Gate 1).",
        epilog="See docs/GATE1.md.",
    )
    parser.add_argument("--data-dir", type=Path, required=True, help="external data directory")
    parser.add_argument("--period", action="append", required=True, metavar="YYYY-MM", help="repeatable")
    parser.add_argument(
        "--init-hours", type=int, nargs="+", required=True, metavar="H", help="UTC, e.g. 0 12"
    )
    parser.add_argument("--out", type=Path, required=True, help="directory for the scorecards")
    parser.add_argument(
        "--truth", type=Path, help="scored year's truth (default <data dir>/truth/isd_conus_2024.nc)"
    )
    parser.add_argument(
        "--train-truth",
        type=Path,
        nargs="+",
        help="train-split truth (default <data dir>/truth/isd_conus_train.nc)",
    )
    parser.add_argument("--thresholds", type=Path, help="default <data dir>/thresholds/tail_p95.nc")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="splits.yaml")
    parser.add_argument(
        "--models", nargs="+", choices=MODELS, default=list(MODELS), help="default: all three"
    )
    parser.add_argument("--overwrite", action="store_true", help="replace a scorecard whose content changed")
    parser.add_argument(
        "--allow-tmp-data-dir", action="store_true", help="tests only: accept a data dir under /tmp"
    )
    args = parser.parse_args(argv)
    try:
        return run(args)
    except Refused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return EXIT_REFUSED


if __name__ == "__main__":
    sys.exit(main())
