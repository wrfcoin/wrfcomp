"""Gate 1 runner: score the three reference forecasts for whole calendar months.

    uv run python scripts/run_gate1.py --data-dir <dir> --period 2024-01 --period 2024-07 \\
        --init-hours 0 12 --out experiments/

A period is a calendar month `YYYY-MM` or a calendar quarter `YYYY-Qn`. A quarter is scored as ONE
period: its forecast stores are the stores of its months joined along init time (run the months
first), and its scorecards are `<model>_YYYY-Qn.json`.

For every period it

  a. builds the climatology and persistence forecasts for every scorable init time of the month at the
     given hours, through `wrfcomp.baselines`, into `<data dir>/gate1/<period>_h<hours>/<model>.zarr`
     (skipped when the store is already there and covers the period);
  b. assembles the per-cycle GEFS point stores `<data dir>/baselines/gefs/<YYYYMMDDHH>.zarr` into
     one forecast store next to them (same skip rule);
  c. runs `wrfcomp-eval` three times: climatology (no baseline), persistence against climatology,
     GEFS against climatology, each on the leads the model has;
  d. writes `<out>/<model>_<period>.json` (`climatology`, `persistence`, `gefs`).

Rules
  * No network. A missing GEFS cycle, truth file or thresholds file is reported, never fetched.
  * Nothing dated in the sealed split is opened. A period must start inside the validation split.
    Its last init is the last one that is inside the validation split and whose longest scored lead
    stays before the sealed start (`scorable_init_times`); later init times of the month are left out
    for every model alike, and the run says so. Truth that contains sealed times is refused on its
    coordinates, before any value is read. GEFS cycles are opened by name for the scored init times
    only.
  * Deterministic. The same inputs give the same scorecard bytes. An existing scorecard that would
    change is NOT overwritten (the run reports it and exits 1) unless `--overwrite` is given.
  * Resumable. Forecast stores that exist are reused, so delete `<data dir>/gate1/` after the truth,
    the station list or the baseline code changes.
  * One model failing (for example GEFS cycles still downloading) does not stop the others.

Exit codes: 0 every requested scorecard is in place; 1 at least one model failed or a scorecard
would change; 2 the request or an input was refused.

Confirmation mode (`--confirmation-train-end <UTC time>`) scores a period that lies INSIDE the train
split, for the rule that a candidate which passes on the validation year must also pass on an earlier,
untouched year. Scoring such a period against the ordinary references would leak: climatology, the
persistence error sample and the tail thresholds are all built from the train split, which contains
the period. In this mode

  * the given time is the one cutoff for everything a reference learns from. It must be earlier than
    the train split end, and the period must start at least the train split's embargo
    (`embargo_days_after`) after it and before the validation split;
  * train truth after the cutoff is left out for climatology and for the persistence error sample
    (persistence still starts from the latest observation at init), and the cutoff is recorded in the
    `train_end_utc` attribute of both stores;
  * `--truth` and `--thresholds` must be given, and the thresholds file must carry exactly this
    cutoff in its `train_end_utc` attribute (`scripts/build_thresholds.py --train-end`);
  * the last init is the last one whose longest scored lead verifies at or before the train split
    end, because truth after it is embargoed.

Nothing else differs: the same builders, the same `wrfcomp-eval`, the same scorecard names.

A second command prints the monthly tables of a year from the scorecards alone (no data needed):

    uv run python scripts/run_gate1.py summary --scorecards experiments/ --year 2024
"""

from __future__ import annotations

import argparse
import calendar
import json
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
from wrfcomp.harness.align import SealedSplitError, check_forecast_not_sealed
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
    label: str  # "2024-01", or "2024-Q1" for a calendar quarter
    init_times: np.ndarray  # datetime64[ns], every scorable init of the period at the requested hours
    n_dropped: int = 0  # init times at the end of the period left out by `scorable_init_times`
    months: tuple[str, ...] = ()  # a quarter only: its months that have scorable init times, in order

    @property
    def cycles(self) -> list[str]:
        return [pd.Timestamp(t).strftime("%Y%m%d%H") for t in self.init_times]


# --------------------------------------------------------------------------- request checks


def parse_utc(text: str) -> np.datetime64:
    """A UTC time from the command line, with or without a zone suffix, as timezone-naive ns."""
    try:
        stamp = pd.Timestamp(text)
    except (ValueError, TypeError) as exc:
        raise Refused(f"not a time: {text!r}") from exc
    if stamp is pd.NaT:
        raise Refused(f"not a time: {text!r}")
    if stamp.tz is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    return stamp.to_datetime64().astype("datetime64[ns]")


def check_confirmation_cutoff(train_end: np.datetime64, cfg: HarnessConfig) -> None:
    if not train_end < cfg.splits.train.end:
        raise Refused(
            f"--confirmation-train-end {train_end} is not earlier than the train split end "
            f"({cfg.splits.train.end}); a confirmation cutoff can only be earlier"
        )


def scorable_init_times(
    inits: np.ndarray, cfg: HarnessConfig, train_end: np.datetime64 | None = None
) -> np.ndarray:
    """The init times of a period that may be scored.

    Confirmation mode (`train_end` given): the period lies inside the train split, and an init time
    is kept if its longest scored lead verifies at or before the train split end. Truth after that
    is embargoed before validation. The rest of this text is the ordinary rule.

    Rule: the last init of a period is the last one that is inside the validation split and whose
    longest scored lead stays before the sealed split. Both limits come from `configs/splits.yaml`.
    The longest scored lead is the longest of `evaluation.lead_hours`, for every model alike (also
    for one that has no such lead), so all models of a period share the same init times. An init
    time is dropped whole; it is never scored at its shorter leads only.

    With the present config the validation split ends on 2024-12-17T18Z (14 days of embargo before
    the sealed split), so that limit is the one that cuts December; the sealed-lead limit alone would
    allow 2024-12-29T12Z.
    """
    longest = np.timedelta64(max(cfg.evaluation.lead_hours), "h")
    if train_end is not None:
        return inits[inits + longest <= cfg.splits.train.end]
    keep = (inits <= cfg.splits.validation.end) & (inits + longest < cfg.splits.sealed_test.start)
    return inits[keep]


def _month_init_times(year: int, month: int, init_hours: list[int]) -> np.ndarray:
    start = np.datetime64(f"{year:04d}-{month:02d}-01T00", "ns")
    last_day = calendar.monthrange(year, month)[1]
    end = np.datetime64(f"{year:04d}-{month:02d}-{last_day:02d}T23", "ns")
    return evaluation_init_times(start, end, init_hours)


def parse_period(
    label: str, init_hours: list[int], cfg: HarnessConfig, train_end: np.datetime64 | None = None
) -> Period:
    """A calendar month `YYYY-MM` or a calendar quarter `YYYY-Qn`, cut by `scorable_init_times`.

    With `train_end` (confirmation mode) the period must lie between that cutoff plus the train
    split's embargo and the validation split, instead of inside the validation split.

    A quarter is ONE period: its init times are those of its three months together, so its scorecard
    is one evaluation over the whole quarter, not an average of three monthly scorecards.
    """
    try:
        year_text, part = label.split("-")
        year = int(year_text)
        if len(year_text) != 4 or not year_text.isdigit():
            raise ValueError
        if len(part) == 2 and part[0] == "Q" and part[1] in "1234":
            first = 3 * int(part[1]) - 2
            month_numbers = [first, first + 1, first + 2]
        elif len(part) == 2 and part.isdigit() and 1 <= int(part) <= 12:
            month_numbers = [int(part)]
        else:
            raise ValueError
    except ValueError:
        raise Refused(f"period {label!r}: expected YYYY-MM or YYYY-Qn") from None
    is_quarter = len(month_numbers) == 3
    per_month = {f"{year:04d}-{m:02d}": _month_init_times(year, m, init_hours) for m in month_numbers}
    all_inits = np.concatenate(list(per_month.values()))
    validation = cfg.splits.validation
    if train_end is not None:
        check_confirmation_cutoff(train_end, cfg)
        embargo_days = cfg.splits.train.embargo_days_after or 0
        earliest = train_end + np.timedelta64(embargo_days, "D")
        if all_inits.min() < earliest:
            raise Refused(
                f"period {label}: starts at {all_inits.min()}, less than "
                f"{embargo_days} days after the confirmation cutoff ({train_end})"
            )
        if all_inits.max() >= validation.start:
            raise Refused(
                f"period {label}: reaches the validation split ({validation.start}); a confirmation "
                "period lies before it. Score it without --confirmation-train-end."
            )
    elif all_inits.min() < validation.start:
        raise Refused(
            f"period {label}: starts at {all_inits.min()}, before the validation split ({validation.start})"
        )
    inits = scorable_init_times(all_inits, cfg, train_end)
    if inits.size == 0 and train_end is not None:
        raise Refused(
            f"period {label}: no init time has its longest lead ({max(cfg.evaluation.lead_hours)} h) "
            f"at or before the train split end ({cfg.splits.train.end})"
        )
    if inits.size == 0:
        raise Refused(
            f"period {label}: no init time is inside the validation split (ends {validation.end}) with its "
            f"longest lead ({max(cfg.evaluation.lead_hours)} h) before the sealed split "
            f"({cfg.splits.sealed_test.start})"
        )
    months = (
        tuple(m for m, times in per_month.items() if scorable_init_times(times, cfg, train_end).size)
        if is_quarter
        else ()
    )
    return Period(label=label, init_times=inits, n_dropped=int(all_inits.size - inits.size), months=months)


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
    train_paths: list[Path],
    truth_path: Path,
    stations: list[str],
    cfg: HarnessConfig,
    train_end: np.datetime64 | None = None,
) -> xr.Dataset:
    """Training truth plus the scored year's truth, in memory, for the baseline models.

    The baselines need the primary variables and their `X_offset_min` companions (data contract
    addendum 1). Stations are cut to the stations file, in its order.

    Confirmation mode (`train_end` given): train truth is cut to times at or before the cutoff and
    the scored truth to times after it, on coordinates, before any value is loaded. The scored truth
    may therefore be the train truth file itself or a slice of it.
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
        if train_end is not None:
            times = ds["valid_time"].values
            keep = times <= train_end if what == "train truth" else times > train_end
            if not keep.any():
                raise Refused(f"{what} {path.name}: no time on its side of the cutoff {train_end}")
            ds = ds.isel(valid_time=np.flatnonzero(keep))
        parts.append(ds[names].sel(station_id=stations).load())
        ds.close()
    archive = xr.concat(parts, dim="valid_time", data_vars="all", coords="minimal", compat="override")
    if not archive.indexes["valid_time"].is_unique:
        raise Refused("train truth and truth overlap in valid_time")
    return archive.sortby("valid_time")


def existing_store_ok(
    path: Path, period: Period, stations: list[str], model: str, train_end: np.datetime64 | None = None
) -> bool:
    """True if a finished store is there and covers the period. A store that does not is refused.

    A climatology or persistence store must also have been built with this run's cutoff: the
    `train_end_utc` attribute is the confirmation cutoff, or absent in an ordinary run.
    """
    if not path.exists():
        return False
    cutoff = None if train_end is None or model == "gefs" else str(train_end)
    with open_store(path, f"{model} forecast") as ds:
        same_inits = ds.sizes.get("init_time") == period.init_times.size and bool(
            (np.sort(ds["init_time"].values) == period.init_times).all()
        )
        has_stations = set(stations) <= set(ds["station_id"].values.astype(str).tolist())
        same_model = ds.attrs.get("model_id") == MODEL_IDS[model]
        same_cutoff = ds.attrs.get("train_end_utc") == cutoff
    if not (same_inits and has_stations and same_model and same_cutoff):
        raise ModelFailed(
            f"{path} exists but does not match this request (init times, stations, model id or "
            "training cutoff). Delete it to rebuild."
        )
    return True


def build_baseline(
    model: str,
    path: Path,
    period: Period,
    archive: xr.Dataset,
    cfg: HarnessConfig,
    train_end: np.datetime64 | None = None,
) -> None:
    module = climatology if model == "climatology" else persistence
    forecast = module.forecast(
        archive, list(period.init_times), tuple(cfg.evaluation.lead_hours), train_end=train_end
    )
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


def assemble_quarter(
    path: Path,
    period: Period,
    month_stores: list[Path],
    months: list[Period],
    model: str,
    sealed_start: np.datetime64,
) -> None:
    """A quarter's forecast store: the stores of its months joined along init_time, nothing else.

    Every month store must already exist and hold exactly that month's scorable init times, so the
    quarter is made of the same forecasts as the monthly references. Init times end up in time order.
    A month store whose init or valid times reach `sealed_start` is refused from its coordinates,
    before any of its values is read.
    """
    parts = []
    for store, month in zip(month_stores, months, strict=True):
        if not store.is_dir():
            raise ModelFailed(
                f"{store} is missing: run the months of {period.label} ({', '.join(period.months)}) first"
            )
        with open_store(store, f"{model} forecast {month.label}") as ds:
            try:
                check_forecast_not_sealed(ds, what=str(store), sealed_start=sealed_start)
            except (SealedSplitError, KeyError, TypeError) as exc:
                raise ModelFailed(f"{store} is refused before it is read: {exc}") from exc
            if ds.sizes.get("init_time") != month.init_times.size or not bool(
                (np.sort(ds["init_time"].values) == month.init_times).all()
            ):
                raise ModelFailed(f"{store} does not hold exactly the init times of {month.label}")
            if ds.attrs.get("model_id") != MODEL_IDS[model]:
                raise ModelFailed(f"{store} is not a {MODEL_IDS[model]} forecast")
            parts.append(ds.assign_coords(station_id=ds["station_id"].values.astype(str)).load())
    first = parts[0]
    for store, part in zip(month_stores, parts, strict=True):
        for dim in ("lead_time", "station_id", "ens_member"):
            if not part.indexes[dim].equals(first.indexes[dim]):
                raise ModelFailed(f"{store}: {dim} differs from {month_stores[0]}")
        if part.attrs != first.attrs or set(part.data_vars) != set(first.data_vars):
            raise ModelFailed(f"{store}: attributes or variables differ from {month_stores[0]}")
    joined = xr.concat(parts, dim="init_time", data_vars="all", coords="minimal", compat="override")
    joined = joined.sortby("init_time")
    if joined.sizes["init_time"] != period.init_times.size or not bool(
        (joined["init_time"].values == period.init_times).all()
    ):
        raise ModelFailed(f"the month stores of {period.label} do not add up to its init times")
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


def check_thresholds_cutoff(thresholds: Path, train_end: np.datetime64) -> None:
    """Attributes only: the thresholds file was built with exactly the confirmation cutoff."""
    with open_store(thresholds, "thresholds") as ds:
        recorded = ds.attrs.get("train_end_utc")
    try:
        same = recorded is not None and np.datetime64(str(recorded), "ns") == train_end
    except ValueError:
        same = False
    if not same:
        raise Refused(
            f"thresholds {thresholds.name}: built from truth up to {recorded!r}, not up to the "
            f"confirmation cutoff {train_end}. Build it with scripts/build_thresholds.py --train-end."
        )


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
    cutoff = None if args.confirmation_train_end is None else parse_utc(args.confirmation_train_end)
    if cutoff is not None:
        check_confirmation_cutoff(cutoff, cfg)
        if args.truth is None or args.thresholds is None:
            raise Refused(
                "confirmation mode needs --truth and --thresholds: the defaults are the validation "
                "year's truth and the thresholds of the whole train split"
            )
    periods = [parse_period(label, hours, cfg, cutoff) for label in args.period]
    models = [m for m in MODELS if m in args.models]
    data_dir = check_data_dir(args.data_dir, args.allow_tmp_data_dir)
    stations = required_stations(args.config, cfg)
    truth = args.truth or data_dir / "truth" / "isd_conus_2024.nc"
    train_truth = args.train_truth or [data_dir / "truth" / "isd_conus_train.nc"]
    thresholds = args.thresholds or data_dir / "thresholds" / "tail_p95.nc"
    for path, what in ((truth, "truth"), (thresholds, "thresholds")):
        if not path.exists():
            raise Refused(f"{what}: no such file or directory: {path}")
    if cutoff is not None:
        check_thresholds_cutoff(thresholds, cutoff)
        print(f"confirmation mode: references learn from truth at or before {cutoff} only")
    check_truth_covers(truth, periods, cfg)
    args.out.mkdir(parents=True, exist_ok=True)
    hour_tag = "-".join(f"{h:02d}" for h in hours)

    archive: xr.Dataset | None = None
    results: list[tuple[str, str, str, str]] = []  # period, model, status, detail
    for period in periods:
        if period.n_dropped:
            print(
                f"[{period.label}] stops at {pd.Timestamp(period.init_times.max()):%Y-%m-%dT%HZ}: the last "
                f"{period.n_dropped} init time(s) of the period are left out for every model "
                + (
                    "(their longest lead verifies after the train split end)"
                    if cutoff is not None
                    else "(after the end of the validation split, or with a longest lead that reaches "
                    "the sealed split)"
                )
            )
        work = data_dir / "gate1" / f"{period.label}_h{hour_tag}"
        stores = {model: work / f"{model}.zarr" for model in MODELS}
        ready: dict[str, bool] = {}
        wanted = [m for m in MODELS if m in models or m == BASELINE]
        for model in wanted:
            try:
                if existing_store_ok(stores[model], period, stations, model, cutoff):
                    print(f"[{period.label}] {model}: forecast store present, reused")
                elif period.months:
                    months = [parse_period(m, hours, cfg, cutoff) for m in period.months]
                    month_stores = [
                        data_dir / "gate1" / f"{m}_h{hour_tag}" / f"{model}.zarr" for m in period.months
                    ]
                    assemble_quarter(
                        stores[model], period, month_stores, months, model, cfg.splits.sealed_test.start
                    )
                    print(f"[{period.label}] {model}: joined the stores of {', '.join(period.months)}")
                elif model == "gefs":
                    assemble_gefs(stores[model], period, data_dir / "baselines" / "gefs", stations)
                    print(f"[{period.label}] gefs: assembled {len(period.cycles)} cycles")
                else:
                    if archive is None:
                        print("loading truth for the baseline models")
                        archive = load_archive(train_truth, truth, stations, cfg, cutoff)
                    build_baseline(model, stores[model], period, archive, cfg, cutoff)
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


# --------------------------------------------------------------------------- summary tables

SUMMARY_GATES = ("skill", "tail", "spread_skill", "dm", "min_members")


def _fmt(value: float | None, pattern: str) -> str:
    return "n/a" if value is None else format(value, pattern)


def _verdict(card: dict, variable: str) -> str:
    gates = card["gates"][variable]
    failed = [name for name in SUMMARY_GATES if not gates[name]]
    return "pass" if not failed else "fails " + ", ".join(failed)


def summary_tables(scorecards: Path, year: int) -> str:
    """Markdown: one table per primary variable, one row per month, from the scorecard files only.

    Every number is the `"all"` entry (all leads of that model) of `metrics[variable]`. Skill is
    `crpss_vs_baseline` (fair members) and `crpss_vs_baseline_ecdf`, both against climatology, in
    percent. A month without a scorecard gives a row of `n/a`.
    """
    cards: dict[tuple[str, str], dict] = {}
    months = [f"{year:04d}-{m:02d}" for m in range(1, 13)]
    for month in months:
        for model in MODELS:
            path = scorecards / f"{model}_{month}.json"
            if path.is_file():
                cards[(model, month)] = json.loads(path.read_text())
    if not cards:
        raise Refused(f"no scorecards for {year} in {scorecards}")
    variables = [
        v
        for v in load_config().evaluation.variables.primary
        if any(v in c["metrics"] for c in cards.values())
    ]
    lines: list[str] = []
    for variable in variables:
        lines += [
            f"### `{variable}`",
            "",
            "| Month | Init times | GEFS skill fair % | GEFS skill ecdf % | GEFS spread-skill | GEFS gate "
            "| Persistence skill fair % | Persistence skill ecdf % | Persistence spread-skill "
            "| Persistence gate | Climatology spread-skill |",
            "| :--- | ---: | ---: | ---: | ---: | :--- | ---: | ---: | ---: | :--- | ---: |",
        ]
        for month in months:
            row = [month]
            first = next((cards[(m, month)] for m in MODELS if (m, month) in cards), None)
            row.append("n/a" if first is None else str(first["n_init_times"]))
            for model in ("gefs", "persistence"):
                card = cards.get((model, month))
                if card is None or variable not in card["metrics"]:
                    row += ["n/a"] * 4
                    continue
                entry = card["metrics"][variable]["all"]
                fair, ecdf = entry.get("crpss_vs_baseline"), entry.get("crpss_vs_baseline_ecdf")
                row.append(_fmt(None if fair is None else 100 * fair, "+.1f"))
                row.append(_fmt(None if ecdf is None else 100 * ecdf, "+.1f"))
                row.append(_fmt(entry.get("spread_skill_ratio"), ".2f"))
                row.append(_verdict(card, variable))
            clim = cards.get(("climatology", month))
            ratio = (
                None
                if clim is None or variable not in clim["metrics"]
                else clim["metrics"][variable]["all"].get("spread_skill_ratio")
            )
            row.append(_fmt(ratio, ".2f"))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    lines += [
        "### Single leads at which GEFS scores below climatology",
        "",
        "| Month | Variable | Lead h | GEFS skill fair % | GEFS skill ecdf % |",
        "| :--- | :--- | ---: | ---: | ---: |",
    ]
    found = False
    for month in months:
        card = cards.get(("gefs", month))
        for variable in variables if card is not None else []:
            per_lead = card["metrics"].get(variable, {})
            for lead in sorted((k for k in per_lead if k != "all"), key=int):
                fair, ecdf = (
                    per_lead[lead].get("crpss_vs_baseline"),
                    per_lead[lead].get("crpss_vs_baseline_ecdf"),
                )
                if fair is not None and fair < 0:
                    found = True
                    ecdf_text = _fmt(None if ecdf is None else 100 * ecdf, "+.1f")
                    lines.append(f"| {month} | `{variable}` | {lead} | {100 * fair:+.1f} | {ecdf_text} |")
    if not found:
        lines.append("| none | | | | |")
    lines.append("")
    return "\n".join(lines)


def summary_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="run_gate1.py summary", description="Monthly tables of a year, from the scorecard files only."
    )
    parser.add_argument(
        "--scorecards", type=Path, required=True, help="directory with the scorecard JSON files"
    )
    parser.add_argument("--year", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        print(summary_tables(args.scorecards, args.year))
    except Refused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "summary":
        return summary_main(argv[1:])
    parser = argparse.ArgumentParser(
        description="Score climatology, persistence and GEFS for whole months (Gate 1).",
        epilog="See docs/GATE1.md.",
    )
    parser.add_argument("--data-dir", type=Path, required=True, help="external data directory")
    parser.add_argument(
        "--period", action="append", required=True, metavar="YYYY-MM|YYYY-Qn", help="repeatable"
    )
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
        "--confirmation-train-end",
        metavar="UTC",
        help="confirmation mode: score a period inside the train split, with references that learn "
        "from truth at or before this time only (needs --truth and --thresholds)",
    )
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
