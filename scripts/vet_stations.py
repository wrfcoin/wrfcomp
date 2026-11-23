"""Vet CONUS NOAA ISD candidate stations using 2024 ISD-Lite completeness counts.

Downloads NOAA ISD-Lite 2024 hourly files (for COUNTING completeness only, never
for ground-truth scoring), applies the Sprint 1 T2 completeness and history
rules, and selects 500 spatially diverse stations via
``scripts/fetch_conus_stations.py:_select_spatially_diverse``.
"""

from __future__ import annotations

import gzip
import importlib.util
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import typer

ISD_HISTORY_URL = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"
# Note: https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/2024/<USAF>-<WBAN>-2024.gz now redirects to
# https://noaa-isd-pds.s3.amazonaws.com/index.html#isd-lite/ (NOAA's official Open Data S3 bucket).
ISD_LITE_2024_URL_TEMPLATE = (
    "https://noaa-isd-pds.s3.amazonaws.com/isd-lite/data/2024/{usaf}-{wban}-2024.gz"
)

HOURS_IN_2024 = 366 * 24  # 2024 is a leap year: 8,784 hours
MONTH_DAYS_2024 = (0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
MONTH_OFFSETS_2024 = (0, 0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335)
MISSING_VALUE = -9999

MIN_TEMP_FRAC = 0.90
MIN_DEWPOINT_FRAC = 0.90
MIN_WIND_SPEED_FRAC = 0.90
MIN_WIND_DIR_FRAC = 0.90
MIN_SLP_FRAC = 0.80
MAX_GAP_HOURS = 10 * 24  # 10 days = 240 hours
MAX_CONCURRENCY = 4
REQUIRED_MIN_BEGIN = 20140101
REQUIRED_MIN_END = 20241231

EXCLUDED_STATES = frozenset({"AK", "HI", "PR", "VI", "GU", "AS", "MP"})

app = typer.Typer(help="Vet CONUS NOAA ISD stations against 2024 ISD-Lite availability.")


def default_data_dir() -> Path:
    """Return the external data directory (never inside the repo or /tmp)."""
    env_dir = os.environ.get("WRFCOMP_DATA_DIR")
    if env_dir:
        return Path(env_dir).expanduser()
    return Path.home() / "wrfcomp-data"


def load_select_spatially_diverse():
    """Import ``_select_spatially_diverse`` from ``scripts/fetch_conus_stations.py``."""
    fetch_path = Path(__file__).resolve().with_name("fetch_conus_stations.py")
    spec = importlib.util.spec_from_file_location("fetch_conus_stations", fetch_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {fetch_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module._select_spatially_diverse


@dataclass(frozen=True)
class StationCompleteness:
    """Completeness summary for a single station in 2024 ISD-Lite."""

    station_id: str
    temp_hours: int
    dewpoint_hours: int
    slp_hours: int
    wind_dir_hours: int
    wind_speed_hours: int
    record_hours: int
    temp_max_gap_hours: int
    dewpoint_max_gap_hours: int
    slp_max_gap_hours: int
    wind_dir_max_gap_hours: int
    wind_speed_max_gap_hours: int
    record_max_gap_hours: int
    total_hours: int = HOURS_IN_2024

    @property
    def temp_frac(self) -> float:
        return self.temp_hours / self.total_hours

    @property
    def dewpoint_frac(self) -> float:
        return self.dewpoint_hours / self.total_hours

    @property
    def slp_frac(self) -> float:
        return self.slp_hours / self.total_hours

    @property
    def wind_dir_frac(self) -> float:
        return self.wind_dir_hours / self.total_hours

    @property
    def wind_speed_frac(self) -> float:
        return self.wind_speed_hours / self.total_hours

    @property
    def max_variable_gap_hours(self) -> int:
        return max(
            self.temp_max_gap_hours,
            self.dewpoint_max_gap_hours,
            self.slp_max_gap_hours,
            self.wind_dir_max_gap_hours,
            self.wind_speed_max_gap_hours,
        )

    def passes(self) -> bool:
        """Return True iff the station satisfies all 2024 completeness and gap rules."""
        return (
            self.temp_frac >= MIN_TEMP_FRAC
            and self.dewpoint_frac >= MIN_DEWPOINT_FRAC
            and self.wind_speed_frac >= MIN_WIND_SPEED_FRAC
            and self.wind_dir_frac >= MIN_WIND_DIR_FRAC
            and self.slp_frac >= MIN_SLP_FRAC
            and self.max_variable_gap_hours <= MAX_GAP_HOURS
        )


def max_missing_run_hours(valid_mask: np.ndarray) -> int:
    """Return the longest contiguous run of False values in a 1-D boolean mask."""
    if valid_mask.size == 0:
        return 0
    padded = np.empty(valid_mask.size + 2, dtype=bool)
    padded[0] = True
    padded[1:-1] = valid_mask
    padded[-1] = True
    true_idx = np.flatnonzero(padded)
    return int(np.max(np.diff(true_idx) - 1))


def parse_isd_lite_2024_lines(
    lines: Iterable[str],
    station_id: str,
    total_hours: int = HOURS_IN_2024,
) -> StationCompleteness:
    """Parse NOAA ISD-Lite 2024 fixed-width lines and count valid hours and max gaps.

    Official NOAA ISD-Lite fixed-width fields (1-indexed columns):
      1-4:   Year
      6-7:   Month
      9-10:  Day
      12-13: Hour (00-23 UTC)
      14-19: Air temperature (tenths of C, -9999 missing)
      20-25: Dew point temperature (tenths of C, -9999 missing)
      26-31: Sea level pressure (tenths of hPa, -9999 missing)
      32-37: Wind direction (degrees, -9999 missing)
      38-43: Wind speed rate (tenths of m/s, -9999 missing)
    """
    temp_ok = np.zeros(total_hours, dtype=bool)
    dewpoint_ok = np.zeros(total_hours, dtype=bool)
    slp_ok = np.zeros(total_hours, dtype=bool)
    wind_dir_ok = np.zeros(total_hours, dtype=bool)
    wind_speed_ok = np.zeros(total_hours, dtype=bool)
    record_ok = np.zeros(total_hours, dtype=bool)

    for raw in lines:
        if len(raw) < 43:
            continue
        try:
            year = int(raw[0:4])
            if year != 2024:
                continue
            month = int(raw[5:7])
            day = int(raw[8:10])
            hour = int(raw[11:13])
            if not (1 <= month <= 12 and 1 <= day <= MONTH_DAYS_2024[month] and 0 <= hour <= 23):
                continue
            hour_idx = (MONTH_OFFSETS_2024[month] + (day - 1)) * 24 + hour
            if hour_idx >= total_hours:
                continue

            temp = int(raw[13:19])
            dewpoint = int(raw[19:25])
            slp = int(raw[25:31])
            wind_dir = int(raw[31:37])
            wind_speed = int(raw[37:43])
        except ValueError:
            continue

        record_ok[hour_idx] = True
        if temp != MISSING_VALUE:
            temp_ok[hour_idx] = True
        if dewpoint != MISSING_VALUE:
            dewpoint_ok[hour_idx] = True
        if slp != MISSING_VALUE:
            slp_ok[hour_idx] = True
        if wind_dir != MISSING_VALUE:
            wind_dir_ok[hour_idx] = True
        if wind_speed != MISSING_VALUE:
            wind_speed_ok[hour_idx] = True

    return StationCompleteness(
        station_id=station_id,
        temp_hours=int(temp_ok.sum()),
        dewpoint_hours=int(dewpoint_ok.sum()),
        slp_hours=int(slp_ok.sum()),
        wind_dir_hours=int(wind_dir_ok.sum()),
        wind_speed_hours=int(wind_speed_ok.sum()),
        record_hours=int(record_ok.sum()),
        temp_max_gap_hours=max_missing_run_hours(temp_ok),
        dewpoint_max_gap_hours=max_missing_run_hours(dewpoint_ok),
        slp_max_gap_hours=max_missing_run_hours(slp_ok),
        wind_dir_max_gap_hours=max_missing_run_hours(wind_dir_ok),
        wind_speed_max_gap_hours=max_missing_run_hours(wind_speed_ok),
        record_max_gap_hours=max_missing_run_hours(record_ok),
        total_hours=total_hours,
    )


def parse_isd_lite_2024_file(path: Path, station_id: str) -> StationCompleteness:
    """Parse a gzipped or plain-text ISD-Lite 2024 file."""
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="ascii", errors="replace") as fh:
            return parse_isd_lite_2024_lines(fh, station_id=station_id)
    with path.open("rt", encoding="ascii", errors="replace") as fh:
        return parse_isd_lite_2024_lines(fh, station_id=station_id)


def filter_conus_candidates(
    history_df: pd.DataFrame,
    min_begin: int = REQUIRED_MIN_BEGIN,
    min_end: int = REQUIRED_MIN_END,
) -> pd.DataFrame:
    """Apply the CONUS geographic and active-history filters to ``isd-history.csv``."""
    df = history_df.dropna(subset=["LAT", "LON", "ELEV(M)", "BEGIN", "END"]).copy()
    conus_mask = (
        (df["CTRY"] == "US")
        & (~df["STATE"].isin(EXCLUDED_STATES))
        & (df["LAT"].between(24.5, 49.5))
        & (df["LON"].between(-125.0, -66.5))
        & (df["BEGIN"].astype(int) <= min_begin)
        & (df["END"].astype(int) >= min_end)
        & (df["USAF"] != "999999")
    )
    filtered = df.loc[conus_mask].copy()
    filtered["USAF"] = filtered["USAF"].str.zfill(6)
    filtered["WBAN"] = filtered["WBAN"].str.zfill(5)
    filtered["station_id"] = filtered["USAF"] + filtered["WBAN"]
    filtered = filtered.drop_duplicates(subset=["station_id"]).sort_values("station_id")
    return filtered.reset_index(drop=True)


def fetch_history_csv(cache_path: Path, refresh: bool = False) -> pd.DataFrame:
    """Download (or load cached) NOAA ``isd-history.csv``."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if refresh or not cache_path.exists():
        req = urllib.request.Request(ISD_HISTORY_URL, headers={"User-Agent": "wrfcomp-vet/0.1"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            cache_path.write_bytes(resp.read())
    return pd.read_csv(cache_path, dtype={"USAF": str, "WBAN": str}, low_memory=False)


def download_station_2024(
    usaf: str,
    wban: str,
    cache_dir: Path,
    timeout: float = 30.0,
    retries: int = 3,
) -> tuple[str, Path | None, int, str]:
    """Download a single 2024 ISD-Lite file if not cached.

    Returns ``(station_id, path_or_none, size_bytes, status)``.
    Status is one of ``cached``, ``downloaded``, ``not_found``, ``error:...``.
    """
    station_id = f"{usaf}{wban}"
    dest = cache_dir / f"{usaf}-{wban}-2024.gz"
    missing_marker = cache_dir / f"{usaf}-{wban}-2024.404"

    if dest.exists() and dest.stat().st_size > 0:
        return station_id, dest, dest.stat().st_size, "cached"
    if missing_marker.exists():
        return station_id, None, 0, "not_found"

    url = ISD_LITE_2024_URL_TEMPLATE.format(usaf=usaf, wban=wban)
    last_err = ""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "wrfcomp-vet/0.1"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            if len(data) < 2 or data[:2] != b"\x1f\x8b":
                raise ValueError("Response is not a gzip archive")
            tmp = dest.with_suffix(".gz.part")
            tmp.write_bytes(data)
            tmp.replace(dest)
            return station_id, dest, len(data), "downloaded"
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                missing_marker.write_text("404\n")
                return station_id, None, 0, "not_found"
            last_err = f"HTTP {exc.code}"
        except Exception as exc:  # noqa: BLE001
            last_err = f"{type(exc).__name__}: {exc}"
        if attempt + 1 < retries:
            time.sleep(1.0 * (attempt + 1))

    return station_id, None, 0, f"error:{last_err}"


def vet_candidates(
    candidates: pd.DataFrame,
    cache_dir: Path,
    concurrency: int = MAX_CONCURRENCY,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Download and evaluate 2024 ISD-Lite completeness for candidate stations."""
    if concurrency < 1 or concurrency > MAX_CONCURRENCY:
        raise ValueError(f"concurrency must be in 1..{MAX_CONCURRENCY}, got {concurrency}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    stats = {
        "total": len(candidates),
        "downloaded": 0,
        "cached": 0,
        "not_found": 0,
        "errors": 0,
        "bytes": 0,
    }

    station_pairs = list(zip(candidates["USAF"].tolist(), candidates["WBAN"].tolist(), strict=True))
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(download_station_2024, usaf, wban, cache_dir): (usaf, wban)
            for usaf, wban in station_pairs
        }
        done = 0
        for fut in as_completed(futures):
            station_id, path, size_bytes, status = fut.result()
            done += 1
            stats["bytes"] += size_bytes
            if status in ("downloaded", "cached", "not_found"):
                stats[status] += 1
            else:
                stats["errors"] += 1

            if path is not None:
                comp = parse_isd_lite_2024_file(path, station_id=station_id)
                rec = asdict(comp)
                rec["status"] = status
                rec["size_bytes"] = size_bytes
                rec["temp_frac"] = comp.temp_frac
                rec["dewpoint_frac"] = comp.dewpoint_frac
                rec["slp_frac"] = comp.slp_frac
                rec["wind_dir_frac"] = comp.wind_dir_frac
                rec["wind_speed_frac"] = comp.wind_speed_frac
                rec["max_variable_gap_hours"] = comp.max_variable_gap_hours
                rec["passes"] = comp.passes()
            else:
                rec = {
                    "station_id": station_id,
                    "status": status,
                    "size_bytes": 0,
                    "temp_hours": 0,
                    "dewpoint_hours": 0,
                    "slp_hours": 0,
                    "wind_dir_hours": 0,
                    "wind_speed_hours": 0,
                    "record_hours": 0,
                    "temp_max_gap_hours": HOURS_IN_2024,
                    "dewpoint_max_gap_hours": HOURS_IN_2024,
                    "slp_max_gap_hours": HOURS_IN_2024,
                    "wind_dir_max_gap_hours": HOURS_IN_2024,
                    "wind_speed_max_gap_hours": HOURS_IN_2024,
                    "record_max_gap_hours": HOURS_IN_2024,
                    "total_hours": HOURS_IN_2024,
                    "temp_frac": 0.0,
                    "dewpoint_frac": 0.0,
                    "slp_frac": 0.0,
                    "wind_dir_frac": 0.0,
                    "wind_speed_frac": 0.0,
                    "max_variable_gap_hours": HOURS_IN_2024,
                    "passes": False,
                }
            rows.append(rec)
            if done % 250 == 0 or done == len(station_pairs):
                typer.echo(
                    f"Vetted {done}/{len(station_pairs)} stations "
                    f"({stats['bytes'] / (1024 * 1024):.1f} MiB, "
                    f"404={stats['not_found']}, err={stats['errors']})"
                )

    metrics_df = pd.DataFrame(rows).sort_values("station_id").reset_index(drop=True)
    merged = candidates.merge(metrics_df, on="station_id", how="inner")
    return merged, stats


def export_station_csv(selected: pd.DataFrame, out: Path) -> None:
    """Write selected stations in the canonical ``configs/stations_conus.csv`` schema."""
    export_df = selected.rename(
        columns={
            "STATION NAME": "station_name",
            "STATE": "state",
            "LAT": "lat",
            "LON": "lon",
            "ELEV(M)": "elevation_m",
            "BEGIN": "begin_yyyymmdd",
            "END": "end_yyyymmdd",
        }
    )[["station_id", "station_name", "state", "lat", "lon", "elevation_m", "begin_yyyymmdd", "end_yyyymmdd"]]
    out.parent.mkdir(parents=True, exist_ok=True)
    export_df.to_csv(out, index=False)


@app.command()
def main(
    out: Path = typer.Option(Path("configs/stations_conus.csv"), help="Output vetted station CSV."),
    count: int = typer.Option(500, help="Number of stations to select."),
    data_dir: Path | None = typer.Option(None, help="External cache directory (default WRFCOMP_DATA_DIR)."),
    concurrency: int = typer.Option(MAX_CONCURRENCY, help="Concurrent NOAA HTTP requests (max 4)."),
    sample: int | None = typer.Option(
        None,
        help="Dry-run on the first N candidate stations to measure download size without writing CSV.",
    ),
    summary_out: Path | None = typer.Option(
        None,
        help="Optional JSON path (outside repo) for full vetting diagnostics.",
    ),
) -> None:
    """Vet CONUS candidate stations on 2024 NOAA ISD-Lite and select 500 spatially even sites."""
    if concurrency < 1 or concurrency > MAX_CONCURRENCY:
        raise typer.BadParameter(f"concurrency must be between 1 and {MAX_CONCURRENCY}")

    base_dir = data_dir.expanduser() if data_dir is not None else default_data_dir()
    isd_root = base_dir / "isd-lite"
    history_path = isd_root / "isd-history.csv"
    cache_2024 = isd_root / "2024"

    history_df = fetch_history_csv(history_path)
    base_2020 = filter_conus_candidates(history_df, min_begin=20200101, min_end=REQUIRED_MIN_END)
    candidates = filter_conus_candidates(history_df, min_begin=REQUIRED_MIN_BEGIN, min_end=REQUIRED_MIN_END)

    typer.echo(
        f"CONUS active in 2024 (BEGIN<=20200101): {len(base_2020)} | "
        f"with 2014-2023 history (BEGIN<=20140101): {len(candidates)}"
    )

    if sample is not None:
        sample_df = candidates.head(sample)
        t0 = time.monotonic()
        vetted_sample, stats = vet_candidates(sample_df, cache_2024, concurrency=concurrency)
        elapsed = time.monotonic() - t0
        found = len(sample_df) - stats["not_found"] - stats["errors"]
        avg_kb = (stats["bytes"] / 1024.0 / found) if found > 0 else 0.0
        est_total_mb = (stats["bytes"] / len(sample_df)) * len(candidates) / (1024 * 1024)
        typer.echo(
            f"Sample {len(sample_df)} stations: {stats['bytes']:,} bytes "
            f"({stats['bytes'] / 1024:.1f} KiB total, {avg_kb:.1f} KiB/found station, "
            f"found={found}, 404={stats['not_found']}, err={stats['errors']}, {elapsed:.1f}s)"
        )
        typer.echo(
            f"Projected full vetting ({len(candidates)} candidates): ~{est_total_mb:.1f} MiB"
        )
        for row in vetted_sample.itertuples():
            typer.echo(
                f"  {row.station_id} ({row.STATE}) status={row.status} bytes={row.size_bytes} "
                f"T={row.temp_frac:.3f} Td={row.dewpoint_frac:.3f} SLP={row.slp_frac:.3f} "
                f"Wdir={row.wind_dir_frac:.3f} Wspd={row.wind_speed_frac:.3f} "
                f"maxGapH={row.max_variable_gap_hours} recGapH={row.record_max_gap_hours} "
                f"pass={row.passes}"
            )
        return

    t0 = time.monotonic()
    vetted, stats = vet_candidates(candidates, cache_2024, concurrency=concurrency)
    elapsed = time.monotonic() - t0

    passing = vetted[vetted["passes"]].copy()
    typer.echo(
        f"Vetted {len(vetted)} candidates in {elapsed:.1f}s "
        f"({stats['bytes'] / (1024 * 1024):.1f} MiB). Passing: {len(passing)}"
    )

    if summary_out is not None:
        summary_out.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "conus_active_2020": len(base_2020),
            "candidates_2014": len(candidates),
            "download_stats": stats,
            "elapsed_seconds": round(elapsed, 2),
            "rule_counts": {
                "has_2024_file": int(vetted["status"].isin(["cached", "downloaded"]).sum()),
                "temp_ge_90": int((vetted["temp_frac"] >= MIN_TEMP_FRAC).sum()),
                "dewpoint_ge_90": int((vetted["dewpoint_frac"] >= MIN_DEWPOINT_FRAC).sum()),
                "wind_speed_ge_90": int((vetted["wind_speed_frac"] >= MIN_WIND_SPEED_FRAC).sum()),
                "wind_dir_ge_90": int((vetted["wind_dir_frac"] >= MIN_WIND_DIR_FRAC).sum()),
                "slp_ge_80": int((vetted["slp_frac"] >= MIN_SLP_FRAC).sum()),
                "max_var_gap_le_240h": int((vetted["max_variable_gap_hours"] <= MAX_GAP_HOURS).sum()),
                "max_rec_gap_le_240h": int((vetted["record_max_gap_hours"] <= MAX_GAP_HOURS).sum()),
                "all_rules_pass": int(len(passing)),
                "all_rules_pass_record_gap_only": int(
                    (
                        (vetted["temp_frac"] >= MIN_TEMP_FRAC)
                        & (vetted["dewpoint_frac"] >= MIN_DEWPOINT_FRAC)
                        & (vetted["wind_speed_frac"] >= MIN_WIND_SPEED_FRAC)
                        & (vetted["wind_dir_frac"] >= MIN_WIND_DIR_FRAC)
                        & (vetted["slp_frac"] >= MIN_SLP_FRAC)
                        & (vetted["record_max_gap_hours"] <= MAX_GAP_HOURS)
                    ).sum()
                ),
            },
        }
        summary_out.write_text(json.dumps(summary, indent=2) + "\n")
        vetted.to_csv(summary_out.with_suffix(".csv"), index=False)

    if len(passing) < count:
        typer.echo(
            f"ERROR: Only {len(passing)} stations passed, fewer than required {count}. "
            "Stopping without modifying station list.",
            err=True,
        )
        raise typer.Exit(code=1)

    select_fn = load_select_spatially_diverse()
    selected = select_fn(passing, target_count=count)
    export_station_csv(selected, out)
    typer.echo(f"Wrote {len(selected)} vetted stations to {out}")


if __name__ == "__main__":
    app()
