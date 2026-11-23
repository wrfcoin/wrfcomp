"""Build hourly NOAA ISD ground-truth NetCDF4 datasets for WRFComp.

Examples
--------
    # 2024 validation truth (with 2023-12-31T18Z lookback for persistence)
    uv run python scripts/build_truth.py \\
        --start 2023-12-31T18:00:00 --end 2024-12-31T23:00:00 \\
        --discard-raw --out ~/wrfcomp-data/truth/isd_conus_2024.nc

    # 2014-2023 training truth (through configs/splits.yaml train.end_utc)
    uv run python scripts/build_truth.py \\
        --start 2014-01-01T00:00:00 --end 2023-12-17T18:00:00 \\
        --discard-raw --out ~/wrfcomp-data/truth/isd_conus_train.nc
"""

from __future__ import annotations

import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import typer

from wrfcomp.harness.truth import (
    AWS_GLOBAL_HOURLY_BASE_URL,
    MAX_CONCURRENT_REQUESTS,
    PARSER_VERSION,
    build_truth,
    compute_completeness,
    download_station_year,
    download_stations,
    get_harness_config,
    get_sealed_test_cutoff,
    get_truth_tolerance_minutes,
    load_parsed_station_year_npz,
    load_stations_metadata,
    parse_isd_csv,
    required_years_for_window,
    save_parsed_station_year_npz,
    write_truth_netcdf,
)

app = typer.Typer(help="Download and build NOAA ISD hourly truth NetCDF4 datasets.")


def _default_data_dir() -> Path:
    return Path(os.environ.get("WRFCOMP_DATA_DIR", str(Path.home() / "wrfcomp-data")))


def _download_parse_discard_one(
    station_id: str,
    year: int,
    raw_cache_dir: Path,
    parsed_cache_dir: Path,
    *,
    base_url: str,
    tolerance_minutes: int,
    max_retries: int = 4,
) -> tuple[str, int, str]:
    """Download a station-year CSV, parse into versioned ``.npz`` observations, and delete raw CSV."""
    npz_dir = parsed_cache_dir / str(year)
    npz_dir.mkdir(parents=True, exist_ok=True)
    npz_path = npz_dir / f"{station_id}.npz"
    if npz_path.is_file() and npz_path.stat().st_size > 0:
        # Refuse stale/mismatched cache loudly instead of silently reusing it
        load_parsed_station_year_npz(npz_path, expected_tolerance_minutes=tolerance_minutes)
        return (station_id, year, "cached")

    csv_path = download_station_year(
        station_id,
        year,
        raw_cache_dir,
        base_url=base_url,
        max_retries=max_retries,
    )
    try:
        obs_df = parse_isd_csv(csv_path)
        save_parsed_station_year_npz(obs_df, npz_path, tolerance_minutes=tolerance_minutes)
    finally:
        csv_path.unlink(missing_ok=True)

    return (station_id, year, "downloaded")


def _resolve_time_bounds(
    *,
    start: str | None,
    end: str | None,
    year: int | None,
    start_year: int | None,
    end_year: int | None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    has_iso = start is not None or end is not None
    has_year = year is not None
    has_year_range = start_year is not None or end_year is not None
    modes = sum([has_iso, has_year, has_year_range])
    if modes != 1:
        typer.echo(
            "Specify exactly one of: (--start and --end), --year, or (--start-year and --end-year).",
            err=True,
        )
        raise typer.Exit(code=2)

    if has_iso:
        if start is None or end is None:
            typer.echo("Provide both --start and --end.", err=True)
            raise typer.Exit(code=2)
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
    elif has_year:
        assert year is not None
        start_ts = pd.Timestamp(f"{year}-01-01T00:00:00")
        end_ts = pd.Timestamp(f"{year}-12-31T23:00:00")
    else:
        if start_year is None or end_year is None:
            typer.echo("Provide both --start-year and --end-year.", err=True)
            raise typer.Exit(code=2)
        start_ts = pd.Timestamp(f"{start_year}-01-01T00:00:00")
        end_ts = pd.Timestamp(f"{end_year}-12-31T23:00:00")

    if start_ts.tzinfo is not None:
        start_ts = start_ts.tz_convert("UTC").tz_localize(None)
    if end_ts.tzinfo is not None:
        end_ts = end_ts.tz_convert("UTC").tz_localize(None)

    if end_ts < start_ts:
        typer.echo(f"end ({end_ts}) must be >= start ({start_ts}).", err=True)
        raise typer.Exit(code=2)
    return start_ts, end_ts


@app.command()
def main(
    out: Path = typer.Option(..., help="Output NetCDF4 path (e.g. ~/wrfcomp-data/truth/isd_conus_2024.nc)."),
    start: str | None = typer.Option(None, help="Inclusive UTC start timestamp (e.g. 2023-12-31T18:00:00)."),
    end: str | None = typer.Option(None, help="Inclusive UTC end timestamp (e.g. 2024-12-31T23:00:00)."),
    year: int | None = typer.Option(None, help="Single calendar year to build (e.g. 2024)."),
    start_year: int | None = typer.Option(None, help="First calendar year (inclusive) for multi-year build."),
    end_year: int | None = typer.Option(None, help="Last calendar year (inclusive) for multi-year build."),
    stations: Path = typer.Option(
        Path("configs/stations_conus.csv"),
        help="Path to stations CSV with station_id, lat, lon, elevation_m.",
    ),
    station_ids: str | None = typer.Option(
        None,
        help="Optional comma-separated subset of station_ids (default: all in --stations).",
    ),
    cache_dir: Path | None = typer.Option(
        None,
        help="Directory for raw ISD CSV cache (default: <WRFCOMP_DATA_DIR>/raw/isd).",
    ),
    parsed_cache_dir: Path | None = typer.Option(
        None,
        help=f"Directory for parsed .npz cache (default: <WRFCOMP_DATA_DIR>/parsed/isd-v{PARSER_VERSION}).",
    ),
    max_workers: int = typer.Option(
        MAX_CONCURRENT_REQUESTS,
        min=1,
        max=MAX_CONCURRENT_REQUESTS,
        help="Concurrent download workers (hard-capped at 4).",
    ),
    discard_raw: bool = typer.Option(
        False,
        help="Parse each station-year CSV into a compact versioned .npz and delete the raw CSV.",
    ),
    skip_download: bool = typer.Option(
        False,
        help="Do not touch the network; build strictly from already-cached files.",
    ),
    base_url: str = typer.Option(
        AWS_GLOBAL_HOURLY_BASE_URL,
        help="Base URL for ISD global-hourly station-year CSVs.",
    ),
    min_free_gb: float = typer.Option(
        40.0,
        help="Abort immediately if free disk space on the data volume drops below this many GB.",
    ),
    max_fail_rate: float = typer.Option(
        0.02,
        help="Abort immediately if failed requests exceed this fraction of total station-years.",
    ),
) -> None:
    cfg = get_harness_config()
    cutoff = get_sealed_test_cutoff(cfg)
    tol_min = get_truth_tolerance_minutes(cfg)

    start_ts, end_ts = _resolve_time_bounds(
        start=start,
        end=end,
        year=year,
        start_year=start_year,
        end_year=end_year,
    )
    if start_ts >= cutoff or end_ts >= cutoff:
        typer.echo(
            f"Refusing range [{start_ts.isoformat()}, {end_ts.isoformat()}]: "
            f"sealed test split starts at {cutoff.isoformat()}.",
            err=True,
        )
        raise typer.Exit(code=2)

    years = required_years_for_window(start_ts, end_ts, tol_min, cfg=cfg)
    data_dir = _default_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = cache_dir or (data_dir / "raw" / "isd")
    parsed_dir = parsed_cache_dir or (data_dir / "parsed" / f"isd-v{PARSER_VERSION}")

    def _check_free_disk() -> None:
        free_gb = shutil.disk_usage(data_dir).free / (1024**3)
        if free_gb < min_free_gb:
            typer.echo(
                f"ABORT: free disk space ({free_gb:.2f} GB) dropped below {min_free_gb:.2f} GB limit.",
                err=True,
            )
            raise typer.Exit(code=3)

    _check_free_disk()

    meta_df = load_stations_metadata(stations)
    if station_ids is not None:
        selected_ids = [s.strip() for s in station_ids.split(",") if s.strip()]
    else:
        selected_ids = meta_df["station_id"].tolist()

    t0 = time.perf_counter()
    failed: list[tuple[str, int, str]] = []
    total_tasks = len(selected_ids) * len(years)
    max_failures = int(total_tasks * max_fail_rate)

    if not skip_download:
        typer.echo(
            f"Fetching {len(selected_ids)} stations x {len(years)} year(s) ({years[0]}..{years[-1]}) "
            f"with max_workers={max_workers}, discard_raw={discard_raw}..."
        )
        if discard_raw:
            tasks = [(sid, yr) for yr in years for sid in selected_ids]
            done = 0
            aborted_reason: str | None = None
            with ThreadPoolExecutor(max_workers=min(max_workers, MAX_CONCURRENT_REQUESTS)) as pool:
                fmap = {
                    pool.submit(
                        _download_parse_discard_one,
                        sid,
                        yr,
                        raw_dir,
                        parsed_dir,
                        base_url=base_url,
                        tolerance_minutes=tol_min,
                    ): (sid, yr)
                    for sid, yr in tasks
                }
                for fut in as_completed(fmap):
                    sid, yr = fmap[fut]
                    done += 1
                    try:
                        fut.result()
                    except Exception as exc:
                        failed.append((sid, yr, str(exc)))
                    if done % 10 == 0 or done == len(tasks):
                        free_gb = shutil.disk_usage(data_dir).free / (1024**3)
                        if free_gb < min_free_gb:
                            aborted_reason = (
                                f"free disk ({free_gb:.2f} GB) dropped below {min_free_gb:.2f} GB limit"
                            )
                    if len(failed) > max_failures:
                        aborted_reason = (
                            f"failures ({len(failed)}/{len(tasks)}) exceeded "
                            f"{max_fail_rate * 100:.1f}% limit ({max_failures})"
                        )
                    if done % 100 == 0 or done == len(tasks):
                        free_gb = shutil.disk_usage(data_dir).free / (1024**3)
                        typer.echo(
                            f"  progress: {done}/{len(tasks)} "
                            f"({len(failed)} failed, {free_gb:.1f} GB free)"
                        )
                    if aborted_reason is not None:
                        for pending in fmap:
                            pending.cancel()
                        break
            failed.sort()
            if aborted_reason is not None:
                typer.echo(f"ABORTING bulk download: {aborted_reason}", err=True)
                if failed:
                    failed_log = out.with_suffix(out.suffix + ".failed.json")
                    failed_log.parent.mkdir(parents=True, exist_ok=True)
                    failed_log.write_text(
                        json.dumps(
                            [{"station_id": s, "year": y, "reason": r} for s, y, r in failed],
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                raise typer.Exit(code=3)
        else:
            _, failed = download_stations(
                selected_ids,
                years,
                raw_dir,
                max_workers=max_workers,
                base_url=base_url,
                cfg=cfg,
            )
            if len(failed) > max_failures:
                typer.echo(
                    f"ABORT: failures ({len(failed)}) exceeded {max_fail_rate * 100:.1f}% limit.",
                    err=True,
                )
                raise typer.Exit(code=3)

    if failed:
        typer.echo(f"Failed station-years ({len(failed)}):", err=True)
        for sid, yr, reason in failed[:30]:
            typer.echo(f"  {yr}/{sid}: {reason}", err=True)
        if len(failed) > 30:
            typer.echo(f"  ... and {len(failed) - 30} more (see failed JSON).", err=True)
        failed_log = out.with_suffix(out.suffix + ".failed.json")
        failed_log.parent.mkdir(parents=True, exist_ok=True)
        failed_log.write_text(
            json.dumps(
                [{"station_id": sid, "year": yr, "reason": reason} for sid, yr, reason in failed],
                indent=2,
            ),
            encoding="utf-8",
        )
        typer.echo(f"Wrote failure list to {failed_log}", err=True)

    _check_free_disk()
    typer.echo(f"Building truth dataset for {start_ts.isoformat()} .. {end_ts.isoformat()}...")
    ds = build_truth(
        selected_ids,
        start=start_ts,
        end=end_ts,
        cache_dir=raw_dir,
        stations_metadata=meta_df,
        tolerance_minutes=tol_min,
        parsed_cache_dir=parsed_dir if discard_raw else None,
        cfg=cfg,
    )

    write_truth_netcdf(ds, out)

    elapsed = time.perf_counter() - t0
    completeness = compute_completeness(ds)
    size_mb = out.stat().st_size / 1e6
    typer.echo(f"Wrote {out} ({size_mb:.2f} MB, NetCDF4/h5netcdf) in {elapsed:.1f}s")
    typer.echo("Per-variable completeness:")
    for var, frac in completeness.items():
        typer.echo(f"  {var:5s}: {frac * 100:6.2f}%")


if __name__ == "__main__":
    app()
