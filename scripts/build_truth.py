"""Build hourly NOAA ISD ground-truth NetCDF datasets for WRFComp.

Examples
--------
    uv run python scripts/build_truth.py --year 2024 --out ~/wrfcomp-data/truth/isd_conus_2024.nc
    uv run python scripts/build_truth.py --start-year 2014 --end-year 2023 --discard-raw \\
        --out ~/wrfcomp-data/truth/isd_conus_2014_2023.nc
"""

from __future__ import annotations

import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from wrfcomp.harness.truth import (
    AWS_GLOBAL_HOURLY_BASE_URL,
    MAX_CONCURRENT_REQUESTS,
    PRIMARY_VARS,
    build_truth,
    compute_completeness,
    download_station_year,
    download_stations,
    get_truth_tolerance_minutes,
    load_stations_metadata,
    match_station_to_hours,
    parse_isd_csv,
    write_truth_netcdf,
)

app = typer.Typer(help="Download and build NOAA ISD hourly truth NetCDF datasets.")


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
    """Download a station-year CSV, match to full hours, save compact .npz, and delete raw CSV."""
    npz_dir = parsed_cache_dir / str(year)
    npz_dir.mkdir(parents=True, exist_ok=True)
    npz_path = npz_dir / f"{station_id}.npz"
    if npz_path.is_file() and npz_path.stat().st_size > 0:
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
        valid_times = pd.date_range(f"{year}-01-01T00:00:00", f"{year}-12-31T23:00:00", freq="h")
        vals, offs = match_station_to_hours(obs_df, valid_times, tolerance_minutes=tolerance_minutes)
        payload: dict[str, np.ndarray] = {}
        for var in PRIMARY_VARS:
            payload[var] = vals[var]
            payload[f"{var}_offset_min"] = offs[var]
        tmp_path = npz_dir / f".{station_id}.tmp.npz"
        np.savez_compressed(tmp_path, **payload)
        tmp_path.replace(npz_path)
    finally:
        csv_path.unlink(missing_ok=True)

    return (station_id, year, "downloaded")


@app.command()
def main(
    out: Path = typer.Option(..., help="Output NetCDF path (e.g. ~/wrfcomp-data/truth/isd_conus_2024.nc)."),
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
    max_workers: int = typer.Option(
        MAX_CONCURRENT_REQUESTS,
        min=1,
        max=MAX_CONCURRENT_REQUESTS,
        help="Concurrent download workers (hard-capped at 4).",
    ),
    discard_raw: bool = typer.Option(
        False,
        help="Parse each station-year CSV into a compact resumable .npz and delete the raw CSV.",
    ),
    skip_download: bool = typer.Option(
        False,
        help="Do not touch the network; build strictly from files already in --cache-dir.",
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
    if year is not None and (start_year is not None or end_year is not None):
        typer.echo("Specify either --year or --start-year/--end-year, not both.", err=True)
        raise typer.Exit(code=2)
    if year is not None:
        y0, y1 = year, year
    elif start_year is not None and end_year is not None:
        y0, y1 = start_year, end_year
    else:
        typer.echo("Provide --year or both --start-year and --end-year.", err=True)
        raise typer.Exit(code=2)

    if y1 < y0:
        typer.echo(f"end_year ({y1}) must be >= start_year ({y0}).", err=True)
        raise typer.Exit(code=2)
    if y0 >= 2025 or y1 >= 2025:
        typer.echo("Refusing to download or build any year >= 2025 (sealed test split).", err=True)
        raise typer.Exit(code=2)

    years = list(range(y0, y1 + 1))
    data_dir = _default_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = cache_dir or (data_dir / "raw" / "isd")
    parsed_dir = data_dir / "parsed" / "isd"
    tol_min = get_truth_tolerance_minutes()

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
    max_failures = int(len(selected_ids) * len(years) * max_fail_rate)

    if not skip_download:
        typer.echo(
            f"Fetching {len(selected_ids)} stations x {len(years)} year(s) ({y0}..{y1}) "
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
    typer.echo(f"Building truth dataset for {y0}-01-01T00:00:00 .. {y1}-12-31T23:00:00...")
    ds = build_truth(
        selected_ids,
        start=f"{y0}-01-01T00:00:00",
        end=f"{y1}-12-31T23:00:00",
        cache_dir=raw_dir,
        stations_metadata=meta_df,
        tolerance_minutes=tol_min,
        parsed_cache_dir=parsed_dir if discard_raw else None,
    )

    write_truth_netcdf(ds, out)

    elapsed = time.perf_counter() - t0
    completeness = compute_completeness(ds)
    size_mb = out.stat().st_size / 1e6
    typer.echo(f"Wrote {out} ({size_mb:.2f} MB) in {elapsed:.1f}s")
    typer.echo("Per-variable completeness:")
    for var, frac in completeness.items():
        typer.echo(f"  {var:5s}: {frac * 100:6.2f}%")


if __name__ == "__main__":
    app()
