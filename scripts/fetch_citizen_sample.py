"""Fetch a small privacy-preserving 2024 CWOP/APRSWXNET citizen-station sample.

Queries NOAA MADIS's open public surface CGI (``madisXmlPublicDir``) for
``APRSWXNET`` observations within 150 km of Amarillo TX and Seattle WA over
2024-07-08 through 2024-07-21. Responses are parsed strictly in memory so no raw
citizen identifiers or exact home coordinates are ever written to disk, printed,
or logged. Only keyed 11-character station tokens, coordinates rounded to 0.1
degree and elevations rounded to 10 m are saved under
``${WRFCOMP_DATA_DIR:-$HOME/wrfcomp-data}/citizen/``. The token key is a random
secret in that folder (``.station_key``); it is never printed.

Log lines carry region names, window times, counts and exception class names
only. An exception message is never printed, because it could quote row content.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import typer

from wrfcomp.assimilation.citizen_obs import (
    CWOP_PROVIDER,
    SEALED_START,
    load_citizen_batch,
    load_or_create_station_key,
    merge_citizen_batches,
    parse_madis_csv,
    save_citizen_batch,
    summarize_batch,
)
from wrfcomp.assimilation.tokens import ObservationBatch

MADIS_CGI_URL = "https://madis-data.ncep.noaa.gov/madisPublic1/cgi-bin/madisXmlPublicDir"
MADIS_NVARS = ("LAT", "LON", "ELEV", "T", "TD", "DD", "FF", "FFGUST", "P", "SLP", "PCP1H")
MAX_CONCURRENCY = 4
HARD_CAP_BYTES = 500 * 1024 * 1024  # 500 MiB hard transfer cap
# From configs/splits.yaml, by way of the loader.
SEALED_DATE = dt.date.fromisoformat(str(SEALED_START.astype("datetime64[D]")))


@dataclass(frozen=True)
class RegionSpec:
    name: str
    center_lat: float
    center_lon: float
    radius_km: float
    bbox: tuple[float, float, float, float]  # (latll, lonll, latur, lonur)


DEFAULT_REGIONS: tuple[RegionSpec, ...] = (
    RegionSpec(
        name="amarillo_tx",
        center_lat=35.222,
        center_lon=-101.831,
        radius_km=150.0,
        bbox=(33.8, -103.5, 36.6, -100.1),
    ),
    RegionSpec(
        name="seattle_wa",
        center_lat=47.606,
        center_lon=-122.332,
        radius_km=150.0,
        bbox=(46.2, -124.4, 49.0, -120.2),
    ),
)

app = typer.Typer(help="Fetch anonymized 2024 CWOP citizen-station sample from NOAA MADIS.")


def default_data_dir() -> Path:
    env_dir = os.environ.get("WRFCOMP_DATA_DIR")
    if env_dir:
        return Path(env_dir).expanduser()
    return Path.home() / "wrfcomp-data"


def build_madis_window_url(
    end_hour_utc: dt.datetime,
    bbox: tuple[float, float, float, float],
    *,
    window_hours: int = 2,
) -> str:
    """Build a 1- or 2-hour ``madisXmlPublicDir`` URL ending at ``end_hour_utc:59`` UTC."""
    if window_hours not in (1, 2):
        raise ValueError(f"window_hours must be 1 or 2, got {window_hours}")
    if end_hour_utc.date() >= SEALED_DATE:
        raise ValueError(f"refusing to request sealed-split date (>= {SEALED_DATE.isoformat()})")
    time_str = end_hour_utc.strftime("%Y%m%d_%H59")
    minbck = -(window_hours * 60 - 1)
    latll, lonll, latur, lonur = bbox
    params: list[tuple[str, str]] = [
        ("rdr", ""),
        ("time", time_str),
        ("minbck", str(minbck)),
        ("minfwd", "0"),
        ("recwin", "0"),  # all sub-hourly reports, not just latest
        ("timefilter", "0"),
        ("dfltrsel", "1"),  # lat/lon bounding box
        ("latll", str(latll)),
        ("lonll", str(lonll)),
        ("latur", str(latur)),
        ("lonur", str(lonur)),
        ("stasel", "0"),
        ("pvdrsel", "1"),
        ("pvd", CWOP_PROVIDER),
        ("varsel", "1"),
        ("qctype", "0"),
        ("qcsel", "0"),  # all observations regardless of MADIS QC
        ("xml", "5"),  # CSV with header
        ("csvmiss", "1"),  # blank missing values
    ]
    params.extend(("nvars", v) for v in MADIS_NVARS)
    return f"{MADIS_CGI_URL}?{urllib.parse.urlencode(params)}"


def iter_windows(start_date: dt.date, end_date: dt.date) -> list[tuple[dt.datetime, int]]:
    """Return non-overlapping 2-hour windows ``(end_hour_utc, window_hours)`` over the range."""
    if start_date > end_date:
        raise ValueError("start_date must be <= end_date")
    if end_date >= SEALED_DATE:
        raise ValueError(f"refusing date range reaching {SEALED_DATE.isoformat()} or later")
    windows: list[tuple[dt.datetime, int]] = []
    cur = dt.datetime(start_date.year, start_date.month, start_date.day, 1, 0)
    last = dt.datetime(end_date.year, end_date.month, end_date.day, 23, 0)
    while cur <= last:
        windows.append((cur, 2))
        cur += dt.timedelta(hours=2)
    return windows


def _fetch_one_window(
    region: RegionSpec,
    end_hour_utc: dt.datetime,
    window_hours: int,
    *,
    key: bytes,
    retries: int = 3,
    timeout: float = 90.0,
) -> tuple[ObservationBatch, int]:
    """Download and immediately anonymize one 1- or 2-hour window of CWOP observations in memory."""
    url = build_madis_window_url(end_hour_utc, region.bbox, window_hours=window_hours)
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "wrfcomp-citizen-sample/0.1"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw_bytes = resp.read()
            text = raw_bytes.decode("utf-8", errors="replace")
            batch = parse_madis_csv(
                text,
                region_center=(region.center_lat, region.center_lon),
                max_radius_km=region.radius_km,
                provider_filter=CWOP_PROVIDER,
                key=key,
            )
            return batch, len(raw_bytes)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt + 1 < retries:
                time.sleep(1.5 * (attempt + 1))
    # Class name only: an exception message could quote response content.
    raise RuntimeError(
        f"failed to fetch {region.name} {end_hour_utc.strftime('%Y-%m-%d %H:59Z')}: {type(last_exc).__name__}"
    ) from None


def fetch_region(
    region: RegionSpec,
    windows: list[tuple[dt.datetime, int]],
    *,
    key: bytes,
    concurrency: int = MAX_CONCURRENCY,
    byte_budget_lock: threading.Lock,
    byte_counter: list[int],
    hard_cap_bytes: int = HARD_CAP_BYTES,
) -> tuple[ObservationBatch, int, int]:
    """Fetch all requested windows for a single region under concurrency and byte caps."""
    if concurrency < 1 or concurrency > MAX_CONCURRENCY:
        raise ValueError(f"concurrency must be in 1..{MAX_CONCURRENCY}, got {concurrency}")

    batches: list[ObservationBatch] = []
    region_bytes = 0
    errors = 0

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(_fetch_one_window, region, end_h, w_hrs, key=key): end_h for end_h, w_hrs in windows
        }
        done = 0
        for fut in as_completed(futures):
            done += 1
            try:
                batch, n_bytes = fut.result()
            except Exception as exc:  # noqa: BLE001
                errors += 1
                # _fetch_one_window raises a RuntimeError whose text is region, time and class name.
                detail = str(exc) if type(exc) is RuntimeError else type(exc).__name__
                typer.echo(f"  warning ({region.name}): {detail}", err=True)
                continue

            with byte_budget_lock:
                byte_counter[0] += n_bytes
                total_so_far = byte_counter[0]
            region_bytes += n_bytes
            batches.append(batch)

            if total_so_far > hard_cap_bytes:
                for pending in futures:
                    pending.cancel()
                raise RuntimeError(
                    f"hard transfer cap exceeded ({total_so_far:,} > {hard_cap_bytes:,} bytes); aborted"
                )

            if done % 24 == 0 or done == len(windows):
                typer.echo(
                    f"  {region.name}: {done}/{len(windows)} windows "
                    f"({region_bytes / (1024 * 1024):.1f} MiB, errors={errors})"
                )

    merged = merge_citizen_batches(batches)
    return merged, region_bytes, errors


@app.command()
def main(
    start_date: str = typer.Option("2024-07-08", help="Start date YYYY-MM-DD (UTC)."),
    end_date: str = typer.Option("2024-07-21", help="End date YYYY-MM-DD inclusive (UTC, < 2025-01-01)."),
    data_dir: Path | None = typer.Option(None, help="Base data dir (default WRFCOMP_DATA_DIR)."),
    concurrency: int = typer.Option(MAX_CONCURRENCY, help="Concurrent HTTP requests (max 4)."),
    sample_windows: int | None = typer.Option(
        None, help="Fetch only the first N 2-hour windows per region to estimate size."
    ),
    overwrite: bool = typer.Option(False, help="Re-fetch even if anonymized NetCDF already exists."),
) -> None:
    """Download anonymized CWOP citizen-station sample for Amarillo TX and Seattle WA."""
    if concurrency < 1 or concurrency > MAX_CONCURRENCY:
        raise typer.BadParameter(f"concurrency must be between 1 and {MAX_CONCURRENCY}")

    s_date = dt.date.fromisoformat(start_date)
    e_date = dt.date.fromisoformat(end_date)
    all_windows = iter_windows(s_date, e_date)
    windows = all_windows[:sample_windows] if sample_windows is not None else all_windows

    out_dir = (data_dir.expanduser() if data_dir is not None else default_data_dir()) / "citizen"
    out_dir.mkdir(parents=True, exist_ok=True)
    key = load_or_create_station_key(out_dir)

    lock = threading.Lock()
    byte_counter = [0]
    summary: dict[str, object] = {
        "source": "NOAA MADIS public surface CGI (madisXmlPublicDir, APRSWXNET)",
        "start_date": start_date,
        "end_date": end_date,
        "windows_requested_per_region": len(windows),
        "hours_covered_per_region": sum(w for _, w in windows),
        "regions": {},
    }

    t0 = time.monotonic()
    for region in DEFAULT_REGIONS:
        nc_path = out_dir / f"{region.name}_{s_date.strftime('%Y%m%d')}_{e_date.strftime('%Y%m%d')}.nc"
        if sample_windows is None and nc_path.exists() and not overwrite:
            typer.echo(f"{region.name}: loading cached anonymized archive {nc_path.name}")
            batch = load_citizen_batch(nc_path)
            r_bytes = 0
            r_errors = 0
        else:
            typer.echo(
                f"Fetching {region.name} ({len(windows)} 2-hour windows, concurrency={concurrency})..."
            )
            batch, r_bytes, r_errors = fetch_region(
                region,
                windows,
                key=key,
                concurrency=concurrency,
                byte_budget_lock=lock,
                byte_counter=byte_counter,
            )
            if sample_windows is None:
                save_citizen_batch(batch, nc_path)

        stats = summarize_batch(batch)
        stats["download_bytes"] = r_bytes
        stats["errors"] = r_errors
        if sample_windows is None and nc_path.exists():
            stats["disk_bytes"] = nc_path.stat().st_size
        summary["regions"][region.name] = stats  # type: ignore[index]
        typer.echo(
            f"Summary {region.name}: stations={stats['n_stations']}, reports={stats['n_reports']}, "
            f"obs={stats['n_observations']}, median_interval_min={stats['median_interval_min']}, "
            f"transfer={r_bytes / (1024 * 1024):.2f} MiB"
        )

    elapsed = time.monotonic() - t0
    summary["total_download_bytes"] = byte_counter[0]
    summary["elapsed_seconds"] = round(elapsed, 2)

    if sample_windows is not None:
        proj_mib = (byte_counter[0] / max(len(windows), 1)) * len(all_windows) / (1024 * 1024)
        typer.echo(
            f"Sample ({len(windows)}x2h/region): {byte_counter[0] / (1024 * 1024):.2f} MiB in {elapsed:.1f}s "
            f"-> projected {len(all_windows)}x2h total: ~{proj_mib:.1f} MiB"
        )
        return

    summary_path = out_dir / f"summary_{s_date.strftime('%Y%m%d')}_{e_date.strftime('%Y%m%d')}.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    typer.echo(
        f"Completed in {elapsed:.1f}s ({byte_counter[0] / (1024 * 1024):.2f} MiB transferred). "
        f"Aggregate summary written to {summary_path}"
    )


if __name__ == "__main__":
    app()
