"""Fetch CONUS CWOP/APRSWXNET reports for the Phase 3 development year from the hourly MADIS archive.

One anonymised file per UTC day under ``<data dir>/citizen/conus/<year>/``. Each hourly archive file
is parsed in memory by :mod:`wrfcomp.assimilation.madis_archive` (restricted providers dropped first,
keyed station tokens, positions rounded); the transfer itself is never written to disk.

Guards, checked before every day:

- only the development year 2024 is accepted (the 2023 confirmation year is not fetched by this script);
- free space on the data filesystem must stay above the stop line (40 GB);
- the data directory must stay under the sprint cap (45 GB).

Log lines carry dates, counts, byte totals and exception class names only.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import typer

from wrfcomp.assimilation.citizen_obs import load_or_create_station_key
from wrfcomp.assimilation.madis_archive import ReportTable, merge_tables, parse_archive_hour, save_table

ARCHIVE_URL = (
    "https://madis-data.ncep.noaa.gov/madisPublic1/data/archive/"
    "{d:%Y/%m/%d}/LDAD/mesonet/netCDF/{d:%Y%m%d_%H}00.gz"
)
ALLOWED_YEARS = (2024,)
LAST_HOUR = dt.datetime(2024, 12, 17, 11)  # the hour before the last 2024 issue (12Z on 17 December)
MAX_CONCURRENCY = 4
STOP_FREE_BYTES = 40 * 10**9
DATA_CAP_BYTES = 45 * 10**9
DAY_HEADROOM_BYTES = 10**9

app = typer.Typer(help=__doc__)


class HourMissingError(Exception):
    """The archive has no file for this hour (HTTP 404)."""


def day_path(citizen_dir: Path, day: dt.date) -> Path:
    return citizen_dir / "conus" / f"{day.year:04d}" / f"{day.isoformat()}.nc"


def hours_of(day: dt.date) -> list[dt.datetime]:
    if day.year not in ALLOWED_YEARS:
        raise ValueError(f"refusing {day.year}: only {ALLOWED_YEARS} may be fetched by this script")
    hours = [dt.datetime(day.year, day.month, day.day, h) for h in range(24)]
    return [h for h in hours if h <= LAST_HOUR]


def data_dir_bytes(data_dir: Path) -> int:
    out = subprocess.run(["du", "-sb", str(data_dir)], capture_output=True, text=True, check=True)
    return int(out.stdout.split()[0])


def check_room(data_dir: Path) -> None:
    free = shutil.disk_usage(data_dir).free
    if free - DAY_HEADROOM_BYTES < STOP_FREE_BYTES:
        raise RuntimeError(f"free space {free / 1e9:.1f} GB is at the stop line; stopping")
    used = data_dir_bytes(data_dir)
    if used + DAY_HEADROOM_BYTES > DATA_CAP_BYTES:
        raise RuntimeError(f"data directory {used / 1e9:.1f} GB is at the cap; stopping")


def fetch_hour(
    hour: dt.datetime, *, key: bytes, cache: dict[str, str], retries: int = 4, timeout: float = 300.0
) -> tuple[ReportTable, int]:
    url = ARCHIVE_URL.format(d=hour)
    last: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "research-sample/0.1"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read()
            return parse_archive_hour(payload, key=key, token_cache=cache), len(payload)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise HourMissingError from None
            last = exc
        except Exception as exc:  # noqa: BLE001
            last = exc
        time.sleep(5.0 * (attempt + 1))
    # Class name only: a message could quote content.
    raise RuntimeError(f"failed {hour:%Y-%m-%d %H}Z: {type(last).__name__}") from None


def fetch_day(
    day: dt.date, citizen_dir: Path, *, key: bytes, cache: dict[str, str], concurrency: int
) -> dict:
    hours = hours_of(day)
    t0 = time.monotonic()
    tables: list[ReportTable] = []
    missing: list[str] = []
    transferred = 0

    def one(hour: dt.datetime):
        try:
            return hour, *fetch_hour(hour, key=key, cache=cache)
        except HourMissingError:
            return hour, None, 0

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for hour, table, n_bytes in pool.map(one, hours):
            transferred += n_bytes
            if table is None:
                missing.append(f"{hour:%H}")
            else:
                tables.append(table)
    merged = merge_tables(tables)
    path = day_path(citizen_dir, day)
    save_table(merged, path)
    return {
        "day": day.isoformat(),
        "hours_requested": len(hours),
        "hours_missing": missing,
        "reports": len(merged),
        "stations": int(len(set(merged.station_id.tolist()))),
        "transferred_bytes": transferred,
        "stored_bytes": path.stat().st_size,
        "seconds": round(time.monotonic() - t0, 1),
    }


@app.command()
def main(
    start_date: str = typer.Option(..., help="First UTC day, YYYY-MM-DD."),
    end_date: str = typer.Option(..., help="Last UTC day, inclusive."),
    data_dir: Path | None = typer.Option(None, help="Base data dir (default WRFCOMP_DATA_DIR)."),
    concurrency: int = typer.Option(MAX_CONCURRENCY, help="Concurrent downloads (max 4)."),
) -> None:
    if not 1 <= concurrency <= MAX_CONCURRENCY:
        raise typer.BadParameter(f"concurrency must be between 1 and {MAX_CONCURRENCY}")
    first, last = dt.date.fromisoformat(start_date), dt.date.fromisoformat(end_date)
    if first > last or first.year not in ALLOWED_YEARS or last.year not in ALLOWED_YEARS:
        raise typer.BadParameter(f"dates must be in order and inside {ALLOWED_YEARS}")
    root = (
        data_dir.expanduser()
        if data_dir
        else Path(os.environ.get("WRFCOMP_DATA_DIR") or Path.home() / "wrfcomp-data")
    )
    citizen_dir = root / "citizen"
    citizen_dir.mkdir(parents=True, exist_ok=True)
    key = load_or_create_station_key(citizen_dir)
    cache: dict[str, str] = {}  # raw identifier -> token, in memory only
    lock = threading.Lock()
    log_path = citizen_dir / "conus" / f"fetch_{first.year}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    day = first
    while day <= last:
        if not hours_of(day):
            break
        if day_path(citizen_dir, day).exists():
            day += dt.timedelta(days=1)
            continue
        check_room(root)
        record = fetch_day(day, citizen_dir, key=key, cache=cache, concurrency=concurrency)
        with lock, log_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
        typer.echo(
            f"{record['day']}: reports={record['reports']} stations={record['stations']} "
            f"missing_hours={len(record['hours_missing'])} "
            f"transferred={record['transferred_bytes'] / 2**20:.0f} MiB "
            f"stored={record['stored_bytes'] / 2**20:.1f} MiB in {record['seconds']:.0f}s"
        )
        day += dt.timedelta(days=1)
    typer.echo("done")


if __name__ == "__main__":
    app()
