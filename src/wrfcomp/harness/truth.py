"""NOAA ISD Global-Hourly station truth parser and dataset builder.

Converts raw NOAA Integrated Surface Database (ISD) global-hourly station-year CSV files
into the canonical WRFComp hourly truth ``xarray.Dataset`` defined in
``docs/VERIFICATION_PROTOCOL.md`` and ``docs/TRUTH_SOURCES.md``.
"""

from __future__ import annotations

import gzip
import math
import random
import time
import urllib.error
import urllib.request
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TextIO

import numpy as np
import pandas as pd
import xarray as xr
import yaml

PRIMARY_VARS: tuple[str, ...] = ("t2m", "td2m", "u10", "v10", "msl")
OFFSET_VARS: tuple[str, ...] = tuple(f"{v}_offset_min" for v in PRIMARY_VARS)
OFFSET_FILL_VALUE: np.int16 = np.int16(-32768)

VAR_UNITS: dict[str, str] = {
    "t2m": "K",
    "td2m": "K",
    "u10": "m s-1",
    "v10": "m s-1",
    "msl": "Pa",
}

# NOAA ISD Format Document (isd-format-document.pdf, pp. 7-12):
#   0 = Passed gross limits check
#   1 = Passed all quality control checks
#   4 = Passed gross limits check, data originate from an NCEI data source
#   5 = Passed all quality control checks, data originate from an NCEI data source
#   9 = Passed gross limits check if element is present (only accepted for non-missing values)
#   A = Data value flagged as suspect, but accepted as a good value (NCEI Interactive QC)
#   C = AWOS temperature/dewpoint reported in whole degrees Celsius; automated QC flags but accepts as valid
PASSING_QC_CODES: frozenset[str] = frozenset({"0", "1", "4", "5", "9", "A", "C"})
PASSING_QC_CODES_NO_AWOS_C: frozenset[str] = frozenset({"0", "1", "4", "5", "9", "A"})

# NCEI's legacy /data/global-hourly/access/ path redirects to an HTML S3 Explorer page;
# the official NOAA Open Data Dissemination (NODD) S3 bucket serves station-year CSVs directly.
AWS_GLOBAL_HOURLY_BASE_URL: str = "https://noaa-global-hourly-pds.s3.amazonaws.com"
NCEI_GLOBAL_HOURLY_BASE_URL: str = "https://www.ncei.noaa.gov/data/global-hourly/access"

SEALED_TEST_CUTOFF: pd.Timestamp = pd.Timestamp("2025-01-01T00:00:00")
MAX_CONCURRENT_REQUESTS: int = 4

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_SPLITS_PATH = _REPO_ROOT / "configs" / "splits.yaml"
_DEFAULT_STATIONS_PATH = _REPO_ROOT / "configs" / "stations_conus.csv"
_REQUIRED_CSV_COLS = ["DATE", "TMP", "DEW", "SLP", "WND"]
_RETRYABLE_HTTP_CODES = frozenset({429, 500, 502, 503, 504})


def load_splits_config(splits_path: Path | None = None) -> dict:
    """Load verification split and tolerance configuration from ``configs/splits.yaml``."""
    path = splits_path or _DEFAULT_SPLITS_PATH
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def get_truth_tolerance_minutes(splits_path: Path | None = None) -> int:
    """Return ``evaluation.truth_tolerance_minutes`` from ``configs/splits.yaml``."""
    cfg = load_splits_config(splits_path)
    return int(cfg["evaluation"]["truth_tolerance_minutes"])


def load_stations_metadata(stations_csv: Path | None = None) -> pd.DataFrame:
    """Load station catalog (``configs/stations_conus.csv``) keyed by 11-char ``station_id``."""
    path = stations_csv or _DEFAULT_STATIONS_PATH
    df = pd.read_csv(path, dtype={"station_id": str})
    df["station_id"] = df["station_id"].str.strip()
    return df


def _to_naive_utc_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def _enforce_pre_sealed_cutoff(ts: pd.Timestamp, label: str = "timestamp") -> None:
    if ts >= SEALED_TEST_CUTOFF:
        raise ValueError(
            f"Refusing {label} {ts.isoformat()}: sealed test split starts at "
            f"{SEALED_TEST_CUTOFF.isoformat()} and is strictly off-limits."
        )


def _enforce_pre_sealed_year(year: int) -> None:
    if year >= SEALED_TEST_CUTOFF.year:
        raise ValueError(
            f"Refusing year {year}: sealed test split starts at "
            f"{SEALED_TEST_CUTOFF.year} and is strictly off-limits."
        )


def wind_to_uv(direction_deg: float, speed_ms: float) -> tuple[float, float]:
    """Convert meteorological wind direction (FROM, deg clockwise from N) and speed (m/s) to (u10, v10).

    Example: north wind (from 0 deg or 360 deg) at 10 m/s blows southward -> ``u10 = 0.0, v10 = -10.0``.
    """
    if math.isnan(direction_deg) or math.isnan(speed_ms) or speed_ms < 0.0:
        return (math.nan, math.nan)
    if speed_ms == 0.0:
        return (0.0, 0.0)
    rad = math.radians(direction_deg % 360.0)
    u = -speed_ms * math.sin(rad)
    v = -speed_ms * math.cos(rad)
    if abs(u) < 1e-12:
        u = 0.0
    if abs(v) < 1e-12:
        v = 0.0
    return (u, v)


def parse_tmp_or_dew(raw: str) -> float:
    """Parse an ISD ``TMP`` or ``DEW`` field (``<signed_tenths_C>,<qc>``) to Kelvin."""
    if not raw or "," not in raw:
        return math.nan
    parts = raw.split(",")
    if len(parts) != 2:
        return math.nan
    val_str, qc = parts[0].strip(), parts[1].strip()
    if qc not in PASSING_QC_CODES or not val_str:
        return math.nan
    try:
        raw_int = int(val_str)
    except ValueError:
        return math.nan
    if abs(raw_int) == 9999:
        return math.nan
    return raw_int / 10.0 + 273.15


def parse_slp(raw: str) -> float:
    """Parse an ISD ``SLP`` field (``<tenths_hPa>,<qc>``) to Pascals."""
    if not raw or "," not in raw:
        return math.nan
    parts = raw.split(",")
    if len(parts) != 2:
        return math.nan
    val_str, qc = parts[0].strip(), parts[1].strip()
    if qc not in PASSING_QC_CODES_NO_AWOS_C or not val_str:
        return math.nan
    try:
        raw_int = int(val_str)
    except ValueError:
        return math.nan
    if raw_int == 99999 or raw_int <= 0:
        return math.nan
    return raw_int * 10.0


def parse_wnd(raw: str) -> tuple[float, float]:
    """Parse an ISD ``WND`` field (``<dir>,<dir_qc>,<type>,<spd>,<spd_qc>``) to ``(u10, v10)``.

    Handling rules (per NOAA ISD Format Document, p. 8):
    - Calm wind: speed ``0000`` with passing speed QC and type ``C`` (or type ``9``, which the ISD
      specification explicitly notes indicates calm winds when speed is ``0000``) -> ``(0.0, 0.0)``
      even though direction is coded as ``999`` with direction QC ``9``.
    - Variable wind: type ``V`` -> ``(nan, nan)`` because wind direction is undefined.
    - All other types (e.g. ``N`` normal, ``H``, ``R``, ``Q``): require both passing direction QC
      (``0 <= dir <= 360``) and passing speed QC (``speed != 9999``).
    """
    if not raw:
        return (math.nan, math.nan)
    parts = raw.split(",")
    if len(parts) != 5:
        return (math.nan, math.nan)
    dir_str, dir_qc, type_code, spd_str, spd_qc = (p.strip() for p in parts)
    if spd_qc not in PASSING_QC_CODES_NO_AWOS_C or not spd_str:
        return (math.nan, math.nan)
    try:
        spd_int = int(spd_str)
    except ValueError:
        return (math.nan, math.nan)
    if spd_int == 9999 or spd_int < 0:
        return (math.nan, math.nan)

    # Calm wind
    if spd_int == 0 and type_code in {"C", "9"}:
        return (0.0, 0.0)

    # Variable wind direction: vector components cannot be resolved
    if type_code == "V":
        return (math.nan, math.nan)

    if dir_qc not in PASSING_QC_CODES_NO_AWOS_C or not dir_str:
        return (math.nan, math.nan)
    try:
        dir_int = int(dir_str)
    except ValueError:
        return (math.nan, math.nan)
    if dir_int == 999 or dir_int < 0 or dir_int > 360:
        return (math.nan, math.nan)

    return wind_to_uv(float(dir_int), spd_int / 10.0)


def _parse_temp_series(series: pd.Series) -> np.ndarray:
    """Vectorized equivalent of :func:`parse_tmp_or_dew` returning float32 Kelvin."""
    n = len(series)
    out = np.full(n, np.nan, dtype=np.float32)
    if n == 0:
        return out
    parts = series.str.split(",", n=1, expand=True)
    if parts.shape[1] < 2:
        return out
    vals = pd.to_numeric(parts[0].str.strip(), errors="coerce")
    qcs = parts[1].str.strip()
    valid = vals.notna() & (vals.abs() != 9999) & qcs.isin(PASSING_QC_CODES)
    if valid.any():
        out[valid.to_numpy()] = (vals[valid].to_numpy(dtype=np.float64) / 10.0 + 273.15).astype(np.float32)
    return out


def _parse_slp_series(series: pd.Series) -> np.ndarray:
    """Vectorized equivalent of :func:`parse_slp` returning float32 Pascals."""
    n = len(series)
    out = np.full(n, np.nan, dtype=np.float32)
    if n == 0:
        return out
    parts = series.str.split(",", n=1, expand=True)
    if parts.shape[1] < 2:
        return out
    vals = pd.to_numeric(parts[0].str.strip(), errors="coerce")
    qcs = parts[1].str.strip()
    valid = vals.notna() & (vals != 99999) & (vals > 0) & qcs.isin(PASSING_QC_CODES_NO_AWOS_C)
    if valid.any():
        out[valid.to_numpy()] = (vals[valid].to_numpy(dtype=np.float64) * 10.0).astype(np.float32)
    return out


def _parse_wnd_series(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized equivalent of :func:`parse_wnd` returning ``(u10, v10)`` as float32 m/s."""
    n = len(series)
    u_out = np.full(n, np.nan, dtype=np.float32)
    v_out = np.full(n, np.nan, dtype=np.float32)
    if n == 0:
        return u_out, v_out

    parts = series.str.split(",", expand=True)
    if parts.shape[1] < 5:
        return u_out, v_out

    dir_vals = pd.to_numeric(parts[0].str.strip(), errors="coerce")
    dir_qcs = parts[1].str.strip()
    type_codes = parts[2].str.strip()
    spd_vals = pd.to_numeric(parts[3].str.strip(), errors="coerce")
    spd_qcs = parts[4].str.strip()

    # Rows with extra commas (more than 5 fields) are malformed
    exact_five = series.str.count(",") == 4

    spd_valid = (
        exact_five
        & spd_vals.notna()
        & (spd_vals != 9999)
        & (spd_vals >= 0)
        & spd_qcs.isin(PASSING_QC_CODES_NO_AWOS_C)
    )

    calm_mask = (spd_valid & (spd_vals == 0) & type_codes.isin({"C", "9"})).to_numpy()
    if calm_mask.any():
        u_out[calm_mask] = 0.0
        v_out[calm_mask] = 0.0

    dir_valid = (
        spd_valid
        & (~calm_mask)
        & (type_codes != "V")
        & dir_vals.notna()
        & (dir_vals != 999)
        & (dir_vals >= 0)
        & (dir_vals <= 360)
        & dir_qcs.isin(PASSING_QC_CODES_NO_AWOS_C)
    ).to_numpy()

    if dir_valid.any():
        speed_ms = spd_vals.to_numpy(dtype=np.float64)[dir_valid] / 10.0
        dir_deg = np.mod(dir_vals.to_numpy(dtype=np.float64)[dir_valid], 360.0)
        rad = np.deg2rad(dir_deg)
        u = -speed_ms * np.sin(rad)
        v = -speed_ms * np.cos(rad)
        u = np.where(np.abs(u) < 1e-12, 0.0, u)
        v = np.where(np.abs(v) < 1e-12, 0.0, v)
        # Zero speed with valid direction gives exact (0.0, 0.0)
        zero_spd = speed_ms == 0.0
        u = np.where(zero_spd, 0.0, u)
        v = np.where(zero_spd, 0.0, v)
        u_out[dir_valid] = u.astype(np.float32)
        v_out[dir_valid] = v.astype(np.float32)

    return u_out, v_out


def parse_isd_csv(source: Path | str | TextIO) -> pd.DataFrame:
    """Parse a NOAA ISD global-hourly station-year CSV into canonical units.

    Returns a DataFrame with columns ``obs_time`` (timezone-naive UTC ``datetime64[ns]``)
    and ``t2m``, ``td2m``, ``u10``, ``v10``, ``msl`` (``float32``). Observations from
    ``2025-01-01T00:00:00`` onward are unconditionally excluded.
    """
    df = pd.read_csv(
        source,
        usecols=_REQUIRED_CSV_COLS,
        dtype=str,
        keep_default_na=False,
    )
    if df.empty:
        return pd.DataFrame(
            {
                "obs_time": pd.Series(dtype="datetime64[ns]"),
                **{v: pd.Series(dtype="float32") for v in PRIMARY_VARS},
            }
        )

    obs_time = pd.to_datetime(df["DATE"], format="ISO8601", errors="coerce", utc=False)
    if obs_time.dt.tz is not None:
        obs_time = obs_time.dt.tz_convert("UTC").dt.tz_localize(None)
    obs_time = obs_time.astype("datetime64[ns]")

    valid_time_mask = obs_time.notna() & (obs_time < SEALED_TEST_CUTOFF)
    if not valid_time_mask.all():
        df = df.loc[valid_time_mask].reset_index(drop=True)
        obs_time = obs_time.loc[valid_time_mask].reset_index(drop=True)

    t2m = _parse_temp_series(df["TMP"])
    td2m = _parse_temp_series(df["DEW"])
    msl = _parse_slp_series(df["SLP"])
    u10, v10 = _parse_wnd_series(df["WND"])

    # Physical consistency (docs/VERIFICATION_PROTOCOL.md): td2m <= t2m in truth
    bad_dew = np.isfinite(t2m) & np.isfinite(td2m) & (td2m > t2m)
    if bad_dew.any():
        td2m[bad_dew] = np.nan

    return pd.DataFrame(
        {
            "obs_time": obs_time,
            "t2m": t2m,
            "td2m": td2m,
            "u10": u10,
            "v10": v10,
            "msl": msl,
        }
    )


def match_station_to_hours(
    obs_df: pd.DataFrame,
    valid_times: pd.DatetimeIndex,
    tolerance_minutes: int | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Match a single station's parsed observations to full hourly ``valid_times``.

    For each variable independently, selects the QC-passing observation nearest to each
    ``valid_time`` within ``±tolerance_minutes``. Exact distance ties pick the earlier
    report. Missing hours are ``NaN`` (never interpolated) and carry offset ``-32768``.

    Returns ``(values_by_var, offsets_by_var)`` where values are ``float32`` arrays of
    shape ``(len(valid_times),)`` and offsets are ``int16`` arrays of whole minutes
    (``measurement_time - valid_time`` in ``[-tolerance, +tolerance]`` or ``-32768``).
    """
    tol_min = get_truth_tolerance_minutes() if tolerance_minutes is None else int(tolerance_minutes)
    tol_ns = np.int64(tol_min) * np.int64(60_000_000_000)

    target_ns = valid_times.astype("datetime64[ns]").asi8
    n_times = len(target_ns)

    values: dict[str, np.ndarray] = {}
    offsets: dict[str, np.ndarray] = {}

    if obs_df.empty or n_times == 0:
        for var in PRIMARY_VARS:
            values[var] = np.full(n_times, np.nan, dtype=np.float32)
            offsets[var] = np.full(n_times, OFFSET_FILL_VALUE, dtype=np.int16)
        return values, offsets

    all_obs_ns = obs_df["obs_time"].astype("datetime64[ns]").to_numpy().view(np.int64)

    for var in PRIMARY_VARS:
        var_vals = obs_df[var].to_numpy(dtype=np.float32)
        finite_mask = np.isfinite(var_vals)
        if not finite_mask.any():
            values[var] = np.full(n_times, np.nan, dtype=np.float32)
            offsets[var] = np.full(n_times, OFFSET_FILL_VALUE, dtype=np.int16)
            continue

        obs_ns = all_obs_ns[finite_mask]
        obs_v = var_vals[finite_mask]

        # Stable sort by timestamp and keep first occurrence for identical timestamps
        order = np.argsort(obs_ns, kind="mergesort")
        obs_ns = obs_ns[order]
        obs_v = obs_v[order]
        if len(obs_ns) > 1:
            unique_mask = np.empty(len(obs_ns), dtype=bool)
            unique_mask[0] = True
            unique_mask[1:] = obs_ns[1:] != obs_ns[:-1]
            obs_ns = obs_ns[unique_mask]
            obs_v = obs_v[unique_mask]

        n_obs = len(obs_ns)
        right_idx = np.searchsorted(obs_ns, target_ns, side="left")
        left_idx = right_idx - 1

        has_left = left_idx >= 0
        has_right = right_idx < n_obs

        max_int64 = np.iinfo(np.int64).max
        dist_left = np.full(n_times, max_int64, dtype=np.int64)
        dist_right = np.full(n_times, max_int64, dtype=np.int64)

        if has_left.any():
            dist_left[has_left] = target_ns[has_left] - obs_ns[left_idx[has_left]]
        if has_right.any():
            dist_right[has_right] = obs_ns[right_idx[has_right]] - target_ns[has_right]

        # Ties (dist_left == dist_right) pick the earlier report (left_idx).
        choose_left = has_left & (dist_left <= dist_right)
        chosen_idx = np.where(choose_left, left_idx, right_idx)
        min_dist = np.where(choose_left, dist_left, dist_right)
        matched = (has_left | has_right) & (min_dist <= tol_ns)

        out_val = np.full(n_times, np.nan, dtype=np.float32)
        out_off = np.full(n_times, OFFSET_FILL_VALUE, dtype=np.int16)
        if matched.any():
            sel = chosen_idx[matched]
            out_val[matched] = obs_v[sel]
            diff_ns = obs_ns[sel] - target_ns[matched]
            out_off[matched] = np.rint(diff_ns / 60_000_000_000.0).astype(np.int16)

        values[var] = out_val
        offsets[var] = out_off

    # Enforce td2m <= t2m after independent per-variable matching
    bad_dew = np.isfinite(values["t2m"]) & np.isfinite(values["td2m"]) & (values["td2m"] > values["t2m"])
    if bad_dew.any():
        values["td2m"][bad_dew] = np.nan
        offsets["td2m"][bad_dew] = OFFSET_FILL_VALUE

    return values, offsets


def _is_valid_cached_csv(path: Path) -> bool:
    """Return True if ``path`` exists, is non-empty, and starts with an ISD CSV header (not HTML)."""
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
            first_line = fh.readline(4096)
    except OSError:
        return False
    upper = first_line.upper()
    return ("DATE" in upper) and ("<HTML" not in upper) and ("<!DOCTYPE" not in upper)


def download_station_year(
    station_id: str,
    year: int,
    cache_dir: Path,
    *,
    base_url: str = AWS_GLOBAL_HOURLY_BASE_URL,
    max_retries: int = 4,
    timeout: float = 60.0,
) -> Path:
    """Download a single NOAA ISD global-hourly station-year CSV with retries and resumability."""
    _enforce_pre_sealed_year(year)
    year_dir = Path(cache_dir) / str(year)
    year_dir.mkdir(parents=True, exist_ok=True)
    dest = year_dir / f"{station_id}.csv"
    if _is_valid_cached_csv(dest):
        return dest

    url = f"{base_url.rstrip('/')}/{year}/{station_id}.csv"
    part = year_dir / f".{station_id}.csv.part"

    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "wrfcomp-truth-loader/0.1"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            first_line = data[:2048].decode("utf-8", errors="replace").splitlines()[0] if data else ""
            if "<html" in first_line.lower() or "<!doctype" in first_line.lower() or "DATE" not in first_line:
                raise RuntimeError(
                    f"Unexpected non-CSV response from {url} (got header: {first_line[:80]!r})"
                )
            part.write_bytes(data)
            part.replace(dest)
            return dest
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise FileNotFoundError(f"Station-year not found (HTTP 404): {url}") from exc
            last_exc = exc
            if exc.code not in _RETRYABLE_HTTP_CODES or attempt == max_retries:
                raise
        except (urllib.error.URLError, TimeoutError, OSError, RuntimeError) as exc:
            last_exc = exc
            if attempt == max_retries:
                raise
        finally:
            if part.exists():
                part.unlink(missing_ok=True)

        backoff = min(30.0, (2.0**attempt) + random.uniform(0.1, 0.6))
        time.sleep(backoff)

    raise RuntimeError(f"Failed downloading {url}: {last_exc}")


def download_stations(
    station_ids: Sequence[str],
    years: Sequence[int],
    cache_dir: Path,
    *,
    max_workers: int = MAX_CONCURRENT_REQUESTS,
    base_url: str = AWS_GLOBAL_HOURLY_BASE_URL,
    max_retries: int = 4,
) -> tuple[list[tuple[str, int]], list[tuple[str, int, str]]]:
    """Download station-year CSVs with at most 4 concurrent requests and retries.

    Returns ``(succeeded, failed)`` where ``failed`` entries are ``(station_id, year, error_message)``.
    """
    for year in years:
        _enforce_pre_sealed_year(year)
    workers = min(max(1, int(max_workers)), MAX_CONCURRENT_REQUESTS)

    tasks = [(sid, yr) for yr in years for sid in station_ids]
    succeeded: list[tuple[str, int]] = []
    failed: list[tuple[str, int, str]] = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {
            pool.submit(
                download_station_year,
                sid,
                yr,
                cache_dir,
                base_url=base_url,
                max_retries=max_retries,
            ): (sid, yr)
            for sid, yr in tasks
        }
        for fut in as_completed(future_map):
            sid, yr = future_map[fut]
            try:
                fut.result()
                succeeded.append((sid, yr))
            except Exception as exc:
                failed.append((sid, yr, str(exc)))

    succeeded.sort()
    failed.sort()
    return succeeded, failed


def _find_cached_station_file(cache_dir: Path, station_id: str, year: int) -> Path | None:
    """Locate a cached station-year CSV (plain or gzipped) in ``cache_dir``."""
    candidates = [
        cache_dir / str(year) / f"{station_id}.csv",
        cache_dir / str(year) / f"{station_id}.csv.gz",
        cache_dir / f"{station_id}_{year}.csv",
        cache_dir / f"{station_id}.csv",
    ]
    for path in candidates:
        if _is_valid_cached_csv(path):
            return path
    return None


def _extract_csv_station_coords(path: Path) -> tuple[float, float, float]:
    """Read LATITUDE, LONGITUDE, ELEVATION from the first valid row of a station CSV."""
    try:
        df = pd.read_csv(path, usecols=["LATITUDE", "LONGITUDE", "ELEVATION"], nrows=5)
        lat = float(pd.to_numeric(df["LATITUDE"], errors="coerce").dropna().iloc[0])
        lon = float(pd.to_numeric(df["LONGITUDE"], errors="coerce").dropna().iloc[0])
        elev = float(pd.to_numeric(df["ELEVATION"], errors="coerce").dropna().iloc[0])
        return (lat, lon, elev)
    except Exception:
        return (math.nan, math.nan, math.nan)


def _resolve_station_coords(
    station_ids: list[str],
    stations_metadata: pd.DataFrame | None,
    fallback_files: dict[str, Path | None],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if stations_metadata is None and _DEFAULT_STATIONS_PATH.is_file():
        stations_metadata = load_stations_metadata(_DEFAULT_STATIONS_PATH)

    lookup: dict[str, tuple[float, float, float]] = {}
    if stations_metadata is not None and not stations_metadata.empty:
        lat_col = "latitude" if "latitude" in stations_metadata.columns else "lat"
        lon_col = "longitude" if "longitude" in stations_metadata.columns else "lon"
        for row in stations_metadata.itertuples(index=False):
            sid = str(row.station_id).strip()
            lookup[sid] = (
                float(getattr(row, lat_col)),
                float(getattr(row, lon_col)),
                float(row.elevation_m),
            )

    n = len(station_ids)
    lats = np.full(n, np.nan, dtype=np.float32)
    lons = np.full(n, np.nan, dtype=np.float32)
    elevs = np.full(n, np.nan, dtype=np.float32)

    for i, sid in enumerate(station_ids):
        if sid in lookup:
            lat, lon, elev = lookup[sid]
        elif fallback_files.get(sid) is not None:
            lat, lon, elev = _extract_csv_station_coords(fallback_files[sid])  # type: ignore[arg-type]
        else:
            lat, lon, elev = (math.nan, math.nan, math.nan)
        lats[i] = np.float32(lat)
        lons[i] = np.float32(lon)
        elevs[i] = np.float32(elev)

    return lats, lons, elevs


def build_truth(
    station_ids: Iterable[str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    cache_dir: Path | str,
    *,
    stations_metadata: pd.DataFrame | None = None,
    tolerance_minutes: int | None = None,
    parsed_cache_dir: Path | str | None = None,
) -> xr.Dataset:
    """Build a contract-compliant hourly truth ``xr.Dataset`` from cached NOAA ISD CSVs.

    Parameters
    ----------
    station_ids:
        Sequence of 11-character ISD ``USAF+WBAN`` station identifiers.
    start, end:
        Inclusive UTC hourly bounds (timezone-naive or UTC-aware). Must be strictly
        before ``2025-01-01T00:00:00``.
    cache_dir:
        Directory containing cached station-year CSV files (``<cache_dir>/<year>/<station_id>.csv``).
    stations_metadata:
        Optional DataFrame with ``station_id``, ``lat``/``latitude``, ``lon``/``longitude``,
        ``elevation_m``. Defaults to ``configs/stations_conus.csv``.
    tolerance_minutes:
        Matching half-window in minutes. Defaults to ``evaluation.truth_tolerance_minutes``
        from ``configs/splits.yaml`` (15).
    parsed_cache_dir:
        Optional directory holding pre-parsed per-station-year ``.npz`` files (used by
        ``scripts/build_truth.py --discard-raw``).
    """
    sids = [str(s).strip() for s in station_ids]
    if not sids:
        raise ValueError("station_ids must not be empty.")
    bad_sids = [s for s in sids if len(s) != 11]
    if bad_sids:
        raise ValueError(f"station_id must be 11 characters (USAF+WBAN); invalid: {bad_sids[:5]}")

    start_ts = _to_naive_utc_timestamp(start)
    end_ts = _to_naive_utc_timestamp(end)
    if end_ts < start_ts:
        raise ValueError(f"end ({end_ts}) must be >= start ({start_ts}).")
    _enforce_pre_sealed_cutoff(start_ts, "start")
    _enforce_pre_sealed_cutoff(end_ts, "end")

    tol_min = get_truth_tolerance_minutes() if tolerance_minutes is None else int(tolerance_minutes)
    valid_times = pd.date_range(start_ts, end_ts, freq="h")

    cache_path = Path(cache_dir)
    parsed_path = Path(parsed_cache_dir) if parsed_cache_dir is not None else None

    years = list(range(start_ts.year, end_ts.year + 1))
    # Include prior year if start is within tolerance of Jan 1 00:00 (e.g. Dec 31 23:53 -> Jan 1 00:00)
    prev_year_needed = (start_ts - pd.Timedelta(minutes=tol_min)).year < start_ts.year
    lookback_years = [start_ts.year - 1, *years] if prev_year_needed else years

    n_times = len(valid_times)
    n_stations = len(sids)

    data_arrays: dict[str, np.ndarray] = {
        v: np.full((n_times, n_stations), np.nan, dtype=np.float32) for v in PRIMARY_VARS
    }
    offset_arrays: dict[str, np.ndarray] = {
        v: np.full((n_times, n_stations), OFFSET_FILL_VALUE, dtype=np.int16) for v in PRIMARY_VARS
    }
    fallback_files: dict[str, Path | None] = {}

    # Fast path: assemble from pre-parsed per-station-year .npz files when exact full years match
    can_use_parsed_npz = (
        parsed_path is not None
        and start_ts == pd.Timestamp(year=start_ts.year, month=1, day=1, hour=0)
        and end_ts == pd.Timestamp(year=end_ts.year, month=12, day=31, hour=23)
    )

    for j, sid in enumerate(sids):
        used_npz = False
        if can_use_parsed_npz and parsed_path is not None:
            npz_files = [parsed_path / str(yr) / f"{sid}.npz" for yr in years]
            if all(p.is_file() for p in npz_files):
                cursor = 0
                for p in npz_files:
                    with np.load(p) as z:
                        nt = len(z["t2m"])
                        for var in PRIMARY_VARS:
                            data_arrays[var][cursor : cursor + nt, j] = z[var]
                            offset_arrays[var][cursor : cursor + nt, j] = z[f"{var}_offset_min"]
                        cursor += nt
                used_npz = cursor == n_times

        if used_npz:
            fallback_files[sid] = None
            continue

        frames: list[pd.DataFrame] = []
        seen_paths: set[Path] = set()
        first_file: Path | None = None
        for yr in lookback_years:
            fpath = _find_cached_station_file(cache_path, sid, yr)
            if fpath is not None and fpath not in seen_paths:
                seen_paths.add(fpath)
                if first_file is None:
                    first_file = fpath
                parsed = parse_isd_csv(fpath)
                if not parsed.empty:
                    # Trim to [start - tol, end + tol] to keep memory small
                    lo = start_ts - pd.Timedelta(minutes=tol_min)
                    hi = min(
                        end_ts + pd.Timedelta(minutes=tol_min),
                        SEALED_TEST_CUTOFF - pd.Timedelta(seconds=1),
                    )
                    sub = parsed[(parsed["obs_time"] >= lo) & (parsed["obs_time"] <= hi)]
                    if not sub.empty:
                        frames.append(sub)

        fallback_files[sid] = first_file
        if not frames:
            continue

        obs_df = frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)
        vals, offs = match_station_to_hours(obs_df, valid_times, tolerance_minutes=tol_min)
        for var in PRIMARY_VARS:
            data_arrays[var][:, j] = vals[var]
            offset_arrays[var][:, j] = offs[var]

    lats, lons, elevs = _resolve_station_coords(sids, stations_metadata, fallback_files)

    data_vars: dict[str, tuple[tuple[str, str], np.ndarray, dict[str, object]]] = {}
    for var in PRIMARY_VARS:
        data_vars[var] = (
            ("valid_time", "station_id"),
            data_arrays[var],
            {"units": VAR_UNITS[var]},
        )
    for var in PRIMARY_VARS:
        data_vars[f"{var}_offset_min"] = (
            ("valid_time", "station_id"),
            offset_arrays[var],
            {
                "units": "minutes",
                "long_name": f"Observation time offset from valid_time for {var} (-32768 when missing)",
            },
        )

    protocol_version = str(load_splits_config().get("protocol_version", "0.1.0"))
    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "valid_time": valid_times.to_numpy(dtype="datetime64[ns]"),
            "station_id": np.array(sids, dtype="U11"),
            "latitude": ("station_id", lats, {"units": "degrees_north"}),
            "longitude": ("station_id", lons, {"units": "degrees_east"}),
            "elevation_m": ("station_id", elevs, {"units": "m"}),
        },
        attrs={
            "source": "NOAA ISD global-hourly (noaa-global-hourly-pds)",
            "protocol_version": protocol_version,
            "truth_tolerance_minutes": tol_min,
            "passing_qc_codes": ",".join(sorted(PASSING_QC_CODES)),
        },
    )

    validate_truth_dataset(ds, tolerance_minutes=tol_min)
    return ds


def validate_truth_dataset(ds: xr.Dataset, tolerance_minutes: int | None = None) -> None:
    """Validate that a truth ``xr.Dataset`` satisfies the WRFComp data contract and Addendum 1."""
    tol_min = get_truth_tolerance_minutes() if tolerance_minutes is None else int(tolerance_minutes)

    expected_dims = ("valid_time", "station_id")
    for dim in expected_dims:
        if dim not in ds.dims:
            raise ValueError(f"Truth dataset missing required dimension: {dim}")

    # Check valid_time coord: datetime64[ns], strictly hourly, strictly before 2025-01-01
    vt = ds["valid_time"].to_numpy()
    if not np.issubdtype(vt.dtype, np.datetime64):
        raise ValueError(f"valid_time must be datetime64[ns], got {vt.dtype}")
    vt_ns = vt.astype("datetime64[ns]")
    if len(vt_ns) > 0:
        if pd.Timestamp(vt_ns.max()) >= SEALED_TEST_CUTOFF:
            raise ValueError(f"valid_time reaches sealed test cutoff ({SEALED_TEST_CUTOFF})")
        if len(vt_ns) > 1:
            diffs = np.diff(vt_ns.view(np.int64))
            one_hour_ns = np.int64(3_600_000_000_000)
            if not np.all(diffs == one_hour_ns):
                raise ValueError("valid_time must be strictly contiguous full hours (1h step).")

    # Check station_id coord: 11-character strings
    sids = [str(s) for s in ds["station_id"].to_numpy()]
    if any(len(s) != 11 for s in sids):
        raise ValueError("All station_id entries must be 11-character ISD USAF+WBAN strings.")

    # Check required station coordinates
    for coord in ("latitude", "longitude", "elevation_m"):
        if coord not in ds.coords:
            raise ValueError(f"Truth dataset missing required coordinate: {coord}")
        if ds.coords[coord].dims != ("station_id",):
            raise ValueError(
                f"Coordinate {coord} must have dims ('station_id',), got {ds.coords[coord].dims}"
            )

    # Check primary variables and companion offset variables
    for var in PRIMARY_VARS:
        if var not in ds.data_vars:
            raise ValueError(f"Truth dataset missing required variable: {var}")
        da = ds[var]
        if da.dims != expected_dims:
            raise ValueError(f"Variable {var} must have dims {expected_dims}, got {da.dims}")
        if da.dtype != np.float32:
            raise ValueError(f"Variable {var} must have dtype float32, got {da.dtype}")

        off_var = f"{var}_offset_min"
        if off_var not in ds.data_vars:
            raise ValueError(f"Truth dataset missing required companion variable: {off_var}")
        off_da = ds[off_var]
        if off_da.dims != expected_dims:
            raise ValueError(f"Variable {off_var} must have dims {expected_dims}, got {off_da.dims}")
        if off_da.dtype != np.int16:
            raise ValueError(f"Variable {off_var} must have dtype int16, got {off_da.dtype}")

        vals = da.to_numpy()
        offs = off_da.to_numpy()
        is_nan = np.isnan(vals)
        is_fill = offs == OFFSET_FILL_VALUE
        if not np.array_equal(is_nan, is_fill):
            raise ValueError(
                f"{off_var} must equal {int(OFFSET_FILL_VALUE)} wherever {var} is NaN and nowhere else."
            )
        valid_offs = offs[~is_fill]
        if len(valid_offs) > 0 and (valid_offs.min() < -tol_min or valid_offs.max() > tol_min):
            raise ValueError(
                f"{off_var} values outside [-{tol_min}, +{tol_min}]: "
                f"[{int(valid_offs.min())}, {int(valid_offs.max())}]"
            )

    # Physical consistency check: td2m <= t2m wherever both are finite
    t2m = ds["t2m"].to_numpy()
    td2m = ds["td2m"].to_numpy()
    both = np.isfinite(t2m) & np.isfinite(td2m)
    if np.any(td2m[both] > t2m[both]):
        raise ValueError("Truth dataset violates physical constraint td2m <= t2m.")


def write_truth_netcdf(ds: xr.Dataset, out: Path | str) -> None:
    """Write a validated truth Dataset atomically to NetCDF.

    Ensures ``X_offset_min`` variables are stored as plain ``int16`` with ``-32768``
    where ``X`` is ``NaN`` and no ``_FillValue`` attribute or encoding.
    """
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    encoding: dict[str, dict[str, object]] = {}
    for off_var in OFFSET_VARS:
        if off_var in ds.data_vars:
            ds[off_var].attrs.pop("_FillValue", None)
            ds[off_var].encoding.pop("_FillValue", None)
            encoding[off_var] = {"dtype": "int16", "_FillValue": None}

    tmp_out = out_path.with_name(f".{out_path.name}.part")
    try:
        ds.to_netcdf(tmp_out, encoding=encoding)
        tmp_out.replace(out_path)
    finally:
        if tmp_out.exists():
            tmp_out.unlink(missing_ok=True)


def compute_completeness(ds: xr.Dataset) -> dict[str, float]:
    """Return fraction of finite observations in ``[0.0, 1.0]`` for each primary variable."""
    result: dict[str, float] = {}
    for var in PRIMARY_VARS:
        arr = ds[var].to_numpy()
        result[var] = float(np.isfinite(arr).mean()) if arr.size > 0 else 0.0
    return result
