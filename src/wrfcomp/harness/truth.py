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

from wrfcomp.harness.config import HarnessConfig, load_config
from wrfcomp.harness.schemas import validate_truth_dataset

PARSER_VERSION: str = "2"

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

# NOAA ISD Format Document (isd-format-document.pdf, pp. 8-12):
#   0 = Passed gross limits check
#   1 = Passed all quality control checks
#   4 = Passed gross limits check, data originate from an NCEI data source
#   5 = Passed all quality control checks, data originate from an NCEI data source
#   9 = Passed gross limits check if element is present (only accepted for non-missing values)
#   A = Data value flagged as suspect, but accepted as a good value (listed on TMP/DEW, pp. 10-11)
#   C = AWOS temperature/dewpoint reported in whole degrees Celsius; automated QC flags but accepts
#       as valid (listed on TMP/DEW, pp. 10-11)
PASSING_QC_CODES_TEMP_DEW: frozenset[str] = frozenset({"0", "1", "4", "5", "9", "A", "C"})
PASSING_QC_CODES_WIND_SLP: frozenset[str] = frozenset({"0", "1", "4", "5", "9"})
PASSING_QC_CODES: frozenset[str] = PASSING_QC_CODES_TEMP_DEW

# Report-type filtering (see docs/TRUTH_SOURCES.md):
# Use METAR-family reports (FM-15, FM-16) whenever a station-year has any, avoiding 3-hourly FM-12
# synoptic copies of the preceding :53 routine METAR. Fall back to FM-12 / SY-MT only for
# station-years with zero FM-15 / FM-16 rows.
METAR_REPORT_TYPES: frozenset[str] = frozenset({"FM-15", "FM-16"})
SYNOPTIC_FALLBACK_REPORT_TYPES: frozenset[str] = frozenset({"FM-12", "SY-MT"})

# Documented ISD physical bounds (isd-format-document.pdf, p. 8):
# Wind direction angle: MIN 001, MAX 360 (999 = missing).
# Wind speed rate: MIN 0000, MAX 0900 tenths of m/s (0.0 to 90.0 m/s; 9999 = missing).
WIND_DIR_MIN_DEG: int = 1
WIND_DIR_MAX_DEG: int = 360
WIND_SPEED_MAX_TENTHS_MS: int = 900

# NCEI's legacy /data/global-hourly/access/ path redirects to an HTML S3 Explorer page;
# the official NOAA Open Data Dissemination (NODD) S3 bucket serves station-year CSVs directly.
AWS_GLOBAL_HOURLY_BASE_URL: str = "https://noaa-global-hourly-pds.s3.amazonaws.com"
NCEI_GLOBAL_HOURLY_BASE_URL: str = "https://www.ncei.noaa.gov/data/global-hourly/access"

MAX_CONCURRENT_REQUESTS: int = 4

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_STATIONS_PATH = _REPO_ROOT / "configs" / "stations_conus.csv"
_REQUIRED_CSV_COLS = ("DATE", "TMP", "DEW", "SLP", "WND")
_OPTIONAL_CSV_COLS = ("REPORT_TYPE",)
_RETRYABLE_HTTP_CODES = frozenset({429, 500, 502, 503, 504})


def get_harness_config(config_path: Path | None = None) -> HarnessConfig:
    """Load the canonical harness configuration from ``configs/splits.yaml``."""
    return load_config(config_path)


def get_sealed_test_cutoff(cfg: HarnessConfig | None = None) -> pd.Timestamp:
    """Return the sealed-test start timestamp from ``configs/splits.yaml``."""
    resolved = cfg or load_config()
    return pd.Timestamp(resolved.splits.sealed_test.start)


def get_truth_tolerance_minutes(cfg: HarnessConfig | None = None) -> int:
    """Return ``evaluation.truth_tolerance_minutes`` from ``configs/splits.yaml``."""
    resolved = cfg or load_config()
    return int(resolved.evaluation.truth_tolerance_minutes)


def get_train_split_end(cfg: HarnessConfig | None = None) -> pd.Timestamp:
    """Return the train split end timestamp from ``configs/splits.yaml``."""
    resolved = cfg or load_config()
    assert resolved.splits.train.end is not None
    return pd.Timestamp(resolved.splits.train.end)


def passing_qc_signature() -> str:
    """Canonical string representing the active passing QC code sets for cache validation."""
    td = ",".join(sorted(PASSING_QC_CODES_TEMP_DEW))
    ws = ",".join(sorted(PASSING_QC_CODES_WIND_SLP))
    return f"temp_dew={td};wind_slp={ws}"


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


def _enforce_pre_sealed_cutoff(
    ts: pd.Timestamp,
    label: str = "timestamp",
    cfg: HarnessConfig | None = None,
) -> None:
    cutoff = get_sealed_test_cutoff(cfg)
    if ts >= cutoff:
        raise ValueError(
            f"Refusing {label} {ts.isoformat()}: sealed test split starts at "
            f"{cutoff.isoformat()} and is strictly off-limits."
        )


def _enforce_pre_sealed_year(year: int, cfg: HarnessConfig | None = None) -> None:
    cutoff = get_sealed_test_cutoff(cfg)
    if year >= cutoff.year:
        raise ValueError(
            f"Refusing year {year}: sealed test split starts at "
            f"{cutoff.isoformat()} and is strictly off-limits."
        )


def required_years_for_window(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    tolerance_minutes: int | None = None,
    *,
    cfg: HarnessConfig | None = None,
) -> list[int]:
    """Return all pre-sealed calendar years needed to match ``[start, end]`` within tolerance.

    Includes the previous year when ``start - tolerance`` crosses into it, and the next year
    when ``end + tolerance`` crosses into it, while never including the sealed-test year or later.
    """
    resolved_cfg = cfg or load_config()
    cutoff = get_sealed_test_cutoff(resolved_cfg)
    tol_min = (
        get_truth_tolerance_minutes(resolved_cfg)
        if tolerance_minutes is None
        else int(tolerance_minutes)
    )
    start_ts = _to_naive_utc_timestamp(start)
    end_ts = _to_naive_utc_timestamp(end)
    _enforce_pre_sealed_cutoff(start_ts, "start", resolved_cfg)
    _enforce_pre_sealed_cutoff(end_ts, "end", resolved_cfg)

    lo = start_ts - pd.Timedelta(minutes=tol_min)
    hi = min(end_ts + pd.Timedelta(minutes=tol_min), cutoff - pd.Timedelta(seconds=1))
    return [yr for yr in range(lo.year, hi.year + 1) if yr < cutoff.year]


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
    if qc not in PASSING_QC_CODES_TEMP_DEW or not val_str:
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
    if qc not in PASSING_QC_CODES_WIND_SLP or not val_str:
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
    - Speed QC must be in ``{0, 1, 4, 5, 9}`` and speed in ``0..900`` tenths of m/s (0..90 m/s).
    - Calm wind: speed ``0000`` with passing speed QC and type ``C`` (or type ``9``, which the ISD
      specification explicitly notes indicates calm winds when speed is ``0000``) -> ``(0.0, 0.0)``
      even though direction is coded as ``999`` with direction QC ``9``.
    - Variable wind (type ``V``) and all directional types: use direction when present in
      ``1..360`` with passing direction QC ``{0, 1, 4, 5, 9}``; return ``(nan, nan)`` when
      direction is ``999`` (missing/variable without direction), outside ``1..360``, or fails QC.
    """
    if not raw:
        return (math.nan, math.nan)
    parts = raw.split(",")
    if len(parts) != 5:
        return (math.nan, math.nan)
    dir_str, dir_qc, type_code, spd_str, spd_qc = (p.strip() for p in parts)
    if spd_qc not in PASSING_QC_CODES_WIND_SLP or not spd_str:
        return (math.nan, math.nan)
    try:
        spd_int = int(spd_str)
    except ValueError:
        return (math.nan, math.nan)
    if spd_int == 9999 or spd_int < 0 or spd_int > WIND_SPEED_MAX_TENTHS_MS:
        return (math.nan, math.nan)

    # Calm wind
    if spd_int == 0 and type_code in {"C", "9"}:
        return (0.0, 0.0)

    if dir_qc not in PASSING_QC_CODES_WIND_SLP or not dir_str:
        return (math.nan, math.nan)
    try:
        dir_int = int(dir_str)
    except ValueError:
        return (math.nan, math.nan)
    if dir_int == 999 or dir_int < WIND_DIR_MIN_DEG or dir_int > WIND_DIR_MAX_DEG:
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
    valid = vals.notna() & (vals.abs() != 9999) & qcs.isin(PASSING_QC_CODES_TEMP_DEW)
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
    valid = vals.notna() & (vals != 99999) & (vals > 0) & qcs.isin(PASSING_QC_CODES_WIND_SLP)
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
        & (spd_vals <= WIND_SPEED_MAX_TENTHS_MS)
        & spd_qcs.isin(PASSING_QC_CODES_WIND_SLP)
    )

    calm_mask = (spd_valid & (spd_vals == 0) & type_codes.isin({"C", "9"})).to_numpy()
    if calm_mask.any():
        u_out[calm_mask] = 0.0
        v_out[calm_mask] = 0.0

    dir_valid = (
        spd_valid
        & (~calm_mask)
        & dir_vals.notna()
        & (dir_vals != 999)
        & (dir_vals >= WIND_DIR_MIN_DEG)
        & (dir_vals <= WIND_DIR_MAX_DEG)
        & dir_qcs.isin(PASSING_QC_CODES_WIND_SLP)
    ).to_numpy()

    if dir_valid.any():
        speed_ms = spd_vals.to_numpy(dtype=np.float64)[dir_valid] / 10.0
        dir_deg = np.mod(dir_vals.to_numpy(dtype=np.float64)[dir_valid], 360.0)
        rad = np.deg2rad(dir_deg)
        u = -speed_ms * np.sin(rad)
        v = -speed_ms * np.cos(rad)
        u = np.where(np.abs(u) < 1e-12, 0.0, u)
        v = np.where(np.abs(v) < 1e-12, 0.0, v)
        zero_spd = speed_ms == 0.0
        u = np.where(zero_spd, 0.0, u)
        v = np.where(zero_spd, 0.0, v)
        u_out[dir_valid] = u.astype(np.float32)
        v_out[dir_valid] = v.astype(np.float32)

    return u_out, v_out


def _empty_parsed_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "obs_time": pd.Series(dtype="datetime64[ns]"),
            **{v: pd.Series(dtype="float32") for v in PRIMARY_VARS},
        }
    )


def parse_isd_csv(
    source: Path | str | TextIO,
    *,
    cfg: HarnessConfig | None = None,
) -> pd.DataFrame:
    """Parse a NOAA ISD global-hourly station-year CSV into canonical units.

    Applies report-type selection (METAR-family ``FM-15``/``FM-16`` whenever the station-year
    contains any; otherwise synoptic fallback ``FM-12``/``SY-MT``), NOAA quality filters, unit
    conversions, physical ``td2m <= t2m`` guard, and the sealed-test cutoff from
    ``configs/splits.yaml``.
    """
    cutoff = get_sealed_test_cutoff(cfg)
    allowed_cols = set(_REQUIRED_CSV_COLS) | set(_OPTIONAL_CSV_COLS)
    df = pd.read_csv(
        source,
        usecols=lambda c: c in allowed_cols,
        dtype=str,
        keep_default_na=False,
    )
    missing = [c for c in _REQUIRED_CSV_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"ISD CSV missing required columns: {missing}")
    if df.empty:
        return _empty_parsed_frame()

    if "REPORT_TYPE" in df.columns:
        rtypes = df["REPORT_TYPE"].str.strip()
        is_metar = rtypes.isin(METAR_REPORT_TYPES)
        if is_metar.any():
            keep_rt = is_metar
        else:
            keep_rt = rtypes.isin(SYNOPTIC_FALLBACK_REPORT_TYPES)
        df = df.loc[keep_rt].reset_index(drop=True)
        if df.empty:
            return _empty_parsed_frame()

    obs_time = pd.to_datetime(df["DATE"], format="ISO8601", errors="coerce", utc=False)
    if obs_time.dt.tz is not None:
        obs_time = obs_time.dt.tz_convert("UTC").dt.tz_localize(None)
    obs_time = obs_time.astype("datetime64[ns]")

    valid_time_mask = obs_time.notna() & (obs_time < cutoff)
    if not valid_time_mask.all():
        df = df.loc[valid_time_mask].reset_index(drop=True)
        obs_time = obs_time.loc[valid_time_mask].reset_index(drop=True)
    if df.empty:
        return _empty_parsed_frame()

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


def save_parsed_station_year_npz(
    obs_df: pd.DataFrame,
    dest: Path,
    *,
    tolerance_minutes: int,
) -> Path:
    """Save parsed station-year observations and cache provenance metadata to a compressed ``.npz``."""
    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    obs_ns = (
        obs_df["obs_time"].astype("datetime64[ns]").to_numpy().view(np.int64)
        if not obs_df.empty
        else np.empty(0, dtype=np.int64)
    )
    payload: dict[str, np.ndarray] = {
        "obs_time_ns": obs_ns,
        "meta_parser_version": np.array(PARSER_VERSION),
        "meta_tolerance_minutes": np.array(int(tolerance_minutes), dtype=np.int32),
        "meta_passing_qc_codes": np.array(passing_qc_signature()),
    }
    for var in PRIMARY_VARS:
        payload[var] = (
            obs_df[var].to_numpy(dtype=np.float32) if not obs_df.empty else np.empty(0, dtype=np.float32)
        )

    tmp_path = dest_path.with_name(f".{dest_path.stem}.tmp.npz")
    try:
        np.savez_compressed(tmp_path, **payload)
        tmp_path.replace(dest_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
    return dest_path


def load_parsed_station_year_npz(
    path: Path,
    *,
    expected_tolerance_minutes: int,
    cfg: HarnessConfig | None = None,
) -> pd.DataFrame:
    """Load a cached parsed station-year ``.npz`` after verifying parser/tolerance/QC metadata.

    Raises ``ValueError`` if any metadata key is missing or does not match the current parser
    version, tolerance, or passing QC code set.
    """
    cutoff = get_sealed_test_cutoff(cfg)
    required_meta = ("meta_parser_version", "meta_tolerance_minutes", "meta_passing_qc_codes")
    with np.load(path, allow_pickle=False) as z:
        missing_meta = [k for k in required_meta if k not in z.files]
        if missing_meta:
            raise ValueError(
                f"Parsed cache {path} is missing required metadata keys {missing_meta}; "
                "refusing to reuse unversioned cache."
            )
        cached_ver = str(z["meta_parser_version"].item())
        cached_tol = int(z["meta_tolerance_minutes"].item())
        cached_qc = str(z["meta_passing_qc_codes"].item())
        expected_qc = passing_qc_signature()

        if cached_ver != PARSER_VERSION:
            raise ValueError(
                f"Parsed cache {path} parser_version mismatch: found {cached_ver!r}, "
                f"expected {PARSER_VERSION!r}."
            )
        if cached_tol != int(expected_tolerance_minutes):
            raise ValueError(
                f"Parsed cache {path} tolerance_minutes mismatch: found {cached_tol}, "
                f"expected {expected_tolerance_minutes}."
            )
        if cached_qc != expected_qc:
            raise ValueError(
                f"Parsed cache {path} passing_qc_codes mismatch: found {cached_qc!r}, "
                f"expected {expected_qc!r}."
            )

        obs_ns = z["obs_time_ns"].astype(np.int64, copy=False)
        obs_time = pd.to_datetime(obs_ns, unit="ns")
        data: dict[str, object] = {"obs_time": obs_time}
        for var in PRIMARY_VARS:
            data[var] = z[var].astype(np.float32, copy=False)

    df = pd.DataFrame(data)
    if not df.empty:
        df = df[df["obs_time"] < cutoff].reset_index(drop=True)
    return df


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
    cfg: HarnessConfig | None = None,
) -> Path:
    """Download a single NOAA ISD global-hourly station-year CSV with retries and resumability."""
    _enforce_pre_sealed_year(year, cfg)
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
            req = urllib.request.Request(url, headers={"User-Agent": "wrfcomp-truth-loader/0.2"})
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
    cfg: HarnessConfig | None = None,
) -> tuple[list[tuple[str, int]], list[tuple[str, int, str]]]:
    """Download station-year CSVs with at most 4 concurrent requests and retries.

    Returns ``(succeeded, failed)`` where ``failed`` entries are ``(station_id, year, error_message)``.
    """
    for year in years:
        _enforce_pre_sealed_year(year, cfg)
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
                cfg=cfg,
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


def _load_station_observations_for_window(
    station_id: str,
    years: Sequence[int],
    lo: pd.Timestamp,
    hi: pd.Timestamp,
    *,
    cache_path: Path,
    parsed_path: Path | None,
    tolerance_minutes: int,
    cfg: HarnessConfig,
) -> tuple[pd.DataFrame, Path | None]:
    """Load and concatenate trimmed station observations from parsed ``.npz`` or raw CSV cache.

    Both paths consume the exact same year window ``[lo, hi]`` (including the last ``tol`` minutes
    of the previous year and first ``tol`` minutes of the next year, clamped strictly below the
    sealed-test cutoff) so they produce identical matched outputs.
    """
    frames: list[pd.DataFrame] = []
    seen_csv_paths: set[Path] = set()
    first_csv: Path | None = None

    for yr in years:
        parsed_df: pd.DataFrame | None = None
        if parsed_path is not None:
            npz_path = parsed_path / str(yr) / f"{station_id}.npz"
            if npz_path.is_file():
                parsed_df = load_parsed_station_year_npz(
                    npz_path,
                    expected_tolerance_minutes=tolerance_minutes,
                    cfg=cfg,
                )

        if parsed_df is None:
            fpath = _find_cached_station_file(cache_path, station_id, yr)
            if fpath is not None and fpath not in seen_csv_paths:
                seen_csv_paths.add(fpath)
                if first_csv is None:
                    first_csv = fpath
                parsed_df = parse_isd_csv(fpath, cfg=cfg)

        if parsed_df is not None and not parsed_df.empty:
            sub = parsed_df[(parsed_df["obs_time"] >= lo) & (parsed_df["obs_time"] <= hi)]
            if not sub.empty:
                frames.append(sub)

    if not frames:
        return _empty_parsed_frame(), first_csv
    obs_df = frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)
    return obs_df, first_csv


def build_truth(
    station_ids: Iterable[str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    cache_dir: Path | str,
    *,
    stations_metadata: pd.DataFrame | None = None,
    tolerance_minutes: int | None = None,
    parsed_cache_dir: Path | str | None = None,
    cfg: HarnessConfig | None = None,
) -> xr.Dataset:
    """Build a contract-compliant hourly truth ``xr.Dataset`` from cached NOAA ISD files.

    Parameters
    ----------
    station_ids:
        Sequence of 11-character ISD ``USAF+WBAN`` station identifiers.
    start, end:
        Inclusive UTC hourly bounds (timezone-naive or UTC-aware). Must be strictly
        before the sealed-test start in ``configs/splits.yaml``.
    cache_dir:
        Directory containing cached raw station-year CSV files
        (``<cache_dir>/<year>/<station_id>.csv``).
    stations_metadata:
        Optional DataFrame with ``station_id``, ``lat``/``latitude``, ``lon``/``longitude``,
        ``elevation_m``. Defaults to ``configs/stations_conus.csv``.
    tolerance_minutes:
        Matching half-window in minutes. Defaults to ``evaluation.truth_tolerance_minutes``
        from ``configs/splits.yaml`` (15).
    parsed_cache_dir:
        Optional directory holding versioned per-station-year parsed observation ``.npz`` files
        (used by ``scripts/build_truth.py --discard-raw``).
    cfg:
        Optional pre-loaded :class:`HarnessConfig` (defaults to ``configs/splits.yaml``).
    """
    resolved_cfg = cfg or load_config()
    cutoff = get_sealed_test_cutoff(resolved_cfg)

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
    _enforce_pre_sealed_cutoff(start_ts, "start", resolved_cfg)
    _enforce_pre_sealed_cutoff(end_ts, "end", resolved_cfg)

    tol_min = (
        get_truth_tolerance_minutes(resolved_cfg)
        if tolerance_minutes is None
        else int(tolerance_minutes)
    )
    valid_times = pd.date_range(start_ts, end_ts, freq="h")

    cache_path = Path(cache_dir)
    parsed_path = Path(parsed_cache_dir) if parsed_cache_dir is not None else None

    lo = start_ts - pd.Timedelta(minutes=tol_min)
    hi = min(end_ts + pd.Timedelta(minutes=tol_min), cutoff - pd.Timedelta(seconds=1))
    years = required_years_for_window(start_ts, end_ts, tol_min, cfg=resolved_cfg)

    n_times = len(valid_times)
    n_stations = len(sids)

    data_arrays: dict[str, np.ndarray] = {
        v: np.full((n_times, n_stations), np.nan, dtype=np.float32) for v in PRIMARY_VARS
    }
    offset_arrays: dict[str, np.ndarray] = {
        v: np.full((n_times, n_stations), OFFSET_FILL_VALUE, dtype=np.int16) for v in PRIMARY_VARS
    }
    fallback_files: dict[str, Path | None] = {}

    for j, sid in enumerate(sids):
        # Always start each station's columns cleanly cleared before loading or fallback
        for var in PRIMARY_VARS:
            data_arrays[var][:, j] = np.nan
            offset_arrays[var][:, j] = OFFSET_FILL_VALUE

        obs_df, first_csv = _load_station_observations_for_window(
            sid,
            years,
            lo,
            hi,
            cache_path=cache_path,
            parsed_path=parsed_path,
            tolerance_minutes=tol_min,
            cfg=resolved_cfg,
        )
        fallback_files[sid] = first_csv
        if obs_df.empty:
            continue

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
            "parser_version": PARSER_VERSION,
            "protocol_version": resolved_cfg.protocol_version,
            "truth_tolerance_minutes": tol_min,
            "passing_qc_codes": passing_qc_signature(),
        },
    )

    validate_truth_dataset(ds, variables=PRIMARY_VARS, cfg=resolved_cfg)
    validate_truth_values(ds, tolerance_minutes=tol_min)
    return ds


def validate_truth_values(ds: xr.Dataset, tolerance_minutes: int | None = None) -> None:
    """Validate value-level invariants beyond :func:`wrfcomp.harness.schemas.validate_truth_dataset`.

    Checks that every primary variable has its ``X_offset_min`` companion, that ``X_offset_min``
    equals ``-32768`` wherever ``X`` is ``NaN`` and lies in ``[-tolerance, +tolerance]`` elsewhere,
    and that ``td2m <= t2m`` wherever both are finite.
    """
    tol_min = get_truth_tolerance_minutes() if tolerance_minutes is None else int(tolerance_minutes)

    for var in PRIMARY_VARS:
        off_var = f"{var}_offset_min"
        if off_var not in ds.data_vars:
            raise ValueError(f"Truth dataset missing required companion variable: {off_var}")

        vals = ds[var].to_numpy()
        offs = ds[off_var].to_numpy()
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

    t2m = ds["t2m"].to_numpy()
    td2m = ds["td2m"].to_numpy()
    both = np.isfinite(t2m) & np.isfinite(td2m)
    if np.any(td2m[both] > t2m[both]):
        raise ValueError("Truth dataset violates physical constraint td2m <= t2m.")


def write_truth_netcdf(ds: xr.Dataset, out: Path | str) -> None:
    """Write a validated truth Dataset atomically to NetCDF4 via ``h5netcdf``.

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
        ds.to_netcdf(tmp_out, engine="h5netcdf", encoding=encoding)
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
