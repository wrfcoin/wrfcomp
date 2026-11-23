"""Privacy-preserving loader for free CWOP / APRSWXNET citizen-station observations.

Parses NOAA MADIS public surface CSV responses (``APRSWXNET`` provider) into
``ObservationBatch`` arrays in contract SI units with exact measurement
timestamps (never rounded).

Privacy, by construction (citizen stations are at people's homes):

- A station identifier never leaves :func:`parse_madis_csv`. It is replaced on ingest by an
  11-character token, ``CW`` plus 9 hex digits of an HMAC-SHA256 keyed with a secret that lives
  only in the local data directory (:func:`load_or_create_station_key`). The key is never in the
  repository, so the tokens cannot be turned back into identifiers by hashing the (small, public)
  list of CWOP identifiers. The same station gets the same token on this machine only.
- Latitude and longitude are rounded to 0.1 degree and elevation to 10 m inside the parser, so no
  batch, file or summary ever holds an exact position. The radius filter uses the exact position
  in memory, before rounding.
- No function here puts row content into an exception message or a log line.
"""

from __future__ import annotations

import csv
import hashlib
import hmac
import io
import math
import os
import re
import secrets
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from wrfcomp.harness.config import load_config

from .qc import distances_km
from .tokens import ObservationBatch

CWOP_PROVIDER = "APRSWXNET"
CITIZEN_SENSOR_TYPE_ID = 2
STATION_KEY_FILENAME = ".station_key"
STATION_KEY_BYTES = 32
COORD_DECIMALS = 1  # 0.1 degree
ELEVATION_STEP_M = 10.0
# The sealed-test start comes from configs/splits.yaml, the single source for split boundaries.
SEALED_START = np.datetime64(load_config().splits.sealed_test.start, "ns")
_SEALED_MESSAGE = "refusing observation at or after the sealed split start"

_PRE_RE = re.compile(r"<pre>(.*?)</pre>", re.IGNORECASE | re.DOTALL)


def load_or_create_station_key(citizen_dir: Path) -> bytes:
    """Return the secret key for station tokens, creating it on first use.

    The key is 32 random bytes in ``<citizen_dir>/.station_key`` (mode 0600). It must never be
    committed, printed or copied next to shared data: whoever holds it can test a guessed
    identifier against a token. Deleting it makes new tokens that no longer match old files.
    """
    path = Path(citizen_dir) / STATION_KEY_FILENAME
    if path.is_file():
        key = path.read_bytes()
        if len(key) != STATION_KEY_BYTES:
            raise ValueError("station key file has the wrong length")
        return key
    path.parent.mkdir(parents=True, exist_ok=True)
    key = secrets.token_bytes(STATION_KEY_BYTES)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(key)
    return key


def anonymize_station_id(raw_id: str, *, key: bytes) -> str:
    """Map a raw citizen station identifier to an 11-character opaque token, keyed with ``key``."""
    cleaned = raw_id.strip()
    if not cleaned:
        raise ValueError("station identifier must be non-empty")
    if len(key) < 16:
        raise ValueError("station key must be at least 16 bytes")
    digest = hmac.new(key, cleaned.encode("utf-8"), hashlib.sha256).hexdigest()[:9].upper()
    return f"CW{digest}"


def round_coord_tenth(value: float) -> float:
    """Round a geographic coordinate to 0.1 degree."""
    return round(float(value), COORD_DECIMALS)


def round_elevation(value: float) -> float:
    """Round an elevation to the nearest 10 m."""
    return round(float(value) / ELEVATION_STEP_M) * ELEVATION_STEP_M


def _parse_float(raw: str | None) -> float | None:
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    try:
        val = float(text)
    except ValueError:
        return None
    if not math.isfinite(val) or val <= -9999.0 or val >= 1e20:
        return None
    return val


def _parse_madis_timestamp(obdate: str, obtime: str) -> np.datetime64:
    """Parse MADIS ``MM/DD/YYYY`` + ``HH:MM`` UTC into timezone-naive ``datetime64[ns]``."""
    d_parts = obdate.strip().split("/")
    t_parts = obtime.strip().split(":")
    if len(d_parts) != 3 or len(t_parts) not in (2, 3) or not all(p.isdigit() for p in d_parts + t_parts):
        raise _MalformedRowError
    month, day, year = int(d_parts[0]), int(d_parts[1]), int(d_parts[2])
    hour, minute = int(t_parts[0]), int(t_parts[1])
    second = int(t_parts[2]) if len(t_parts) == 3 else 0
    if not (1 <= month <= 12 and 1 <= day <= 31 and hour <= 23 and minute <= 59 and second <= 59):
        raise _MalformedRowError
    try:
        ts = np.datetime64(
            f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:02d}",
            "ns",
        )
    except ValueError:
        raise _MalformedRowError from None
    if np.isnat(ts):
        raise _MalformedRowError
    if ts >= SEALED_START:
        raise ValueError(_SEALED_MESSAGE)
    return ts


class _MalformedRowError(Exception):
    """A row that cannot be read. Carries no row content, so it is safe to count and log."""


def _extract_csv_body(text: str) -> str:
    match = _PRE_RE.search(text)
    return match.group(1).strip() if match is not None else text.strip()


def _column_indices(header: list[str]) -> dict[str, int]:
    """Return the first column index for each stripped non-QC column name."""
    indices: dict[str, int] = {}
    for idx, col in enumerate(header):
        name = col.strip()
        if not name or name == "D" or name in indices:
            continue
        indices[name] = idx
    return indices


def _wind_uv(speed_ms: float, dir_deg: float) -> tuple[float, float]:
    """Convert meteorological wind speed (m/s) and from-direction (deg) to (u10, v10)."""
    rad = math.radians(dir_deg)
    u = -speed_ms * math.sin(rad)
    v = -speed_ms * math.cos(rad)
    return (0.0 if u == 0.0 else u), (0.0 if v == 0.0 else v)


def parse_madis_csv(
    text: str,
    *,
    region_center: tuple[float, float] | None = None,
    max_radius_km: float | None = None,
    provider_filter: str | None = CWOP_PROVIDER,
    key: bytes,
) -> ObservationBatch:
    """Parse a MADIS ``xml=5`` CSV payload into an anonymized ``ObservationBatch``.

    - Preserves real measurement times (``OBDATE`` + ``OBTIME``, UTC, never rounded).
    - Converts MADIS units to contract SI units (``T``/``TD`` K, ``FF``/``FFGUST``/``u10``/``v10``
      m/s, ``P``/``SLP`` Pa, ``PCP1H`` m -> ``precip`` mm).
    - MADIS already serves SI units (``T``/``TD`` K, ``FF``/``FFGUST`` m/s, ``P``/``SLP`` Pa);
      only ``PCP1H`` (m) is converted, to mm. The MADIS quality descriptor columns (``D``) are not
      read: the point of this sample is to run our own quality control on raw reports.
    - Filters by ``provider_filter`` (default ``APRSWXNET``) and optional Haversine radius
      computed on exact coordinates, then rounds latitude/longitude to 0.1 degree and elevation
      to 10 m. The returned batch never holds an exact position.
    - Replaces every station identifier with an opaque 11-character token keyed with ``key``
      (see :func:`load_or_create_station_key`).
    - Skips rows it cannot read (wrong field count, unreadable time or position) without quoting
      them anywhere. A time at or after the sealed split start is refused with ``ValueError``.
    - Deduplicates ``(station_id, obs_time, variable)`` deterministically (last wins) so the
      returned batch feeds directly into ``wrfcomp.assimilation.qc.quality_control``.
    """
    body = _extract_csv_body(text)
    if not body:
        return empty_citizen_batch()

    reader = csv.reader(io.StringIO(body))
    header = next(reader, None)
    if not header:
        return empty_citizen_batch()

    cols = _column_indices(header)
    required_cols = ("STAID", "OBDATE", "OBTIME", "LAT", "LON", "ELEV")
    missing = [c for c in required_cols if c not in cols]
    if missing:
        raise ValueError(f"MADIS CSV missing required columns: {missing}")

    # Deduplicate keyed by (anon_id, time_ns, variable) keeping latest row deterministically.
    records: dict[tuple[str, int, str], tuple[float, float, float, float]] = {}

    for row in reader:
        if not row or len(row) <= max(cols[c] for c in required_cols):
            continue
        if provider_filter is not None and "PVDR" in cols:
            if row[cols["PVDR"]].strip() != provider_filter:
                continue

        raw_id = row[cols["STAID"]].strip()
        if not raw_id:
            continue

        lat = _parse_float(row[cols["LAT"]])
        lon = _parse_float(row[cols["LON"]])
        elev = _parse_float(row[cols["ELEV"]])
        if lat is None or lon is None or elev is None:
            continue
        if abs(lat) > 90.0 or abs(lon) > 180.0:
            continue

        if region_center is not None and max_radius_km is not None:
            dist = float(
                distances_km(
                    region_center[0],
                    region_center[1],
                    np.array([lat], dtype=float),
                    np.array([lon], dtype=float),
                )[0]
            )
            if dist > max_radius_km:
                continue

        try:
            obs_ts = _parse_madis_timestamp(row[cols["OBDATE"]], row[cols["OBTIME"]])
        except _MalformedRowError:
            continue
        time_ns = int(obs_ts.astype(np.int64))
        anon_id = anonymize_station_id(raw_id, key=key)
        out_lat = round_coord_tenth(lat)
        out_lon = round_coord_tenth(lon)
        elev = round_elevation(elev)

        def _get(col_name: str, current_row: list[str] = row) -> float | None:
            idx = cols.get(col_name)
            if idx is None or idx >= len(current_row):
                return None
            return _parse_float(current_row[idx])

        temp = _get("T")
        if temp is not None:
            records[(anon_id, time_ns, "t2m")] = (out_lat, out_lon, elev, temp)

        dewp = _get("TD")
        if dewp is not None:
            records[(anon_id, time_ns, "td2m")] = (out_lat, out_lon, elev, dewp)

        wspd = _get("FF")
        wdir = _get("DD")
        if wspd is not None:
            records[(anon_id, time_ns, "wind_speed")] = (out_lat, out_lon, elev, wspd)
            if wdir is not None and 0.0 <= wdir <= 360.0:
                u10, v10 = _wind_uv(wspd, wdir)
                records[(anon_id, time_ns, "u10")] = (out_lat, out_lon, elev, u10)
                records[(anon_id, time_ns, "v10")] = (out_lat, out_lon, elev, v10)

        wgust = _get("FFGUST")
        if wgust is not None:
            records[(anon_id, time_ns, "wind_gust")] = (out_lat, out_lon, elev, wgust)

        stnp = _get("P")
        if stnp is not None:
            records[(anon_id, time_ns, "station_pressure")] = (out_lat, out_lon, elev, stnp)

        slp = _get("SLP")
        if slp is not None:
            records[(anon_id, time_ns, "msl")] = (out_lat, out_lon, elev, slp)

        pcp1h_m = _get("PCP1H")
        if pcp1h_m is not None:
            # MADIS PCP1H is in meters; convert to contract millimeters.
            records[(anon_id, time_ns, "precip")] = (out_lat, out_lon, elev, pcp1h_m * 1000.0)

    return _records_to_batch(records)


def empty_citizen_batch() -> ObservationBatch:
    """Return an empty ``ObservationBatch`` with canonical dtypes."""
    return ObservationBatch(
        station_ids=np.empty(0, dtype="U11"),
        lat=np.empty(0, dtype=np.float64),
        lon=np.empty(0, dtype=np.float64),
        elevation_m=np.empty(0, dtype=np.float64),
        obs_time=np.empty(0, dtype="datetime64[ns]"),
        sensor_type_id=np.empty(0, dtype=np.int32),
        variable=np.empty(0, dtype="U16"),
        values=np.empty(0, dtype=np.float32),
        qc_flags=np.empty(0, dtype=np.uint32),
    )


def _records_to_batch(
    records: dict[tuple[str, int, str], tuple[float, float, float, float]],
) -> ObservationBatch:
    if not records:
        return empty_citizen_batch()
    # Sort deterministically by (time_ns, anon_id, variable).
    keys = sorted(records.keys(), key=lambda k: (k[1], k[0], k[2]))
    n = len(keys)
    station_ids = np.empty(n, dtype="U11")
    lats = np.empty(n, dtype=np.float64)
    lons = np.empty(n, dtype=np.float64)
    elevs = np.empty(n, dtype=np.float64)
    times = np.empty(n, dtype="datetime64[ns]")
    variables = np.empty(n, dtype="U16")
    values = np.empty(n, dtype=np.float32)

    for i, key in enumerate(keys):
        anon_id, time_ns, var = key
        lat, lon, elev, val = records[key]
        station_ids[i] = anon_id
        lats[i] = lat
        lons[i] = lon
        elevs[i] = elev
        times[i] = np.datetime64(time_ns, "ns")
        variables[i] = var
        values[i] = np.float32(val)

    return ObservationBatch(
        station_ids=station_ids,
        lat=lats,
        lon=lons,
        elevation_m=elevs,
        obs_time=times,
        sensor_type_id=np.full(n, CITIZEN_SENSOR_TYPE_ID, dtype=np.int32),
        variable=variables,
        values=values,
        qc_flags=np.zeros(n, dtype=np.uint32),
    )


def merge_citizen_batches(batches: list[ObservationBatch]) -> ObservationBatch:
    """Merge batches, deduplicating by (station, time, variable); the later batch wins.

    The result is sorted by (time, station, variable), like :func:`parse_madis_csv` output.
    """
    batches = [b for b in batches if len(b.values)]
    if not batches:
        return empty_citizen_batch()
    frame = pd.DataFrame(
        {
            "station_id": np.concatenate([b.station_ids for b in batches]),
            "time_ns": np.concatenate(
                [b.obs_time.astype("datetime64[ns]").astype(np.int64) for b in batches]
            ),
            "variable": np.concatenate([b.variable for b in batches]),
            "lat": np.concatenate([b.lat for b in batches]),
            "lon": np.concatenate([b.lon for b in batches]),
            "elevation_m": np.concatenate([b.elevation_m for b in batches]),
            "value": np.concatenate([b.values for b in batches]),
        }
    )
    if (frame["time_ns"] >= SEALED_START.astype(np.int64)).any():
        raise ValueError(_SEALED_MESSAGE)
    frame = frame.drop_duplicates(["station_id", "time_ns", "variable"], keep="last")
    frame = frame.sort_values(["time_ns", "station_id", "variable"], kind="mergesort")
    n = len(frame)
    return ObservationBatch(
        station_ids=frame["station_id"].to_numpy(dtype="U11"),
        lat=frame["lat"].to_numpy(dtype=np.float64),
        lon=frame["lon"].to_numpy(dtype=np.float64),
        elevation_m=frame["elevation_m"].to_numpy(dtype=np.float64),
        obs_time=frame["time_ns"].to_numpy(dtype=np.int64).astype("datetime64[ns]"),
        sensor_type_id=np.full(n, CITIZEN_SENSOR_TYPE_ID, dtype=np.int32),
        variable=frame["variable"].to_numpy(dtype="U16"),
        values=frame["value"].to_numpy(dtype=np.float32),
        qc_flags=np.zeros(n, dtype=np.uint32),
    )


def save_citizen_batch(batch: ObservationBatch, path: Path) -> None:
    """Persist an anonymized ``ObservationBatch`` to NetCDF.

    Positions are rounded again here (0.1 degree, 10 m), so even a batch built by other code
    cannot put an exact position on disk through this function.
    """
    if len(batch.obs_time) and np.any(batch.obs_time >= SEALED_START):
        raise ValueError(_SEALED_MESSAGE)
    rounded_lat = np.round(batch.lat.astype(np.float64), COORD_DECIMALS)
    rounded_lon = np.round(batch.lon.astype(np.float64), COORD_DECIMALS)
    rounded_elev = np.round(batch.elevation_m.astype(np.float64) / ELEVATION_STEP_M) * ELEVATION_STEP_M
    ds = xr.Dataset(
        {
            "station_id": ("obs", np.asarray(batch.station_ids, dtype="U11")),
            "lat": ("obs", rounded_lat.astype(np.float32)),
            "lon": ("obs", rounded_lon.astype(np.float32)),
            "elevation_m": ("obs", rounded_elev.astype(np.float32)),
            "obs_time": ("obs", batch.obs_time.astype("datetime64[ns]")),
            "sensor_type_id": ("obs", batch.sensor_type_id.astype(np.int32)),
            "variable": ("obs", np.asarray(batch.variable, dtype="U16")),
            "value": ("obs", batch.values.astype(np.float32)),
            "qc_flags": ("obs", batch.qc_flags.astype(np.uint32)),
        },
        attrs={
            "source": "NOAA MADIS public archive (APRSWXNET / CWOP)",
            "privacy": "station_id is a keyed 11-char token; lat/lon rounded to 0.1 deg, elevation to 10 m",
        },
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.part")
    try:
        ds.to_netcdf(tmp, engine="h5netcdf")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def load_citizen_batch(path: Path) -> ObservationBatch:
    """Load a persisted citizen-station NetCDF archive into an ``ObservationBatch``."""
    with xr.open_dataset(path, engine="h5netcdf") as ds:
        obs_time = ds["obs_time"].values.astype("datetime64[ns]")
        if obs_time.size and np.any(obs_time >= SEALED_START):
            raise ValueError(_SEALED_MESSAGE)
        raw_ids = ds["station_id"].values
        raw_vars = ds["variable"].values
        station_ids = np.array(
            [x.decode("utf-8") if isinstance(x, bytes | np.bytes_) else str(x) for x in raw_ids],
            dtype="U11",
        )
        variables = np.array(
            [x.decode("utf-8") if isinstance(x, bytes | np.bytes_) else str(x) for x in raw_vars],
            dtype="U16",
        )
        return ObservationBatch(
            station_ids=station_ids,
            lat=ds["lat"].values.astype(np.float64),
            lon=ds["lon"].values.astype(np.float64),
            elevation_m=ds["elevation_m"].values.astype(np.float64),
            obs_time=obs_time,
            sensor_type_id=ds["sensor_type_id"].values.astype(np.int32),
            variable=variables,
            values=ds["value"].values.astype(np.float32),
            qc_flags=ds["qc_flags"].values.astype(np.uint32),
        )


def summarize_batch(batch: ObservationBatch) -> dict[str, Any]:
    """Return aggregate-only statistics (zero station identifiers, names, or coordinates)."""
    n_obs = int(len(batch.values))
    if n_obs == 0:
        return {
            "n_observations": 0,
            "n_reports": 0,
            "n_stations": 0,
            "variables": {},
            "time_start_utc": None,
            "time_end_utc": None,
            "median_interval_min": None,
            "p25_interval_min": None,
            "p75_interval_min": None,
        }

    unique_stations = int(len(set(batch.station_ids.tolist())))
    var_names, var_counts = np.unique(batch.variable, return_counts=True)
    var_dict = {str(v): int(c) for v, c in zip(var_names, var_counts, strict=True)}

    # Group distinct report timestamps per opaque station to compute reporting intervals.
    times_by_station: dict[str, set[int]] = defaultdict(set)
    for sid, ts in zip(batch.station_ids.tolist(), batch.obs_time.astype(np.int64).tolist(), strict=True):
        times_by_station[sid].add(ts)

    n_reports = sum(len(ts_set) for ts_set in times_by_station.values())
    deltas_min: list[float] = []
    for ts_set in times_by_station.values():
        if len(ts_set) < 2:
            continue
        arr = np.sort(np.fromiter(ts_set, dtype=np.int64))
        diffs = np.diff(arr) / 60e9
        # Exclude multi-hour outages (> 120 min) when characterizing typical cadence.
        typical = diffs[(diffs > 0.0) & (diffs <= 120.0)]
        if typical.size:
            deltas_min.extend(typical.tolist())

    if deltas_min:
        d_arr = np.asarray(deltas_min, dtype=np.float64)
        p25, med, p75 = (float(x) for x in np.percentile(d_arr, [25, 50, 75]))
    else:
        p25 = med = p75 = None

    t_min = np.datetime_as_string(batch.obs_time.min().astype("datetime64[s]")) + "Z"
    t_max = np.datetime_as_string(batch.obs_time.max().astype("datetime64[s]")) + "Z"

    return {
        "n_observations": n_obs,
        "n_reports": int(n_reports),
        "n_stations": unique_stations,
        "variables": var_dict,
        "time_start_utc": t_min,
        "time_end_utc": t_max,
        "median_interval_min": None if med is None else round(med, 2),
        "p25_interval_min": None if p25 is None else round(p25, 2),
        "p75_interval_min": None if p75 is None else round(p75, 2),
    }
