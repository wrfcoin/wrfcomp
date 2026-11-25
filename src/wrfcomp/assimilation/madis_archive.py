"""Privacy-preserving reader for the hourly NOAA MADIS mesonet archive (CWOP / ``APRSWXNET`` only).

One archive file holds one hour of reports from every mesonet provider. Most providers restrict
redistribution, so everything that is not ``APRSWXNET`` is dropped in memory, before any other step,
and never reaches a table, a file or a log line.

The privacy rules are those of :mod:`wrfcomp.assimilation.citizen_obs`, by construction:

- a station identifier never leaves :func:`parse_archive_hour`; it is replaced by the keyed
  11-character token of :func:`~wrfcomp.assimilation.citizen_obs.anonymize_station_id`;
- latitude and longitude are rounded to 0.1 degree and elevation to 10 m inside the parser. The CONUS
  box test uses the exact position in memory, before rounding;
- no function here puts record content into an exception message or a log line;
- a time at or after the sealed split start is refused.

Arrival time (Amendment 1 of ``docs/PHASE3_DESIGN.md``). A record is kept only if it was both measured
and received at or before the first issue time (00Z or 12Z) at or after its measurement. That is
decided on the archive's own timestamps, **before** repeated records are merged: a late repeat of a
report can never displace the timely one, and nothing that arrived after an issue time is in a table.

Timestamps keep their precision. Measurement and arrival times are stored as whole nanoseconds
(:func:`seconds_to_ns`), which keeps every archive timestamp distinct and on the same side of every
whole second, so the arrival test and the 15-minute window give the same answer on stored times as on
the archive's. Two records are the same report only if their measurement times are exactly equal.
Which version of a report is kept does not depend on the order of the records: the one that arrived
last, and among those the one with the largest content key.
"""

from __future__ import annotations

import gzip
import io
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.io import netcdf_file

from .citizen_obs import (
    COORD_DECIMALS,
    CWOP_PROVIDER,
    ELEVATION_STEP_M,
    SEALED_START,
    anonymize_station_id,
)

CONUS_BOX = (24.5, -125.0, 49.5, -66.5)  # lat south, lon west, lat north, lon east
ISSUE_STEP_S = 12 * 3600  # issue times are 00Z and 12Z
ISSUE_STEP_NS = ISSUE_STEP_S * 10**9
# Written into every stored table. A file without exactly this text is refused on load.
ARRIVAL_POLICY = (
    "amendment-1.v3: measured and received at or before the first 00Z/12Z at or after measurement; "
    "nanosecond times"
)
# Archive name -> (table column, required ``units`` attribute).
VALUE_FIELDS = {
    "temperature": ("t2m", "kelvin"),
    "dewpoint": ("td2m", "kelvin"),
    "windSpeed": ("wind_speed", "meter/sec"),
    "windDir": ("wind_dir", "degree"),
}
VALUE_COLUMNS = tuple(column for column, _ in VALUE_FIELDS.values())
CONTENT_COLUMNS = (*VALUE_COLUMNS, "lat", "lon", "elevation_m")
_SEALED_MESSAGE = "refusing observation at or after the sealed split start"
_MISSING_ABOVE = 1e30
_MISSING_BELOW = -9000.0


class ArchiveFormatError(ValueError):
    """The file is not the expected archive layout. Carries no record content."""


class StaleTableError(ValueError):
    """A stored table was not written under the current arrival policy."""


def seconds_to_ns(seconds) -> np.ndarray:
    """Archive seconds (float64) to whole nanoseconds, without moving a time across a whole second.

    The whole seconds are converted exactly; the fraction is rounded up to the nanosecond. A time
    after a whole second stays after it, a time before stays before, and distinct archive timestamps
    stay distinct (float64 seconds near today resolve about 240 ns).
    """
    seconds = np.asarray(seconds, dtype=np.float64)
    whole = np.floor(seconds)
    return whole.astype(np.int64) * 10**9 + np.ceil((seconds - whole) * 1e9).astype(np.int64)


def arrived_by_next_issue(measured_s, received_s) -> np.ndarray:
    """Amendment 1 on the archive's float seconds. True where a report was received at or before the
    first issue time at or after its measurement. A missing time is never eligible."""
    measured = np.asarray(measured_s, dtype=np.float64)
    received = np.asarray(received_s, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        next_issue = np.ceil(measured / ISSUE_STEP_S) * ISSUE_STEP_S
        return np.isfinite(measured) & np.isfinite(received) & (received <= next_issue)


def arrived_by_next_issue_ns(measured_ns, received_ns) -> np.ndarray:
    """The same test on stored nanosecond times (exact integer arithmetic)."""
    measured = np.asarray(measured_ns).astype(np.int64)
    received = np.asarray(received_ns).astype(np.int64)
    return received <= -(-measured // ISSUE_STEP_NS) * ISSUE_STEP_NS


@dataclass(frozen=True, repr=False)
class ReportTable:
    """One row per (station, measurement time). Array contents are omitted from repr."""

    station_id: np.ndarray  # U11 keyed tokens
    obs_time: np.ndarray  # datetime64[ns], UTC, measurement time
    received_time: np.ndarray  # datetime64[ns], UTC, arrival time
    lat: np.ndarray  # float32, 0.1 degree
    lon: np.ndarray  # float32, 0.1 degree
    elevation_m: np.ndarray  # float32, 10 m
    t2m: np.ndarray  # float32 K, NaN when not reported
    td2m: np.ndarray  # float32 K
    wind_speed: np.ndarray  # float32 m/s
    wind_dir: np.ndarray  # float32 degrees, direction the wind comes from

    def __len__(self) -> int:
        return len(self.station_id)


def empty_table() -> ReportTable:
    return ReportTable(
        station_id=np.empty(0, dtype="U11"),
        obs_time=np.empty(0, dtype="datetime64[ns]"),
        received_time=np.empty(0, dtype="datetime64[ns]"),
        lat=np.empty(0, dtype=np.float32),
        lon=np.empty(0, dtype=np.float32),
        elevation_m=np.empty(0, dtype=np.float32),
        t2m=np.empty(0, dtype=np.float32),
        td2m=np.empty(0, dtype=np.float32),
        wind_speed=np.empty(0, dtype=np.float32),
        wind_dir=np.empty(0, dtype=np.float32),
    )


def latest_version_rows(station, obs_ns, received_ns, content) -> np.ndarray:
    """Row indices, one per (station, exact measurement time), sorted by station and time.

    Of several versions of one report the one that arrived last is kept; among versions that arrived
    at the same instant, the one whose content (``content``: the value and position columns, in a
    fixed order) sorts last, with a missing value sorting first. The choice depends on the records
    only, never on the order they came in. Every row passed in has already passed the arrival test,
    so the version kept had arrived in time, and so had the ones it replaces.
    """
    station = np.asarray(station)
    obs = np.asarray(obs_ns).astype(np.int64)
    received = np.asarray(received_ns).astype(np.int64)
    if not station.size:
        return np.empty(0, dtype=np.int64)
    keys = [np.where(np.isnan(c), -np.inf, c) for c in (np.asarray(c, dtype=np.float64) for c in content)]
    order = np.lexsort((*reversed(keys), received, obs, station))
    s, t = station[order], obs[order]
    last = np.r_[(s[1:] != s[:-1]) | (t[1:] != t[:-1]), True]
    return order[last]


def _content(columns: dict) -> list:
    return [columns[name] for name in CONTENT_COLUMNS]


def _text_column(nc: netcdf_file, name: str) -> np.ndarray:
    data = np.ascontiguousarray(nc.variables[name][:])
    if data.ndim != 2 or data.dtype.kind != "S" or data.dtype.itemsize != 1:
        raise ArchiveFormatError(f"archive field {name} has an unexpected layout")
    joined = data.view(f"S{data.shape[1]}").ravel()
    return np.char.strip(np.char.decode(joined, "ascii", "replace"), "\x00 ")


def _number_column(nc: netcdf_file, name: str) -> np.ndarray:
    values = np.array(nc.variables[name][:], dtype=np.float64)
    values[~np.isfinite(values) | (values > _MISSING_ABOVE) | (values < _MISSING_BELOW)] = np.nan
    return values


def _units(nc: netcdf_file, name: str) -> str:
    raw = getattr(nc.variables[name], "units", b"")
    return raw.decode("ascii", "replace") if isinstance(raw, bytes) else str(raw)


def parse_archive_hour(
    payload: bytes,
    *,
    key: bytes,
    not_before: np.datetime64,
    token_cache: dict[str, str] | None = None,
    counts: dict[str, int] | None = None,
) -> ReportTable:
    """Parse one hourly archive file (gzip or plain NetCDF-3 bytes) into an anonymised table.

    Keeps ``APRSWXNET`` records inside the CONUS box that were measured at or after ``not_before``
    (the start of the year being built) and that pass the arrival rule of the module docstring.
    Skips a record with an unreadable identifier, time or position, without quoting it.
    ``token_cache`` may be passed to reuse tokens between hours; it holds raw identifiers in memory
    and must not be stored or logged by the caller. ``counts``, if given, is updated with totals.
    """
    try:
        body = gzip.decompress(payload) if payload[:2] == b"\x1f\x8b" else payload
        nc = netcdf_file(io.BytesIO(body), "r", mmap=False, maskandscale=False)
    except Exception as exc:  # noqa: BLE001 - class name only, a message could quote content
        raise ArchiveFormatError(f"cannot read archive file ({type(exc).__name__})") from None
    try:
        needed = ("dataProvider", "stationId", "latitude", "longitude", "elevation", "observationTime")
        missing = [name for name in (*needed, "receivedTime", *VALUE_FIELDS) if name not in nc.variables]
        if missing:
            raise ArchiveFormatError(f"archive file lacks fields: {missing}")
        for name, (_, units) in VALUE_FIELDS.items():
            if _units(nc, name) != units:
                raise ArchiveFormatError(f"archive field {name} is not in {units}")
        for name in ("observationTime", "receivedTime"):
            if not _units(nc, name).startswith("seconds since 1970-1-1 00:00:00"):
                raise ArchiveFormatError(f"archive {name} has an unexpected epoch")

        # Restricted providers are dropped here, before anything else is read for them.
        keep = np.flatnonzero(_text_column(nc, "dataProvider") == CWOP_PROVIDER)
        if not keep.size:
            return empty_table()
        raw_id = _text_column(nc, "stationId")[keep]
        lat = _number_column(nc, "latitude")[keep]
        lon = _number_column(nc, "longitude")[keep]
        elev = _number_column(nc, "elevation")[keep]
        measured = _number_column(nc, "observationTime")[keep]
        received = _number_column(nc, "receivedTime")[keep]
        values = {column: _number_column(nc, name)[keep] for name, (column, _) in VALUE_FIELDS.items()}
    finally:
        nc.close()

    south, west, north, east = CONUS_BOX
    with np.errstate(invalid="ignore"):
        readable = (
            (np.char.str_len(raw_id) > 0)
            & np.isfinite(lat)
            & np.isfinite(lon)
            & np.isfinite(elev)
            & np.isfinite(measured)
            & (measured >= 0)
            & (lat >= south)
            & (lat <= north)
            & (lon >= west)
            & (lon <= east)
        )
        sealed_s = float(SEALED_START.astype("datetime64[s]").astype(np.int64))
        if np.any(readable & (measured >= sealed_s)):
            raise ValueError(_SEALED_MESSAGE)
        start_s = float(np.datetime64(not_before, "s").astype(np.int64))
        in_year = readable & (measured >= start_s)
        # The arrival rule, on the source timestamps, before repeats are merged.
        ok = in_year & arrived_by_next_issue(measured, received)
    if counts is not None:
        for name, n in (
            ("records", readable.sum()),
            ("before_start", (readable & ~in_year).sum()),
            ("not_arrived", (in_year & ~ok).sum()),
        ):
            counts[name] = counts.get(name, 0) + int(n)
    if not ok.any():
        return empty_table()
    raw_id, lat, lon, elev, measured, received = (a[ok] for a in (raw_id, lat, lon, elev, measured, received))
    columns = {column: a[ok].astype(np.float32) for column, a in values.items()}
    columns["lat"] = np.round(lat, COORD_DECIMALS).astype(np.float32)
    columns["lon"] = np.round(lon, COORD_DECIMALS).astype(np.float32)
    columns["elevation_m"] = (np.round(elev / ELEVATION_STEP_M) * ELEVATION_STEP_M).astype(np.float32)
    obs_ns, received_ns = seconds_to_ns(measured), seconds_to_ns(received)

    cache = token_cache if token_cache is not None else {}
    unique, inverse = np.unique(raw_id, return_inverse=True)
    tokens = np.empty(unique.size, dtype="U11")
    for j, identifier in enumerate(unique.tolist()):
        token = cache.get(identifier)
        if token is None:
            token = cache[identifier] = anonymize_station_id(identifier, key=key)
        tokens[j] = token
    station = tokens[inverse]

    pick = latest_version_rows(station, obs_ns, received_ns, _content(columns))
    if counts is not None:
        counts["repeats_merged"] = counts.get("repeats_merged", 0) + int(station.size - pick.size)
    return ReportTable(
        station_id=station[pick],
        obs_time=obs_ns[pick].astype("datetime64[ns]"),
        received_time=received_ns[pick].astype("datetime64[ns]"),
        **{name: a[pick] for name, a in columns.items()},
    )


def merge_tables(tables: list[ReportTable]) -> ReportTable:
    """Concatenate; of the versions of one report keep one (see :func:`latest_version_rows`); sort."""
    tables = [t for t in tables if len(t)]
    if not tables:
        return empty_table()
    columns = {f.name: np.concatenate([getattr(t, f.name) for t in tables]) for f in fields(ReportTable)}
    if np.any(columns["obs_time"] >= SEALED_START):
        raise ValueError(_SEALED_MESSAGE)
    if not arrived_by_next_issue_ns(columns["obs_time"], columns["received_time"]).all():
        raise ValueError("a table holds a report that had not arrived by its issue time")
    pick = latest_version_rows(
        columns["station_id"], columns["obs_time"], columns["received_time"], _content(columns)
    )
    return ReportTable(**{name: a[pick] for name, a in columns.items()})


def save_table(table: ReportTable, path: Path, *, attrs: dict | None = None) -> None:
    """Write an anonymised table. Positions are rounded again here, whatever built the table."""
    if len(table) and np.any(table.obs_time >= SEALED_START):
        raise ValueError(_SEALED_MESSAGE)
    if not arrived_by_next_issue_ns(table.obs_time, table.received_time).all():
        raise ValueError("refusing to store a report that had not arrived by its issue time")
    ds = xr.Dataset(
        {
            "station_id": ("obs", np.asarray(table.station_id, dtype="U11")),
            "obs_time": ("obs", table.obs_time.astype("datetime64[ns]").astype(np.int64)),
            "received_time": ("obs", table.received_time.astype("datetime64[ns]").astype(np.int64)),
            "lat": ("obs", np.round(table.lat.astype(np.float64), COORD_DECIMALS).astype(np.float32)),
            "lon": ("obs", np.round(table.lon.astype(np.float64), COORD_DECIMALS).astype(np.float32)),
            "elevation_m": (
                "obs",
                (np.round(table.elevation_m.astype(np.float64) / ELEVATION_STEP_M) * ELEVATION_STEP_M).astype(
                    np.float32
                ),
            ),
            **{column: ("obs", getattr(table, column).astype(np.float32)) for column in VALUE_COLUMNS},
        },
        attrs={
            **(attrs or {}),
            "source": "NOAA MADIS public mesonet archive (APRSWXNET / CWOP only)",
            "privacy": "station_id is a keyed 11-char token; lat/lon rounded to 0.1 deg, elevation to 10 m",
            "times": "obs_time and received_time are nanoseconds since 1970-01-01T00:00:00Z",
            "arrival_policy": ARRIVAL_POLICY,
        },
    )
    encoding = {name: {"zlib": True, "complevel": 4, "shuffle": True} for name in ds.data_vars}
    encoding["station_id"] = {"zlib": True, "complevel": 4, "dtype": "S11"}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.part")
    try:
        ds.to_netcdf(tmp, engine="h5netcdf", encoding=encoding)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def load_table(path: Path, *, with_attrs: bool = False):
    """Read a stored table. A file written under another arrival policy, or none, is refused."""
    with xr.open_dataset(path, engine="h5netcdf", decode_times=False) as ds:
        if ds.attrs.get("arrival_policy") != ARRIVAL_POLICY:
            raise StaleTableError(f"{Path(path).name} was not written under the current arrival policy")
        obs_time = ds["obs_time"].values.astype(np.int64).astype("datetime64[ns]")
        received_time = ds["received_time"].values.astype(np.int64).astype("datetime64[ns]")
        if obs_time.size and np.any(obs_time >= SEALED_START):
            raise ValueError(_SEALED_MESSAGE)
        raw = ds["station_id"].values
        station = np.char.decode(raw, "ascii") if raw.dtype.kind == "S" else raw.astype("U11")
        table = ReportTable(
            station_id=station.astype("U11"),
            obs_time=obs_time,
            received_time=received_time,
            lat=ds["lat"].values.astype(np.float32),
            lon=ds["lon"].values.astype(np.float32),
            elevation_m=ds["elevation_m"].values.astype(np.float32),
            **{column: ds[column].values.astype(np.float32) for column in VALUE_COLUMNS},
        )
        attrs = dict(ds.attrs)
    if not arrived_by_next_issue_ns(obs_time, received_time).all():
        raise StaleTableError(f"{Path(path).name} holds a report that had not arrived by its issue time")
    return (table, attrs) if with_attrs else table
