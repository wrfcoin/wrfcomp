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
decided on the archive's own timestamps, at their full precision, **before** repeated records are
merged and before any time is converted: a late repeat of a report can therefore never displace the
timely one, and nothing that arrived after an issue time is in a table at all. Stored times are rounded
up to the whole second, which cannot move a time to or before an issue time it was after.
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
# Written into every stored table. A file without exactly this text is refused on load.
ARRIVAL_POLICY = (
    "amendment-1.v2: measured and received at or before the first 00Z/12Z at or after measurement"
)
# Archive name -> (table column, required ``units`` attribute).
VALUE_FIELDS = {
    "temperature": ("t2m", "kelvin"),
    "dewpoint": ("td2m", "kelvin"),
    "windSpeed": ("wind_speed", "meter/sec"),
    "windDir": ("wind_dir", "degree"),
}
VALUE_COLUMNS = tuple(column for column, _ in VALUE_FIELDS.values())
_SEALED_MESSAGE = "refusing observation at or after the sealed split start"
_MISSING_ABOVE = 1e30
_MISSING_BELOW = -9000.0


class ArchiveFormatError(ValueError):
    """The file is not the expected archive layout. Carries no record content."""


class StaleTableError(ValueError):
    """A stored table was not written under the current arrival policy."""


def arrived_by_next_issue(measured_s, received_s) -> np.ndarray:
    """Amendment 1. True where a report was received at or before the first issue time at or after
    its measurement. Works on the archive's float seconds and on stored whole seconds alike; a
    missing time is never eligible."""
    measured = np.asarray(measured_s, dtype=np.float64)
    received = np.asarray(received_s, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        next_issue = np.ceil(measured / ISSUE_STEP_S) * ISSUE_STEP_S
        return np.isfinite(measured) & np.isfinite(received) & (received <= next_issue)


@dataclass(frozen=True, repr=False)
class ReportTable:
    """One row per (station, measurement time). Array contents are omitted from repr."""

    station_id: np.ndarray  # U11 keyed tokens
    obs_time: np.ndarray  # datetime64[s], UTC, measurement time rounded up to the second
    lat: np.ndarray  # float32, 0.1 degree
    lon: np.ndarray  # float32, 0.1 degree
    elevation_m: np.ndarray  # float32, 10 m
    t2m: np.ndarray  # float32 K, NaN when not reported
    td2m: np.ndarray  # float32 K
    wind_speed: np.ndarray  # float32 m/s
    wind_dir: np.ndarray  # float32 degrees, direction the wind comes from
    received_delay_s: np.ndarray  # int32: arrival (rounded up to the second) minus obs_time

    def __len__(self) -> int:
        return len(self.station_id)

    def received_time_s(self) -> np.ndarray:
        return self.obs_time.astype("datetime64[s]").astype(np.int64) + self.received_delay_s.astype(np.int64)


def empty_table() -> ReportTable:
    return ReportTable(
        station_id=np.empty(0, dtype="U11"),
        obs_time=np.empty(0, dtype="datetime64[s]"),
        lat=np.empty(0, dtype=np.float32),
        lon=np.empty(0, dtype=np.float32),
        elevation_m=np.empty(0, dtype=np.float32),
        t2m=np.empty(0, dtype=np.float32),
        td2m=np.empty(0, dtype=np.float32),
        wind_speed=np.empty(0, dtype=np.float32),
        wind_dir=np.empty(0, dtype=np.float32),
        received_delay_s=np.empty(0, dtype=np.int32),
    )


def latest_arrival_rows(station, obs_time_s, received_s) -> np.ndarray:
    """Row indices, one per (station, time): the version that arrived last; on a tie, the later row.

    Every row passed in is already eligible, so whichever version is kept had arrived by the issue
    time, and so had the ones it replaces. Result is sorted by station, then time.
    """
    station = np.asarray(station)
    obs = np.asarray(obs_time_s).astype(np.int64)
    received = np.asarray(received_s).astype(np.int64)
    order = np.lexsort((np.arange(station.size), received, obs, station))
    s, t = station[order], obs[order]
    last = np.r_[(s[1:] != s[:-1]) | (t[1:] != t[:-1]), True] if station.size else np.empty(0, bool)
    return order[last]


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
        # The arrival rule, on the source timestamps, before repeats are merged or times converted.
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
    values = {column: a[ok] for column, a in values.items()}

    # Rounded up: a time after an issue time stays after it.
    obs_s = np.ceil(measured).astype(np.int64)
    received_s = np.ceil(received).astype(np.int64)

    cache = token_cache if token_cache is not None else {}
    unique, inverse = np.unique(raw_id, return_inverse=True)
    tokens = np.empty(unique.size, dtype="U11")
    for j, identifier in enumerate(unique.tolist()):
        token = cache.get(identifier)
        if token is None:
            token = cache[identifier] = anonymize_station_id(identifier, key=key)
        tokens[j] = token
    station = tokens[inverse]

    pick = latest_arrival_rows(station, obs_s, received_s)
    if counts is not None:
        counts["repeats_merged"] = counts.get("repeats_merged", 0) + int(station.size - pick.size)
    return ReportTable(
        station_id=station[pick],
        obs_time=obs_s[pick].astype("datetime64[s]"),
        lat=np.round(lat[pick], COORD_DECIMALS).astype(np.float32),
        lon=np.round(lon[pick], COORD_DECIMALS).astype(np.float32),
        elevation_m=(np.round(elev[pick] / ELEVATION_STEP_M) * ELEVATION_STEP_M).astype(np.float32),
        received_delay_s=(received_s[pick] - obs_s[pick]).astype(np.int32),
        **{column: a[pick].astype(np.float32) for column, a in values.items()},
    )


def merge_tables(tables: list[ReportTable]) -> ReportTable:
    """Concatenate; of a repeated (station, time) keep the version that arrived last; sort."""
    tables = [t for t in tables if len(t)]
    if not tables:
        return empty_table()
    columns = {f.name: np.concatenate([getattr(t, f.name) for t in tables]) for f in fields(ReportTable)}
    if np.any(columns["obs_time"] >= SEALED_START):
        raise ValueError(_SEALED_MESSAGE)
    obs = columns["obs_time"].astype("datetime64[s]").astype(np.int64)
    received = obs + columns["received_delay_s"].astype(np.int64)
    if not arrived_by_next_issue(obs, received).all():
        raise ValueError("a table holds a report that had not arrived by its issue time")
    pick = latest_arrival_rows(columns["station_id"], obs, received)
    return ReportTable(**{name: a[pick] for name, a in columns.items()})


def save_table(table: ReportTable, path: Path, *, attrs: dict | None = None) -> None:
    """Write an anonymised table. Positions are rounded again here, whatever built the table."""
    if len(table) and np.any(table.obs_time >= SEALED_START):
        raise ValueError(_SEALED_MESSAGE)
    if not arrived_by_next_issue(
        table.obs_time.astype("datetime64[s]").astype(np.int64), table.received_time_s()
    ).all():
        raise ValueError("refusing to store a report that had not arrived by its issue time")
    ds = xr.Dataset(
        {
            "station_id": ("obs", np.asarray(table.station_id, dtype="U11")),
            "obs_time": ("obs", table.obs_time.astype("datetime64[s]").astype(np.int64)),
            "lat": ("obs", np.round(table.lat.astype(np.float64), COORD_DECIMALS).astype(np.float32)),
            "lon": ("obs", np.round(table.lon.astype(np.float64), COORD_DECIMALS).astype(np.float32)),
            "elevation_m": (
                "obs",
                (np.round(table.elevation_m.astype(np.float64) / ELEVATION_STEP_M) * ELEVATION_STEP_M).astype(
                    np.float32
                ),
            ),
            "received_delay_s": ("obs", table.received_delay_s.astype(np.int32)),
            **{column: ("obs", getattr(table, column).astype(np.float32)) for column in VALUE_COLUMNS},
        },
        attrs={
            **(attrs or {}),
            "source": "NOAA MADIS public mesonet archive (APRSWXNET / CWOP only)",
            "privacy": "station_id is a keyed 11-char token; lat/lon rounded to 0.1 deg, elevation to 10 m",
            "obs_time": "seconds since 1970-01-01T00:00:00Z, measurement time rounded up to the second",
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
        obs_time = ds["obs_time"].values.astype(np.int64).astype("datetime64[s]")
        if obs_time.size and np.any(obs_time >= SEALED_START):
            raise ValueError(_SEALED_MESSAGE)
        raw = ds["station_id"].values
        station = np.char.decode(raw, "ascii") if raw.dtype.kind == "S" else raw.astype("U11")
        table = ReportTable(
            station_id=station.astype("U11"),
            obs_time=obs_time,
            lat=ds["lat"].values.astype(np.float32),
            lon=ds["lon"].values.astype(np.float32),
            elevation_m=ds["elevation_m"].values.astype(np.float32),
            received_delay_s=ds["received_delay_s"].values.astype(np.int32),
            **{column: ds[column].values.astype(np.float32) for column in VALUE_COLUMNS},
        )
        attrs = dict(ds.attrs)
    if not arrived_by_next_issue(obs_time.astype(np.int64), table.received_time_s()).all():
        raise StaleTableError(f"{Path(path).name} holds a report that had not arrived by its issue time")
    return (table, attrs) if with_attrs else table
