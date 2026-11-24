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

Times are measurement times (``observationTime``), kept to the second and never moved.
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
_MAX_DELAY_S = 32767


class ArchiveFormatError(ValueError):
    """The file is not the expected archive layout. Carries no record content."""


@dataclass(frozen=True, repr=False)
class ReportTable:
    """One row per (station, measurement time). Array contents are omitted from repr."""

    station_id: np.ndarray  # U11 keyed tokens
    obs_time: np.ndarray  # datetime64[s], UTC
    lat: np.ndarray  # float32, 0.1 degree
    lon: np.ndarray  # float32, 0.1 degree
    elevation_m: np.ndarray  # float32, 10 m
    t2m: np.ndarray  # float32 K, NaN when not reported
    td2m: np.ndarray  # float32 K
    wind_speed: np.ndarray  # float32 m/s
    wind_dir: np.ndarray  # float32 degrees, direction the wind comes from
    received_delay_s: np.ndarray  # int16, arrival minus measurement, -1 when unknown

    def __len__(self) -> int:
        return len(self.station_id)


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
        received_delay_s=np.empty(0, dtype=np.int16),
    )


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
    token_cache: dict[str, str] | None = None,
) -> ReportTable:
    """Parse one hourly archive file (gzip or plain NetCDF-3 bytes) into an anonymised table.

    Keeps ``APRSWXNET`` records inside the CONUS box only. Skips a record with an unreadable
    identifier, time or position, without quoting it. One row per (station, time); the last record
    wins, as in the CSV loader. ``token_cache`` may be passed to reuse tokens between hours; it
    holds raw identifiers in memory and must not be stored or logged by the caller.
    """
    try:
        body = gzip.decompress(payload) if payload[:2] == b"\x1f\x8b" else payload
        nc = netcdf_file(io.BytesIO(body), "r", mmap=False, maskandscale=False)
    except Exception as exc:  # noqa: BLE001 - class name only, a message could quote content
        raise ArchiveFormatError(f"cannot read archive file ({type(exc).__name__})") from None
    try:
        needed = ("dataProvider", "stationId", "latitude", "longitude", "elevation", "observationTime")
        missing = [name for name in (*needed, *VALUE_FIELDS) if name not in nc.variables]
        if missing:
            raise ArchiveFormatError(f"archive file lacks fields: {missing}")
        for name, (_, units) in VALUE_FIELDS.items():
            if _units(nc, name) != units:
                raise ArchiveFormatError(f"archive field {name} is not in {units}")
        if not _units(nc, "observationTime").startswith("seconds since 1970-1-1 00:00:00"):
            raise ArchiveFormatError("archive observationTime has an unexpected epoch")

        # Restricted providers are dropped here, before anything else is read for them.
        keep = np.flatnonzero(_text_column(nc, "dataProvider") == CWOP_PROVIDER)
        if not keep.size:
            return empty_table()
        raw_id = _text_column(nc, "stationId")[keep]
        lat = _number_column(nc, "latitude")[keep]
        lon = _number_column(nc, "longitude")[keep]
        elev = _number_column(nc, "elevation")[keep]
        seconds = _number_column(nc, "observationTime")[keep]
        received = (
            _number_column(nc, "receivedTime")[keep]
            if "receivedTime" in nc.variables
            else np.full(keep.size, np.nan)
        )
        values = {column: _number_column(nc, name)[keep] for name, (column, _) in VALUE_FIELDS.items()}
    finally:
        nc.close()

    south, west, north, east = CONUS_BOX
    with np.errstate(invalid="ignore"):
        ok = (
            (np.char.str_len(raw_id) > 0)
            & np.isfinite(lat)
            & np.isfinite(lon)
            & np.isfinite(elev)
            & np.isfinite(seconds)
            & (seconds >= 0)
            & (lat >= south)
            & (lat <= north)
            & (lon >= west)
            & (lon <= east)
        )
    if not ok.any():
        return empty_table()
    raw_id, lat, lon, elev, seconds, received = (a[ok] for a in (raw_id, lat, lon, elev, seconds, received))
    values = {column: a[ok] for column, a in values.items()}

    obs_time = np.floor(seconds).astype(np.int64).astype("datetime64[s]")
    if np.any(obs_time >= SEALED_START):
        raise ValueError(_SEALED_MESSAGE)

    cache = token_cache if token_cache is not None else {}
    unique, inverse = np.unique(raw_id, return_inverse=True)
    tokens = np.empty(unique.size, dtype="U11")
    for j, identifier in enumerate(unique.tolist()):
        token = cache.get(identifier)
        if token is None:
            token = cache[identifier] = anonymize_station_id(identifier, key=key)
        tokens[j] = token
    station = tokens[inverse]

    with np.errstate(invalid="ignore"):
        delay = received - seconds
        delay = np.where(np.isfinite(delay) & (delay >= 0), np.minimum(delay, _MAX_DELAY_S), -1)

    # Last record wins for a repeated (station, time), then sort by station and time.
    row = np.arange(station.size)
    order = np.lexsort((row, obs_time, station))
    last = np.r_[
        (station[order][1:] != station[order][:-1]) | (obs_time[order][1:] != obs_time[order][:-1]), True
    ]
    pick = order[last]
    return ReportTable(
        station_id=station[pick],
        obs_time=obs_time[pick],
        lat=np.round(lat[pick], COORD_DECIMALS).astype(np.float32),
        lon=np.round(lon[pick], COORD_DECIMALS).astype(np.float32),
        elevation_m=(np.round(elev[pick] / ELEVATION_STEP_M) * ELEVATION_STEP_M).astype(np.float32),
        received_delay_s=delay[pick].astype(np.int16),
        **{column: a[pick].astype(np.float32) for column, a in values.items()},
    )


def merge_tables(tables: list[ReportTable]) -> ReportTable:
    """Concatenate, keep the last row of a repeated (station, time), sort by station and time."""
    tables = [t for t in tables if len(t)]
    if not tables:
        return empty_table()
    columns = {f.name: np.concatenate([getattr(t, f.name) for t in tables]) for f in fields(ReportTable)}
    if np.any(columns["obs_time"] >= SEALED_START):
        raise ValueError(_SEALED_MESSAGE)
    station, when = columns["station_id"], columns["obs_time"]
    order = np.lexsort((np.arange(station.size), when, station))
    last = np.r_[(station[order][1:] != station[order][:-1]) | (when[order][1:] != when[order][:-1]), True]
    pick = order[last]
    return ReportTable(**{name: a[pick] for name, a in columns.items()})


def save_table(table: ReportTable, path: Path) -> None:
    """Write an anonymised table. Positions are rounded again here, whatever built the table."""
    if len(table) and np.any(table.obs_time >= SEALED_START):
        raise ValueError(_SEALED_MESSAGE)
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
            "received_delay_s": ("obs", table.received_delay_s.astype(np.int16)),
            **{column: ("obs", getattr(table, column).astype(np.float32)) for column in VALUE_COLUMNS},
        },
        attrs={
            "source": "NOAA MADIS public mesonet archive (APRSWXNET / CWOP only)",
            "privacy": "station_id is a keyed 11-char token; lat/lon rounded to 0.1 deg, elevation to 10 m",
            "obs_time": "seconds since 1970-01-01T00:00:00Z, measurement time",
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


def load_table(path: Path) -> ReportTable:
    with xr.open_dataset(path, engine="h5netcdf", decode_times=False) as ds:
        obs_time = ds["obs_time"].values.astype(np.int64).astype("datetime64[s]")
        if obs_time.size and np.any(obs_time >= SEALED_START):
            raise ValueError(_SEALED_MESSAGE)
        raw = ds["station_id"].values
        station = np.char.decode(raw, "ascii") if raw.dtype.kind == "S" else raw.astype("U11")
        return ReportTable(
            station_id=station.astype("U11"),
            obs_time=obs_time,
            lat=ds["lat"].values.astype(np.float32),
            lon=ds["lon"].values.astype(np.float32),
            elevation_m=ds["elevation_m"].values.astype(np.float32),
            received_delay_s=ds["received_delay_s"].values.astype(np.int16),
            **{column: ds[column].values.astype(np.float32) for column in VALUE_COLUMNS},
        )
