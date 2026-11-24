"""Offline tests for the hourly MADIS archive reader. Every station here is invented."""

from __future__ import annotations

import datetime as dt
import gzip
import importlib.util
import io
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.io import netcdf_file

from wrfcomp.assimilation.citizen_obs import SEALED_START, anonymize_station_id
from wrfcomp.assimilation.madis_archive import (
    ArchiveFormatError,
    load_table,
    merge_tables,
    parse_archive_hour,
    save_table,
)

KEY = bytes(range(32))
FILL = np.float32(3.4028235e38)
T0 = int(np.datetime64("2024-03-01T11:00:00", "s").astype(np.int64))
FETCH = Path(__file__).resolve().parents[1] / "scripts" / "fetch_citizen_year.py"

# id, provider, lat, lon, elevation, seconds after T0, temperature, dewpoint, speed, direction
ROWS = [
    ("SYN001", "APRSWXNET", 35.2349, -101.8312, 1094.0, 600, 290.5, 280.25, 3.0, 180.0),
    ("SYN001", "APRSWXNET", 35.2349, -101.8312, 1094.0, 1205, 291.0, 280.5, 2.5, 190.0),
    ("SYN002", "APRSWXNET", 47.6061, -122.3328, 56.0, 30, 285.0, FILL, FILL, FILL),
    ("SECRET", "MesoWest", 40.0, -100.0, 500.0, 60, 280.0, 270.0, 1.0, 90.0),  # restricted provider
    ("SYN003", "APRSWXNET", 51.5, -0.1, 20.0, 60, 283.0, 280.0, 1.0, 10.0),  # outside CONUS
    ("SYN004", "APRSWXNET", FILL, -100.0, 10.0, 60, 283.0, 280.0, 1.0, 10.0),  # no position
    ("SYN001", "APRSWXNET", 35.2349, -101.8312, 1094.0, 600, 299.0, 280.25, 3.0, 180.0),  # repeat: last wins
]


def archive_bytes(rows=ROWS, *, units="kelvin", compress=True, base=T0) -> bytes:
    buffer = io.BytesIO()
    nc = netcdf_file(buffer, "w")
    nc.createDimension("recNum", len(rows))
    nc.createDimension("maxStaIdLen", 6)
    nc.createDimension("maxProviderIdLen", 11)

    def text(name, dim, width, column):
        var = nc.createVariable(name, "c", ("recNum", dim))
        var[:] = np.array([list(r[column].ljust(width, "\x00")) for r in rows], dtype="S1")

    text("stationId", "maxStaIdLen", 6, 0)
    text("dataProvider", "maxProviderIdLen", 11, 1)
    for name, column, unit in (
        ("latitude", 2, "degree_north"),
        ("longitude", 3, "degree_east"),
        ("elevation", 4, "meter"),
        ("temperature", 6, units),
        ("dewpoint", 7, "kelvin"),
        ("windSpeed", 8, "meter/sec"),
        ("windDir", 9, "degree"),
    ):
        var = nc.createVariable(name, "f4", ("recNum",))
        var[:] = np.array([r[column] for r in rows], dtype="f4")
        var.units = unit
    for name, extra in (("observationTime", 0), ("receivedTime", 345)):
        var = nc.createVariable(name, "f8", ("recNum",))
        var[:] = np.array([base + r[5] + extra for r in rows], dtype="f8")
        var.units = "seconds since 1970-1-1 00:00:00.0"
    nc.flush()
    payload = buffer.getvalue()
    return gzip.compress(payload) if compress else payload


def test_only_citizen_records_inside_conus_are_kept_and_anonymised():
    table = parse_archive_hour(archive_bytes(), key=KEY)
    assert len(table) == 3
    assert set(table.station_id.tolist()) == {
        anonymize_station_id("SYN001", key=KEY),
        anonymize_station_id("SYN002", key=KEY),
    }
    # The restricted provider's station never reaches the table, under its name or its token.
    assert anonymize_station_id("SECRET", key=KEY) not in table.station_id.tolist()
    assert not any("SYN" in s or "SECRET" in s for s in table.station_id.tolist())
    assert all(len(s) == 11 and s.startswith("CW") for s in table.station_id.tolist())
    np.testing.assert_array_equal(np.round(table.lat * 10), table.lat * 10)
    np.testing.assert_array_equal(table.elevation_m % 10, 0)
    one = table.station_id == anonymize_station_id("SYN001", key=KEY)
    np.testing.assert_array_equal(table.lat[one], np.float32([35.2, 35.2]))
    np.testing.assert_array_equal(table.elevation_m[one], np.float32([1090, 1090]))


def test_times_are_measurement_times_to_the_second_and_the_last_repeat_wins():
    table = parse_archive_hour(archive_bytes(compress=False), key=KEY)
    one = table.station_id == anonymize_station_id("SYN001", key=KEY)
    np.testing.assert_array_equal(table.obs_time[one].astype(np.int64), [T0 + 600, T0 + 1205])
    np.testing.assert_array_equal(table.t2m[one], np.float32([299.0, 291.0]))
    np.testing.assert_array_equal(table.received_delay_s[one], [345, 345])
    two = ~one
    assert np.isnan(table.td2m[two]).all() and np.isnan(table.wind_speed[two]).all()
    assert table.obs_time.dtype == np.dtype("datetime64[s]")


def test_sealed_time_is_refused_and_errors_quote_no_record():
    sealed = int(SEALED_START.astype("datetime64[s]").astype(np.int64))
    with pytest.raises(ValueError, match="sealed") as caught:
        parse_archive_hour(archive_bytes(base=sealed), key=KEY)
    assert "SYN" not in str(caught.value)
    with pytest.raises(ArchiveFormatError) as caught:
        parse_archive_hour(archive_bytes(units="celsius"), key=KEY)
    assert "SYN" not in str(caught.value) and "35.2" not in str(caught.value)
    with pytest.raises(ArchiveFormatError) as caught:
        parse_archive_hour(b"SYN001 is not a NetCDF file", key=KEY)
    assert "SYN001" not in str(caught.value)
    assert "SYN" not in repr(parse_archive_hour(archive_bytes(), key=KEY))


def test_save_and_load_round_trip_and_positions_are_rounded_again(tmp_path):
    table = parse_archive_hour(archive_bytes(), key=KEY)
    exact = type(table)(
        **{**table.__dict__, "lat": table.lat + np.float32(0.0333), "elevation_m": table.elevation_m + 3}
    )
    path = tmp_path / "day.nc"
    save_table(exact, path)
    back = load_table(path)
    for name in table.__dict__:
        np.testing.assert_array_equal(getattr(back, name), getattr(table, name))
    merged = merge_tables([back, back])
    assert len(merged) == len(back)
    assert not list(tmp_path.glob(".*.part"))


def test_fetch_script_refuses_the_confirmation_year_and_stops_at_the_last_issue():
    spec = importlib.util.spec_from_file_location("fetch_citizen_year", FETCH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    with pytest.raises(ValueError, match="2023"):
        module.hours_of(dt.date(2023, 7, 1))
    with pytest.raises(ValueError):
        module.hours_of(dt.date(2025, 1, 1))
    assert len(module.hours_of(dt.date(2024, 12, 16))) == 24
    assert [h.hour for h in module.hours_of(dt.date(2024, 12, 17))] == list(range(12))
    assert module.hours_of(dt.date(2024, 12, 18)) == []
    assert "2023" not in module.ARCHIVE_URL
