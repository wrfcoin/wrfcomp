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
import xarray as xr
from scipy.io import netcdf_file

from wrfcomp.assimilation.citizen_obs import SEALED_START, anonymize_station_id
from wrfcomp.assimilation.madis_archive import (
    ARRIVAL_POLICY,
    ArchiveFormatError,
    StaleTableError,
    arrived_by_next_issue,
    load_table,
    merge_tables,
    parse_archive_hour,
    save_table,
)

KEY = bytes(range(32))
FILL = np.float32(3.4028235e38)
YEAR = np.datetime64("2024-01-01T00:00:00", "s")
ISSUE = int(np.datetime64("2024-03-01T12:00:00", "s").astype(np.int64))  # an issue time, T
T0 = ISSUE - 3600
FETCH = Path(__file__).resolve().parents[1] / "scripts" / "fetch_citizen_year.py"
HERE = (35.2349, -101.8312, 1094.0)

# id, provider, lat, lon, elevation, measured (s), temperature, dewpoint, speed, direction, received (s)
ROWS = [
    ("SYN001", "APRSWXNET", *HERE, T0 + 600, 290.5, 280.25, 3.0, 180.0, T0 + 900),
    ("SYN001", "APRSWXNET", *HERE, T0 + 1205, 291.0, 280.5, 2.5, 190.0, T0 + 1500),
    ("SYN002", "APRSWXNET", 47.6061, -122.3328, 56.0, T0 + 30, 285.0, FILL, FILL, FILL, T0 + 400),
    ("SECRET", "MesoWest", 40.0, -100.0, 500.0, T0 + 60, 280.0, 270.0, 1.0, 90.0, T0 + 90),  # restricted
    ("SYN003", "APRSWXNET", 51.5, -0.1, 20.0, T0 + 60, 283.0, 280.0, 1.0, 10.0, T0 + 90),  # outside CONUS
    ("SYN004", "APRSWXNET", FILL, -100.0, 10.0, T0 + 60, 283.0, 280.0, 1.0, 10.0, T0 + 90),  # no position
    ("SYN001", "APRSWXNET", *HERE, T0 + 600, 299.0, 280.25, 3.0, 180.0, T0 + 1000),  # repeat, arrived later
]


def row(station, measured, received, temperature=290.0):
    return (station, "APRSWXNET", *HERE, measured, temperature, 280.0, 2.0, 90.0, received)


def archive_bytes(rows=ROWS, *, units="kelvin", compress=True, received=True) -> bytes:
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
    for name, column in (("observationTime", 5), ("receivedTime", 10)):
        if name == "receivedTime" and not received:
            continue
        var = nc.createVariable(name, "f8", ("recNum",))
        var[:] = np.array([r[column] for r in rows], dtype="f8")
        var.units = "seconds since 1970-1-1 00:00:00.0"
    nc.flush()
    payload = buffer.getvalue()
    return gzip.compress(payload) if compress else payload


def parse(rows=ROWS, **kwargs):
    return parse_archive_hour(archive_bytes(rows, **kwargs), key=KEY, not_before=YEAR)


def test_only_citizen_records_inside_conus_are_kept_and_anonymised():
    table = parse()
    assert len(table) == 3
    one, two = (anonymize_station_id(s, key=KEY) for s in ("SYN001", "SYN002"))
    assert set(table.station_id.tolist()) == {one, two}
    # The restricted provider's station never reaches the table, under its name or its token.
    assert anonymize_station_id("SECRET", key=KEY) not in table.station_id.tolist()
    assert not any("SYN" in s or "SECRET" in s for s in table.station_id.tolist())
    assert all(len(s) == 11 and s.startswith("CW") for s in table.station_id.tolist())
    np.testing.assert_array_equal(np.round(table.lat * 10), table.lat * 10)
    np.testing.assert_array_equal(table.elevation_m % 10, 0)
    np.testing.assert_array_equal(table.lat[table.station_id == one], np.float32([35.2, 35.2]))
    np.testing.assert_array_equal(table.elevation_m[table.station_id == one], np.float32([1090, 1090]))


def test_of_two_timely_versions_the_one_that_arrived_last_is_kept():
    table = parse(compress=False)
    one = table.station_id == anonymize_station_id("SYN001", key=KEY)
    np.testing.assert_array_equal(table.obs_time[one].astype(np.int64), [T0 + 600, T0 + 1205])
    np.testing.assert_array_equal(table.t2m[one], np.float32([299.0, 291.0]))
    np.testing.assert_array_equal(table.received_delay_s[one], [400, 295])
    assert np.isnan(table.td2m[~one]).all() and np.isnan(table.wind_speed[~one]).all()
    assert table.obs_time.dtype == np.dtype("datetime64[s]")
    counts: dict = {}
    parse_archive_hour(archive_bytes(), key=KEY, not_before=YEAR, counts=counts)
    assert counts == {"records": 4, "before_start": 0, "not_arrived": 0, "repeats_merged": 1}


def test_a_late_repeat_never_displaces_the_timely_report():
    """Review case: measured T-600 s, received T-100 s; then a repeat received at T+1 s is added."""
    timely = row("SYN001", ISSUE - 600, ISSUE - 100, temperature=290.0)
    alone = parse([timely])
    for late in (
        row("SYN001", ISSUE - 600, ISSUE + 1, temperature=305.0),
        row("SYN001", ISSUE - 600, ISSUE + 86400, temperature=305.0),
    ):
        for rows in ([timely, late], [late, timely]):
            both = parse(rows)
            assert len(both) == len(alone) == 1
            for name in alone.__dict__:
                np.testing.assert_array_equal(getattr(both, name), getattr(alone, name))
    assert len(parse([row("SYN001", ISSUE - 600, ISSUE + 1)])) == 0  # only the late version: nothing


def test_arrival_is_judged_on_the_source_timestamps_at_full_precision():
    """Review cases: an arrival 0.25 s after T is late; a known early arrival with a long delay is kept."""
    assert len(parse([row("SYN001", ISSUE - 600, ISSUE + 0.25)])) == 0
    assert len(parse([row("SYN001", ISSUE - 600.75, ISSUE)])) == 1
    assert len(parse([row("SYN001", ISSUE - 0.5, ISSUE - 0.25)])) == 1
    long_delay = parse([row("SYN001", ISSUE - 34100, ISSUE - 100)])
    assert len(long_delay) == 1 and long_delay.received_delay_s[0] == 34000
    # Measured a quarter second after T: it belongs to the next issue, and its stored time is after T.
    after = parse([row("SYN001", ISSUE + 0.25, ISSUE + 300)])
    assert len(after) == 1 and after.obs_time[0].astype(np.int64) == ISSUE + 1
    assert len(parse([row("SYN001", ISSUE - 600, FILL)])) == 0  # no arrival time
    with pytest.raises(ArchiveFormatError):
        parse([row("SYN001", ISSUE - 600, ISSUE - 100)], received=False)
    np.testing.assert_array_equal(
        arrived_by_next_issue(
            [ISSUE - 600, ISSUE - 600, ISSUE, ISSUE + 1, np.nan], [ISSUE, ISSUE + 0.001, ISSUE, ISSUE + 5, 1]
        ),
        [True, False, True, True, False],
    )


def test_nothing_measured_before_the_start_of_the_year_is_kept():
    start = int(YEAR.astype(np.int64))
    rows = [
        row("SYN001", start - 1, start + 200),
        row("SYN001", start - 0.5, start + 200),
        row("SYN001", start + 1, start + 200),
    ]
    counts: dict = {}
    table = parse_archive_hour(archive_bytes(rows), key=KEY, not_before=YEAR, counts=counts)
    assert len(table) == 1 and table.obs_time[0] == YEAR + np.timedelta64(1, "s")
    assert counts["before_start"] == 2


def test_sealed_time_is_refused_and_errors_quote_no_record():
    sealed = int(SEALED_START.astype("datetime64[s]").astype(np.int64))
    with pytest.raises(ValueError, match="sealed") as caught:
        parse([row("SYN001", sealed, sealed + 5)])
    assert "SYN" not in str(caught.value)
    with pytest.raises(ArchiveFormatError) as caught:
        parse(units="celsius")
    assert "SYN" not in str(caught.value) and "35.2" not in str(caught.value)
    with pytest.raises(ArchiveFormatError) as caught:
        parse_archive_hour(b"SYN001 is not a NetCDF file", key=KEY, not_before=YEAR)
    assert "SYN001" not in str(caught.value)
    assert "SYN" not in repr(parse())


def test_save_and_load_round_trip_and_positions_are_rounded_again(tmp_path):
    table = parse()
    exact = type(table)(
        **{**table.__dict__, "lat": table.lat + np.float32(0.0333), "elevation_m": table.elevation_m + 3}
    )
    path = tmp_path / "day.nc"
    save_table(exact, path, attrs={"day": "2024-03-01", "hours_missing": "07"})
    back, attrs = load_table(path, with_attrs=True)
    for name in table.__dict__:
        np.testing.assert_array_equal(getattr(back, name), getattr(table, name))
    assert attrs["arrival_policy"] == ARRIVAL_POLICY and attrs["hours_missing"] == "07"
    assert len(merge_tables([back, back])) == len(back)
    assert not list(tmp_path.glob(".*.part"))


def test_a_table_from_before_the_amendment_or_with_a_late_report_is_refused(tmp_path):
    table = parse()
    path = tmp_path / "day.nc"
    save_table(table, path)
    with xr.open_dataset(path, engine="h5netcdf", decode_times=False) as ds:
        old = ds.load()
    for change in ({"arrival_policy": "none"}, None):
        stale = old.copy()
        if change is None:
            del stale.attrs["arrival_policy"]
        else:
            stale.attrs.update(change)
        stale.to_netcdf(tmp_path / "stale.nc", engine="h5netcdf")
        with pytest.raises(StaleTableError):
            load_table(tmp_path / "stale.nc")
    late = type(table)(**{**table.__dict__, "received_delay_s": table.received_delay_s + 86400})
    with pytest.raises(ValueError, match="not arrived"):
        save_table(late, tmp_path / "late.nc")
    with pytest.raises(ValueError, match="not arrived"):
        merge_tables([late])


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
