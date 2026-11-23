"""Offline tests for the citizen-station loader and the fetch script's guards.

Every station in these tests is invented (``SYN...``, ``PRIVATE99``); no real citizen data is used.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import stat
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from wrfcomp.assimilation.citizen_obs import (
    CITIZEN_SENSOR_TYPE_ID,
    SEALED_START,
    STATION_KEY_FILENAME,
    anonymize_station_id,
    load_citizen_batch,
    load_or_create_station_key,
    merge_citizen_batches,
    parse_madis_csv,
    save_citizen_batch,
    summarize_batch,
)
from wrfcomp.assimilation.qc import quality_control

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_CSV = REPO_ROOT / "tests" / "data" / "citizen_sample.csv"
FETCH_SCRIPT = REPO_ROOT / "scripts" / "fetch_citizen_sample.py"
KEY = bytes(range(32))  # a fixed test key; the real one is random and lives in the data dir
CENTER = (35.222, -101.831)
HEADER = "STAID,OBDATE,OBTIME,PVDR,LAT,D,LON,D,ELEV,D,T,D\n"


def _load_fetch_module():
    spec = importlib.util.spec_from_file_location("fetch_citizen_sample", FETCH_SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _sample_batch():
    return parse_madis_csv(
        SAMPLE_CSV.read_text(encoding="utf-8"), region_center=CENTER, max_radius_km=150.0, key=KEY
    )


def test_parse_madis_csv_units_timestamps_filters_and_dedup() -> None:
    raw = SAMPLE_CSV.read_text(encoding="utf-8")
    # Wrap in HTML <pre> to mirror live MADIS CGI output.
    batch = parse_madis_csv(
        f"<html><body><pre>\n{raw}\n</pre></body></html>", region_center=CENTER, max_radius_km=150.0, key=KEY
    )

    # SYN999 (other provider), SYNFAR (>150 km) and SYNBAD (unreadable date) are left out.
    expected_ids = {anonymize_station_id(f"SYN00{i}", key=KEY) for i in range(1, 5)}
    assert set(batch.station_ids.tolist()) == expected_ids
    assert all(len(sid) == 11 for sid in batch.station_ids.tolist())
    assert (batch.sensor_type_id == CITIZEN_SENSOR_TYPE_ID).all()
    assert (batch.qc_flags == 0).all()

    # Real measurement times, to the minute, never moved to the hour.
    assert set(np.unique(batch.obs_time).tolist()) == {
        np.datetime64("2024-07-08T00:05:00", "ns").item(),
        np.datetime64("2024-07-08T00:15:00", "ns").item(),
    }

    syn1 = anonymize_station_id("SYN001", key=KEY)
    t_0005 = np.datetime64("2024-07-08T00:05:00", "ns")
    t_0015 = np.datetime64("2024-07-08T00:15:00", "ns")

    def val(sid: str, ts: np.datetime64, var: str) -> float:
        m = (batch.station_ids == sid) & (batch.obs_time == ts) & (batch.variable == var)
        assert int(m.sum()) == 1
        return float(batch.values[m][0])

    # MADIS serves SI units; values pass through unchanged, except precipitation (m -> mm).
    assert val(syn1, t_0005, "t2m") == pytest.approx(300.15, abs=1e-4)
    assert val(syn1, t_0005, "td2m") == pytest.approx(285.15, abs=1e-4)
    assert val(syn1, t_0005, "wind_speed") == pytest.approx(4.0, abs=1e-5)
    assert val(syn1, t_0005, "wind_gust") == pytest.approx(6.5, abs=1e-5)
    assert val(syn1, t_0005, "station_pressure") == pytest.approx(89200.0, abs=1e-2)
    assert val(syn1, t_0005, "msl") == pytest.approx(101300.0, abs=1e-2)
    assert val(syn1, t_0005, "precip") == pytest.approx(2.5, abs=1e-4)
    # Wind FROM the east (90 deg) at 4 m/s blows westward: u10 = -4, v10 = 0.
    assert val(syn1, t_0005, "u10") == pytest.approx(-4.0, abs=1e-5)
    assert val(syn1, t_0005, "v10") == pytest.approx(0.0, abs=1e-5)
    # Wind FROM the north (360 deg) at 2 m/s blows southward: u10 = 0, v10 = -2.
    syn4 = anonymize_station_id("SYN004", key=KEY)
    assert val(syn4, t_0005, "u10") == pytest.approx(0.0, abs=1e-5)
    assert val(syn4, t_0005, "v10") == pytest.approx(-2.0, abs=1e-5)
    # The duplicated 00:15 report keeps its last row.
    assert val(syn1, t_0015, "t2m") == pytest.approx(300.75, abs=1e-4)


def test_no_identifier_or_exact_position_survives_the_parser() -> None:
    batch = _sample_batch()
    raw = SAMPLE_CSV.read_text(encoding="utf-8")
    everything = " ".join(str(getattr(batch, f).tolist()) for f in batch.__dataclass_fields__)
    for line in raw.splitlines()[1:]:
        fields = line.split(",")
        assert fields[0] not in everything  # raw identifier
        assert fields[5] not in everything and fields[7] not in everything  # exact latitude, longitude
    # Positions are on the 0.1 degree grid and elevations on the 10 m grid, in memory already.
    np.testing.assert_allclose(batch.lat * 10, np.round(batch.lat * 10), atol=1e-9)
    np.testing.assert_allclose(batch.lon * 10, np.round(batch.lon * 10), atol=1e-9)
    np.testing.assert_allclose(batch.elevation_m % 10, 0, atol=1e-9)
    syn1 = batch.station_ids == anonymize_station_id("SYN001", key=KEY)
    assert set(batch.lat[syn1].tolist()) == {35.2} and set(batch.lon[syn1].tolist()) == {-101.8}
    assert set(batch.elevation_m[syn1].tolist()) == {1090.0}
    with pytest.raises(TypeError):
        parse_madis_csv(raw, round_coords=False, key=KEY)  # there is no way to ask for exact positions
    with pytest.raises(TypeError):
        parse_madis_csv(raw)  # and no default key


def test_station_tokens_depend_on_the_secret_key(tmp_path: Path) -> None:
    other = bytes(range(1, 33))
    a, b = anonymize_station_id("SYN001", key=KEY), anonymize_station_id("SYN001", key=other)
    assert a != b and a == anonymize_station_id(" SYN001 ", key=KEY)
    assert a.startswith("CW") and len(a) == 11 and "SYN" not in a
    with pytest.raises(ValueError, match="at least 16 bytes"):
        anonymize_station_id("SYN001", key=b"short")
    with pytest.raises(ValueError, match="non-empty"):
        anonymize_station_id("  ", key=KEY)

    key = load_or_create_station_key(tmp_path / "citizen")
    path = tmp_path / "citizen" / STATION_KEY_FILENAME
    assert len(key) == 32 and path.read_bytes() == key
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert load_or_create_station_key(tmp_path / "citizen") == key  # stable on this machine
    assert load_or_create_station_key(tmp_path / "elsewhere") != key  # and different anywhere else
    path.write_bytes(b"too short")
    with pytest.raises(ValueError, match="wrong length"):
        load_or_create_station_key(tmp_path / "citizen")


def test_netcdf_roundtrip_holds_only_rounded_positions(tmp_path: Path) -> None:
    csv_text = HEADER + "PRIVATE99,07/10/2024,12:07,APRSWXNET,47.612345,V,-122.334567,V,56.4,V,293.15,V\n"
    batch = parse_madis_csv(csv_text, key=KEY)
    nc_path = tmp_path / "citizen.nc"
    save_citizen_batch(batch, nc_path)
    assert not list(tmp_path.glob(".*.part"))

    blob = nc_path.read_bytes()
    for secret in (b"PRIVATE99", b"47.612", b"122.334", b"612345", b"334567"):
        assert secret not in blob
    with xr.open_dataset(nc_path, engine="h5netcdf") as ds:
        assert float(ds["lat"].values[0]) == pytest.approx(47.6, abs=1e-5)
        assert float(ds["lon"].values[0]) == pytest.approx(-122.3, abs=1e-5)
        assert float(ds["elevation_m"].values[0]) == 60.0
        assert ds["station_id"].values[0].decode() == anonymize_station_id("PRIVATE99", key=KEY)
        # Stored compactly: float32 values, fixed-width byte strings, compressed.
        assert ds["value"].dtype == np.float32 and ds["lat"].dtype == np.float32
        assert ds["station_id"].dtype == np.dtype("S11") and ds["variable"].dtype == np.dtype("S16")
        assert all(ds[name].encoding.get("zlib") for name in ds.data_vars)

    reloaded = load_citizen_batch(nc_path)
    assert reloaded.station_ids.tolist() == batch.station_ids.tolist()
    np.testing.assert_array_equal(reloaded.obs_time, batch.obs_time)
    assert reloaded.obs_time[0] == np.datetime64("2024-07-10T12:07:00", "ns")
    np.testing.assert_allclose(reloaded.values, batch.values)


def test_save_rounds_even_a_batch_built_elsewhere(tmp_path: Path) -> None:
    batch = _sample_batch()
    exact = type(batch)(
        station_ids=batch.station_ids,
        lat=batch.lat + 0.0123,
        lon=batch.lon - 0.0456,
        elevation_m=batch.elevation_m + 3.3,
        obs_time=batch.obs_time,
        sensor_type_id=batch.sensor_type_id,
        variable=batch.variable,
        values=batch.values,
    )
    save_citizen_batch(exact, tmp_path / "x.nc")
    back = load_citizen_batch(tmp_path / "x.nc")
    np.testing.assert_allclose(back.lat, batch.lat, atol=1e-5)
    np.testing.assert_allclose(back.lon, batch.lon, atol=1e-5)
    np.testing.assert_allclose(back.elevation_m, batch.elevation_m)


def test_unreadable_rows_are_skipped_without_quoting_them() -> None:
    good = "SYN001,07/08/2024,00:05,APRSWXNET,35.2346,V,-101.8127,V,1093.0,V,300.15,V\n"
    bad_rows = (
        "SECRETID,notadate,00:05,APRSWXNET,35.2346,V,-101.8127,V,1093.0,V,300.15,V\n"
        "SECRETID,07/08/2024,25:61,APRSWXNET,35.2346,V,-101.8127,V,1093.0,V,300.15,V\n"
        "SECRETID,02/30/2024,00:05,APRSWXNET,35.2346,V,-101.8127,V,1093.0,V,300.15,V\n"
        "SECRETID,07/08/2024,00:05,APRSWXNET,north,V,-101.8127,V,1093.0,V,300.15,V\n"
        "SECRETID,07/08/2024,00:05,APRSWXNET,95.0,V,-101.8127,V,1093.0,V,300.15,V\n"
        "SECRETID,07/08/2024\n"
        ",07/08/2024,00:05,APRSWXNET,35.2346,V,-101.8127,V,1093.0,V,300.15,V\n"
    )
    batch = parse_madis_csv(HEADER + bad_rows + good, key=KEY)
    assert batch.station_ids.tolist() == [anonymize_station_id("SYN001", key=KEY)]
    # A missing required column is the one error raised; its text names columns, not data.
    with pytest.raises(ValueError) as info:
        parse_madis_csv(
            "STAID,OBDATE,OBTIME,PVDR,LAT,D\nSECRETID,07/08/2024,00:05,APRSWXNET,35.2346,V\n", key=KEY
        )
    assert "SECRETID" not in str(info.value) and "35.2346" not in str(info.value)
    assert parse_madis_csv("", key=KEY).values.size == 0


def test_quality_control_runs_directly_on_loaded_citizen_batch() -> None:
    batch = _sample_batch()
    # Background equal to the observed values (zero innovation).
    res = quality_control(batch, batch.values, sigma_total=2.0, sigma_bg=2.0, official=False)
    assert len(res.weights) == len(batch.values)
    assert np.all(np.isfinite(res.weights))
    # All synthetic reports pass static/temporal/background gates and get new-station prior 0.5.
    assert np.all(res.weights == pytest.approx(0.5))


def test_sealed_observations_are_refused(tmp_path: Path) -> None:
    assert SEALED_START == np.datetime64("2025-01-01T00:00:00", "ns")  # from configs/splits.yaml
    csv_2025 = HEADER + "SECRETID,01/01/2025,00:00,APRSWXNET,35.2,V,-101.8,V,1000.0,V,280.0,V\n"
    with pytest.raises(ValueError, match="sealed split") as info:
        parse_madis_csv(csv_2025, key=KEY)
    assert "SECRETID" not in str(info.value)
    last_ok = HEADER + "SYN001,12/31/2024,23:59,APRSWXNET,35.2,V,-101.8,V,1000.0,V,280.0,V\n"
    batch = parse_madis_csv(last_ok, key=KEY)
    assert batch.obs_time[0] == np.datetime64("2024-12-31T23:59:00", "ns")
    sealed = type(batch)(
        station_ids=batch.station_ids,
        lat=batch.lat,
        lon=batch.lon,
        elevation_m=batch.elevation_m,
        obs_time=np.array(["2025-01-01T00:00:00"], dtype="datetime64[ns]"),
        sensor_type_id=batch.sensor_type_id,
        variable=batch.variable,
        values=batch.values,
    )
    with pytest.raises(ValueError, match="sealed split"):
        save_citizen_batch(sealed, tmp_path / "no.nc")
    with pytest.raises(ValueError, match="sealed split"):
        merge_citizen_batches([batch, sealed])
    assert not (tmp_path / "no.nc").exists()


def test_merge_and_summary_are_aggregates_only() -> None:
    batch = _sample_batch()
    merged = merge_citizen_batches([batch, batch])  # idempotent
    assert merged.station_ids.tolist() == batch.station_ids.tolist()
    np.testing.assert_array_equal(merged.obs_time, batch.obs_time)
    assert merged.variable.tolist() == batch.variable.tolist()
    np.testing.assert_array_equal(merged.values, batch.values)
    assert merge_citizen_batches([]).values.size == 0

    # The later batch wins on a duplicate (station, time, variable).
    later = type(batch)(
        station_ids=batch.station_ids[:1],
        lat=batch.lat[:1],
        lon=batch.lon[:1],
        elevation_m=batch.elevation_m[:1],
        obs_time=batch.obs_time[:1],
        sensor_type_id=batch.sensor_type_id[:1],
        variable=batch.variable[:1],
        values=batch.values[:1] + 1,
    )
    both = merge_citizen_batches([batch, later])
    assert len(both.values) == len(batch.values) and both.values[0] == batch.values[0] + 1

    summary = summarize_batch(merged)
    assert summary["n_stations"] == 4
    assert summary["n_reports"] == 5
    assert summary["median_interval_min"] == pytest.approx(10.0)
    assert summary["variables"]["t2m"] == 5 and summary["variables"]["precip"] == 5
    text = json.dumps(summary)
    for sid in batch.station_ids.tolist():
        assert sid not in text  # not even the tokens
    assert "SYN" not in text and "35.2" not in text and "101.8" not in text


def test_stored_size_is_capped_and_a_large_sample_stays_small(tmp_path: Path) -> None:
    mod = _load_fetch_module()
    # 400 000 random rows: under 30 bytes a row (the first, uncompressed layout took about 123).
    rng = np.random.default_rng(7)
    n = 400_000
    stations = np.array([anonymize_station_id(f"SYN{i:03d}", key=KEY) for i in range(300)])
    batch = type(_sample_batch())(
        station_ids=stations[rng.integers(0, 300, n)],
        lat=np.round(rng.uniform(46.5, 48.5, n), 1),
        lon=np.round(rng.uniform(-123.5, -121.0, n), 1),
        elevation_m=np.round(rng.uniform(0, 900, n), -1),
        obs_time=np.datetime64("2024-07-08T00:00", "ns")
        + rng.integers(0, 14 * 1440, n) * np.timedelta64(1, "m"),
        sensor_type_id=np.full(n, CITIZEN_SENSOR_TYPE_ID),
        variable=np.array(
            ["t2m", "td2m", "u10", "v10", "wind_speed", "wind_gust", "precip", "station_pressure"]
        )[rng.integers(0, 8, n)],
        values=rng.normal(290, 5, n).astype(np.float32),
    )
    path = tmp_path / "citizen" / "big.nc"
    save_citizen_batch(batch, path)
    assert path.stat().st_size < 30 * n

    # The cap counts what is stored, not only what is transferred.
    assert mod.stored_bytes(tmp_path / "citizen") == path.stat().st_size
    mod.check_stored_cap(tmp_path / "citizen", cap_bytes=path.stat().st_size)
    with pytest.raises(RuntimeError, match="stored-size cap"):
        mod.check_stored_cap(tmp_path / "citizen", cap_bytes=path.stat().st_size - 1)


def test_fetch_script_guards_and_logs(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_fetch_module()
    with pytest.raises(ValueError, match="2025-01-01"):
        mod.iter_windows(dt.date(2024, 12, 31), dt.date(2025, 1, 1))
    with pytest.raises(ValueError, match="2025-01-01"):
        mod.build_madis_window_url(dt.datetime(2025, 1, 1, 1, 0), (33.8, -103.5, 36.6, -100.1))
    windows = mod.iter_windows(dt.date(2024, 7, 8), dt.date(2024, 7, 21))
    assert len(windows) == 14 * 12
    assert sum(w for _, w in windows) == 14 * 24
    assert mod.MAX_CONCURRENCY == 4 and mod.HARD_CAP_BYTES == 500 * 1024 * 1024
    with pytest.raises(ValueError, match="concurrency"):
        mod.fetch_region(
            mod.DEFAULT_REGIONS[0],
            windows[:1],
            key=KEY,
            concurrency=5,
            byte_budget_lock=mod.threading.Lock(),
            byte_counter=[0],
        )

    # A failure must not leak response content: the error carries the exception class name only.
    def boom(*_args, **_kwargs):
        raise ValueError("row SECRETID at 47.612345 could not be read")

    monkeypatch.setattr(mod.urllib.request, "urlopen", boom)
    monkeypatch.setattr(mod.time, "sleep", lambda _s: None)
    with pytest.raises(RuntimeError) as info:
        mod._fetch_one_window(mod.DEFAULT_REGIONS[0], windows[0][0], 2, key=KEY)
    assert "SECRETID" not in str(info.value) and "47.6" not in str(info.value)
    assert "ValueError" in str(info.value)
    assert info.value.__cause__ is None and info.value.__suppress_context__
