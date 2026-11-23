"""Unit tests for the NOAA ISD station truth loader (never touches the network)."""

from __future__ import annotations

import io
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy.io import netcdf_file

from wrfcomp.harness.schemas import validate_truth_dataset as validate_schema_truth
from wrfcomp.harness.truth import (
    OFFSET_FILL_VALUE,
    OFFSET_VARS,
    PRIMARY_VARS,
    build_truth,
    compute_completeness,
    download_station_year,
    match_station_to_hours,
    parse_isd_csv,
    parse_slp,
    parse_tmp_or_dew,
    parse_wnd,
    validate_truth_dataset,
    wind_to_uv,
    write_truth_netcdf,
)

DATA_DIR = Path(__file__).resolve().parent / "data"
SAMPLE_AMARILLO = DATA_DIR / "72363023047_2024_sample.csv"
SAMPLE_LLANO_AWOS = DATA_DIR / "72011053983_2024_awos_sample.csv"


def test_sample_data_budget_under_200kb() -> None:
    files = [p for p in DATA_DIR.iterdir() if p.is_file()]
    assert len(files) >= 1
    total_bytes = sum(p.stat().st_size for p in files)
    assert total_bytes < 200_000


def test_wind_to_uv_cardinal_and_diagonal() -> None:
    # North wind (from 0 or 360 deg) at 10 m/s blows southward: u10 = 0, v10 = -10
    for north_deg in (0.0, 360.0):
        u, v = wind_to_uv(north_deg, 10.0)
        assert u == pytest.approx(0.0, abs=1e-12)
        assert v == pytest.approx(-10.0, abs=1e-12)

    # East wind (from 90 deg) blows westward: u10 = -10, v10 = 0
    u, v = wind_to_uv(90.0, 10.0)
    assert u == pytest.approx(-10.0, abs=1e-12)
    assert v == pytest.approx(0.0, abs=1e-12)

    # South wind (from 180 deg) blows northward: u10 = 0, v10 = +10
    u, v = wind_to_uv(180.0, 10.0)
    assert u == pytest.approx(0.0, abs=1e-12)
    assert v == pytest.approx(10.0, abs=1e-12)

    # West wind (from 270 deg) blows eastward: u10 = +10, v10 = 0
    u, v = wind_to_uv(270.0, 10.0)
    assert u == pytest.approx(10.0, abs=1e-12)
    assert v == pytest.approx(0.0, abs=1e-12)

    # SW wind (from 225 deg) blows northeastward
    u, v = wind_to_uv(225.0, 10.0)
    expected = 10.0 / math.sqrt(2.0)
    assert u == pytest.approx(expected, rel=1e-6)
    assert v == pytest.approx(expected, rel=1e-6)


def test_scalar_parsers_edge_cases() -> None:
    # Temperatures & missing/QC
    assert parse_tmp_or_dew("+0123,1") == pytest.approx(285.45)
    assert parse_tmp_or_dew("-0044,1") == pytest.approx(268.75)
    assert parse_tmp_or_dew("+0070,C") == pytest.approx(280.15)  # AWOS whole-deg C accepted
    assert math.isnan(parse_tmp_or_dew("+9999,9"))  # missing sentinel
    assert math.isnan(parse_tmp_or_dew("+0183,6"))  # suspect NCEI
    assert math.isnan(parse_tmp_or_dew("+0183,2"))  # suspect
    assert math.isnan(parse_tmp_or_dew("+0183,3"))  # erroneous
    assert math.isnan(parse_tmp_or_dew("+0183,I"))  # validator-inserted (rejected)

    # SLP
    assert parse_slp("10246,1") == pytest.approx(102460.0)
    assert math.isnan(parse_slp("99999,9"))
    assert math.isnan(parse_slp("10132,6"))

    # Winds: normal, calm (C and 9 with 0000), variable (V), failed QC
    u, v = parse_wnd("360,1,N,0100,1")
    assert u == pytest.approx(0.0, abs=1e-12)
    assert v == pytest.approx(-10.0, abs=1e-12)

    assert parse_wnd("999,9,C,0000,5") == (0.0, 0.0)
    assert parse_wnd("999,9,9,0000,1") == (0.0, 0.0)
    assert all(math.isnan(x) for x in parse_wnd("999,9,V,0026,5"))
    assert all(math.isnan(x) for x in parse_wnd("999,9,9,9999,9"))
    assert all(math.isnan(x) for x in parse_wnd("180,6,N,0050,5"))  # suspect dir QC
    assert all(math.isnan(x) for x in parse_wnd("180,5,N,0050,2"))  # suspect speed QC


def test_parse_real_sample_file_amarillo() -> None:
    df = parse_isd_csv(SAMPLE_AMARILLO)
    assert not df.empty
    assert list(df.columns) == ["obs_time", "t2m", "td2m", "u10", "v10", "msl"]

    by_time = df.set_index("obs_time")

    # 1. Negative temperature, dewpoint, SLP, and wind on 2024-01-17T00:00:00
    #    Raw: WND=230,1,N,0046,1 TMP=-0044,1 DEW=-0172,1 SLP=10246,1
    row_neg = by_time.loc[pd.Timestamp("2024-01-17T00:00:00")]
    assert row_neg["t2m"] == pytest.approx(273.15 - 4.4, abs=1e-3)
    assert row_neg["td2m"] == pytest.approx(273.15 - 17.2, abs=1e-3)
    assert row_neg["msl"] == pytest.approx(102460.0, abs=1e-1)
    exp_u, exp_v = wind_to_uv(230.0, 4.6)
    assert row_neg["u10"] == pytest.approx(exp_u, abs=1e-4)
    assert row_neg["v10"] == pytest.approx(exp_v, abs=1e-4)

    # 2. Failed quality codes: TMP=+0183,6 at 20:53 and TMP=+0183,2 at 21:00
    for ts_str in ("2024-01-17T20:53:00", "2024-01-17T21:00:00"):
        row_bad = by_time.loc[pd.Timestamp(ts_str)]
        assert np.isnan(row_bad["t2m"])
        assert row_bad["td2m"] == pytest.approx(273.15 - 16.7, abs=1e-3)
        assert row_bad["msl"] == pytest.approx(100330.0, abs=1e-1)

    # 3. Missing sentinels on SOD row (2024-01-17T05:59:00)
    row_sod = by_time.loc[pd.Timestamp("2024-01-17T05:59:00")]
    for var in PRIMARY_VARS:
        assert np.isnan(row_sod[var])

    # 4. Calm wind: 2024-01-18T17:53:00 WND=999,9,C,0000,5
    row_calm = by_time.loc[pd.Timestamp("2024-01-18T17:53:00")]
    assert row_calm["u10"] == 0.0
    assert row_calm["v10"] == 0.0
    assert row_calm["t2m"] == pytest.approx(273.15 + 11.7, abs=1e-3)

    # 5. Variable wind: 2024-01-18T18:53:00 WND=999,9,V,0026,5 -> u10/v10 NaN
    row_var = by_time.loc[pd.Timestamp("2024-01-18T18:53:00")]
    assert np.isnan(row_var["u10"])
    assert np.isnan(row_var["v10"])
    assert row_var["t2m"] == pytest.approx(273.15 + 14.4, abs=1e-3)


def test_parse_real_sample_file_awos_qc_c() -> None:
    df = parse_isd_csv(SAMPLE_LLANO_AWOS)
    by_time = df.set_index("obs_time")
    # 2024-01-01T05:55:00 WND=999,9,C,0000,5 TMP=+0070,C DEW=+0050,C SLP=99999,9
    row = by_time.loc[pd.Timestamp("2024-01-01T05:55:00")]
    assert row["t2m"] == pytest.approx(280.15, abs=1e-3)
    assert row["td2m"] == pytest.approx(278.15, abs=1e-3)
    assert row["u10"] == 0.0
    assert row["v10"] == 0.0
    assert np.isnan(row["msl"])


def test_hour_matching_rules_and_offsets() -> None:
    obs_df = pd.DataFrame(
        {
            "obs_time": pd.to_datetime(
                [
                    "2024-01-10T00:53:00",  # belongs to 01:00 (-7 min)
                    "2024-01-10T01:20:00",  # 20 min past 01:00, 40 min before 02:00 -> matches nothing
                    "2024-01-10T02:55:00",  # tie for 03:00 (-5 min) -> earlier wins!
                    "2024-01-10T03:05:00",  # tie for 03:00 (+5 min) -> loses tie
                    "2024-01-10T04:15:00",  # exact +15 min boundary for 04:00 -> matches (+15 min)
                    "2024-01-10T04:45:00",  # exact -15 min boundary for 05:00 -> matches (-15 min)
                    "2024-01-10T06:16:00",  # outside +15 min for 06:00 -> matches nothing
                ]
            ),
            "t2m": np.array([281.0, 282.0, 283.0, 284.0, 285.0, 286.0, 287.0], dtype=np.float32),
            "td2m": np.array([275.0, 276.0, 277.0, 278.0, 279.0, 280.0, 281.0], dtype=np.float32),
            "u10": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float32),
            "v10": np.array([-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0], dtype=np.float32),
            "msl": np.array([101000.0] * 7, dtype=np.float32),
        }
    )
    valid_times = pd.date_range("2024-01-10T00:00:00", "2024-01-10T06:00:00", freq="h")
    vals, offs = match_station_to_hours(obs_df, valid_times, tolerance_minutes=15)

    # 00:00 -> no match
    assert np.isnan(vals["t2m"][0])
    assert offs["t2m"][0] == OFFSET_FILL_VALUE

    # 01:00 -> matched 00:53 (-7 min)
    assert vals["t2m"][1] == pytest.approx(281.0)
    assert offs["t2m"][1] == -7

    # 02:00 -> 01:20 is outside ±15 min -> no match
    assert np.isnan(vals["t2m"][2])
    assert offs["t2m"][2] == OFFSET_FILL_VALUE

    # 03:00 -> tie between 02:55 (-5 min) and 03:05 (+5 min) -> earlier (283.0, -5) wins
    assert vals["t2m"][3] == pytest.approx(283.0)
    assert offs["t2m"][3] == -5

    # 04:00 -> exact +15 min boundary (04:15) matches
    assert vals["t2m"][4] == pytest.approx(285.0)
    assert offs["t2m"][4] == 15

    # 05:00 -> exact -15 min boundary (04:45) matches
    assert vals["t2m"][5] == pytest.approx(286.0)
    assert offs["t2m"][5] == -15

    # 06:00 -> 06:16 is outside ±15 min -> no match
    assert np.isnan(vals["t2m"][6])
    assert offs["t2m"][6] == OFFSET_FILL_VALUE


def test_td2m_gt_t2m_enforced_after_per_variable_matching() -> None:
    # At 00:53, both t2m=280 and td2m=279 are valid.
    # At 01:00, t2m=278 is valid (colder) while td2m is NaN.
    # Per-variable matching would pair t2m=278 (01:00) with td2m=279 (00:53) -> guard must mask td2m!
    obs_df = pd.DataFrame(
        {
            "obs_time": pd.to_datetime(["2024-01-10T00:53:00", "2024-01-10T01:00:00"]),
            "t2m": np.array([280.0, 278.0], dtype=np.float32),
            "td2m": np.array([279.0, np.nan], dtype=np.float32),
            "u10": np.array([0.0, 0.0], dtype=np.float32),
            "v10": np.array([0.0, 0.0], dtype=np.float32),
            "msl": np.array([101320.0, np.nan], dtype=np.float32),
        }
    )
    valid_times = pd.date_range("2024-01-10T01:00:00", periods=1, freq="h")
    vals, offs = match_station_to_hours(obs_df, valid_times, tolerance_minutes=15)
    assert vals["t2m"][0] == pytest.approx(278.0)
    assert offs["t2m"][0] == 0
    assert np.isnan(vals["td2m"][0])
    assert offs["td2m"][0] == OFFSET_FILL_VALUE
    assert vals["msl"][0] == pytest.approx(101320.0)
    assert offs["msl"][0] == -7


def test_build_truth_contract_and_netcdf_roundtrip(tmp_path: Path) -> None:
    cache_2024 = tmp_path / "2024"
    cache_2024.mkdir(parents=True)
    shutil.copy(SAMPLE_AMARILLO, cache_2024 / "72363023047.csv")
    shutil.copy(SAMPLE_LLANO_AWOS, cache_2024 / "72011053983.csv")

    ds = build_truth(
        station_ids=["72363023047", "72011053983"],
        start="2024-01-17T00:00:00",
        end="2024-01-18T19:00:00",
        cache_dir=tmp_path,
    )

    # Contract validation
    validate_truth_dataset(ds)
    assert tuple(ds.dims) == ("valid_time", "station_id")
    assert ds.sizes["valid_time"] == 44
    assert ds.sizes["station_id"] == 2
    assert ds["valid_time"].dtype == np.dtype("datetime64[ns]")
    assert list(ds["station_id"].to_numpy()) == ["72363023047", "72011053983"]
    for coord in ("latitude", "longitude", "elevation_m"):
        assert coord in ds.coords
        assert np.all(np.isfinite(ds.coords[coord].to_numpy()))

    for var in PRIMARY_VARS:
        assert ds[var].dims == ("valid_time", "station_id")
        assert ds[var].dtype == np.float32
    for off_var in OFFSET_VARS:
        assert ds[off_var].dims == ("valid_time", "station_id")
        assert ds[off_var].dtype == np.int16

    # Check known values on Amarillo (station 0):
    # - 2024-01-17T01:00:00 comes from 00:53 (-7 min), TMP=-0067,5 -> 266.45 K
    t_01 = ds.sel(valid_time="2024-01-17T01:00:00", station_id="72363023047")
    assert float(t_01["t2m"]) == pytest.approx(273.15 - 6.7, abs=1e-3)
    assert int(t_01["t2m_offset_min"]) == -7

    # - 2024-01-17T06:00:00 MUST NOT be clobbered by the 05:59 SOD row
    t_06 = ds.sel(valid_time="2024-01-17T06:00:00", station_id="72363023047")
    assert np.isfinite(float(t_06["t2m"]))
    assert int(t_06["t2m_offset_min"]) in range(-15, 16)

    # - 2024-01-17T21:00:00 failed TMP QC (2 at 21:00 and 6 at 20:53) -> t2m NaN, offset -32768
    t_21 = ds.sel(valid_time="2024-01-17T21:00:00", station_id="72363023047")
    assert np.isnan(float(t_21["t2m"]))
    assert int(t_21["t2m_offset_min"]) == int(OFFSET_FILL_VALUE)
    assert np.isfinite(float(t_21["td2m"]))
    assert int(t_21["td2m_offset_min"]) == 0

    # - 2024-01-18T18:00:00 calm wind -> u10=0, v10=0, offset=0
    t_calm = ds.sel(valid_time="2024-01-18T18:00:00", station_id="72363023047")
    assert float(t_calm["u10"]) == 0.0
    assert float(t_calm["v10"]) == 0.0
    assert int(t_calm["u10_offset_min"]) == 0

    # - 2024-01-18T19:00:00 variable wind (18:53) -> u10/v10 NaN, offset -32768
    t_var = ds.sel(valid_time="2024-01-18T19:00:00", station_id="72363023047")
    assert np.isnan(float(t_var["u10"]))
    assert int(t_var["u10_offset_min"]) == int(OFFSET_FILL_VALUE)

    # Primary variables pass harness schema validator
    validate_schema_truth(ds[list(PRIMARY_VARS)])

    # NetCDF roundtrip must preserve plain int16 dtype, -32768 sentinel, and NO _FillValue attr/encoding
    nc_path = tmp_path / "truth_roundtrip.nc"
    write_truth_netcdf(ds, nc_path)
    with netcdf_file(nc_path, "r") as nf:
        for off_var in OFFSET_VARS:
            assert "_FillValue" not in nf.variables[off_var]._attributes
    with xr.open_dataset(nc_path) as reloaded:
        validate_truth_dataset(reloaded)
        for off_var in OFFSET_VARS:
            assert reloaded[off_var].dtype == np.int16
            assert "_FillValue" not in reloaded[off_var].attrs
            assert "_FillValue" not in reloaded[off_var].encoding
        reloaded_missing = reloaded.sel(valid_time="2024-01-17T21:00:00", station_id="72363023047")
        assert int(reloaded_missing["t2m_offset_min"]) == -32768

    comp = compute_completeness(ds)
    assert set(comp.keys()) == set(PRIMARY_VARS)
    assert 0.0 <= comp["t2m"] <= 1.0


def test_anti_leakage_guards_reject_2025(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="2025"):
        download_station_year("72363023047", 2025, tmp_path)

    with pytest.raises(ValueError, match="2025"):
        build_truth(["72363023047"], "2024-12-31T23:00:00", "2025-01-01T00:00:00", tmp_path)

    # Even if a CSV contained a 2025 row, parse_isd_csv must strip it
    buf = io.StringIO(
        "DATE,TMP,DEW,SLP,WND\n"
        "2024-12-31T23:53:00,+0050,5,+0010,5,10132,5,180,5,N,0030,5\n"
        "2025-01-01T00:00:00,+0060,1,+0020,1,10130,1,180,1,N,0030,1\n"
    )
    parsed = parse_isd_csv(buf)
    assert len(parsed) == 1
    assert parsed["obs_time"].iloc[0] == pd.Timestamp("2024-12-31T23:53:00")
