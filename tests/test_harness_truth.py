"""Unit tests for the NOAA ISD station truth loader (never touches the network)."""

from __future__ import annotations

import io
import math
import shutil
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from wrfcomp.harness.schemas import validate_truth_dataset
from wrfcomp.harness.truth import (
    OFFSET_FILL_VALUE,
    OFFSET_VARS,
    PARSER_VERSION,
    PRIMARY_VARS,
    build_truth,
    compute_completeness,
    count_report_types,
    download_station_year,
    load_parsed_station_year_npz,
    match_station_to_hours,
    parse_isd_csv,
    parse_slp,
    parse_tmp_or_dew,
    parse_wnd,
    read_parsed_npz_report_types,
    required_years_for_window,
    save_parsed_station_year_npz,
    validate_truth_values,
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


def test_scalar_parsers_qc_domains_and_range_checks() -> None:
    # Temperatures & dewpoint: passing set {0, 1, 4, 5, 9, A, C}
    assert parse_tmp_or_dew("+0123,1") == pytest.approx(285.45)
    assert parse_tmp_or_dew("-0044,1") == pytest.approx(268.75)
    assert parse_tmp_or_dew("+0070,C") == pytest.approx(280.15)  # AWOS whole-deg C accepted
    assert parse_tmp_or_dew("+0100,A") == pytest.approx(283.15)  # Interactive QC A listed for TMP/DEW
    assert math.isnan(parse_tmp_or_dew("+9999,9"))  # missing sentinel
    assert math.isnan(parse_tmp_or_dew("+0183,6"))  # suspect NCEI
    assert math.isnan(parse_tmp_or_dew("+0183,2"))  # suspect
    assert math.isnan(parse_tmp_or_dew("+0183,3"))  # erroneous
    assert math.isnan(parse_tmp_or_dew("+0183,I"))  # validator-inserted (rejected)

    # SLP: passing set {0, 1, 4, 5, 9} only (A and C are NOT listed in ISD format doc p. 12)
    assert parse_slp("10246,1") == pytest.approx(102460.0)
    assert math.isnan(parse_slp("99999,9"))
    assert math.isnan(parse_slp("10132,6"))
    assert math.isnan(parse_slp("10132,A"))
    assert math.isnan(parse_slp("10132,C"))

    # Winds: passing set {0, 1, 4, 5, 9} only; dir 1..360; speed 0..900 tenths of m/s (<= 90 m/s)
    u, v = parse_wnd("360,1,N,0100,1")
    assert u == pytest.approx(0.0, abs=1e-12)
    assert v == pytest.approx(-10.0, abs=1e-12)

    # Calm winds (C or 9 with 0000)
    assert parse_wnd("999,9,C,0000,5") == (0.0, 0.0)
    assert parse_wnd("999,9,9,0000,1") == (0.0, 0.0)

    # Variable winds (type V): use direction when present and passing; NaN only for 999 / invalid
    u_var, v_var = parse_wnd("190,5,V,0041,5")
    exp_u, exp_v = wind_to_uv(190.0, 4.1)
    assert u_var == pytest.approx(exp_u, rel=1e-6)
    assert v_var == pytest.approx(exp_v, rel=1e-6)
    assert all(math.isnan(x) for x in parse_wnd("999,9,V,0026,5"))

    # Missing, failed QC, QC='A' on WND (not listed pp. 8-9), and range checks
    assert all(math.isnan(x) for x in parse_wnd("999,9,9,9999,9"))
    assert all(math.isnan(x) for x in parse_wnd("180,6,N,0050,5"))
    assert all(math.isnan(x) for x in parse_wnd("180,5,N,0050,2"))
    assert all(math.isnan(x) for x in parse_wnd("180,5,N,0050,A"))  # A not in WND speed domain
    assert all(math.isnan(x) for x in parse_wnd("180,A,N,0050,5"))  # A not in WND dir domain
    assert all(math.isnan(x) for x in parse_wnd("000,5,N,0050,5"))  # dir 000 outside 1..360
    assert all(math.isnan(x) for x in parse_wnd("361,5,N,0050,5"))  # dir 361 outside 1..360
    assert parse_wnd("180,5,N,0900,5") == pytest.approx((0.0, 90.0), abs=1e-12)  # max 90.0 m/s
    assert all(math.isnan(x) for x in parse_wnd("180,5,N,0901,5"))  # > 90.0 m/s rejected


def test_parse_real_sample_file_amarillo_and_fm12_filtering() -> None:
    df = parse_isd_csv(SAMPLE_AMARILLO)
    assert not df.empty
    assert list(df.columns) == ["obs_time", "t2m", "td2m", "u10", "v10", "msl"]

    by_time = df.set_index("obs_time")

    # Station-year has FM-15 reports, so FM-12 copies (e.g., 00:00, 21:00, 18:00) and SOD rows are dropped!
    assert pd.Timestamp("2024-01-17T00:00:00") not in by_time.index  # FM-12 copy dropped
    assert pd.Timestamp("2024-01-17T05:59:00") not in by_time.index  # SOD dropped
    assert pd.Timestamp("2024-01-17T21:00:00") not in by_time.index  # FM-12 copy dropped
    assert pd.Timestamp("2024-01-18T18:00:00") not in by_time.index  # FM-12 copy dropped

    # 1. Negative temperature, dewpoint, SLP, and wind on 2024-01-17T00:53:00 (FM-15)
    #    Raw: WND=220,5,N,0036,5 TMP=-0067,5 DEW=-0167,5 SLP=10251,5
    row_neg = by_time.loc[pd.Timestamp("2024-01-17T00:53:00")]
    assert row_neg["t2m"] == pytest.approx(273.15 - 6.7, abs=1e-3)
    assert row_neg["td2m"] == pytest.approx(273.15 - 16.7, abs=1e-3)
    assert row_neg["msl"] == pytest.approx(102510.0, abs=1e-1)
    exp_u, exp_v = wind_to_uv(220.0, 3.6)
    assert row_neg["u10"] == pytest.approx(exp_u, abs=1e-4)
    assert row_neg["v10"] == pytest.approx(exp_v, abs=1e-4)

    # 2. Failed quality code on FM-15: TMP=+0183,6 at 20:53 -> t2m NaN
    row_bad = by_time.loc[pd.Timestamp("2024-01-17T20:53:00")]
    assert np.isnan(row_bad["t2m"])
    assert row_bad["td2m"] == pytest.approx(273.15 - 16.7, abs=1e-3)
    assert row_bad["msl"] == pytest.approx(100330.0, abs=1e-1)

    # 3. Calm wind on FM-15: 2024-01-18T17:53:00 WND=999,9,C,0000,5
    row_calm = by_time.loc[pd.Timestamp("2024-01-18T17:53:00")]
    assert row_calm["u10"] == 0.0
    assert row_calm["v10"] == 0.0
    assert row_calm["t2m"] == pytest.approx(273.15 + 11.7, abs=1e-3)

    # 4. Variable wind with 999 direction: 2024-01-18T18:53:00 WND=999,9,V,0026,5 -> u10/v10 NaN
    row_var = by_time.loc[pd.Timestamp("2024-01-18T18:53:00")]
    assert np.isnan(row_var["u10"])
    assert np.isnan(row_var["v10"])
    assert row_var["t2m"] == pytest.approx(273.15 + 14.4, abs=1e-3)

    # 5. Real SY-MT routine report at measurement time (2024-07-14T23:53:00) MUST be retained!
    #    Raw: WND=210,5,N,0062,5 TMP=+0372,5 DEW=+0117,5 SLP=10116,5
    row_symt = by_time.loc[pd.Timestamp("2024-07-14T23:53:00")]
    assert row_symt["t2m"] == pytest.approx(273.15 + 37.2, abs=1e-3)
    assert row_symt["td2m"] == pytest.approx(273.15 + 11.7, abs=1e-3)
    assert row_symt["msl"] == pytest.approx(101160.0, abs=1e-1)
    exp_su, exp_sv = wind_to_uv(210.0, 6.2)
    assert row_symt["u10"] == pytest.approx(exp_su, abs=1e-4)
    assert row_symt["v10"] == pytest.approx(exp_sv, abs=1e-4)

    # Verify report-type counting on real sample
    rt_counts = count_report_types(SAMPLE_AMARILLO)
    assert rt_counts["FM-15"] > 0
    assert rt_counts["FM-12"] > 0
    assert rt_counts["SY-MT"] == 1
    assert rt_counts["SOD"] > 0


def test_real_symt_row_matches_00z_in_build_truth(tmp_path: Path) -> None:
    cache_2024 = tmp_path / "2024"
    cache_2024.mkdir(parents=True)
    shutil.copy(SAMPLE_AMARILLO, cache_2024 / "72363023047.csv")

    ds = build_truth(
        station_ids=["72363023047"],
        start="2024-07-14T23:00:00",
        end="2024-07-15T01:00:00",
        cache_dir=tmp_path,
    )
    t_00z = ds.sel(valid_time="2024-07-15T00:00:00", station_id="72363023047")
    assert float(t_00z["t2m"]) == pytest.approx(310.35, abs=1e-3)
    assert float(t_00z["td2m"]) == pytest.approx(284.85, abs=1e-3)
    assert float(t_00z["msl"]) == pytest.approx(101160.0, abs=1e-1)
    assert int(t_00z["t2m_offset_min"]) == -7
    assert int(t_00z["msl_offset_min"]) == -7


def test_report_type_fm12_copy_rejected_and_synoptic_fallback() -> None:
    # Case 1: FM-15 at 05:53 has suspect SLP (6) and variable wind WITH direction (190,5,V,0041,5);
    # FM-12 copy at 06:00 falsely carries passing SLP (1). FM-12 must be dropped so:
    #   - 06:00 gets offset -7 (true measurement time), NOT 0
    #   - msl stays NaN (no QC bypass via FM-12)
    #   - variable wind with direction 190 is preserved!
    mixed_csv = io.StringIO(
        'DATE,REPORT_TYPE,TMP,DEW,SLP,WND\n'
        '"2024-06-01T05:53:00","FM-15","+0200,5","+0100,5","10077,6","190,5,V,0041,5"\n'
        '"2024-06-01T06:00:00","FM-12","+0200,1","+0100,1","10077,1","190,1,V,0041,1"\n'
        '"2024-06-01T06:00:00","FM-16","+0205,5","+0100,5","99999,9","200,5,N,0050,5"\n'
    )
    parsed = parse_isd_csv(mixed_csv)
    assert len(parsed) == 2
    valid_times = pd.date_range("2024-06-01T06:00:00", periods=1, freq="h")
    vals, offs = match_station_to_hours(parsed, valid_times, tolerance_minutes=15)
    # Real FM-16 at 06:00 wins over FM-15 at 05:53 for t2m and wind, while msl stays NaN (FM-12 dropped)
    assert vals["t2m"][0] == pytest.approx(293.65, abs=1e-3)
    assert offs["t2m"][0] == 0
    assert np.isnan(vals["msl"][0])
    assert offs["msl"][0] == OFFSET_FILL_VALUE

    # Also verify pure FM-15 + FM-12 pair without FM-16 gives offset -7 (never 0 from FM-12)
    pair_csv = io.StringIO(
        'DATE,REPORT_TYPE,TMP,DEW,SLP,WND\n'
        '"2024-06-01T05:53:00","FM-15","+0200,5","+0100,5","10077,6","190,5,V,0041,5"\n'
        '"2024-06-01T06:00:00","FM-12","+0200,1","+0100,1","10077,1","190,1,V,0041,1"\n'
    )
    parsed_pair = parse_isd_csv(pair_csv)
    vals2, offs2 = match_station_to_hours(parsed_pair, valid_times, tolerance_minutes=15)
    assert vals2["t2m"][0] == pytest.approx(293.15, abs=1e-3)
    assert offs2["t2m"][0] == -7
    assert np.isnan(vals2["msl"][0])
    exp_u, exp_v = wind_to_uv(190.0, 4.1)
    assert vals2["u10"][0] == pytest.approx(exp_u, abs=1e-4)
    assert vals2["v10"][0] == pytest.approx(exp_v, abs=1e-4)

    # Case 2: Station-year with NO FM-15/FM-16/SY-MT falls back to FM-12 only (dropping SOD)
    synoptic_only = io.StringIO(
        'DATE,REPORT_TYPE,TMP,DEW,SLP,WND\n'
        '"2024-06-01T00:00:00","FM-12","+0150,1","+0050,1","10132,1","180,1,N,0050,1"\n'
        '"2024-06-01T03:00:00","FM-12","+0160,1","+0060,1","10135,1","180,1,N,0050,1"\n'
        '"2024-06-01T05:59:00","SOD  ","+9999,9","+9999,9","99999,9","999,9,9,9999,9"\n'
    )
    parsed_syn = parse_isd_csv(synoptic_only)
    assert len(parsed_syn) == 2
    assert list(parsed_syn["obs_time"]) == [
        pd.Timestamp("2024-06-01T00:00:00"),
        pd.Timestamp("2024-06-01T03:00:00"),
    ]

    # Case 3: SY-MT alongside FM-12 keeps SY-MT and drops FM-12
    symt_and_fm12 = io.StringIO(
        'DATE,REPORT_TYPE,TMP,DEW,SLP,WND\n'
        '"2024-06-01T02:53:00","SY-MT","+0160,5","+0060,5","10135,5","180,5,N,0050,5"\n'
        '"2024-06-01T03:00:00","FM-12","+0160,1","+0060,1","10135,1","180,1,N,0050,1"\n'
    )
    parsed_symt = parse_isd_csv(symt_and_fm12)
    assert len(parsed_symt) == 1
    assert parsed_symt["obs_time"].iloc[0] == pd.Timestamp("2024-06-01T02:53:00")


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

    assert np.isnan(vals["t2m"][0])
    assert offs["t2m"][0] == OFFSET_FILL_VALUE

    assert vals["t2m"][1] == pytest.approx(281.0)
    assert offs["t2m"][1] == -7

    assert np.isnan(vals["t2m"][2])
    assert offs["t2m"][2] == OFFSET_FILL_VALUE

    assert vals["t2m"][3] == pytest.approx(283.0)
    assert offs["t2m"][3] == -5

    assert vals["t2m"][4] == pytest.approx(285.0)
    assert offs["t2m"][4] == 15

    assert vals["t2m"][5] == pytest.approx(286.0)
    assert offs["t2m"][5] == -15

    assert np.isnan(vals["t2m"][6])
    assert offs["t2m"][6] == OFFSET_FILL_VALUE


def test_td2m_gt_t2m_enforced_after_per_variable_matching() -> None:
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


def test_year_boundary_both_paths_identical_and_never_2025(tmp_path: Path) -> None:
    # Two synthetic station-years spanning 2022 -> 2023 -> 2024 boundary:
    # - 2022-12-31T23:55 (-5 min) beats 2023-01-01T00:15 (+15 min) for 2023-01-01T00:00
    # - 2024-01-01T00:05 (+5 min) matches 2024-01-01T00:00 (next year's first 15 min)
    sid = "72363023047"
    raw_dir = tmp_path / "raw"
    parsed_dir = tmp_path / "parsed"
    for yr, rows in [
        (
            2022,
            '"2022-12-31T23:55:00","FM-15","+0111,5","+0050,5","10120,5","180,5,N,0040,5"\n',
        ),
        (
            2023,
            '"2023-01-01T00:15:00","FM-15","+0222,5","+0060,5","10130,5","180,5,N,0050,5"\n'
            '"2023-12-31T23:40:00","FM-15","+0050,5","+0010,5","10140,5","180,5,N,0030,5"\n',
        ),
        (
            2024,
            '"2024-01-01T00:05:00","FM-15","+0080,5","+0020,5","10150,5","180,5,N,0060,5"\n',
        ),
    ]:
        ydir = raw_dir / str(yr)
        ydir.mkdir(parents=True)
        csv_path = ydir / f"{sid}.csv"
        csv_path.write_text("DATE,REPORT_TYPE,TMP,DEW,SLP,WND\n" + rows, encoding="utf-8")
        save_parsed_station_year_npz(
            parse_isd_csv(csv_path),
            parsed_dir / str(yr) / f"{sid}.npz",
            tolerance_minutes=15,
        )

    assert required_years_for_window("2023-01-01T00:00:00", "2024-01-01T00:00:00", 15) == [2022, 2023, 2024]
    # 2024-12-31T23:00 never includes 2025
    assert required_years_for_window("2023-12-31T18:00:00", "2024-12-31T23:00:00", 15) == [2023, 2024]

    ds_raw = build_truth(
        [sid],
        start="2023-01-01T00:00:00",
        end="2024-01-01T00:00:00",
        cache_dir=raw_dir,
    )
    ds_parsed = build_truth(
        [sid],
        start="2023-01-01T00:00:00",
        end="2024-01-01T00:00:00",
        cache_dir=tmp_path / "empty_raw",
        parsed_cache_dir=parsed_dir,
    )
    xr.testing.assert_identical(ds_raw, ds_parsed)

    # Mixed fallback (e.g., 2023 in parsed cache, 2024 in raw CSV only) also matches identically
    (parsed_dir / "2024" / f"{sid}.npz").unlink()
    ds_mixed = build_truth(
        [sid],
        start="2023-01-01T00:00:00",
        end="2024-01-01T00:00:00",
        cache_dir=raw_dir,
        parsed_cache_dir=parsed_dir,
    )
    xr.testing.assert_identical(ds_raw, ds_mixed)

    # Check exact boundary matches
    t0 = ds_parsed.sel(valid_time="2023-01-01T00:00:00", station_id=sid)
    assert float(t0["t2m"]) == pytest.approx(273.15 + 11.1, abs=1e-3)
    assert int(t0["t2m_offset_min"]) == -5

    t_end = ds_parsed.sel(valid_time="2024-01-01T00:00:00", station_id=sid)
    assert float(t_end["t2m"]) == pytest.approx(273.15 + 8.0, abs=1e-3)
    assert int(t_end["t2m_offset_min"]) == 5


def test_parsed_cache_refuses_metadata_mismatch(tmp_path: Path) -> None:
    obs_df = parse_isd_csv(SAMPLE_AMARILLO)
    rt_counts = count_report_types(SAMPLE_AMARILLO)
    npz_path = save_parsed_station_year_npz(
        obs_df,
        tmp_path / "2024" / "72363023047.npz",
        tolerance_minutes=15,
        report_type_counts=rt_counts,
    )

    # Matching metadata loads cleanly and preserves report-type counts
    loaded = load_parsed_station_year_npz(npz_path, expected_tolerance_minutes=15)
    assert len(loaded) == len(obs_df)
    assert read_parsed_npz_report_types(npz_path) == rt_counts

    # Tolerance mismatch raises ValueError
    with pytest.raises(ValueError, match="tolerance_minutes mismatch"):
        load_parsed_station_year_npz(npz_path, expected_tolerance_minutes=10)

    # Previous parser version ("2") mismatch raises ValueError
    with np.load(npz_path) as z:
        payload = {k: z[k] for k in z.files}
    payload["meta_parser_version"] = np.array("2")
    bad_ver = tmp_path / "bad_ver.npz"
    np.savez_compressed(bad_ver, **payload)
    with pytest.raises(ValueError, match="parser_version mismatch"):
        load_parsed_station_year_npz(bad_ver, expected_tolerance_minutes=15)

    # Passing QC codes mismatch raises ValueError
    payload["meta_parser_version"] = np.array(PARSER_VERSION)
    payload["meta_passing_qc_codes"] = np.array("temp_dew=0,1;wind_slp=0,1")
    bad_qc = tmp_path / "bad_qc.npz"
    np.savez_compressed(bad_qc, **payload)
    with pytest.raises(ValueError, match="passing_qc_codes mismatch"):
        load_parsed_station_year_npz(bad_qc, expected_tolerance_minutes=15)

    # Unversioned legacy npz raises ValueError
    legacy = tmp_path / "legacy.npz"
    np.savez_compressed(legacy, t2m=np.zeros(2, dtype=np.float32))
    with pytest.raises(ValueError, match="missing required metadata"):
        load_parsed_station_year_npz(legacy, expected_tolerance_minutes=15)


def test_missing_station_year_emits_warning(tmp_path: Path) -> None:
    cache_2024 = tmp_path / "2024"
    cache_2024.mkdir(parents=True)
    shutil.copy(SAMPLE_AMARILLO, cache_2024 / "72363023047.csv")

    # Requesting 2024-01-01T00:00 needs 2023 lookback; missing 2023/72363023047 must warn
    with pytest.warns(UserWarning, match=r"2023/72363023047"):
        build_truth(
            ["72363023047"],
            start="2024-01-01T00:00:00",
            end="2024-01-01T02:00:00",
            cache_dir=tmp_path,
        )


def test_build_truth_contract_and_h5netcdf_roundtrip(tmp_path: Path) -> None:
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

    # Passes canonical harness/schemas.py validator AND value-level validator
    validate_truth_dataset(ds, variables=PRIMARY_VARS)
    validate_truth_values(ds)
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

    # 2024-01-17T01:00:00 comes from 00:53 (-7 min), TMP=-0067,5 -> 266.45 K
    t_01 = ds.sel(valid_time="2024-01-17T01:00:00", station_id="72363023047")
    assert float(t_01["t2m"]) == pytest.approx(273.15 - 6.7, abs=1e-3)
    assert int(t_01["t2m_offset_min"]) == -7

    # 2024-01-17T06:00:00 comes from 05:53 FM-15 (-7 min), NOT clobbered by 05:59 SOD or 06:00 FM-12
    t_06 = ds.sel(valid_time="2024-01-17T06:00:00", station_id="72363023047")
    assert np.isfinite(float(t_06["t2m"]))
    assert int(t_06["t2m_offset_min"]) == -7

    # 2024-01-17T21:00:00: 20:53 FM-15 has failed TMP QC (6); FM-12 copy at 21:00 was dropped!
    t_21 = ds.sel(valid_time="2024-01-17T21:00:00", station_id="72363023047")
    assert np.isnan(float(t_21["t2m"]))
    assert int(t_21["t2m_offset_min"]) == int(OFFSET_FILL_VALUE)
    assert np.isfinite(float(t_21["td2m"]))
    assert int(t_21["td2m_offset_min"]) == -7

    # 2024-01-18T18:00:00: 17:53 FM-15 calm wind (offset -7; 18:00 FM-12 copy dropped!)
    t_calm = ds.sel(valid_time="2024-01-18T18:00:00", station_id="72363023047")
    assert float(t_calm["u10"]) == 0.0
    assert float(t_calm["v10"]) == 0.0
    assert int(t_calm["u10_offset_min"]) == -7

    # 2024-01-18T19:00:00: 18:53 FM-15 variable wind with 999 -> u10/v10 NaN, offset -32768
    t_var = ds.sel(valid_time="2024-01-18T19:00:00", station_id="72363023047")
    assert np.isnan(float(t_var["u10"]))
    assert int(t_var["u10_offset_min"]) == int(OFFSET_FILL_VALUE)

    # NetCDF4 (h5netcdf) roundtrip preserves plain int16, -32768, and NO _FillValue attr/encoding
    nc_path = tmp_path / "truth_roundtrip.nc"
    write_truth_netcdf(ds, nc_path)
    with h5py.File(nc_path, "r") as hf:
        for off_var in OFFSET_VARS:
            assert "_FillValue" not in hf[off_var].attrs
    with xr.open_dataset(nc_path, engine="h5netcdf") as reloaded:
        validate_truth_dataset(reloaded, variables=PRIMARY_VARS)
        validate_truth_values(reloaded)
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

    buf = io.StringIO(
        'DATE,REPORT_TYPE,TMP,DEW,SLP,WND\n'
        '"2024-12-31T23:53:00","FM-15","+0050,5","+0010,5","10132,5","180,5,N,0030,5"\n'
        '"2025-01-01T00:00:00","FM-15","+0060,1","+0020,1","10130,1","180,1,N,0030,1"\n'
    )
    parsed = parse_isd_csv(buf)
    assert len(parsed) == 1
    assert parsed["obs_time"].iloc[0] == pd.Timestamp("2024-12-31T23:53:00")
