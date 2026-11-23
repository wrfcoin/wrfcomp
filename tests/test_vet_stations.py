"""Offline unit tests for scripts/vet_stations.py (ISD-Lite parser and vetting rules)."""

from __future__ import annotations

import gzip
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
VET_SCRIPT = REPO_ROOT / "scripts" / "vet_stations.py"


def _load_vet_module():
    spec = importlib.util.spec_from_file_location("vet_stations", VET_SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


vet = _load_vet_module()

# Tiny committed NOAA ISD-Lite sample (fixed-width 61-char format, leap year 2024).
# Columns: YYYY MM DD HH  TEMP  DEWP   SLP  WDIR  WSPD   SKY  PR1H  PR6H
SAMPLE_ISD_LITE_LINES = [
    "2024 01 01 00   -50   -80 10132   180    35     0 -9999 -9999\n",
    "2024 01 01 01   -55   -85 10135     0     0     0 -9999 -9999\n",  # calm wind: dir=0, spd=0
    "2024 01 01 02   -60 -9999 -9999   270    40     2 -9999 -9999\n",  # missing dewp & slp
    "2024 01 01 03 -9999   -90 10140 -9999 -9999     4 -9999 -9999\n",  # missing temp & wind
    "2024 02 29 23   120    40 10200    90    20     0 -9999 -9999\n",  # 2024 leap day
    "2023 12 31 23    10     0 10100   100    10     0 -9999 -9999\n",  # non-2024 ignored
    "2025 01 01 00    10     0 10100   100    10     0 -9999 -9999\n",  # 2025 ignored
]


def test_parse_isd_lite_sample_lines_and_gz(tmp_path: Path) -> None:
    comp = vet.parse_isd_lite_2024_lines(SAMPLE_ISD_LITE_LINES, station_id="72011053983")
    assert comp.station_id == "72011053983"
    assert comp.total_hours == 8784
    assert comp.record_hours == 5
    assert comp.temp_hours == 4
    assert comp.dewpoint_hours == 4
    assert comp.slp_hours == 4
    assert comp.wind_dir_hours == 4
    assert comp.wind_speed_hours == 4
    assert not comp.passes()

    gz_path = tmp_path / "720110-53983-2024.gz"
    with gzip.open(gz_path, "wt", encoding="ascii") as fh:
        fh.writelines(SAMPLE_ISD_LITE_LINES)
    comp_gz = vet.parse_isd_lite_2024_file(gz_path, station_id="72011053983")
    assert comp_gz == comp


def test_parse_isd_lite_mini_window_gaps() -> None:
    """Verify exact gap counting across a 10-hour window using tiny committed lines."""
    lines = [
        "2024 01 01 00   100    50 10120   180    20 -9999 -9999 -9999\n",
        # hours 1, 2 missing completely (gap = 2)
        "2024 01 01 03   110    55 -9999   190    25 -9999 -9999 -9999\n",  # slp missing (gap = 3)
        "2024 01 01 04   120    60 10125   200    30 -9999 -9999 -9999\n",
        # hours 5..9 missing (trailing gap = 5)
    ]
    comp = vet.parse_isd_lite_2024_lines(lines, station_id="99999800001", total_hours=10)
    assert comp.record_hours == 3
    assert comp.temp_hours == 3
    assert comp.slp_hours == 2
    assert comp.temp_max_gap_hours == 5
    assert comp.slp_max_gap_hours == 5
    assert comp.record_max_gap_hours == 5


def test_max_missing_run_hours() -> None:
    assert vet.max_missing_run_hours(np.array([], dtype=bool)) == 0
    assert vet.max_missing_run_hours(np.array([True, True, True])) == 0
    assert vet.max_missing_run_hours(np.array([False, False, False])) == 3
    assert vet.max_missing_run_hours(np.array([False, True, False, False, True])) == 2
    assert vet.max_missing_run_hours(np.array([True, False, True, False, False, False])) == 3


def _make_completeness(**overrides: int) -> vet.StationCompleteness:
    total = vet.HOURS_IN_2024
    # Exact threshold counts: ceil(0.90 * 8784) = 7906, ceil(0.80 * 8784) = 7028
    base = {
        "station_id": "72405013743",
        "temp_hours": 7906,
        "dewpoint_hours": 7906,
        "slp_hours": 7028,
        "wind_dir_hours": 7906,
        "wind_speed_hours": 7906,
        "record_hours": 8500,
        "temp_max_gap_hours": 240,
        "dewpoint_max_gap_hours": 240,
        "slp_max_gap_hours": 240,
        "wind_dir_max_gap_hours": 240,
        "wind_speed_max_gap_hours": 240,
        "record_max_gap_hours": 240,
        "total_hours": total,
    }
    base.update(overrides)
    return vet.StationCompleteness(**base)


def test_pass_rule_exact_boundaries() -> None:
    assert _make_completeness().passes() is True


@pytest.mark.parametrize(
    "field,value",
    [
        ("temp_hours", 7905),
        ("dewpoint_hours", 7905),
        ("wind_dir_hours", 7905),
        ("wind_speed_hours", 7905),
        ("slp_hours", 7027),
        ("temp_max_gap_hours", 241),
        ("dewpoint_max_gap_hours", 241),
        ("slp_max_gap_hours", 241),
        ("wind_dir_max_gap_hours", 241),
        ("wind_speed_max_gap_hours", 241),
    ],
)
def test_pass_rule_fails_single_violation(field: str, value: int) -> None:
    assert _make_completeness(**{field: value}).passes() is False


def test_filter_conus_candidates_enforces_2014_history_and_geography() -> None:
    history = pd.DataFrame(
        [
            # Valid CONUS with 2014-2024 span
            {
                "USAF": "724050",
                "WBAN": "13743",
                "STATION NAME": "REAGAN NATL",
                "CTRY": "US",
                "STATE": "VA",
                "LAT": 38.85,
                "LON": -77.04,
                "ELEV(M)": 4.6,
                "BEGIN": 19730101,
                "END": 20250826,
            },
            # Begins after 2014-01-01 -> rejected
            {
                "USAF": "720999",
                "WBAN": "00123",
                "STATION NAME": "TOO NEW",
                "CTRY": "US",
                "STATE": "TX",
                "LAT": 32.0,
                "LON": -97.0,
                "ELEV(M)": 200.0,
                "BEGIN": 20140102,
                "END": 20250826,
            },
            # Ended before 2024-12-31 -> rejected
            {
                "USAF": "720998",
                "WBAN": "00124",
                "STATION NAME": "ENDED EARLY",
                "CTRY": "US",
                "STATE": "TX",
                "LAT": 32.0,
                "LON": -97.0,
                "ELEV(M)": 200.0,
                "BEGIN": 20100101,
                "END": 20241230,
            },
            # Alaska -> rejected
            {
                "USAF": "702730",
                "WBAN": "26451",
                "STATION NAME": "ANCHORAGE",
                "CTRY": "US",
                "STATE": "AK",
                "LAT": 61.17,
                "LON": -149.99,
                "ELEV(M)": 36.6,
                "BEGIN": 19730101,
                "END": 20250826,
            },
            # USAF 999999 -> rejected
            {
                "USAF": "999999",
                "WBAN": "13743",
                "STATION NAME": "PLACEHOLDER",
                "CTRY": "US",
                "STATE": "VA",
                "LAT": 38.85,
                "LON": -77.04,
                "ELEV(M)": 4.6,
                "BEGIN": 19730101,
                "END": 20250826,
            },
        ]
    )
    filtered = vet.filter_conus_candidates(history)
    assert filtered["station_id"].tolist() == ["72405013743"]


def test_reuses_fetch_conus_spatial_selector_deterministically() -> None:
    select_fn = vet.load_select_spatially_diverse()
    df = pd.DataFrame(
        [
            {
                "station_id": "72000100001",
                "WBAN": "00001",
                "LAT": 30.0,
                "LON": -100.0,
                "BEGIN": 20000101,
                "END": 20250101,
            },
            {
                "station_id": "72000200002",
                "WBAN": "00002",
                "LAT": 30.1,
                "LON": -100.1,
                "BEGIN": 20050101,
                "END": 20250101,
            },
            {
                "station_id": "72000300003",
                "WBAN": "00003",
                "LAT": 45.0,
                "LON": -75.0,
                "BEGIN": 20050101,
                "END": 20250101,
            },
            {
                "station_id": "72000400004",
                "WBAN": "00004",
                "LAT": 45.0,
                "LON": -120.0,
                "BEGIN": 20050101,
                "END": 20250101,
            },
        ]
    )
    pick1 = select_fn(df, target_count=3)
    pick2 = select_fn(df.iloc[::-1].reset_index(drop=True), target_count=3)
    assert pick1["station_id"].tolist() == pick2["station_id"].tolist()
    assert pick1["station_id"].tolist() == ["72000100001", "72000300003", "72000400004"]


def test_concurrency_cap_enforced(tmp_path: Path) -> None:
    empty = pd.DataFrame(columns=["USAF", "WBAN", "station_id"])
    with pytest.raises(ValueError, match="concurrency"):
        vet.vet_candidates(empty, tmp_path, concurrency=5)


def test_vetted_stations_csv_invariants() -> None:
    csv_path = REPO_ROOT / "configs" / "stations_conus.csv"
    df = pd.read_csv(csv_path, dtype={"station_id": str})
    assert len(df) == 500
    assert df["station_id"].nunique() == 500
    assert (df["station_id"].str.len() == 11).all()
    assert (df["begin_yyyymmdd"].astype(int) <= 20140101).all()
    assert (df["end_yyyymmdd"].astype(int) >= 20241231).all()
    assert (~df["state"].isin(vet.EXCLUDED_STATES)).all()
