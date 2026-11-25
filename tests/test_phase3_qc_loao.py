"""Synthetic acceptance tests for the official-free Phase 3 Design 2 Track B QC driver."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from wrfcomp.assimilation import phase3_qc as p3
from wrfcomp.assimilation import phase3_qc_loao as loao
from wrfcomp.assimilation.qc import Flag, cells, spatial_series

START = np.datetime64("2024-03-01T00", "ns")
H = np.timedelta64(1, "h")
M = np.timedelta64(1, "m")


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _synthetic_forecasts(trial, n_sites: int, days: int = 5):
    rng = np.random.default_rng(123)
    issued, valid = [], []
    means = {var: [] for var in (*trial.DIRECT, "wind_speed")}
    for cycle in np.arange(START, START + days * 24 * H, 12 * H):
        for lead in (3, 6, 12, 24, 48):
            issued.append(cycle)
            valid.append(cycle + lead * H)
            hour = (cycle + lead * H - START) / H
            level = 285.0 + 6.0 * np.sin(2.0 * np.pi * (hour - 9.0) / 24.0)
            means["t2m"].append(level + rng.normal(0.0, 0.4, n_sites))
            means["td2m"].append(level - 6.0 + rng.normal(0.0, 0.4, n_sites))
            means["u10"].append(2.0 + rng.normal(0.0, 0.3, n_sites))
            means["v10"].append(-1.0 + rng.normal(0.0, 0.3, n_sites))
            means["wind_speed"].append(2.5 + rng.normal(0.0, 0.3, n_sites))
    return p3.archive_from_arrays(
        np.array(issued), np.array(valid), {k: np.array(v) for k, v in means.items()}, trial
    )


def _make_station(lat, lon, height, times, t2m, td2m=None):
    t2m = np.asarray(t2m, dtype=np.float32)
    td2m = (t2m - 6.0).astype(np.float32) if td2m is None else np.asarray(td2m, dtype=np.float32)
    zeros = np.full(len(times), np.nan, dtype=np.float32)
    return p3.StationData(lat, lon, height, times.astype("datetime64[ns]"), t2m, td2m, zeros, zeros)


def test_official_free_qc_runs_with_no_official_series_and_rejects_faults():
    trial = p3.load_trial_module(START)
    sites = loao.StaticSites(
        station_id=np.array(["00000000001", "00000000002"]),
        lat=np.array([40.0, 41.5]),
        lon=np.array([-100.0, -100.0]),
        height=np.array([500.0, 800.0]),
    )
    archive = _synthetic_forecasts(trial, len(sites.station_id), days=3)
    rng = np.random.default_rng(7)
    positions = [
        (40.0, -100.0, 500.0),
        (40.1, -100.0, 510.0),
        (40.0, -100.1, 490.0),
        (40.1, -100.1, 500.0),
    ]
    stations = []
    for k, (lat, lon, h) in enumerate(positions):
        times = np.arange(START + k * M, START + 3 * 24 * H, 10 * M)
        hour = (times - START) / H
        t2m = 285.0 + 6.0 * np.sin(2.0 * np.pi * (hour - 9.0) / 24.0) + rng.normal(0.0, 0.3, len(times))
        if k == 0:
            t2m[120] += 25.0  # spike -> temporal fail
            t2m[160:170] = t2m[160]  # flatline -> temporal fail
            t2m[220] += 18.0  # uncorroborated gross outlier (>8 sigma)
        stations.append(_make_station(lat, lon, h, times, t2m))
    sources = [loao.station_source_loao(st, sites, archive, trial) for st in stations]
    res0 = loao.station_weights_loao(
        stations[0], [(stations[j], sources[j]) for j in range(1, 4)], sites, archive, trial
    )
    w = res0["t2m"].weights
    assert w.max() > 0.3 and w.max() <= 0.5  # new_prior = 0.5, no verified tier
    assert not res0["t2m"].earned.any() and not res0["t2m"].copycat.any()
    assert w[120] == 0.0 and w[165] == 0.0 and w[220] == 0.0


def test_asynchronous_staggered_buddies_15min_and_failed_supersedes_good():
    trial = p3.load_trial_module(START)
    sites = loao.StaticSites(
        station_id=np.array(["00000000001"]),
        lat=np.array([40.0]),
        lon=np.array([-100.0]),
        height=np.array([500.0]),
    )
    archive = _synthetic_forecasts(trial, 1, days=2)
    target_t = np.array([START + 12 * H], dtype="datetime64[ns]")
    # Buddy 1 reports 14 min earlier (in window), buddy 2 reports 16 min earlier (out of window),
    # buddy 3 reports good at -10 min then fails with a 25 K spike at -2 min (failed supersedes good).
    b1 = _make_station(40.1, -100.0, 500.0, np.array([START + 11 * H + 46 * M]), [285.0])
    b2 = _make_station(40.1, -100.0, 500.0, np.array([START + 11 * H + 44 * M]), [285.0])
    b3 = _make_station(
        40.1,
        -100.0,
        500.0,
        np.array([START + 11 * H + 50 * M, START + 11 * H + 58 * M]),
        [285.0, 310.0],
    )
    s1 = loao.station_source_loao(b1, sites, archive, trial)["t2m"]
    s2 = loao.station_source_loao(b2, sites, archive, trial)["t2m"]
    s3 = loao.station_source_loao(b3, sites, archive, trial)["t2m"]
    assert np.isfinite(trial.align(s1, target_t)[0][0])
    assert np.isnan(trial.align(s2, target_t)[0][0])
    assert np.isnan(trial.align(s3, target_t)[0][0])  # newer failed spike masks older good report


def test_three_buddies_in_one_neighbouring_cell_qualify_under_shipped_spatial_rule():
    """Shipped spatial_series requires >= 3 buddies with >= 1 outside the subject's cell."""
    trial = p3.load_trial_module(START)
    subj_lat, subj_lon = 40.0, -100.0
    bud_lats = np.array([40.1, 40.1, 40.1])
    bud_lons = np.array([-100.0, -100.0, -100.0])
    is_off = np.zeros(3, dtype=bool)
    external = cells(bud_lats, bud_lons) != cells(subj_lat, subj_lon)
    assert external.all()
    flags, med = spatial_series(
        np.array([285.0]), np.array([[285.1, 284.9, 285.0]]), is_off, external, trial.SIGMA, trial.CFG
    )
    assert flags[0] == 0 and not (flags[0] & int(Flag.FEW_BUDDIES)) and np.isfinite(med[0])
    # Same-cell-only buddies do NOT satisfy independence.
    same_ext = np.zeros(3, dtype=bool)
    flags_same, _ = spatial_series(
        np.array([285.0]), np.array([[285.1, 284.9, 285.0]]), is_off, same_ext, trial.SIGMA, trial.CFG
    )
    assert flags_same[0] == int(Flag.FEW_BUDDIES)


def test_stale_buddy_median_does_not_decay_correlation_guard():
    """Round 3 #2: missing current buddy median after low-correlation history must not decay guard."""
    trial = p3.load_trial_module(START)
    sites = loao.StaticSites(
        station_id=np.array(["00000000001"]),
        lat=np.array([40.0]),
        lon=np.array([-100.0]),
        height=np.array([500.0]),
    )
    archive = _synthetic_forecasts(trial, 1, days=2)
    times = np.arange(START + 6 * H, START + 36 * H, 15 * M)
    hour = (times - START) / H
    # Subject is anti-correlated with its 3 buddies during the first half, then buddies go silent.
    buddy_sig = 285.0 + 3.0 * np.sin(2.0 * np.pi * hour / 12.0)
    subj_sig = 285.0 - 3.0 * np.sin(2.0 * np.pi * hour / 12.0)
    cut = 60  # after report 60, buddies stop reporting so current median becomes NaN
    subj = _make_station(40.0, -100.0, 500.0, times, subj_sig)
    buddies = [_make_station(40.1, -100.0, 500.0, times[:cut], buddy_sig[:cut] + 0.05 * k) for k in range(3)]
    sources = [loao.station_source_loao(b, sites, archive, trial) for b in buddies]
    res = loao.station_weights_loao(
        subj, [(b, s) for b, s in zip(buddies, sources, strict=True)], sites, archive, trial
    )
    w = res["t2m"].weights
    # During active low-correlation window (indices ~20..59), weights decay strictly down toward 0.
    assert w[55] < w[25]
    # After buddies stop reporting (indices >= cut + 2), current median is NaN; guard must NOT keep
    # decaying exponentially (weight ratio stays flat up to solar/huber changes, not 0.9^50).
    assert w[cut + 2] > 0.0
    assert w[cut + 30] / w[cut + 2] > 0.1  # 0.9^28 ~= 0.052 would fail if stale decay occurred


def test_loao_causality_reports_after_T_change_nothing():
    trial = p3.load_trial_module(START)
    sites = loao.StaticSites(
        station_id=np.array(["00000000001"]),
        lat=np.array([40.0]),
        lon=np.array([-100.0]),
        height=np.array([500.0]),
    )
    archive = _synthetic_forecasts(trial, 1, days=3)
    rng = np.random.default_rng(19)
    positions = [(40.0, -100.0, 500.0), (40.1, -100.0, 500.0), (40.0, -100.1, 500.0), (40.1, -100.1, 500.0)]
    times = np.arange(START + 3 * H, START + 3 * 24 * H, 10 * M)
    hour = (times - START) / H
    issue_cut = START + 36 * H

    def run_all(shift_after: bool):
        sts = []
        for lat, lon, h in positions:
            t2m = 285.0 + 4.0 * np.sin(2.0 * np.pi * hour / 24.0) + rng.normal(0.0, 0.2, len(times))
            if shift_after:
                t2m = np.where(times > issue_cut, t2m + 12.0, t2m)
            sts.append(_make_station(lat, lon, h, times, t2m))
        srcs = [loao.station_source_loao(st, sites, archive, trial) for st in sts]
        return loao.station_weights_loao(
            sts[0], [(sts[j], srcs[j]) for j in range(1, 4)], sites, archive, trial
        )

    rng = np.random.default_rng(19)
    r0 = run_all(False)
    rng = np.random.default_rng(19)
    r1 = run_all(True)
    n = int(np.count_nonzero(r0["t2m"].when <= issue_cut))
    np.testing.assert_array_equal(r0["t2m"].weights[:n], r1["t2m"].weights[:n])
    for a, b in zip(
        p3.issue_snapshot(r0["t2m"], [issue_cut]),
        p3.issue_snapshot(r1["t2m"], [issue_cut]),
        strict=True,
    ):
        np.testing.assert_array_equal(a, b)


TILE_NAMES_ON_WINDOWS = pytest.mark.skipif(sys.platform == "win32", reason="tile names use POSIX paths")


@TILE_NAMES_ON_WINDOWS
def test_loao_digest_binds_all_dependencies_and_timing_folder_refused(tmp_path, monkeypatch):
    from wrfcomp.assimilation.madis_archive import ReportTable, save_table

    qc_std = _load_script("run_phase3_qc")
    qc_loao = _load_script("run_phase3_qc_loao")
    days = ("2024-01-01", "2024-01-02")
    monkeypatch.setattr(qc_std, "expected_days", lambda year: list(days))
    citizen = tmp_path / "citizen"
    for day in days:
        base = np.datetime64(day + "T06:00:00", "s")
        tbl = ReportTable(
            station_id=np.array(["CW000000001", "CW000000002"]),
            obs_time=(base + np.arange(2) * 60 * np.timedelta64(1, "s")).astype("datetime64[ns]"),
            received_time=(base + (np.arange(2) * 60 + 300) * np.timedelta64(1, "s")).astype(
                "datetime64[ns]"
            ),
            lat=np.float32([40.0, 40.1]),
            lon=np.float32([-100.0, -100.0]),
            elevation_m=np.float32([500.0, 500.0]),
            t2m=np.float32([280.0, 280.5]),
            td2m=np.float32([275.0, 275.5]),
            wind_speed=np.float32([2.0, 2.0]),
            wind_dir=np.float32([90.0, 90.0]),
        )
        save_table(tbl, citizen / "conus" / "2024" / f"{day}.nc", attrs={"day": day, "hours_missing": ""})
    manifest = qc_std.partition(citizen, 2024)
    stations_csv = tmp_path / "stations.csv"
    stations_csv.write_text("station_id,lat,lon,elevation_m\n00000000001,40.0,-100.0,500.0\n")
    gefs_means = {"t2m": np.zeros((2, 1))}
    cycle = np.array(["2024-01-01T00"], dtype="datetime64[ns]")
    fake_fc = SimpleNamespace(issued=cycle, valid=cycle + 3 * H, means=gefs_means)
    sites = loao.load_static_sites(stations_csv)
    monkeypatch.setattr(loao, "load_loao_inputs", lambda *args, **kwargs: (None, sites, fake_fc))
    rec0 = loao.current_run_record(tmp_path, 2024, stations_csv=stations_csv)
    # Changing stations_csv changes digest.
    stations_csv.write_text("station_id,lat,lon,elevation_m\n00000000001,40.0,-100.0,510.0\n")
    rec_st = loao.current_run_record(tmp_path, 2024, stations_csv=stations_csv)
    assert rec_st["digest"] != rec0["digest"]
    stations_csv.write_text("station_id,lat,lon,elevation_m\n00000000001,40.0,-100.0,500.0\n")
    # Changing frozen QC_CODE digest changes LOAO digest.
    orig_code_digest = p3.code_digest
    monkeypatch.setattr(p3, "code_digest", lambda: "changed-frozen-digest")
    rec_code = loao.current_run_record(tmp_path, 2024, stations_csv=stations_csv)
    assert rec_code["digest"] != rec0["digest"]
    monkeypatch.setattr(p3, "code_digest", orig_code_digest)
    # Changing GEFS content changes digest.
    gefs_means["t2m"] = np.ones((2, 1))
    assert loao.current_run_record(tmp_path, 2024, stations_csv=stations_csv)["digest"] != rec0["digest"]
    gefs_means["t2m"] = np.zeros((2, 1))
    # Timing folder cannot be merged.
    timing = citizen / "conus" / "2024-qc-loao.timing"
    timing.mkdir()
    (timing / "run.json").write_text(json.dumps(rec0))
    with pytest.raises(p3.IncompleteError, match="no manifest"):
        qc_loao.merge(tmp_path, 2024, stations_csv=stations_csv, min_free_bytes=0)
    # Proper merge writes phase3-extract-loao.v1 with official_observations_used=0 and no truth_sha256.
    out = citizen / "conus" / "2024-qc-loao"
    out.mkdir()
    (out / "run.json").write_text(json.dumps(rec0))
    rows = {
        "station": [0],
        "issue": [1],
        "var": [0],
        "value": [280.0],
        "weight": [0.4],
        "age_ns": [60 * 10**9],
    }
    for name in manifest["tiles"]:
        qc_loao.write_tile_output(
            out / f"{name}.npz",
            rec0["digest"],
            ["CW000000001"],
            [40.0],
            [-100.0],
            [500.0],
            [[5, 5, 4, 4]],
            rows,
        )
    summary = qc_loao.merge(tmp_path, 2024, stations_csv=stations_csv, min_free_bytes=0)
    assert summary["format"] == loao.EXTRACT_FORMAT
    with xr.open_dataset(citizen / "conus" / "issue_extract_loao_2024.nc", engine="h5netcdf") as ds:
        assert ds.attrs["format"] == "phase3-extract-loao.v1"
        assert int(ds.attrs["official_observations_used"]) == 0
        assert "truth_sha256" not in ds.attrs
        assert ds.attrs["stations_sha256"] == rec0["stations_sha256"]


def test_default_load_static_sites_reads_registered_stations_conus(tmp_path):
    """Default LOAO loader must work against checked-in configs/stations_conus.csv (lat/lon schema)."""
    sites = loao.load_static_sites()
    csv_df = pd.read_csv(loao.STATIONS_CSV, dtype={"station_id": str})
    assert len(sites.station_id) == 500
    assert sites.station_id.tolist() == csv_df["station_id"].tolist()
    assert np.array_equal(np.sort(sites.station_id), sites.station_id)
    assert np.isfinite(sites.lat).all() and np.isfinite(sites.lon).all() and np.isfinite(sites.height).all()
    bad_csv = tmp_path / "bad_stations.csv"
    bad_csv.write_text("station_id,latitude,longitude,elevation_m\n00000000001,40.0,-100.0,500.0\n")
    with pytest.raises(ValueError, match="missing required columns"):
        loao.load_static_sites(bad_csv)


@TILE_NAMES_ON_WINDOWS
def test_loao_qc_logs_and_summaries_contain_no_tokens_or_coordinate_tile_labels(
    tmp_path, monkeypatch, capsys
):
    """Operational logs, summaries, stdout/stderr, and errors must carry opaque tile ordinals only."""
    import re

    from wrfcomp.assimilation.madis_archive import ReportTable, save_table

    qc_std = _load_script("run_phase3_qc")
    qc_loao = _load_script("run_phase3_qc_loao")
    days = ("2024-01-01", "2024-01-02")
    monkeypatch.setattr(qc_std, "expected_days", lambda year: list(days))
    citizen = tmp_path / "citizen"
    for day in days:
        base = np.datetime64(day + "T11:50:00", "s")
        tbl = ReportTable(
            station_id=np.array(["CW000000001", "CW000000002", "CW000000003", "CW000000004"]),
            obs_time=(base + np.arange(4) * 60 * np.timedelta64(1, "s")).astype("datetime64[ns]"),
            received_time=(base + (np.arange(4) * 60 + 30) * np.timedelta64(1, "s")).astype("datetime64[ns]"),
            lat=np.float32([40.0, 40.1, 40.0, 40.1]),
            lon=np.float32([-100.0, -100.0, -100.1, -100.1]),
            elevation_m=np.float32([500.0, 500.0, 500.0, 500.0]),
            t2m=np.float32([280.0, 280.1, 279.9, 280.0]),
            td2m=np.float32([275.0, 275.1, 274.9, 275.0]),
            wind_speed=np.float32([2.0, 2.0, 2.0, 2.0]),
            wind_dir=np.float32([90.0, 90.0, 90.0, 90.0]),
        )
        save_table(tbl, citizen / "conus" / "2024" / f"{day}.nc", attrs={"day": day, "hours_missing": ""})
    manifest = qc_std.partition(citizen, 2024)
    stations_csv = tmp_path / "stations.csv"
    stations_csv.write_text("station_id,lat,lon,elevation_m\n00000000001,40.0,-100.0,500.0\n")
    sites = loao.load_static_sites(stations_csv)
    trial = p3.load_trial_module(np.datetime64("2024-01-01T00", "ns"))
    issues = np.array(
        ["2024-01-01T00", "2024-01-01T12", "2024-01-02T00", "2024-01-02T12"], dtype="datetime64[ns]"
    )
    f_iss, f_val = [], []
    f_means = {var: [] for var in (*trial.DIRECT, "wind_speed")}
    for cyc in issues:
        for lead in (3, 6, 12, 24, 48):
            f_iss.append(cyc)
            f_val.append(cyc + lead * H)
            f_means["t2m"].append(np.array([280.0]))
            f_means["td2m"].append(np.array([275.0]))
            f_means["u10"].append(np.array([2.0]))
            f_means["v10"].append(np.array([0.0]))
            f_means["wind_speed"].append(np.array([2.0]))
    fc = p3.archive_from_arrays(
        np.array(f_iss), np.array(f_val), {k: np.array(v) for k, v in f_means.items()}, trial
    )
    monkeypatch.setattr(loao, "issue_times", lambda year: issues)
    monkeypatch.setattr(loao, "load_loao_inputs", lambda *args, **kwargs: (trial, sites, fc))
    rec = loao.run_record(2024, manifest, fc, stations_csv=stations_csv)
    out = citizen / "conus" / "2024-qc-loao"
    out.mkdir()
    (out / "run.json").write_text(json.dumps(rec, indent=1) + "\n")
    qc_loao._G.update(
        trial=trial,
        sites=sites,
        forecasts=fc,
        issues=issues,
        tiles=citizen / "conus" / "2024-tiles",
        out=out,
        digest=rec["digest"],
        start=np.datetime64("2024-01-01T00", "ns"),
    )
    jsonl_lines = []
    for ordinal, name in enumerate(manifest["tiles"]):
        row = qc_loao.run_tile((ordinal, name))
        assert "tile" not in row and row["tile_ordinal"] == ordinal
        jsonl_lines.append(json.dumps(row))
    (out / "run.jsonl").write_text("\n".join(jsonl_lines) + "\n")
    qc_loao.merge(tmp_path, 2024, stations_csv=stations_csv, min_free_bytes=0)
    captured = capsys.readouterr()

    # Check digest-mismatch error message uses opaque tile ordinal, never coordinate label.
    qc_loao._G["digest"] = "different-digest"
    with pytest.raises(p3.IncompleteError) as exc_info:
        qc_loao.run_tile((0, manifest["tiles"][0]))
    err_msg = str(exc_info.value)
    assert "tile ordinal 0" in err_msg

    # Verify zero citizen tokens and zero coordinate-derived tile labels across every log/summary/output.
    coord_tile_re = re.compile(r"[+-]\d{2,3}_[+-]\d{3,4}")
    token_re = re.compile(r"CW\d{9}")
    surfaces = [
        (out / "run.jsonl").read_text(),
        (out / "run.json").read_text(),
        (citizen / "conus" / "issue_extract_loao_2024.summary.json").read_text(),
        captured.out,
        captured.err,
        err_msg,
    ]
    for text in surfaces:
        assert not coord_tile_re.search(text), f"coordinate-derived tile label leaked: {text}"
        assert not token_re.search(text), f"citizen token leaked: {text}"
