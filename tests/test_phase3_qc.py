"""The year QC driver against the shipped trial runner. All observations are seeded synthetic fixtures."""

from __future__ import annotations

from collections import Counter, defaultdict
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from wrfcomp.assimilation import phase3_qc as p3
from wrfcomp.assimilation.qc import distances_km

START = np.datetime64("2024-03-01T00", "ns")
H = np.timedelta64(1, "h")
M = np.timedelta64(1, "m")
S = np.timedelta64(1, "s")
DAYS = 5


@pytest.fixture(scope="module")
def world():
    trial = p3.load_trial_module(START)
    rng = np.random.default_rng(2024)
    labels = np.arange(START, START + DAYS * 24 * H, H)
    hours = np.arange(len(labels))
    official = xr.Dataset(
        coords={
            "valid_time": labels,
            "station_id": ["00000000001", "00000000002", "00000000003"],
            "latitude": ("station_id", np.float32([40.0, 40.15, 41.5])),
            "longitude": ("station_id", np.float32([-100.0, -100.1, -100.0])),
            "elevation_m": ("station_id", np.float32([500.0, 520.0, 800.0])),
        }
    )
    diurnal = 285 + 6 * np.sin(2 * np.pi * (hours - 9) / 24)
    for var, base, noise in (
        ("t2m", diurnal, 0.3),
        ("td2m", diurnal - 6, 0.3),
        ("u10", 2 + 0 * diurnal, 1.0),
        ("v10", -1 + 0 * diurnal, 1.0),
    ):
        official[var] = (
            ("valid_time", "station_id"),
            (base[:, None] + rng.normal(0, noise, (len(labels), 3))).astype("f4"),
        )
        official[var + "_offset_min"] = (
            ("valid_time", "station_id"),
            np.full((len(labels), 3), -7, dtype=np.int16),
        )
    official["t2m"].values[30:33, 0] = np.nan
    issued, valid = [], []
    means = {var: [] for var in (*trial.DIRECT, "wind_speed")}
    for cycle in np.arange(START, START + DAYS * 24 * H, 12 * H):
        for lead in (3, 6, 12, 24, 48):
            issued.append(cycle)
            valid.append(cycle + lead * H)
            hour = (cycle + lead * H - START) / H
            level = 285 + 6 * np.sin(2 * np.pi * (hour - 9) / 24)
            means["t2m"].append(level + rng.normal(0, 0.5, 3))
            means["td2m"].append(level - 6 + rng.normal(0, 0.5, 3))
            means["u10"].append(2 + rng.normal(0, 0.5, 3))
            means["v10"].append(-1 + rng.normal(0, 0.5, 3))
            means["wind_speed"].append(2.5 + rng.normal(0, 0.5, 3))
    fc = (np.array(issued), np.array(valid), {k: np.array(v) for k, v in means.items()})
    series = trial.official_series(official, fc)

    # Seven citizen stations near the first official station, reporting every 10 minutes.
    position = [
        (40.0, -100.0, 500.0),
        (40.0, -100.1, 510.0),
        (40.1, -100.0, 490.0),
        (40.1, -100.1, 500.0),
        (40.0, -100.0, 500.0),
        (40.2, -100.0, 530.0),
        (41.5, -100.1, 800.0),
    ]
    frames = []
    for k, _ in enumerate(position):
        times = np.arange(START + (k % 5) * M, START + DAYS * 24 * H, 10 * M)
        if k == 2:
            times = times + 20 * S  # reports off the whole minute
        hour = (times - START) / H
        t2m = (
            285
            + 6 * np.sin(2 * np.pi * (hour - 9) / 24)
            + rng.normal(0, 0.4, len(times))
            + (2.5 if k == 3 else 0)
        )
        if k == 4:  # repeats the official station's readings
            idx = np.clip(np.searchsorted(labels - 7 * M, times, side="right") - 1, 0, len(labels) - 1)
            t2m = np.nan_to_num(official.t2m.values[idx, 0].astype(float), nan=285.0)
        if k == 1:
            t2m[200] += 25  # spike
            t2m[300:310] = t2m[300]  # stuck
        td2m = t2m - 6 + rng.normal(0, 0.3, len(times))
        speed = np.abs(rng.normal(2.5, 1.0, len(times)))
        direction = rng.uniform(0, 360, len(times))
        if k == 5:
            td2m[::7] = np.nan
        frames.append(
            pd.DataFrame(
                {
                    "station": k,
                    "time": times,
                    "t2m": t2m.astype("f4"),
                    "td2m": td2m.astype("f4"),
                    "wind_speed": speed.astype("f4"),
                    "wind_dir": direction.astype("f4"),
                }
            )
        )
    geo = pd.DataFrame(position, columns=["lat", "lon", "height"])
    return trial, official, fc, series, frames, geo


def _trial_weights(trial, official, fc, series, frames, geo):
    wides = {}
    for k, frame in enumerate(frames):
        rad = np.radians(frame.wind_dir.to_numpy(float))
        speed = frame.wind_speed.to_numpy(float)
        wide = pd.DataFrame(
            {
                "t2m": frame.t2m.to_numpy(),
                "td2m": frame.td2m.to_numpy(),
                "u10": (-speed * np.sin(rad)).astype("f4"),
                "v10": (-speed * np.cos(rad)).astype("f4"),
                "wind_speed": frame.wind_speed.to_numpy(),
            },
            index=pd.DatetimeIndex(frame.time),
        )
        wides[k] = wide
    sources, faults = trial.citizen_sources(wides, geo, official, fc, series)
    out = {}
    for k, wide in wides.items():
        samples = defaultdict(list)
        trial.aggregate_station(
            wide,
            geo.iloc[k][["lat", "lon", "height"]].to_numpy(),
            official,
            fc,
            series,
            defaultdict(Counter),
            defaultdict(set),
            samples,
            k,
            geo=geo,
            sources=sources,
            copy_times=faults.get(k, ()),
        )
        for var in p3.THERMAL:
            if samples[var]:
                out[k, var] = samples[var][0]
    return out, faults


def _driver(trial, official_ds, fc, series, frames, geo, until=None):
    official = p3.Official(
        official_ds.latitude.values, official_ds.longitude.values, official_ds.elevation_m.values, series
    )
    archive = p3.archive_from_arrays(*fc, trial)
    stations = {}
    for k, frame in enumerate(frames):
        if until is not None:
            frame = frame[frame.time.to_numpy() <= until]
        stations[k] = p3.StationData(
            *geo.iloc[k][["lat", "lon", "height"]],
            frame.time.to_numpy().astype("datetime64[ns]"),
            frame.t2m.to_numpy(),
            frame.td2m.to_numpy(),
            frame.wind_speed.to_numpy(),
            frame.wind_dir.to_numpy(),
        )
    first = {k: p3.station_source(st, official, archive, trial) for k, st in stations.items()}
    results = {}
    for k, st in stations.items():
        d = distances_km(st.lat, st.lon, geo.lat.values, geo.lon.values)
        near = [j for j in range(len(geo)) if j != k and d[j] <= trial.CFG.radius_km]
        results[k] = p3.station_weights(
            st, [(stations[j], first[j][0]) for j in near], first[k][1], official, archive, trial
        )
    return results, first


def test_forecast_choice_table_equals_the_trial_rule(world):
    trial, _, fc, *_ = world
    archive = p3.archive_from_arrays(*fc, trial)
    rng = np.random.default_rng(5)
    span = int((fc[1].max() - START) / S) + 6 * 3600
    times = START - 3 * H + rng.integers(0, span, 5000) * S
    edges = np.arange(START - 3 * H, fc[1].max() + 4 * H, 30 * M)
    times = np.concatenate(
        [times, edges, edges + S, edges - S, [START - 5 * 24 * H, fc[1].max() + 30 * 24 * H]]
    )
    np.testing.assert_array_equal(archive.choose(times), trial.forecast_indices(fc[0], fc[1], times))


def test_driver_gives_the_trial_runner_weights_exactly(world):
    trial, official, fc, series, frames, geo = world
    reference, faults = _trial_weights(trial, official, fc, series, frames, geo)
    results, first = _driver(trial, official, fc, series, frames, geo)
    compared = 0
    for k in range(len(frames)):
        np.testing.assert_array_equal(first[k][1], np.asarray(faults.get(k, ()), dtype=np.int64))
        for var in p3.THERMAL:
            mine = results[k][var].weights[results[k][var].scored]
            np.testing.assert_array_equal(mine, reference[k, var])
            compared += len(mine)
    assert compared > 5000
    # The fixture exercises the rules: an honest station is used, the copy and the faults are not.
    assert results[0]["t2m"].weights.max() > 0.2
    assert results[4]["t2m"].copycat.any() and len(first[4][1])
    assert results[1]["t2m"].weights[200] == 0 and results[1]["t2m"].weights[305] == 0


def test_reports_after_the_issue_time_change_nothing_at_or_before_it(world):
    trial, official, fc, series, frames, geo = world
    full, _ = _driver(trial, official, fc, series, frames, geo)
    issue = START + 3 * 24 * H + 12 * H
    cut, _ = _driver(trial, official, fc, series, frames, geo, until=issue)
    changed = [
        f.assign(t2m=np.where(f.time.to_numpy() > issue, f.t2m + 9, f.t2m).astype("f4")) for f in frames
    ]
    moved, _ = _driver(trial, official, fc, series, changed, geo)
    for k in range(len(frames)):
        for var in p3.THERMAL:
            n = len(cut[k][var].weights)
            assert n and np.all(cut[k][var].when <= issue)
            for other in (full, moved):
                np.testing.assert_array_equal(other[k][var].weights[:n], cut[k][var].weights)
                np.testing.assert_array_equal(other[k][var].earned[:n], cut[k][var].earned)
            for other in (full, moved):
                for a, b in zip(
                    p3.issue_snapshot(other[k][var], [issue]),
                    p3.issue_snapshot(cut[k][var], [issue]),
                    strict=True,
                ):
                    np.testing.assert_array_equal(a, b)


def _channel(seconds_before, values, weights, issue):
    when = issue.astype("datetime64[ns]") + (np.array(seconds_before, float) * 1e9).astype("timedelta64[ns]")
    n = len(when)
    return p3.ChannelResult(
        when,
        np.array(values, float),
        np.array(weights, float),
        np.ones(n, bool),
        np.zeros(n, bool),
        np.zeros(n, bool),
    )


def test_issue_snapshot_takes_the_latest_positive_weight_report_of_the_last_15_minutes():
    """Review case (rule 3): a zero-weight report does not suppress an earlier eligible one."""
    issue = START + 12 * H
    result = _channel(
        [-40 * 60, -16 * 60, -15 * 60, -4 * 60, 1], [1, 2, 3, 4, 5], [0.5, 0.4, 0.3, 0.0, 0.9], issue
    )
    has, value, weight, _, age = p3.issue_snapshot(result, [issue])
    assert (
        has[0] and value[0] == 3 and weight[0] == 0.3 and age[0] == 900 * 10**9
    )  # the 4-minute-old report has weight 0
    has, value, *_ = p3.issue_snapshot(result, [issue + M])
    assert has[0] and value[0] == 5  # the report measured one second after T counts later, never at T
    earlier = _channel([-40 * 60, -16 * 60, -15 * 60, -4 * 60], [1, 2, 3, 4], [0.5, 0.4, 0.3, 0.0], issue)
    has, value, weight, _, _ = p3.issue_snapshot(earlier, [issue + M])
    assert not has[0] and np.isnan(value[0]) and weight[0] == 0  # 16 minutes old does not count
    # Review case: 900.25 s before T is outside the inclusive 15 minutes; 900 s exactly is inside.
    edge = _channel([-900.25], [7], [0.5], issue)
    assert not p3.issue_snapshot(edge, [issue])[0][0]
    assert p3.issue_snapshot(_channel([-900], [7], [0.5], issue), [issue])[0][0]
    assert not p3.issue_snapshot(_channel([0.25], [7], [0.5], issue), [issue])[0][0]  # measured after T
    # Review case: the age is exact, so 60.9 s and 60.1 s are different ages (not both "60").
    ages = [p3.issue_snapshot(_channel([s], [7], [0.5], issue), [issue])[4][0] for s in (-60.9, -60.1)]
    assert ages == [60_900_000_000, 60_100_000_000] and ages[1] < ages[0]
    only_zero = _channel([-300, -100], [1, 2], [0.0, 0.0], issue)
    assert not p3.issue_snapshot(only_zero, [issue])[0][0]
    newest = _channel([-600, -100], [1, 2], [0.4, 0.2], issue)
    assert p3.issue_snapshot(newest, [issue])[1][0] == 2  # of two eligible reports, the later one


def test_official_readings_measured_before_the_year_are_removed():
    """Review case: a 00Z label on 1 January with a negative offset was measured the year before."""
    start = np.datetime64("2024-01-01T00", "ns")
    labels = np.arange(start - 2 * H, start + 3 * H, H)
    truth = xr.Dataset(coords={"valid_time": labels, "station_id": ["00000000001", "00000000002"]})
    for var in ("t2m", "td2m"):
        truth[var] = (("valid_time", "station_id"), np.full((5, 2), 280.0, dtype="f4"))
        truth[var + "_offset_min"] = (("valid_time", "station_id"), np.array([[-7, 0]] * 5, dtype=np.int16))
    truth["t2m_offset_min"].values[3, 0] = 7
    out = p3.within_year(truth, start)
    assert out.valid_time.values[0] == start and out.sizes["valid_time"] == 3
    np.testing.assert_array_equal(np.isfinite(out.t2m.values), [[False, True], [True, True], [True, True]])
    np.testing.assert_array_equal(np.isfinite(out.td2m.values), [[False, True], [True, True], [True, True]])
    assert np.isfinite(truth.t2m.values).all()  # the input is not changed


# ----------------------------------------------------------------------------- incomplete or stale products


def _script():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "run_phase3_qc", Path(__file__).parents[1] / "scripts/run_phase3_qc.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _day_table(day, n=4):
    from wrfcomp.assimilation.madis_archive import ReportTable

    base = np.datetime64(day + "T06:00:00", "s")
    return ReportTable(
        station_id=np.array([f"CW{i:09d}" for i in range(n)]),
        obs_time=(base + np.arange(n) * 60 * np.timedelta64(1, "s")).astype("datetime64[ns]"),
        received_time=(base + (np.arange(n) * 60 + 300) * np.timedelta64(1, "s")).astype("datetime64[ns]"),
        lat=np.float32([40.0, 40.1, 41.2, 40.0][:n]),
        lon=np.float32([-100.0, -99.9, -101.3, -100.0][:n]),
        elevation_m=np.full(n, 500, dtype=np.float32),
        t2m=np.full(n, 280, dtype=np.float32),
        td2m=np.full(n, 275, dtype=np.float32),
        wind_speed=np.full(n, 2, dtype=np.float32),
        wind_dir=np.full(n, 90, dtype=np.float32),
    )


def _year_dir(tmp_path, script, monkeypatch, days=("2024-01-01", "2024-01-02", "2024-02-01")):
    from wrfcomp.assimilation.madis_archive import save_table

    monkeypatch.setattr(script, "expected_days", lambda year: list(days))
    citizen = tmp_path / "citizen"
    for day in days:
        attrs = {"day": day, "hours_requested": 24, "hours_missing": "07" if day.endswith("02") else ""}
        save_table(_day_table(day), citizen / "conus" / "2024" / f"{day}.nc", attrs=attrs)
    return citizen


def test_partition_needs_every_day_and_records_missing_source_hours(tmp_path, monkeypatch):
    script = _script()
    citizen = _year_dir(tmp_path, script, monkeypatch)
    (citizen / "conus" / "2024" / "2024-01-02.nc").rename(tmp_path / "aside.nc")
    with pytest.raises(script.IncompleteError, match="1 of 3 day files are missing"):
        script.partition(citizen, 2024)
    assert not (citizen / "conus" / "2024-tiles").exists()
    (tmp_path / "aside.nc").rename(citizen / "conus" / "2024" / "2024-01-02.nc")
    manifest = script.partition(citizen, 2024)
    assert manifest["days"] == 3 and manifest["reports"] == 12 and len(manifest["tiles"]) == 2
    assert manifest["missing_source_hours"] == {"2024-01-02": ["07"]}  # a fact about the archive, kept
    assert script.partition(citizen, 2024)["digest"] == manifest["digest"]  # same inputs: accepted as done


def test_partition_refuses_tiles_made_from_other_files_and_stale_day_files(tmp_path, monkeypatch):
    from wrfcomp.assimilation.madis_archive import StaleTableError, save_table

    script = _script()
    citizen = _year_dir(tmp_path, script, monkeypatch)
    script.partition(citizen, 2024)
    day = citizen / "conus" / "2024" / "2024-02-01.nc"
    save_table(_day_table("2024-02-01", n=3), day, attrs={"day": "2024-02-01", "hours_missing": ""})
    with pytest.raises(script.IncompleteError, match="other daily files"):
        script.partition(citizen, 2024)
    # A day file written before the amendment (no arrival policy) is refused.
    import shutil

    shutil.rmtree(citizen / "conus" / "2024-tiles")
    with xr.open_dataset(day, engine="h5netcdf", decode_times=False) as ds:
        old = ds.load()
    del old.attrs["arrival_policy"]
    day.unlink()
    old.to_netcdf(day, engine="h5netcdf")
    with pytest.raises(StaleTableError):
        script.partition(citizen, 2024)
    assert not (citizen / "conus" / "2024-tiles").exists()


GEFS: dict = {}


def _fake_run(tmp_path, script, monkeypatch, tiles):
    citizen = _year_dir(tmp_path, script, monkeypatch)
    manifest = script.partition(citizen, 2024)
    truth = xr.Dataset(
        coords={"valid_time": [np.datetime64("2024-01-01", "ns")], "station_id": ["00000000001"]}
    )
    (tmp_path / "truth").mkdir()
    truth.to_netcdf(tmp_path / "truth" / "isd_conus_2024.nc")
    # Stand-in GEFS content: the real loader is replaced, the content digest is the real one.
    GEFS["t2m"] = np.zeros((2, 1))
    cycle = np.array(["2024-01-01T00"], dtype="datetime64[ns]")
    fake = lambda *_: (None, None, SimpleNamespace(issued=cycle, valid=cycle + 3 * H, means=dict(GEFS)))  # noqa: E731
    monkeypatch.setattr(p3, "load_year_inputs", fake)
    record = p3.current_run_record(tmp_path, 2024)
    out = citizen / "conus" / "2024-qc"
    out.mkdir()
    (out / "run.json").write_text(__import__("json").dumps(record))
    rows = {
        "station": [0],
        "issue": [3],
        "var": [0],
        "value": [280.0],
        "weight": [0.4],
        "earned": [False],
        "age_ns": [60 * 10**9],
    }
    for name in tiles:
        script.write_tile_output(
            out / f"{name}.npz",
            record["digest"],
            ["CW000000000"],
            [40.0],
            [-100.0],
            [500.0],
            [[9, 9, 4, 4, 0, 0]],
            rows,
        )
    return manifest, record, out


def test_merge_refuses_a_partial_run_and_a_run_from_other_inputs(tmp_path, monkeypatch):
    script = _script()
    manifest, record, out = _fake_run(tmp_path, script, monkeypatch, tiles=())
    first, second = manifest["tiles"]
    rows = {
        "station": [0],
        "issue": [3],
        "var": [0],
        "value": [280.0],
        "weight": [0.4],
        "earned": [False],
        "age_ns": [60 * 10**9],
    }
    args = (["CW000000000"], [40.0], [-100.0], [500.0], [[9, 9, 4, 4, 0, 0]], rows)
    script.write_tile_output(out / f"{first}.npz", record["digest"], *args)
    with pytest.raises(script.IncompleteError, match="1 of 2 tiles have no QC output"):
        script.merge(tmp_path, 2024)  # a run on a subset of tiles cannot become an extract
    script.write_tile_output(out / f"{second}.npz", "made-from-something-else", *args)
    with pytest.raises(script.IncompleteError, match="other inputs"):
        script.merge(tmp_path, 2024)
    script.write_tile_output(out / f"{second}.npz", record["digest"], *args)
    summary = script.merge(tmp_path, 2024)
    assert (
        summary["stations_weighted"] == 2
        and summary["issue_rows"] == 2
        and summary["missing_source_hours"] == 1
    )
    with xr.open_dataset(tmp_path / "citizen" / "conus" / "issue_extract_2024.nc", engine="h5netcdf") as ds:
        assert ds.attrs["complete"] == 1 and ds.attrs["run_digest"] == record["digest"]
        assert ds.sizes["issue"] == 704 and ds.official_station_id.values.astype(str).tolist() == [
            "00000000001"
        ]
    # A GEFS store with the same name and other content: refused (the digest is of content, not names).
    GEFS["t2m"] = np.ones((2, 1))
    with pytest.raises(script.IncompleteError, match="other inputs or code"):
        script.merge(tmp_path, 2024)
    GEFS["t2m"] = np.zeros((2, 1))
    # A tile-month set aside, altered or added after partition: refused by partition, run and merge.
    tiles = tmp_path / "citizen" / "conus" / "2024-tiles"
    part = sorted(tiles.glob("*/*.npz"))[0]
    part.rename(tmp_path / "aside.npz")
    for step in (
        lambda: script.partition(tmp_path / "citizen", 2024),
        lambda: script.merge(tmp_path, 2024),
        lambda: script.run(tmp_path, 2024, 1, None),
        lambda: p3.verify_tiles(tiles),
    ):
        with pytest.raises(script.IncompleteError, match="1 tile-month files are missing"):
            step()
    part.write_bytes((tmp_path / "aside.npz").read_bytes() + b"x")
    with pytest.raises(script.IncompleteError, match="not the ones the manifest lists"):
        script.merge(tmp_path, 2024)
    (tmp_path / "aside.npz").replace(part)
    (part.parent / "2031-01.npz").write_bytes(b"unlisted")
    with pytest.raises(script.IncompleteError, match="1 are not in the manifest"):
        p3.verify_tiles(tiles)
    (part.parent / "2031-01.npz").unlink()
    assert script.merge(tmp_path, 2024)["run_digest"] == record["digest"]
    # The truth file changes after the run: what is on disk no longer describes the outputs.
    xr.Dataset(
        coords={"valid_time": [np.datetime64("2024-01-02", "ns")], "station_id": ["00000000001"]}
    ).to_netcdf(tmp_path / "truth" / "isd_conus_2024.nc")
    with pytest.raises(script.IncompleteError, match="other inputs or code"):
        script.merge(tmp_path, 2024)


def test_a_timing_run_has_its_own_folder_and_merge_never_reads_it(tmp_path, monkeypatch):
    script = _script()
    manifest, record, out = _fake_run(tmp_path, script, monkeypatch, tiles=())
    timing = out.with_name("2024-qc.timing")
    out.rename(timing)
    rows = {
        "station": [0],
        "issue": [3],
        "var": [0],
        "value": [280.0],
        "weight": [0.4],
        "earned": [False],
        "age_ns": [60 * 10**9],
    }
    for name in manifest["tiles"]:
        script.write_tile_output(
            timing / f"{name}.npz",
            record["digest"],
            ["CW000000000"],
            [40.0],
            [-100.0],
            [500.0],
            [[9, 9, 4, 4, 0, 0]],
            rows,
        )
    with pytest.raises(script.IncompleteError, match="no manifest"):
        script.merge(tmp_path, 2024)
    assert not (tmp_path / "citizen" / "conus" / "issue_extract_2024.nc").exists()


def test_tile_part_names_use_forward_slashes_on_every_system(tmp_path, monkeypatch):
    """The manifest names a part ``<tile>/<month>.npz``; the check must build the same name on Windows."""
    import json
    from pathlib import PurePosixPath, PureWindowsPath

    win = PureWindowsPath(r"C:\data\citizen\conus\2024-tiles")
    assert str((win / "+40_-100" / "2024-01.npz").relative_to(win)) == "+40_-100\\2024-01.npz"  # the bug
    assert p3.part_name(win / "+40_-100" / "2024-01.npz", win) == "+40_-100/2024-01.npz"
    posix = PurePosixPath("/data/citizen/conus/2024-tiles")
    assert p3.part_name(posix / "+40_-100" / "2024-01.npz", posix) == "+40_-100/2024-01.npz"
    # verify_tiles compares the manifest with these names, so a set of tiles written with "/" is
    # accepted; names built the Windows way would all be refused as missing and unlisted.
    tiles = tmp_path / "2024-tiles"
    (tiles / "+40_-100").mkdir(parents=True)
    (tiles / "+40_-100" / "2024-01.npz").write_bytes(b"part")
    parts = {"+40_-100/2024-01.npz": p3.sha256_file(tiles / "+40_-100" / "2024-01.npz")}
    (tiles / "manifest.json").write_text(json.dumps({"digest": "d", "parts": parts}))
    assert p3.verify_tiles(tiles)["parts"] == parts
    windows_style = lambda path, root: str(PureWindowsPath(path.relative_to(root)))  # noqa: E731
    monkeypatch.setattr(p3, "part_name", windows_style)
    with pytest.raises(p3.IncompleteError, match="1 tile-month files are missing and 1 are not"):
        p3.verify_tiles(tiles)
