"""The year QC driver against the shipped trial runner. All observations are seeded synthetic fixtures."""

from __future__ import annotations

from collections import Counter, defaultdict

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


def test_issue_snapshot_takes_the_latest_report_of_the_last_15_minutes_only():
    issue = START + 12 * H
    when = issue + np.array([-40 * 60, -16 * 60, -15 * 60, -4 * 60, 1]) * S
    result = p3.ChannelResult(
        when.astype("datetime64[ns]"),
        np.array([1.0, 2, 3, 4, 5]),
        np.array([0.5, 0.4, 0.3, 0.0, 0.9]),
        np.ones(5, bool),
        np.array([False, False, False, True, False]),
        np.zeros(5, bool),
    )
    has, value, weight, earned = p3.issue_snapshot(result, [issue])
    assert has[0] and value[0] == 4 and weight[0] == 0 and earned[0]  # a later zero-weight report supersedes
    shorter = p3.ChannelResult(
        *(getattr(result, f)[:3] for f in ("when", "values", "weights", "scored", "earned", "copycat"))
    )
    has, value, weight, _ = p3.issue_snapshot(shorter, [issue, issue + M])
    assert has[0] and value[0] == 3 and weight[0] == 0.3  # exactly 15 minutes old counts
    assert not has[1] and np.isnan(value[1]) and weight[1] == 0  # 16 minutes old does not
    assert not p3.issue_snapshot(shorter, [when[0].astype("datetime64[ns]") - S])[0][0]


def test_a_report_that_had_not_arrived_by_the_next_issue_time_is_never_used():
    noon = int(np.datetime64("2024-03-02T12:00:00", "s").astype(np.int64))
    #            measured            delay   kept?
    cases = [
        (noon - 600, 300, True),  # measured 11:50, arrived 11:55
        (noon - 600, 600, True),  # arrived exactly at the issue time
        (noon - 600, 601, False),  # arrived one second after it
        (noon, 0, True),  # measured and arrived at the issue time
        (noon, 1, False),
        (noon + 1, 700, True),  # after the issue: the next one is twelve hours away
        (noon - 3600, -1, False),  # no arrival time
        (noon - 3600, p3.UNKNOWN_DELAY_S, False),  # saturated delay
        (noon + 12 * 3600 - 5, 4, True),  # just before midnight, arrived before it
        (noon + 12 * 3600 - 5, 6, False),
    ]
    measured, delay, expected = (np.array(c) for c in zip(*cases, strict=True))
    kept = p3.arrived_by_next_issue(measured.astype("datetime64[s]"), delay.astype(np.int16))
    np.testing.assert_array_equal(kept, expected.astype(bool))
    # Whatever is kept and measured at or before an issue time had arrived by that issue time.
    rng = np.random.default_rng(8)
    m = noon + rng.integers(-40 * 3600, 40 * 3600, 20000)
    d = rng.integers(-1, 1500, 20000)
    k = p3.arrived_by_next_issue(m, d)
    for issue in noon + np.arange(-3, 4) * 12 * 3600:
        assert np.all((m + d)[k & (m <= issue)] <= issue)
