import json
from dataclasses import asdict

import numpy as np
import pytest

from wrfcomp.assimilation.qc import Flag, quality_control, static_gate
from wrfcomp.assimilation.tokens import ObservationBatch, QCConfig
from wrfcomp.assimilation.trust import StationState, qod_score

CFG = QCConfig()


def batch(values, *, var="t2m", times=None, stations=None, lon=None, elev=0):
    n = len(values)
    idx = np.arange(n) if stations is None else np.asarray(stations)
    return ObservationBatch(
        station_ids=[f"{s:011}" for s in idx],
        lat=np.full(n, 35),
        lon=np.broadcast_to(0 if lon is None else lon, n),
        elevation_m=np.broadcast_to(elev, n),
        obs_time=np.broadcast_to(np.datetime64("2024-04-01T12") if times is None else times, n),
        sensor_type_id=np.zeros(n),
        variable=np.broadcast_to(var, n),
        values=values,
    )


@pytest.mark.parametrize(
    ("var", "value", "peers", "elev"),
    [
        ("t2m", 145, {}, 0),
        ("td2m", 290, {"t2m": 280}, 0),
        ("wind_gust", 4, {"u10": 3, "v10": 4}, 0),
        ("station_pressure", 101325, {}, 2500),
    ],
)
def test_static(var, value, peers, elev):
    assert static_gate(var, value, elev, peers, CFG) == Flag.STATIC


def test_temporal_flatline_precip_and_step():
    times = np.datetime64("2024-03-01") + np.arange(6) * np.timedelta64(15, "m")
    for var, values, flat in [("precip", [0] * 6, False), ("wind_speed", [3] * 6, True)]:
        b = batch(values, var=var, times=times, stations=np.zeros(6, int))
        r = quality_control(b, values, 1, 1)
        assert bool(r.qc_flags[-1] & Flag.FLATLINE) == flat
        assert r.weights[-1] == (0 if flat else 0.5)
    values = np.random.default_rng(41).normal(280, 0.1, 6)
    values[-1] += 11
    b = batch(values, times=times, stations=np.zeros(6, int))
    r = quality_control(b, values, 1, 1)
    assert r.qc_flags[-1] & Flag.TEMPORAL
    assert not b.qc_flags.any()
    np.testing.assert_array_equal(b.values, values.astype("float32"))


@pytest.mark.parametrize(
    ("drop", "longitude", "expected"),
    [
        (4.5, [0, 1, 1], 2.5 / 4.5),
        (9, [0, 0.1, -0.1], 2.5 / 9),
        (9, [0, 0.1, 1], 0),
        (9, [0, 1, 1], 0),
    ],
)
def test_storm(drop, longitude, expected):
    bg = np.random.default_rng(42).integers(285, 290, 3)
    r = quality_control(batch(bg - [drop, 4, 4], lon=longitude), bg, 1, 2, official=True)
    assert r.qc_flags[0] & Flag.BACKGROUND
    assert r.weights[0] == pytest.approx(expected)


def test_robust_buddies_and_mountain_lapse():
    heights = np.random.default_rng(43).integers(0, 500, 9)
    heights[0] = 0
    values = 280 - CFG.lapse_k_per_km * heights / 1000
    values[-3:] += 20
    r = quality_control(batch(values, elev=heights), values, 1, 1)
    assert r.buddy_median[0] == pytest.approx(280)
    assert r.qc_flags[-1] & Flag.SPATIAL
    assert not (r.qc_flags[0] & Flag.SPATIAL)


def test_solar_day_penalty_and_replay():
    rng = np.random.default_rng(44)
    day_weights, night_weights = [], []
    states = {}
    for hour in range(24 * 12):
        time = np.datetime64("2024-03-20") + np.timedelta64(hour, "h")
        h = hour % 24
        if h not in (0, 1, 2, 10, 11, 12):
            continue
        bg = 285 + rng.normal()
        b = batch([bg + (4.5 if h >= 10 else 0.2)], times=time)
        r = quality_control(b, bg, 2, 10, states=states, official=True)
        restored = {
            k: StationState(**json.loads(json.dumps(asdict(v), allow_nan=False))) for k, v in states.items()
        }
        replay = quality_control(b, bg, 2, 10, states=restored, official=True)
        assert replay.states == r.states
        np.testing.assert_array_equal(replay.weights, r.weights)
        states = r.states
        (day_weights if h >= 10 else night_weights).append(r.weights[0])
    assert day_weights[-1] < 0.4
    assert night_weights[-1] > 0.8


def test_indoor_sensor_and_copycat():
    states = {}
    rng = np.random.default_rng(45)
    for hour in range(24 * 7):
        time = np.datetime64("2024-01-01") + np.timedelta64(hour, "h")
        outdoor = 290 + 5 * np.sin(hour * np.pi / 12) + rng.normal(0, 0.1)
        values = np.array([294, outdoor, outdoor + 0.1, outdoor - 0.2, outdoor + 0.3, outdoor])
        refs = [np.nan] * 5 + [outdoor]
        r = quality_control(batch(values, times=time), outdoor, 2, 5, reference=refs, states=states)
        states = r.states
    assert states[("00000000000", "t2m")].correlation_guard < 0.01
    copy = states[("00000000005", "t2m")]
    assert r.qc_flags[-1] & Flag.COPYCAT
    assert copy.correlation_guard == qod_score(copy) == 0
    assert not states[("00000000001", "t2m")].copycat


def test_buddies_distinct_and_same_time_only():
    time = np.datetime64("2024-02-02")
    b = batch([280, 280, 280], times=[time, time, time + np.timedelta64(1, "h")])
    r = quality_control(b, 290, 1, 3)
    assert r.weights[0] == 0
    assert r.qc_flags[0] & Flag.FEW_BUDDIES
    with pytest.raises(ValueError, match="duplicate"):
        quality_control(batch([280, 280], stations=[1, 1]), 280, 1, 3)


def test_batch_order_matches_streaming_and_excludes_future_reports():
    rng = np.random.default_rng(46)
    times = np.datetime64("2024-02-02") + np.arange(8) * np.timedelta64(1, "h")
    values = rng.normal(281, 1, (8, 4))
    values[-1, 0] -= 15
    stations = np.tile(np.arange(4), 8)
    order = rng.permutation(values.size)
    combined = quality_control(
        batch(values.ravel()[order], stations=stations[order], times=np.repeat(times, 4)[order]), 281, 1, 3
    )
    states, weights, flags = {}, [], []
    for t, reports in zip(times, values, strict=True):
        step = quality_control(batch(reports, times=t), 281, 1, 3, states=states)
        states = step.states
        weights.extend(step.weights)
        flags.extend(step.qc_flags)
    np.testing.assert_array_equal(combined.weights, np.asarray(weights)[order])
    np.testing.assert_array_equal(combined.qc_flags, np.asarray(flags)[order])
    assert states == combined.states
