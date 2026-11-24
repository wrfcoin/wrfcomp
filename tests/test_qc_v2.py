import json
from dataclasses import asdict

import numpy as np
import pytest

from wrfcomp.assimilation.qc import Flag, background_gate, official_hour_bias, quality_control, spatial_gate
from wrfcomp.assimilation.tokens import ObservationBatch, QCConfig
from wrfcomp.assimilation.trust import StationState

CFG = QCConfig()


def batch(values, times, *, stations=None, var="t2m", lon=None):
    n = len(values)
    if stations is None:
        stations = np.zeros(n, dtype=int)
    return ObservationBatch(
        [f"{i:011}" for i in stations],
        np.zeros(n),
        np.zeros(n) if lon is None else lon,
        np.zeros(n),
        times,
        np.zeros(n),
        np.repeat(var, n),
        values,
    )


def test_collocated_block_cannot_supply_spatial_or_storm_evidence():
    t = np.datetime64("2024-07-12T12")
    for lon, official, expected in [
        ([0, 0, 0, 0], [False] * 4, False),
        ([0, 0, 0, 0.1], [False] * 4, True),
        ([0, 0, 0, 0], [False, False, False, True], True),
    ]:
        r = quality_control(
            batch([280] * 4, [t] * 4, stations=range(4), lon=lon),
            290,
            1,
            2,
            official=official,
            official_background_bias=0,
        )
        assert bool(r.qc_flags[0] & Flag.FEW_BUDDIES) != expected
        assert bool(r.huber[0] > 0) == expected
    # An independent station that disagrees cannot validate the two agreeing in-cell readings.
    _, w = background_gate(-10, np.array([-9, -9, 4]), CFG, independent=np.array([False, False, True]))
    assert w == 0
    # With only two other stations the target cannot count itself to reach three.
    r = quality_control(
        batch([280] * 3, [t] * 3, stations=range(3), lon=[0, 0.1, 0.1]), 280, 1, 1, official_background_bias=0
    )
    assert np.all(r.qc_flags & Flag.FEW_BUDDIES)


def test_official_spatial_weight_and_quantised_wind():
    _, m = spatial_gate(
        280,
        np.array([280.0, 284.0, 288.0]),
        2,
        CFG,
        official=np.array([True, False, False]),
        independent=np.ones(3, bool),
    )
    assert m == 282
    times = np.datetime64("2024-07-12T06") + np.arange(7) * np.timedelta64(15, "m")
    for value, ref, stuck in [(0, 3, False), (0.4, 3, False), (3, 3.15, False), (3, 5, True)]:
        out = quality_control(batch([value] * 7, times, var="wind_speed"), 3, 2, 2, reference=ref)
        assert bool(out.qc_flags[5] & Flag.FLATLINE) == stuck
        assert not out.qc_flags[4] & Flag.FLATLINE
    fast = times[0] + np.arange(7) * np.timedelta64(5, "m")
    assert not np.any(
        quality_control(batch([3] * 7, fast, var="wind_speed"), 3, 2, 2).qc_flags & Flag.FLATLINE
    )


def test_correction_uses_only_prior_official_measurements_in_the_same_hour():
    times = np.array(["2024-07-08T12", "2024-07-08T13", "2024-07-09T12", "2024-07-10T12"], "datetime64[ns]")
    target = times[2] + np.array([0, 1]) * np.timedelta64(1, "m")
    np.testing.assert_array_equal(official_hour_bias(times, [4.0, 90.0, 8.0, 100.0], target), [4, 6])
    # An official reading from today at the exact same time cannot affect the citizen's corrected EMA.
    b = batch([284, 300, 284], [times[0], times[2], times[2]], stations=[1, 1, 0])
    out = quality_control(b, 280, 2, 2, official=[True, True, False])
    assert out.states[("00000000000", "t2m")].bias_day == 0


def test_temperature_tier_uses_prior_complete_days_and_distinct_reference_times():
    rng = np.random.default_rng(84)
    hours = np.arange(24 * 9)
    times = np.datetime64("2024-07-01T00", "ns") + hours * np.timedelta64(1, "h")
    reference = 285 + 4 * np.sin(hours * np.pi / 12)
    values = reference + rng.uniform(0.1, 0.25, len(hours))

    def step(start, stop, states=None, stamps=None):
        sl = slice(start, stop)
        return quality_control(
            batch(values[sl], times[sl]),
            reference[sl],
            2,
            2,
            reference=reference[sl],
            states=states,
            official_background_bias=0,
            reference_time_ns=(times.astype(np.int64) if stamps is None else stamps)[sl],
        )

    key = ("00000000000", "t2m")
    first = step(0, 24 * 7)
    assert not first.states[key].earned
    resumed = {
        k: StationState(**json.loads(json.dumps(asdict(v), allow_nan=False))) for k, v in first.states.items()
    }
    second = step(24 * 7, None, resumed)
    full = step(0, None)
    assert second.states == full.states and full.states[key].earned
    np.testing.assert_array_equal(second.weights, full.weights[24 * 7 :])
    assert full.weights[-1] > 0.8
    repeated = np.repeat(times[::24].astype(np.int64), 24)
    assert not step(0, None, stamps=repeated).states[key].earned
    with pytest.raises(ValueError, match="later"):
        step(0, 1, stamps=times.astype(np.int64) + 1)


def test_unstable_and_copycat_temperatures_do_not_qualify():
    h = np.arange(24 * 9)
    times = np.datetime64("2024-07-01T00", "ns") + h * np.timedelta64(1, "h")
    ref = 285 + 5 * np.sin(h * np.pi / 12)
    for v in (ref, ref + np.where(h // 24 % 2, 1.6, -1.6), np.full(len(h), 285.0)):
        out = quality_control(
            batch(v, times),
            ref,
            2,
            2,
            reference=ref,
            reference_time_ns=times.astype(np.int64),
            official_background_bias=0,
        )
        assert not out.states[("00000000000", "t2m")].earned
        assert np.max(out.weights) <= CFG.new_prior


def test_another_copycat_channel_suspends_an_earned_temperature_credential():
    ns = int(np.datetime64("2024-07-12", "ns").astype(np.int64))
    day = ns // 86_400_000_000_000
    key = ("00000000000", "t2m")
    old = StationState(earned=True, verification_days=[[day, 0, 0.0, False]])
    wind = StationState(last_ns=ns - 1, last_copycat_ns=ns - 1)
    result = quality_control(
        batch([285.0], [np.datetime64(ns, "ns")]),
        285.0,
        2.0,
        2.0,
        states={key: old, (key[0], "wind_speed"): wind},
        official_background_bias=0.0,
    )
    assert not result.states[key].earned
    assert result.weights[0] == CFG.new_prior
    assert old.earned  # no mutation of the incoming snapshot
