"""A rapid temperature step needs independent evidence of the change itself."""

import numpy as np
import pytest

from wrfcomp.assimilation.qc import Flag, quality_control
from wrfcomp.assimilation.tokens import ObservationBatch, QCConfig


@pytest.mark.parametrize(
    ("offsets", "official", "drops", "retained"),
    [
        ([0, 0.12], [False, True], [12, 12], True),
        ([0, -0.12, 0.12], [False] * 3, [12, 12, 12], True),
        ([0, 0.10, 0.11], [False] * 3, [12, 12, 12], False),
        ([0, -0.12, 0.12], [False] * 3, [12, 0, 0], False),
    ],
)
def test_temperature_step_needs_a_matching_official_or_separated_witnesses(
    offsets, official, drops, retained
):
    rng = np.random.default_rng(72)
    n = len(drops)
    start = np.datetime64("2024-04-24T18")
    before = rng.normal(295, 0.1, n)
    batch = ObservationBatch(
        station_ids=[f"{i:011}" for i in range(n)] * 2,
        lat=np.zeros(2 * n),
        lon=np.tile(offsets, 2),
        elevation_m=np.zeros(2 * n),
        obs_time=np.repeat([start, start + np.timedelta64(15, "m")], n),
        sensor_type_id=np.zeros(2 * n),
        variable=["t2m"] * (2 * n),
        values=np.r_[before, before - drops],
    )
    result = quality_control(batch, 295, 1, 2, official=np.tile(official, 2), official_background_bias=0)
    assert bool(result.qc_flags[n] & Flag.TEMPORAL) != retained
    assert bool(result.weights[n] > 0) == retained


def test_official_controls_spatial_flag_and_citizens_supply_separate_trust_history():
    rng = np.random.default_rng(142)
    n = 5
    wind = np.r_[rng.normal(3, 0.02, n - 1), 10.0]
    report_time = np.datetime64("2024-04-24T18")
    batch = ObservationBatch(
        station_ids=[f"{i:011}" for i in range(n)],
        lat=np.zeros(n),
        lon=[0, -0.12, 0.11, 0.13, 0],
        elevation_m=np.zeros(n),
        obs_time=np.repeat(report_time, n),
        sensor_type_id=np.zeros(n),
        variable=["wind_speed"] * n,
        values=wind,
    )
    result = quality_control(batch, wind, 1, 2, official=[False] * 4 + [True])
    assert result.qc_flags[0] & Flag.SPATIAL
    assert result.buddy_median[0] == 10.0
    assert result.states[(batch.station_ids[0], "wind_speed")].pairs[-1][2] == pytest.approx(
        np.median(wind[1:4])
    )


def test_reused_official_measurement_does_not_repeat_correlation_decay():
    rng = np.random.default_rng(143)
    n = 24
    hours = np.datetime64("2024-04-01", "ns") + np.arange(n) * np.timedelta64(1, "h")
    times = np.repeat(hours, 2) + np.tile(np.array([0, 10]) * np.timedelta64(1, "m"), n)
    reference = np.repeat(280 + np.arange(n) * 0.3, 2)
    values = np.repeat(295 - np.arange(n) * 0.3, 2) + rng.uniform(-0.05, 0.05, len(times))
    batch = ObservationBatch(
        station_ids=["00000000000"] * len(times),
        lat=np.zeros(len(times)),
        lon=np.zeros(len(times)),
        elevation_m=np.zeros(len(times)),
        obs_time=times,
        sensor_type_id=np.zeros(len(times)),
        variable=["t2m"] * len(times),
        values=values,
    )
    result = quality_control(
        batch,
        values,
        2,
        2,
        reference=reference,
        reference_time_ns=np.repeat(hours.astype(np.int64), 2),
        official_background_bias=0,
    )
    state = result.states[(batch.station_ids[0], "t2m")]
    assert len([p for p in state.pairs if p[3] is not None]) == n

    cfg = QCConfig()
    assert state.correlation_guard == pytest.approx(cfg.guard_decay ** (n - cfg.min_pairs + 1))


def test_hourly_official_can_confirm_rapid_temperature_step_but_must_match_levels():
    rng = np.random.default_rng(145)
    t = np.datetime64("2024-07-11T18:00")
    stamps = t + np.array([-60, -15, 0, 0]) * np.timedelta64(1, "m")
    for background_offset, kept in ((0.0, True), (6.0, False)):
        values = np.array([298.0 + background_offset, 298.0, 286.0, 286.0 + background_offset]) + rng.uniform(
            -0.1, 0.1, 4
        )
        b = ObservationBatch(
            ["00000000001", "00000000000", "00000000000", "00000000001"],
            np.zeros(4),
            [0.12, 0, 0, 0.12],
            np.zeros(4),
            stamps,
            np.zeros(4),
            ["t2m"] * 4,
            values,
        )
        out = quality_control(
            b, 298.0, 1.0, 2.0, official=[True, False, False, True], official_background_bias=0
        )
        assert bool(out.weights[2] > 0) == kept


def test_single_official_anchors_centre_without_forcing_zero_spread():
    from wrfcomp.assimilation.qc import spatial_gate

    refs = np.array([True, False, False, False, False])
    values = np.array([280.0, 278.0, 280.0, 282.0, 284.0])
    flag, center = spatial_gate(285.0, values, 2.0, QCConfig(), official=refs, independent=np.ones(5, bool))
    assert center == 280 and not flag & Flag.SPATIAL


def test_corrected_thermal_innovations_do_not_pay_for_raw_geographic_offset():
    rng = np.random.default_rng(146)
    hours = np.arange(24 * 9)
    times = np.datetime64("2024-07-09T00", "ns") + hours * np.timedelta64(1, "h")
    bg = 286 + 4 * np.sin(hours * np.pi / 12)
    values = bg + rng.uniform(-0.1, 0.1, len(hours))
    b = ObservationBatch(
        ["00000000000"] * len(hours),
        np.zeros(len(hours)),
        np.zeros(len(hours)),
        np.zeros(len(hours)),
        times,
        np.zeros(len(hours)),
        ["t2m"] * len(hours),
        values,
    )
    # Raw reference separated by 6 K, while elevation-adjusted and bias-corrected forecast is accurate.
    out = quality_control(
        b,
        bg - 3,
        2,
        2,
        reference=bg - 6,
        official_background_bias=3,
        reference_time_ns=times.astype(np.int64),
    )
    assert np.median(out.weights[-24:]) > 0.45


def test_near_copy_with_point_one_kelvin_scatter_needs_sustained_unique_evidence():
    rng = np.random.default_rng(147)
    hours = np.arange(24 * 9)
    times = np.datetime64("2024-07-09T00", "ns") + hours * np.timedelta64(1, "h")
    reference = 287 + 6 * np.sin(hours * np.pi / 12)
    values = reference + rng.normal(0.0, 0.1, len(hours))
    b = ObservationBatch(
        ["00000000000"] * len(hours),
        np.zeros(len(hours)),
        np.zeros(len(hours)),
        np.zeros(len(hours)),
        times,
        np.zeros(len(hours)),
        ["t2m"] * len(hours),
        values,
    )
    out = quality_control(
        b,
        reference,
        2,
        2,
        reference=reference,
        official_background_bias=0,
        reference_time_ns=times.astype(np.int64),
    )
    assert out.qc_flags[-1] & Flag.COPYCAT
    assert not out.states[(b.station_ids[0], "t2m")].earned
    assert out.weights[-1] == 0


@pytest.mark.parametrize("has_reference", [False, True])
def test_wind_drift_falls_in_weight_but_a_stable_exposure_offset_does_not(has_reference):
    rng = np.random.default_rng(148)
    n = 24 * 16
    times = np.datetime64("2024-07-01", "ns") + np.arange(n) * np.timedelta64(1, "h")
    wind = 10 + np.sin(np.arange(n) * np.pi / 12) + rng.normal(0, 0.25, n)

    def check(offset):
        b = ObservationBatch(
            ["00000000000"] * n,
            np.zeros(n),
            np.zeros(n),
            np.zeros(n),
            times,
            np.zeros(n),
            ["wind_speed"] * n,
            wind + offset,
        )
        return quality_control(
            b,
            wind,
            2,
            2,
            reference=wind if has_reference else np.nan,
            reference_time_ns=times.astype(np.int64),
        )

    drift = check(np.linspace(0, 8, n))
    stable = check(4.0)
    assert np.mean(drift.weights[-24:]) < 0.1
    assert np.mean(stable.weights[-24:]) > 0.4
