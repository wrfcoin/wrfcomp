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
