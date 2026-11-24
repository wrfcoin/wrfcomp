"""A rapid temperature step needs independent evidence of the change itself."""

import numpy as np
import pytest

from wrfcomp.assimilation.qc import Flag, quality_control
from wrfcomp.assimilation.tokens import ObservationBatch


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
