"""Acceptance checks for the registered Phase 3 arms (docs/PHASE3_DESIGN.md, 1.2). Synthetic data only."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from wrfcomp.baselines.io import UNITS, layout
from wrfcomp.models import citizen_arms as arms

H = np.timedelta64(1, "h")
D = np.timedelta64(1, "D")
START = np.datetime64("2024-01-01", "ns")
LEADS = [1, 3, 6, 12, 24, 48]
CHANGED = [0, 1, 2]  # positions of the 1, 3 and 6 h leads
N_DAYS = 45
# Five citizen stations in three cells around the first official station, two near the second
# (too few), one near the third but 400 m higher, none near the fourth.
CITIZEN = np.array(
    [
        (40.0, -100.0, 500),
        (40.0, -100.1, 500),
        (40.1, -100.0, 520),
        (40.1, -100.0, 480),
        (40.2, -100.1, 510),
        (42.0, -100.0, 300),
        (42.1, -100.0, 300),
        (44.0, -100.0, 900),
    ],
    dtype=float,
)


def world(*, truth_shift_after=None, citizen_shift_after=None, own_effect=0.5, citizen_effect=0.6):
    rng = np.random.default_rng(77)
    stations = xr.Dataset(
        coords={
            "station_id": ["00000000001", "00000000002", "00000000003", "00000000004"],
            "latitude": ("station_id", [40.05, 42.0, 44.0, 46.0]),
            "longitude": ("station_id", [-100.05, -100.0, -100.0, -100.0]),
            "elevation_m": ("station_id", [500.0, 300.0, 500.0, 400.0]),
        }
    )
    inits = np.arange(START, START + N_DAYS * D, 12 * H)
    labels = np.arange(START, START + (N_DAYS + 3) * D, H)
    truth = xr.Dataset(coords={"valid_time": labels, **dict(stations.coords)})
    hours = np.arange(len(labels))
    smooth = 280 + 5 * np.sin(2 * np.pi * hours / 24)[:, None] + np.zeros((1, 4))
    # A slow local anomaly at each airport, and a neighbourhood signal that reaches the first airport later.
    local = np.cumsum(rng.normal(0, 0.25, (len(labels), 4)), axis=0)
    local -= np.linspace(0, 1, len(labels))[:, None] * local[-1]
    nearby = 3.0 * np.sin(2 * np.pi * hours / 37.0)
    for k in UNITS:
        scale = 100 if k == "msl" else 1
        centre = 100000 if k == "msl" else (274 if k == "td2m" else 280)
        value = centre + scale * (smooth - 280 + local) + scale * rng.normal(0, 0.05, smooth.shape)
        truth[k] = (("valid_time", "station_id"), value.astype("f4"))
        truth[k + "_offset_min"] = xr.full_like(truth[k], -7, dtype=np.int16)
    # The first airport moves towards its neighbourhood over the next hours.
    for k in ("t2m", "td2m"):
        v = truth[k].values.astype(float)
        v[3:, 0] += citizen_effect * nearby[:-3]
        truth[k].values[:] = v.astype("f4")
    truth.t2m.values[200:206, 0] = np.nan
    truth.t2m_offset_min.values[400, 0] = 5  # measured after the label: not a reading at T
    if truth_shift_after is not None:
        later = labels > truth_shift_after
        truth.t2m.values[later] += 3
        truth.td2m.values[later] += 3
    issues, snapshots = [], []
    for t in inits:
        a = layout(stations, [t], LEADS, 31, "gefs-station-corrected")
        a.attrs["model_version"] = "3"
        i0 = int((t - START) / H)
        for k in UNITS:
            scale = 100 if k == "msl" else 1
            centre = 100000 if k == "msl" else (274 if k == "td2m" else 280)
            for j, lead in enumerate(LEADS):
                level = centre + scale * (smooth[i0 + lead] - 280)
                a[k].values[0, j] = (level[:, None] + scale * rng.normal(0, 1, (4, 31))).astype("f4")
        a["td2m"].values[:] = np.minimum(a["td2m"].values, a["t2m"].values)
        issues.append(a)
        snap = {}
        for k in ("t2m", "td2m"):
            site = np.array([0, 0, 0, 0, 0, 1, 1, 2])
            usual = np.array([0.5, -0.3, 0.2, 0.0, 1.0, 0.0, 0.2, 0.1])
            value = truth[k].values[i0, site].astype(float) + usual + rng.normal(0, 0.1, 8)
            value[:5] += nearby[i0]
            weight = np.array([0.4, 0.3, 0.5, 0.2, 0.3, 0.4, 0.4, 0.4])
            if i0 % 9 == 0:
                weight[1] = 0.0
            if citizen_shift_after is not None and t > citizen_shift_after:
                value += 4
            snap[k] = (value, weight)
        snapshots.append(snap)
    return stations, truth, issues, snapshots


def run(**kwargs):
    stations, truth, issues, snapshots = world(**kwargs)
    pairs = arms.neighbour_pairs(
        *CITIZEN.T, truth.latitude.values, truth.longitude.values, truth.elevation_m.values
    )
    replay = arms.Replay(2024, truth, pairs, flagged=np.array([False, True] + [False] * 6))
    out = [replay.issue(a, s) for a, s in zip(issues, snapshots, strict=True)]
    return truth, issues, out, replay, pairs


@pytest.fixture(scope="module")
def base():
    return run()


def test_neighbour_pairs_apply_radius_and_height(base):
    *_, pairs = base
    assert sorted(pairs.citizen[pairs.official == 0].tolist()) == [0, 1, 2, 3, 4]
    assert sorted(pairs.citizen[pairs.official == 1].tolist()) == [5, 6]
    assert not (pairs.official == 2).any()  # in range, but 400 m higher
    assert not (pairs.official == 3).any()
    assert pairs.distance_km.max() <= arms.RADIUS_KM


def test_only_the_six_cells_change_and_everything_else_is_the_v3_bytes(base):
    _, issues, out, *_ = base
    for a, (b, c, _) in zip(issues, out, strict=True):
        for arm, model in ((b, arms.MODEL_OWN), (c, arms.MODEL_CITIZEN)):
            assert arm.attrs["model_id"] == model and arm.attrs["model_version"] == "1"
            for k in UNITS:
                assert arm[k].dtype == np.dtype("float32") and arm[k].attrs == a[k].attrs
                keep = [j for j in range(len(LEADS)) if k not in arms.VARIABLES or j not in CHANGED]
                assert arm[k].values[:, keep].tobytes() == a[k].values[:, keep].tobytes()
            xr.testing.assert_identical(
                arm.coords.to_dataset().drop_attrs(), a.coords.to_dataset().drop_attrs()
            )
        assert a.attrs["model_id"] == "gefs-station-corrected"


def test_cold_start_changes_nothing_and_the_spread_is_never_changed(base):
    _, issues, out, *_ = base
    for a, (b, c, diag) in list(zip(issues, out, strict=True))[:2]:
        xr.testing.assert_identical(b.drop_attrs(), a.drop_attrs())
        xr.testing.assert_identical(c.drop_attrs(), a.drop_attrs())
        assert diag.phi == {"t2m": [0.0] * 3, "td2m": [0.0] * 3} and diag.gamma == diag.phi
    a, (b, c, _) = issues[-1], out[-1]
    for arm in (b, c):
        shift = arm.t2m.values.astype(float) - a.t2m.values.astype(float)
        assert np.ptp(shift, axis=-1).max() < 1e-4  # one shift for all 31 members
        assert np.all(arm.td2m.values <= arm.t2m.values)


def test_gains_are_learned_and_the_citizen_term_acts_only_where_it_is_defined(base):
    truth, issues, out, replay, _ = base
    last = out[-1][2]
    assert all(0 < g <= 1 for g in last.phi["t2m"])
    assert last.gamma["t2m"][1] > 0.05  # the planted three-hour neighbourhood signal
    assert last.citizen_defined == {"t2m": 1, "td2m": 1} and last.weight_flagged["t2m"] > 0
    # Usual difference needs 10 earlier same-hour issues: nothing is defined before day 10.
    assert all(d.citizen_defined["t2m"] == 0 for _, _, d in out[:20])
    assert out[20][2].citizen_defined["t2m"] == 1
    for b, c, _ in out:
        same = c.t2m.values[0, :, 1:] == b.t2m.values[0, :, 1:]
        assert same.all()  # stations two to four have no usable neighbourhood: C is B there
    assert arms.covered_stations(truth.station_id.values, replay.citizen_defined_count, len(issues)) == [
        "00000000001"
    ]
    assert arms.covered_stations(["a", "b"], [5, 4], 10) == ["a"]


def test_planted_signal_lowers_the_error_in_the_order_a_b_c(base):
    truth, issues, out, *_ = base
    errors = {"a": [], "b": [], "c": []}
    for a, (b, c, _) in list(zip(issues, out, strict=True))[40:]:
        t = a.init_time.values[0]
        y = truth.t2m.sel(valid_time=t + 3 * H).values[0]
        for name, arm in (("a", a), ("b", b), ("c", c)):
            errors[name].append(arm.t2m.values[0, 1, 0].astype(float).mean() - y)
    rmse = {k: float(np.sqrt(np.nanmean(np.square(v)))) for k, v in errors.items()}
    assert rmse["c"] < rmse["b"] < rmse["a"]


def test_nothing_after_the_issue_time_is_read():
    cut = START + 30 * D
    _, issues, reference, *_ = run()
    _, _, moved, *_ = run(truth_shift_after=cut, citizen_shift_after=cut)
    for a, (b0, c0, d0), (b1, c1, d1) in zip(issues, reference, moved, strict=True):
        if a.init_time.values[0] > cut:
            break
        xr.testing.assert_identical(b0, b1)
        xr.testing.assert_identical(c0, c1)
        assert d0 == d1
    assert not moved[-1][1].t2m.equals(reference[-1][1].t2m)


def test_year_isolation_and_order_are_enforced(base):
    truth, issues, *_ = base
    early = truth.assign_coords(valid_time=truth.valid_time.values - 2 * D)
    with pytest.raises(ValueError, match="before the start of the year"):
        arms.Replay(2024, early, arms.neighbour_pairs(*CITIZEN.T, [40.0], [-100.0], [500.0]))
    replay = arms.Replay(
        2024,
        truth,
        arms.neighbour_pairs(
            *CITIZEN.T, truth.latitude.values, truth.longitude.values, truth.elevation_m.values
        ),
    )
    empty = {k: (np.full(8, np.nan), np.zeros(8)) for k in arms.VARIABLES}
    replay.issue(issues[1], empty)
    with pytest.raises(ValueError, match="increase"):
        replay.issue(issues[0], empty)
    wrong = issues[2].copy()
    wrong.attrs["model_version"] = "2"
    with pytest.raises(ValueError, match="version 3"):
        replay.issue(wrong, empty)


def test_gain_weighted_median_and_neighbourhood_rules():
    assert arms.gain(np.empty(0), np.empty(0), 500) == 0.0
    assert arms.gain(np.zeros(4), np.ones(4), 500) == 0.0
    x = np.array([1.0, -2.0, 0.5, 3.0])
    assert arms.gain(x, 0.4 * x, 4) == pytest.approx(0.5 * 0.4)  # shrunk by N / (N + S)
    assert arms.gain(x, -0.4 * x, 4) == 0.0 and arms.gain(x, 9 * x, 0) == 1.0  # clipped to [0, 1]
    assert arms.weighted_lower_median(np.array([3.0, 1.0, 2.0]), np.array([1.0, 1.0, 1.0])) == 2.0
    assert arms.weighted_lower_median(np.array([1.0, 2.0]), np.array([1.0, 1.0])) == 1.0  # lower median
    assert arms.weighted_lower_median(np.array([1.0, 2.0, 50.0]), np.array([0.1, 0.1, 5.0])) == 50.0
    assert (
        arms.weighted_lower_median(np.array([-9.0, 0.1, 0.2, 0.3]), np.ones(4)) == 0.1
    )  # one bad station cannot move it far
    assert (arms.RADIUS_KM, arms.DISTANCE_SCALE_KM, arms.MAX_HEIGHT_DIFFERENCE_M) == (50.0, 25.0, 200.0)
    assert (arms.USUAL_DAYS, arms.USUAL_MIN, arms.CLIP_K, arms.MIN_NEIGHBOURS, arms.MIN_CELLS) == (
        30,
        10,
        5.0,
        3,
        2,
    )
    assert arms.LEAD_HOURS == (1, 3, 6) and arms.VARIABLES == ("t2m", "td2m")
    assert arms.GAIN_WINDOW == np.timedelta64(30, "D")


def test_one_cell_or_two_stations_never_form_a_neighbourhood_value():
    stations, truth, issues, snapshots = world()
    one_cell = CITIZEN.copy()
    one_cell[:5, :2] = (40.0, -100.0)
    for positions, weights in ((one_cell, None), (CITIZEN, np.array([0.4, 0.3, 0, 0, 0, 0.4, 0.4, 0.4]))):
        pairs = arms.neighbour_pairs(
            *positions.T, truth.latitude.values, truth.longitude.values, truth.elevation_m.values
        )
        replay = arms.Replay(2024, truth, pairs)
        for a, snap in zip(issues, snapshots, strict=True):
            if weights is not None:
                snap = {k: (v, weights) for k, (v, _) in snap.items()}
            b, c, diag = replay.issue(a, snap)
            assert diag.citizen_defined == {"t2m": 0, "td2m": 0}
            xr.testing.assert_identical(c.drop_attrs(), b.drop_attrs())
