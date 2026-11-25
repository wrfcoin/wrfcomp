"""Live v4 builder (wrfcomp.live). Synthetic data only, fixed seeds, no network."""

from __future__ import annotations

import json
import os
import shutil
import threading
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from wrfcomp.baselines import gefs
from wrfcomp.baselines import io as baselines_io
from wrfcomp.baselines.io import UNITS, layout
from wrfcomp.harness import schemas
from wrfcomp.harness.hashing import content_sha256
from wrfcomp.live import metar
from wrfcomp.live import v4 as live
from wrfcomp.live.dates import LiveRefusedError, LiveYear
from wrfcomp.models import citizen_arms as arms
from wrfcomp.models import gefs_station_v2 as v2
from wrfcomp.models import gefs_station_v3 as v3

H = np.timedelta64(1, "h")
D = np.timedelta64(1, "D")
M = np.timedelta64(1, "m")
S = np.timedelta64(1, "s")
RAW_LEADS = [3, 6, 12, 24, 48]
DAYS = 34
SHIFT = np.datetime64("2026-01-01", "ns") - np.datetime64("2024-01-01", "ns")  # 731 days
ICAO = ["KAAA", "KBBB", "KCCC", "KDDD"]
POLICY_2026 = LiveYear(2026)


def site() -> xr.Dataset:
    return xr.Dataset(
        coords={
            "station_id": ["00000000001", "00000000002", "00000000003", "00000000004"],
            "latitude": ("station_id", [40.0, 42.0, 44.0, 46.0]),
            "longitude": ("station_id", [-100.0, -101.0, -102.0, -103.0]),
            "elevation_m": ("station_id", [500.0, 300.0, 800.0, 400.0]),
        }
    )


def signal(start, days=DAYS, seed=11):
    """Hourly 'true weather' per variable (labels, 4 stations), smooth enough to be learnable."""
    rng = np.random.default_rng(seed)
    labels = np.arange(start, start + (days + 3) * D, H)
    hours = np.arange(len(labels))[:, None]
    walk = np.cumsum(rng.normal(0, 0.2, (len(labels), 4)), axis=0)
    t2m = 280 + 5 * np.sin(2 * np.pi * hours / 24) + walk
    out = {
        "t2m": t2m,
        "td2m": t2m - 4 - np.abs(rng.normal(0, 1, t2m.shape)),
        "u10": 3 + 2 * np.sin(2 * np.pi * hours / 31) + 0 * walk,
        "v10": -1 + 2 * np.cos(2 * np.pi * hours / 29) + 0 * walk,
        "msl": 101300 + 300 * np.sin(2 * np.pi * hours / 97) + 50 * walk,
    }
    return labels, out


def truth_world(start, seed=11):
    labels, sig = signal(start, seed=seed)
    rng = np.random.default_rng(seed + 1)
    truth = xr.Dataset(coords={"valid_time": labels, **dict(site().coords)})
    for k in UNITS:
        value = (sig[k] + rng.normal(0, 0.3 if k != "msl" else 30, sig[k].shape)).astype("f4")
        offset = rng.choice(np.array([-7, -7, -7, -3, 0, 4, 12, -14], dtype=np.int16), size=value.shape)
        gone = rng.random(value.shape) < 0.05
        value[gone] = np.nan
        offset[gone] = metar.OFFSET_FILL_VALUE
        truth[k] = (("valid_time", "station_id"), value)
        truth[k + "_offset_min"] = (("valid_time", "station_id"), offset)
    return truth


def raw_world(start, seed=11, days=DAYS):
    """31-member raw GEFS issues at 00Z and 12Z with a station bias the model can learn."""
    labels, sig = signal(start, seed=seed)
    rng = np.random.default_rng(seed + 2)
    bias = {"t2m": 1.5, "td2m": -1.0, "u10": 0.5, "v10": -0.5, "msl": 80.0}
    year = pd.Timestamp(start).year
    make = layout if year < 2025 else LiveYear(year).io.layout
    raws = []
    for t in np.arange(start, start + days * D, 12 * H):
        raw = make(site(), [t], RAW_LEADS, 31, "noaa-gefs")
        i0 = int((t - start) / H)
        for k in UNITS:
            scale = 30 if k == "msl" else 0.8
            for j, lead in enumerate(RAW_LEADS):
                centre = sig[k][i0 + lead] + bias[k]
                raw[k].values[0, j] = (centre[:, None] + rng.normal(0, scale, (4, 31))).astype("f4")
        raw["td2m"].values[:] = np.minimum(raw.td2m.values, raw.t2m.values)
        raws.append(raw)
    return raws


def frozen(raws, truth, year):
    """What scripts/build_candidate.py (v3) and scripts/build_v4.py (arm B) do with the same inputs."""
    start, _ = v2.year_bounds(year)
    truth = truth.sel(valid_time=slice(np.datetime64(start, "ns"), None))
    models = live.frozen_models()
    rv3, rv4 = v3.Replay(year), arms.Replay(year, truth, live.no_pairs(models))
    out = []
    for raw in raws:
        a, _ = rv3.issue(raw, truth)
        b, _, _ = rv4.issue(a, live.no_citizen_snapshot(models))
        out.append((a, b))
    return out


def _values(ds):
    return {k: ds[k].values.tobytes() for k in UNITS}


def _masked(truth):
    return lambda t: metar.causal(truth, t)


def _shared_state():
    return (
        v2.END,
        v2.year_bounds,
        v2.validate,
        baselines_io.load_config,
        baselines_io.layout,
        baselines_io.validate_forecast_dataset,
        schemas.load_config,
        gefs.layout,
        v3.v2,
        arms.v2,
    )


# -- the live date policy ----------------------------------------------------------------------


def _ordinary_guards_refuse():
    """The three ordinary checks, called the ordinary way, on a live-year time."""
    with pytest.raises(ValueError, match="sealed"):
        v2.year_bounds(2026)
    with pytest.raises(ValueError, match="sealed"):
        v2.year_bounds(2025)
    with pytest.raises(ValueError, match="sealed"):
        layout(site(), [np.datetime64("2026-03-01T00")], RAW_LEADS, 31, "noaa-gefs")
    live_store = POLICY_2026.io.layout(site(), [np.datetime64("2026-03-01T00")], RAW_LEADS, 31, "noaa-gefs")
    with pytest.raises(ValueError, match="sealed"):
        schemas.validate_forecast_dataset(live_store)
    with pytest.raises(ValueError, match="sealed"):
        baselines_io.validate(live_store)


def test_live_policy_changes_nothing_shared():
    before = _shared_state()
    policy = LiveYear(2026)
    assert _shared_state() == before
    _ordinary_guards_refuse()
    assert policy.v2.year_bounds(2026)[1] == np.datetime64("2027-01-01")
    assert policy.config.splits.sealed_test.start == np.datetime64("2027-01-01", "ns")
    store = policy.io.layout(site(), [np.datetime64("2026-12-01T00")], RAW_LEADS, 31, "noaa-gefs")
    policy.io.validate(store)
    assert set(policy.sources) == {
        "wrfcomp.baselines.io",
        "wrfcomp.baselines.gefs",
        "wrfcomp.models.gefs_station_v2",
        "wrfcomp.models.gefs_station_v3",
        "wrfcomp.models.citizen_arms",
    }


def test_ordinary_guards_refuse_in_another_thread_while_a_live_replay_runs():
    start = np.datetime64("2024-01-01", "ns")
    truth, raws = truth_world(start), raw_world(start, days=8)
    shifted_truth = truth.assign_coords(valid_time=truth.valid_time.values + SHIFT)
    shifted = [r.assign_coords(init_time=r.init_time.values + SHIFT) for r in raws]
    running, failures, checks = threading.Event(), [], []

    def ordinary():
        running.wait(30)
        while running.is_set():
            try:
                _ordinary_guards_refuse()
                checks.append(1)
            except BaseException as exc:  # noqa: BLE001
                failures.append(exc)
                return

    worker = threading.Thread(target=ordinary)
    worker.start()
    try:
        steps = live.replay(shifted, _masked(shifted_truth), 2026, LiveYear(2026))
        running.set()
        assert len(list(steps)) == 16
    finally:
        running.clear()
        worker.join(30)
    assert not failures and checks


def test_live_policy_refuses_every_other_year():
    policy = LiveYear(2026)
    for year in (2024, 2025, 2027):
        with pytest.raises(ValueError, match="sealed"):
            policy.v2.year_bounds(year)
    with pytest.raises(ValueError, match="sealed"):
        policy.io.layout(site(), [np.datetime64("2027-01-01T00")], RAW_LEADS, 31, "noaa-gefs")
    store_2025 = LiveYear(2025).io.layout(site(), [np.datetime64("2025-06-01T00")], RAW_LEADS, 31, "x")
    with pytest.raises(LiveRefusedError, match="outside the live year"):
        policy.write(store_2025, "unused")
    start = np.datetime64("2025-01-01", "ns")
    raws_2025 = raw_world(start, days=2)
    truth_2025 = truth_world(start)
    with pytest.raises(ValueError, match="sealed|outside|start of the year"):
        list(live.replay(raws_2025, _masked(truth_2025), 2026, policy))
    with pytest.raises(ValueError, match="sealed"):
        list(live.replay(raws_2025, _masked(truth_2025), 2025, policy))
    for bad in (2024, True, "2026"):
        with pytest.raises(LiveRefusedError):
            LiveYear(bad)


def test_live_replay_is_byte_identical_to_the_frozen_build_on_synthetic_2024():
    start = np.datetime64("2024-01-01", "ns")
    truth, raws = truth_world(start), raw_world(start)
    expected = frozen(raws, truth, 2024)
    got = list(live.replay(raws, _masked(truth), 2024, live.frozen_models()))
    assert len(got) == len(expected) == 2 * DAYS
    for (_, a, b, _), (ea, eb) in zip(got, expected, strict=True):
        assert content_sha256(a) == content_sha256(ea)
        assert content_sha256(b) == content_sha256(eb)
        assert b.attrs == eb.attrs and b.attrs["model_id"] == arms.MODEL_OWN
    _, a, b, diag = got[-1]
    assert not np.array_equal(a.t2m.values, b.t2m.values)
    assert any(phi > 0 for phi in diag.phi["t2m"])


def test_the_same_world_in_2026_gives_the_same_numbers_under_the_live_policy():
    start = np.datetime64("2024-01-01", "ns")
    truth, raws = truth_world(start), raw_world(start)
    expected = frozen(raws, truth, 2024)
    shifted_truth = truth.assign_coords(valid_time=truth.valid_time.values + SHIFT)
    shifted = [r.assign_coords(init_time=r.init_time.values + SHIFT) for r in raws]
    with pytest.raises(ValueError, match="sealed"):
        list(live.replay(shifted, _masked(shifted_truth), 2026, live.frozen_models()))
    got = list(live.replay(shifted, _masked(shifted_truth), 2026, LiveYear(2026)))
    for (t, a, b, _), (ea, eb) in zip(got, expected, strict=True):
        assert t == ea.init_time.values[0] + SHIFT
        assert _values(a) == _values(ea) and _values(b) == _values(eb)


def test_causal_truth_refuses_and_masks_anything_after_t():
    start = np.datetime64("2026-01-01", "ns")
    truth = truth_world(start)
    t = start + 5 * D
    seen = metar.causal(truth, t)
    assert seen.valid_time.values.max() == t
    late = seen.t2m_offset_min.values[-1] > 0
    assert np.isnan(seen.t2m.values[-1][late]).all()
    with pytest.raises(ValueError, match="label after"):
        metar.assert_causal(truth, t)
    bad = seen.copy(deep=True)
    bad.t2m.values[-1, 0], bad.t2m_offset_min.values[-1, 0] = 280.0, 5
    with pytest.raises(ValueError, match="measured after"):
        metar.assert_causal(bad, t)


def test_every_frozen_step_sees_only_causal_truth(monkeypatch):
    start = np.datetime64("2024-01-01", "ns")
    truth, raws = truth_world(start), raw_world(start, days=6)
    seen = []
    original_v3, original_v4 = v3.Replay.issue, arms.Replay.issue

    def spy_v3(self, raw, truth_in):
        metar.assert_causal(truth_in, raw.init_time.values[0])
        seen.append("v3")
        return original_v3(self, raw, truth_in)

    def spy_v4(self, a, snapshot):
        metar.assert_causal(self.truth, a.init_time.values[0])
        seen.append("v4")
        return original_v4(self, a, snapshot)

    monkeypatch.setattr(v3.Replay, "issue", spy_v3)
    monkeypatch.setattr(arms.Replay, "issue", spy_v4)
    assert len(list(live.replay(raws, _masked(truth), 2024, live.frozen_models()))) == 12
    assert len(seen) == 24
    assert (truth.t2m_offset_min.values > 0).any()  # not vacuous
    with pytest.raises(ValueError, match="after the issue time"):
        list(live.replay(raws, lambda t: truth.sel(valid_time=slice(None, t)), 2024, live.frozen_models()))


# -- issue times and places --------------------------------------------------------------------


@pytest.mark.parametrize(
    "t, message",
    [
        ("2026-12-02T06", "00Z and 12Z"),
        ("2024-06-01T00", "2025 or later"),
        ("2026-12-30T00", "crosses 1 January"),
        ("2026-12-31T12", "crosses 1 January"),
    ],
)
def test_check_issue_refuses(t, message):
    with pytest.raises(LiveRefusedError, match=message):
        live.check_issue(np.datetime64(t))


def test_check_issue_accepts_the_last_buildable_issue():
    assert live.check_issue(np.datetime64("2026-12-29T12")) == np.datetime64("2026-12-29T12", "ns")


def test_live_root_refuses_temp_repository_and_evaluation_folders(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "TEMP_DIRS", (str(tmp_path),))
    with pytest.raises(LiveRefusedError, match="external"):
        live.live_root(tmp_path / "data")
    with pytest.raises(LiveRefusedError, match="external"):
        live.live_root(baselines_io.CONFIG.parents[1] / "data")
    monkeypatch.setattr(live, "TEMP_DIRS", ())
    data = tmp_path / "wrfcomp-data"
    for evaluation in ("truth", "candidates/gefs-station-corrected-v3", "phase3/2024", "baselines"):
        (data / evaluation).mkdir(parents=True, exist_ok=True)
        with pytest.raises(LiveRefusedError, match="evaluation folder"):
            live.live_root(data / evaluation)
    with pytest.raises(LiveRefusedError, match="evaluation folder"):
        live.live_root(data / "live")
    assert live.live_root(data) == data.resolve() / "live"


def test_live_root_refuses_links_and_evaluation_folders_inside(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "TEMP_DIRS", ())
    data = tmp_path / "wrfcomp-data"
    (data / "truth").mkdir(parents=True)
    os.symlink(data / "truth", data / "live")
    with pytest.raises(LiveRefusedError, match="not a link"):
        live.live_root(data)
    (data / "live").unlink()
    (data / "live" / "truth").mkdir(parents=True)
    with pytest.raises(LiveRefusedError, match="holds an evaluation folder"):
        live.live_root(data)
    (data / "live" / "truth").rmdir()
    root = live.live_root(data)
    os.symlink(data / "truth", root / "2026-03-20T00Z")
    with pytest.raises(LiveRefusedError, match="linked path"):
        live.safe(root, root / "2026-03-20T00Z" / "v4.zarr")


def test_icao_table_covers_the_registered_stations():
    table = live.icao_map()
    assert len(table) == 500 and all(k.startswith("K") and len(k) == 4 for k in table)
    assert sorted(table.values()) == live.stations().station_id.values.astype(str).tolist()


def test_cli_has_no_data_directory_option():
    text = (baselines_io.CONFIG.parents[1] / "scripts" / "live_v4.py").read_text()
    assert "--data-dir" not in text and "live_root()" in text


# -- METAR -------------------------------------------------------------------------------------


def test_decode_prefers_the_t_group_and_converts_like_isd():
    raw = "KAAA 021253Z 27005KT 10SM CLR M02/M05 A3012 RMK AO2 SLP206 T10171050"
    out = metar.decode(raw)
    assert out["t2m"] == pytest.approx(273.15 - 1.7) and out["td2m"] == pytest.approx(273.15 - 5.0)
    assert out["msl"] == pytest.approx(102060.0)
    assert out["u10"] == pytest.approx(2.6) and out["v10"] == pytest.approx(0.0, abs=1e-12)  # 5 kt -> 2.6


@pytest.mark.parametrize(
    "raw, expected",
    [
        (
            "KAAA 021253Z 00000KT 10SM CLR 07/02 A3000 RMK AO2 SLP980",
            {"u10": 0.0, "v10": 0.0, "msl": 99800.0},
        ),
        ("KAAA 021253Z VRB03KT 10SM CLR 07/02 A3000 RMK AO2 SLPNO", {"u10": np.nan, "msl": np.nan}),
        ("KAAA 021253Z 36010G20KT 10SM 07/ A3000", {"v10": -5.1, "td2m": np.nan, "t2m": 280.15}),
        ("KAAA 021253Z 18010MPS 10SM 07/02 A3000", {"u10": np.nan, "v10": np.nan}),
        ("KAAA 021253Z 18010KT 10SM A3000", {"t2m": np.nan, "td2m": np.nan, "v10": 5.1}),
    ],
)
def test_decode_edge_cases(raw, expected):
    out = metar.decode(raw)
    for k, v in expected.items():
        if np.isnan(v):
            assert np.isnan(out[k]), k
        else:
            assert out[k] == pytest.approx(v, abs=1e-9), k


def test_report_time_must_match_the_raw_report():
    t = int(datetime(2026, 12, 2, 12, 53, tzinfo=UTC).timestamp())
    good = {"obsTime": t, "rawOb": "KAAA 021253Z 27005KT 10SM CLR 07/02 A3000"}
    assert metar.report_time(good) == np.datetime64("2026-12-02T12:53", "ns")
    with pytest.raises(ValueError, match="does not match"):
        metar.report_time({"obsTime": t, "rawOb": "KAAA 021153Z 27005KT 10SM CLR 07/02 A3000"})


def test_fetch_batches_and_stores_append_only(tmp_path):
    asked = []

    def get(url):
        asked.append(url)
        return json.dumps([{"icaoId": "KAAA", "obsTime": 0, "rawOb": "x"}]).encode()

    now = datetime(2026, 12, 2, 4, 30, tzinfo=UTC)
    icaos = [f"K{i:03d}" for i in range(250)]
    path, received = metar.fetch(tmp_path / "metar", icaos, 30, now=now, get=get)
    assert len(asked) == 3 and all("hours=30" in u and u.startswith(metar.API) for u in asked)
    assert path.name == "20261202T043000Z.json" and received > 0
    assert len(json.loads(path.read_text())["reports"]) == 3
    assert metar.fetch_time(path) == np.datetime64("2026-12-02T04:30:00", "ns")
    with pytest.raises(FileExistsError):
        metar.fetch(tmp_path / "metar", icaos, 30, now=now, get=get)
    with pytest.raises(ValueError):
        metar.fetch(tmp_path / "metar", icaos, 49, now=now, get=get)


def _raw_report(icao, when, values):
    ts = pd.Timestamp(when)
    c, d = values["t2m"] - 273.15, values["td2m"] - 273.15

    def body(x):
        return f"M{abs(round(x)):02d}" if round(x) < 0 else f"{round(x):02d}"

    def group(x):
        return f"{1 if x < 0 else 0}{abs(round(x * 10)):03d}"

    slp = round(values["msl"] / 10) % 1000
    return {
        "icaoId": icao,
        "obsTime": int(ts.tz_localize("UTC").timestamp()),
        "rawOb": f"{icao} {ts:%d%H%M}Z 27010KT 10SM CLR {body(c)}/{body(d)} A3000 "
        f"RMK AO2 SLP{slp:03d} T{group(c)}{group(d)}",
    }


def test_hourly_truth_follows_the_harness_rule(tmp_path):
    reports = []
    base = np.datetime64("2026-12-02T12:00", "ns")
    for minutes, temp in ((-7, 280.0), (5, 281.0), (52, 282.0), (60 + 8, 283.0), (120 + 30, 284.0)):
        values = {"t2m": temp, "td2m": 270.0, "msl": 101300.0}
        reports.append(_raw_report("KAAA", base + minutes * M, values))
    one = site().isel(station_id=[0])
    frame = metar.reports_frame([_write_fetch(reports, tmp_path)], {"KAAA": "00000000001"})
    truth = metar.hourly_truth(frame, one, base, base + 2 * H)
    # Label 0: +5 beats -7. Label 1: -8 and +8 tie, the earlier wins. Label 2: nothing within 15 min.
    assert truth.t2m.values[:, 0].tolist() == pytest.approx([281.0, 282.0, np.nan], abs=0.06, nan_ok=True)
    assert truth.t2m_offset_min.values[:, 0].tolist() == [5, -8, -32768]
    assert truth.u10.values[0, 0] == pytest.approx(5.1) and truth.v10.values[0, 0] == pytest.approx(
        0, abs=1e-6
    )


def _write_fetch(reports, directory, stamp="20261202T160000Z"):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{stamp}.json"
    path.write_text(json.dumps({"reports": reports, "fetched_at": stamp}))
    return path


def test_decode_discards_a_dewpoint_above_its_own_temperature():
    out = metar.decode("KAAA 021253Z 27005KT 10SM CLR 07/09 A3000 RMK AO2 SLP206 T00720091")
    assert out["t2m"] == pytest.approx(280.35) and np.isnan(out["td2m"])


def test_a_bad_dewpoint_never_hides_a_good_earlier_one(tmp_path):
    """As parse_isd_csv: the discard happens per report, BEFORE the nearest-report selection."""
    label = np.datetime64("2026-12-02T12:00", "ns")
    reports = [
        _raw_report("KAAA", label - 10 * M, {"t2m": 280.0, "td2m": 275.0, "msl": 101300.0}),
        _raw_report("KAAA", label - 2 * M, {"t2m": 279.0, "td2m": 285.0, "msl": 101300.0}),
    ]
    frame = metar.reports_frame([_write_fetch(reports, tmp_path)], {"KAAA": "00000000001"})
    truth = metar.hourly_truth(frame, site().isel(station_id=[0]), label, label)
    assert (
        truth.t2m.values[0, 0] == pytest.approx(279.0, abs=0.06) and truth.t2m_offset_min.values[0, 0] == -2
    )
    assert (
        truth.td2m.values[0, 0] == pytest.approx(275.0, abs=0.06)
        and truth.td2m_offset_min.values[0, 0] == -10
    )


def test_reports_after_t_are_dropped_before_the_slot_is_chosen(tmp_path):
    """A T-5 min reading is the T input; one at T+1 min or T+15 s can neither replace nor hide it."""
    t = np.datetime64("2026-12-02T12:00", "ns")
    base = [_raw_report("KAAA", t - 5 * M, {"t2m": 280.0, "td2m": 270.0, "msl": 101300.0})]
    later = [
        _raw_report("KAAA", t + 1 * M, {"t2m": 300.0, "td2m": 270.0, "msl": 101300.0}),
        _raw_report("KAAA", t + 15 * S, {"t2m": 310.0, "td2m": 270.0, "msl": 101300.0}),
    ]
    icao = {"KAAA": "00000000001"}
    one = site().isel(station_id=[0])
    for extra in ([], later[:1], later[1:], later):
        frame = metar.reports_frame(
            [_write_fetch(base + extra, tmp_path / str(len(extra)) / repr(extra)[:8])], icao
        )
        truth = metar.truth_as_of(frame, one, t - H, t)
        assert truth.t2m.values[-1, 0] == pytest.approx(280.0, abs=0.06)
        assert truth.t2m_offset_min.values[-1, 0] == -5
    # The old way (choose among all, then mask) lost the T-5 min reading: the T+15 s report won the
    # slot at a rounded offset of 0 and survived the mask. That is what this test forbids.
    frame = metar.reports_frame([_write_fetch(base + later, tmp_path / "all")], icao)
    old = metar.causal(metar.hourly_truth(frame, one, t - H, t), t).t2m.values[-1, 0]
    assert not old == pytest.approx(280.0, abs=0.06)


# -- GEFS fetch --------------------------------------------------------------------------------


def _index(t, member, lead, *, wrong=None):
    ens = "ENS=low-res ctl" if member == "c00" else f"ENS=+{int(member[1:])}"
    step = "anl" if lead == 0 else f"{lead} hour fcst"
    fields = [gefs.HEIGHT, *gefs.FIELDS.values()]
    lines = []
    for i, (v, level) in enumerate(fields):
        d, s = f"d={pd.Timestamp(t):%Y%m%d%H}", step
        if wrong == "time" and i == 1:
            d = "d=2026120112"
        if wrong == "step" and i == 1:
            s = "3 hour fcst" if lead != 3 else "6 hour fcst"
        lines.append(f"{i + 1}:{i}:{d}:{v}:{level}:{s}:{ens}")
    return "\n".join(lines).encode()


def test_fetch_cycle_checks_every_index_and_keeps_the_raw_store(tmp_path, monkeypatch):
    t = np.datetime64("2026-12-02T00", "ns")
    asked = []

    def fake_get(self, url, interval=None):
        asked.append(url)
        name = url.rsplit("/", 1)[1]
        member, lead = name[2:5], int(name.split(".f")[1][:3])
        if url.endswith(".idx"):
            return _index(t, member, lead)
        return [100.0, 280.0, 90.0, -2.0, 1.0, 99000.0][interval[0]]

    monkeypatch.setattr(gefs.Transfer, "get", fake_get)
    policy = LiveYear(2026)
    monkeypatch.setattr(
        policy.gefs, "points", lambda value, stations: np.full(stations.sizes["station_id"], value)
    )
    monkeypatch.setattr(live, "stations", site)
    monkeypatch.setattr(live, "TEMP_DIRS", ())
    root = live.live_root(tmp_path / "wrfcomp-data")
    out = live.fetch_cycle(root, t, policy=policy)
    assert all(u.startswith(f"{live.S3}/gefs.20261202/00/atmos/pgrb2ap5/") for u in asked)
    with xr.open_zarr(out["path"], chunks=None) as kept:
        assert np.array_equal(kept.init_time.values, [t]) and kept.sizes["ens_member"] == 31
    assert (
        live.fetch_cycle(root, t, policy=policy)["content_sha256"] == out["content_sha256"]
    )  # kept, not refetched


@pytest.mark.parametrize("wrong", ["time", "step"])
def test_checked_transfer_refuses_another_time(monkeypatch, wrong):
    t = np.datetime64("2026-12-02T00", "ns")
    monkeypatch.setattr(
        gefs.Transfer, "get", lambda self, url, interval=None: _index(t, "p03", 24, wrong=wrong)
    )
    transfer = live.CheckedTransfer(t)
    url = f"{transfer.prefix}gep03.t00z.pgrb2a.0p50.f024.idx"
    with pytest.raises(LiveRefusedError, match="not cycle T"):
        transfer.get(url)


def test_checked_transfer_refuses_files_of_another_cycle(monkeypatch):
    monkeypatch.setattr(gefs.Transfer, "get", lambda self, url, interval=None: b"")
    transfer = live.CheckedTransfer(np.datetime64("2026-12-02T00"))
    for url in (
        f"{live.S3}/gefs.20261202/12/atmos/pgrb2ap5/gec00.t12z.pgrb2a.0p50.f024.idx",
        f"{live.S3}/gefs.20261203/00/atmos/pgrb2ap5/gec00.t00z.pgrb2a.0p50.f024.idx",
        "https://example.org/gec00.t00z.pgrb2a.0p50.f024",
    ):
        with pytest.raises(LiveRefusedError, match="not a file of cycle"):
            transfer.get(url)


# -- one live issue, end to end ----------------------------------------------------------------


def _world_reports(labels, sig, seed=4):
    rng = np.random.default_rng(seed)
    reports = []
    for i, label in enumerate(labels):
        for s, icao in enumerate(ICAO):
            if rng.random() < 0.05:
                continue
            values = {k: float(sig[k][i, s] + rng.normal(0, 0.3 if k != "msl" else 30)) for k in sig}
            reports.append(_raw_report(icao, label - 7 * M, values))
    return reports


def _stamp(t):
    return f"{pd.Timestamp(t):%Y%m%dT%H%M%SZ}"


def make_live_world(data, monkeypatch, *, extra_reports=()):
    """Twenty days of kept 2026 cycles and one METAR fetch (hourly reports at :53), 4 stations."""
    start = np.datetime64("2026-03-01", "ns")
    labels, sig = signal(start, days=20, seed=3)
    raws = raw_world(start, seed=3, days=20)
    monkeypatch.setattr(live, "TEMP_DIRS", ())
    monkeypatch.setattr(live, "stations", site)
    monkeypatch.setattr(
        live, "icao_map", lambda: dict(zip(ICAO, site().station_id.values.tolist(), strict=True))
    )
    root = live.live_root(data)
    for raw in raws:
        POLICY_2026.write(raw, live.raw_path(root, raw.init_time.values[0]))
    last = raws[-1].init_time.values[0]
    _write_fetch(_world_reports(labels, sig) + list(extra_reports), root / "metar", _stamp(last + 4 * H))
    return root, raws, last


@pytest.fixture
def live_world(tmp_path, monkeypatch):
    data = tmp_path / "wrfcomp-data"
    root, raws, last = make_live_world(data, monkeypatch)
    return data, root, raws, last


def test_build_issue_writes_every_issue_once_with_pinned_inputs(live_world):
    data, root, raws, last = live_world
    record = live.build_issue(last, data)
    out = live.issue_dir(root, last)
    assert out.name == f"{pd.Timestamp(last):%Y-%m-%dT%HZ}"
    with xr.open_zarr(out / "v4.zarr", chunks=None) as store:
        assert content_sha256(store) == record["content_sha256"]
        assert store.attrs["model_id"] == arms.MODEL_OWN and np.array_equal(store.init_time.values, [last])
    assert (out / "v4.sha256").read_text() == f"{record['content_sha256']}  v4.zarr\n"
    assert record["live_start"] == "2026-03-01T00" and record["missing_cycles"] == []
    assert set(record["frozen_sources_sha256"]) == set(POLICY_2026.sources)
    pinned = json.loads((out / "inputs.json").read_text())
    assert len(pinned["metar"]) == 1 and pinned["gefs_sha256"] == content_sha256(raws[-1])
    assert all((live.issue_dir(root, r.init_time.values[0]) / "v4.zarr").is_dir() for r in raws)
    # Same numbers as the replay function on the same pinned inputs.
    frame = metar.reports_frame(metar.stored(root / "metar"), live.icao_map())
    start = np.datetime64("2026-01-01T00", "ns")
    steps = live.replay(raws, lambda t: metar.truth_as_of(frame, site(), start, t), 2026, POLICY_2026)
    *_, (_, _, b, diag) = steps
    assert content_sha256(b) == record["content_sha256"]
    assert any(phi > 0 for phi in diag.phi["t2m"])
    files = {p: p.read_bytes() for p in out.iterdir() if p.is_file()}
    assert live.build_issue(last, data) == record  # rebuild: identical, nothing rewritten
    assert {p: p.read_bytes() for p in out.iterdir() if p.is_file()} == files


def test_future_reports_at_the_raw_report_entry_point_change_nothing(tmp_path, monkeypatch):
    """Readings measured after an issue (T+1 min, T+15 s) never change that issue or any before it."""
    plain = make_live_world(tmp_path / "a" / "wrfcomp-data", monkeypatch)
    root_a, raws, last = plain
    pivot = last - 12 * H
    later = []
    for icao in ICAO:
        later += [
            _raw_report(icao, pivot + 1 * M, {"t2m": 330.0, "td2m": 300.0, "msl": 95000.0}),
            _raw_report(icao, pivot + 15 * S, {"t2m": 320.0, "td2m": 300.0, "msl": 96000.0}),
        ]
    root_b, _, _ = make_live_world(tmp_path / "b" / "wrfcomp-data", monkeypatch, extra_reports=later)
    live.build_issue(last, tmp_path / "a" / "wrfcomp-data")
    live.build_issue(last, tmp_path / "b" / "wrfcomp-data")
    changed = []
    for r in raws:
        t = r.init_time.values[0]
        same = (live.issue_dir(root_a, t) / "v4.sha256").read_text() == (
            live.issue_dir(root_b, t) / "v4.sha256"
        ).read_text()
        if t <= pivot:
            assert same, t
        else:
            changed.append(not same)
    assert any(changed)  # the extra reports are real inputs once they are in the past


def test_a_later_fetch_never_rewrites_an_earlier_issue(live_world, monkeypatch):
    """A correction fetched later (for a reading before the earlier issue) reaches only later issues."""
    data, root, raws, last = live_world
    earlier = last - 12 * H
    first = live.build_issue(earlier, data)
    snapshot = {p: p.read_bytes() for p in live.issue_dir(root, earlier).iterdir() if p.is_file()}
    corrected = [
        _raw_report(icao, earlier - 7 * M, {"t2m": 290.0, "td2m": 280.0, "msl": 100000.0}) for icao in ICAO
    ]
    _write_fetch(corrected, root / "metar", _stamp(last + 6 * H))
    record = live.build_issue(last, data)
    assert {p: p.read_bytes() for p in live.issue_dir(root, earlier).iterdir() if p.is_file()} == snapshot
    assert live.build_issue(earlier, data) == first
    assert len(json.loads((live.issue_dir(root, last) / "inputs.json").read_text())["metar"]) == 2
    # The later issue does use the correction (it verifies an earlier same-hour forecast at 12 h).
    clean = data.parent / "clean" / "wrfcomp-data"
    make_live_world(clean, monkeypatch)
    assert live.build_issue(last, clean)["content_sha256"] != record["content_sha256"]


def test_a_changed_pinned_fetch_or_cycle_is_refused(live_world):
    data, root, raws, last = live_world
    live.build_issue(last, data)
    fetch = metar.stored(root / "metar")[0]
    fetch.write_text(fetch.read_text().replace("27010KT", "27011KT", 1))
    with pytest.raises(LiveRefusedError, match="missing or changed"):
        live.build_issue(last, data)


def test_build_issue_refuses_without_final_readings_or_the_cycle(live_world):
    data, root, raws, last = live_world
    for path in metar.stored(root / "metar"):
        path.rename(path.with_name(f"{_stamp(last + 10 * M)}.json"))
    with pytest.raises(LiveRefusedError, match="readings not final"):
        live.build_issue(last, data)
    with pytest.raises(LiveRefusedError, match="not kept"):
        live.build_issue(last + 12 * H, data)


def test_build_issue_refuses_a_linked_issue_folder(live_world, tmp_path):
    data, root, raws, last = live_world
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    os.symlink(elsewhere, live.issue_dir(root, last))
    with pytest.raises(LiveRefusedError, match="linked path"):
        live.build_issue(last, data)
    assert not any(elsewhere.iterdir())


def test_build_issue_records_missing_cycles(live_world):
    data, root, raws, last = live_world
    gone = raws[10].init_time.values[0]
    shutil.rmtree(live.raw_path(root, gone))
    record = live.build_issue(last, data)
    assert record["missing_cycles"] == [str(np.datetime_as_string(gone, unit="h"))]
