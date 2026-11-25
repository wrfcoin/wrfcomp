"""Live v4 builder (wrfcomp.live). Synthetic data only, fixed seeds, no network."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from wrfcomp.baselines import gefs
from wrfcomp.baselines import io as baselines_io
from wrfcomp.baselines.io import UNITS, layout, write
from wrfcomp.harness import schemas
from wrfcomp.harness.hashing import content_sha256
from wrfcomp.live import metar
from wrfcomp.live import v4 as live
from wrfcomp.live.dates import LiveRefusedError, live_dates
from wrfcomp.models import citizen_arms as arms
from wrfcomp.models import gefs_station_v2 as v2
from wrfcomp.models import gefs_station_v3 as v3

H = np.timedelta64(1, "h")
D = np.timedelta64(1, "D")
M = np.timedelta64(1, "m")
RAW_LEADS = [3, 6, 12, 24, 48]
DAYS = 34
SHIFT = np.datetime64("2026-01-01", "ns") - np.datetime64("2024-01-01", "ns")  # 731 days
ICAO = ["KAAA", "KBBB", "KCCC", "KDDD"]


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
    raws = []
    for t in np.arange(start, start + days * D, 12 * H):
        with _dates_for(t):
            raw = layout(site(), [t], RAW_LEADS, 31, "noaa-gefs")
        i0 = int((t - start) / H)
        for k in UNITS:
            scale = 30 if k == "msl" else 0.8
            for j, lead in enumerate(RAW_LEADS):
                centre = sig[k][i0 + lead] + bias[k]
                raw[k].values[0, j] = (centre[:, None] + rng.normal(0, scale, (4, 31))).astype("f4")
        raw["td2m"].values[:] = np.minimum(raw.td2m.values, raw.t2m.values)
        raws.append(raw)
    return raws


class _dates_for:
    """live_dates for a live year, nothing for 2024."""

    def __init__(self, t):
        self.year = pd.Timestamp(t).year
        self.ctx = live_dates(self.year) if self.year >= 2025 else None

    def __enter__(self):
        return self.ctx.__enter__() if self.ctx else None

    def __exit__(self, *exc):
        return self.ctx.__exit__(*exc) if self.ctx else False


def frozen(raws, truth, year):
    """What scripts/build_candidate.py (v3) and scripts/build_v4.py (arm B) do with the same inputs."""
    start, _ = v2.year_bounds(year)
    truth = truth.sel(valid_time=slice(np.datetime64(start, "ns"), None))
    rv3, rv4 = v3.Replay(year), arms.Replay(year, truth, live.NO_PAIRS)
    out = []
    for raw in raws:
        a, _ = rv3.issue(raw, truth)
        b, _, _ = rv4.issue(a, live.no_citizen_snapshot())
        out.append((a, b))
    return out


def _values(ds):
    return {k: ds[k].values.tobytes() for k in UNITS}


# -- the date checks ---------------------------------------------------------------------------


def test_live_dates_lifts_exactly_three_checks_and_restores_them():
    saved = (v2.END, baselines_io.load_config, schemas.load_config)
    with pytest.raises(ValueError, match="sealed"):
        v2.year_bounds(2026)
    with pytest.raises(ValueError, match="sealed"):
        layout(site(), [np.datetime64("2026-12-01T00")], RAW_LEADS, 31, "noaa-gefs")
    with live_dates(2026) as cfg:
        assert v2.year_bounds(2026)[1] == np.datetime64("2027-01-01")
        assert cfg.splits.sealed_test.start == np.datetime64("2027-01-01", "ns")
        ds = layout(site(), [np.datetime64("2026-12-01T00")], RAW_LEADS, 31, "noaa-gefs")
        schemas.validate_forecast_dataset(ds)
        with pytest.raises(ValueError, match="sealed"):
            v2.year_bounds(2027)  # one live year, nothing after it
        with pytest.raises(ValueError, match="sealed"):
            layout(site(), [np.datetime64("2026-12-30T12")], RAW_LEADS, 31, "noaa-gefs")
        with pytest.raises(LiveRefusedError, match="already"):
            with live_dates(2026):
                pass
    assert (v2.END, baselines_io.load_config, schemas.load_config) == saved
    with pytest.raises(ValueError, match="sealed"):
        v2.year_bounds(2026)
    with pytest.raises(LiveRefusedError):
        with live_dates(2024):
            pass
    with pytest.raises(RuntimeError, match="boom"):
        with live_dates(2026):
            raise RuntimeError("boom")
    assert (v2.END, baselines_io.load_config, schemas.load_config) == saved


def test_live_replay_is_byte_identical_to_the_frozen_build_on_synthetic_2024():
    start = np.datetime64("2024-01-01", "ns")
    truth, raws = truth_world(start), raw_world(start)
    expected = frozen(raws, truth, 2024)
    got = list(live.replay(raws, truth, 2024))
    assert len(got) == len(expected) == 2 * DAYS
    for (_, a, b, _), (ea, eb) in zip(got, expected, strict=True):
        assert content_sha256(a) == content_sha256(ea)
        assert content_sha256(b) == content_sha256(eb)
        assert b.attrs == eb.attrs and b.attrs["model_id"] == arms.MODEL_OWN
    # v4 really differs from v3 by the end (the own-reading gain has been learned).
    _, a, b, diag = got[-1]
    assert not np.array_equal(a.t2m.values, b.t2m.values)
    assert any(phi > 0 for phi in diag.phi["t2m"])


def test_the_same_world_in_2026_gives_the_same_numbers_under_live_dates():
    start = np.datetime64("2024-01-01", "ns")
    truth, raws = truth_world(start), raw_world(start)
    expected = frozen(raws, truth, 2024)
    shifted_truth = truth.assign_coords(valid_time=truth.valid_time.values + SHIFT)
    shifted = [r.assign_coords(init_time=r.init_time.values + SHIFT) for r in raws]
    with pytest.raises(ValueError, match="sealed"):
        list(live.replay(shifted, shifted_truth, 2026))
    with live_dates(2026):
        got = list(live.replay(shifted, shifted_truth, 2026))
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


# -- issue times -------------------------------------------------------------------------------


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


def test_live_root_refuses_tmp_and_the_repository(tmp_path):
    with pytest.raises(LiveRefusedError):
        live.live_root(tmp_path)
    with pytest.raises(LiveRefusedError):
        live.live_root(baselines_io.CONFIG.parents[1] / "data")


def test_icao_table_covers_the_registered_stations():
    table = live.icao_map()
    assert len(table) == 500 and all(k.startswith("K") and len(k) == 4 for k in table)
    assert sorted(table.values()) == live.stations().station_id.values.astype(str).tolist()


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
    monkeypatch.setattr(gefs, "points", lambda value, stations: np.full(stations.sizes["station_id"], value))
    monkeypatch.setattr(live, "stations", site)
    root = tmp_path / "live"
    out = live.fetch_cycle(root, t)
    assert all(u.startswith(f"{live.S3}/gefs.20261202/00/atmos/pgrb2ap5/") for u in asked)
    with xr.open_zarr(out["path"], chunks=None) as kept:
        assert np.array_equal(kept.init_time.values, [t]) and kept.sizes["ens_member"] == 31
    assert live.fetch_cycle(root, t)["content_sha256"] == out["content_sha256"]  # kept, not refetched


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


@pytest.fixture
def live_world(tmp_path, monkeypatch):
    """Twenty days of kept 2026 cycles and hourly METARs at :53 for four stations."""
    start = np.datetime64("2026-03-01", "ns")
    days = 20
    labels, sig = signal(start, days=days, seed=3)
    rng = np.random.default_rng(4)
    raws = raw_world(start, seed=3, days=days)
    root = tmp_path / "live"
    monkeypatch.setattr(live, "live_root", lambda data: data / "live")
    monkeypatch.setattr(live, "stations", site)
    monkeypatch.setattr(
        live, "icao_map", lambda: dict(zip(ICAO, site().station_id.values.tolist(), strict=True))
    )
    with live_dates(2026):
        for raw in raws:
            write(raw, live.raw_path(root, raw.init_time.values[0]))
    reports = []
    for i, label in enumerate(labels):
        for s, icao in enumerate(ICAO):
            if rng.random() < 0.05:
                continue
            values = {k: float(sig[k][i, s] + rng.normal(0, 0.3 if k != "msl" else 30)) for k in sig}
            reports.append(_raw_report(icao, label - 7 * M, values))
    last = raws[-1].init_time.values[0]
    _write_fetch(reports, root / "metar", f"{pd.Timestamp(last + 4 * H):%Y%m%dT%H%M%SZ}")
    return tmp_path, root, raws, last


def test_build_issue_writes_the_store_hash_and_record(live_world):
    data, root, raws, last = live_world
    record = live.build_issue(data, last)
    out = live.issue_dir(root, last)
    assert out.name == f"{pd.Timestamp(last):%Y-%m-%dT%HZ}"
    with xr.open_zarr(out / "v4.zarr", chunks=None) as store:
        assert content_sha256(store) == record["content_sha256"]
        assert store.attrs["model_id"] == arms.MODEL_OWN and np.array_equal(store.init_time.values, [last])
    assert (out / "v4.sha256").read_text() == f"{record['content_sha256']}  v4.zarr\n"
    saved = json.loads((out / "build.json").read_text())
    assert saved["live_start"] == "2026-03-01T00" and saved["missing_cycles"] == []
    assert len(saved["gefs"]) == len(raws) and len(saved["metar"]) == 1
    # Same numbers as the replay function on the same inputs.
    fetches = metar.stored(root / "metar")
    truth = metar.hourly_truth(
        metar.reports_frame(fetches, live.icao_map()), site(), np.datetime64("2026-01-01T00", "ns"), last
    )
    with live_dates(2026):
        *_, (_, _, b, diag) = live.replay(raws, truth, 2026)
    assert content_sha256(b) == record["content_sha256"]
    assert any(phi > 0 for phi in diag.phi["t2m"])
    assert live.build_issue(data, last)["content_sha256"] == record["content_sha256"]  # rebuild


def test_build_issue_ignores_readings_after_t(live_world):
    data, root, raws, last = live_world
    before = live.build_issue(data, last)["content_sha256"]
    later = [_raw_report(icao, last + 20 * M, {"t2m": 330.0, "td2m": 300.0, "msl": 95000.0}) for icao in ICAO]
    _write_fetch(later, root / "metar", f"{pd.Timestamp(last + 5 * H):%Y%m%dT%H%M%SZ}")
    assert live.build_issue(data, last)["content_sha256"] == before


def test_build_issue_refuses_without_final_readings_or_the_cycle(live_world):
    data, root, raws, last = live_world
    for path in metar.stored(root / "metar"):
        path.rename(path.with_name(f"{pd.Timestamp(last + 10 * M):%Y%m%dT%H%M%SZ}.json"))
    with pytest.raises(LiveRefusedError, match="readings not final"):
        live.build_issue(data, last)
    with pytest.raises(LiveRefusedError, match="not kept"):
        live.build_issue(data, last + 12 * H)


def test_build_issue_records_missing_cycles(live_world):
    import shutil

    data, root, raws, last = live_world
    gone = raws[10].init_time.values[0]
    shutil.rmtree(live.raw_path(root, gone))
    record = live.build_issue(data, last)
    assert record["missing_cycles"] == [str(np.datetime_as_string(gone, unit="h"))]


def test_every_frozen_step_sees_only_causal_truth(monkeypatch):
    start = np.datetime64("2024-01-01", "ns")
    truth, raws = truth_world(start), raw_world(start, days=6)
    seen = []
    original_v3, original_v4 = v3.Replay.issue, arms.Replay.issue

    def spy_v3(self, raw, truth_in):
        t = raw.init_time.values[0]
        metar.assert_causal(truth_in, t)
        seen.append(("v3", t))
        return original_v3(self, raw, truth_in)

    def spy_v4(self, a, snapshot):
        t = a.init_time.values[0]
        metar.assert_causal(self.truth, t)
        seen.append(("v4", t))
        return original_v4(self, a, snapshot)

    monkeypatch.setattr(v3.Replay, "issue", spy_v3)
    monkeypatch.setattr(arms.Replay, "issue", spy_v4)
    assert len(list(live.replay(raws, truth, 2024))) == 12
    assert len(seen) == 24
    # The world has readings measured after their label (offsets +4 and +12), so this is not vacuous.
    assert (truth.t2m_offset_min.values > 0).any()
    with pytest.raises(ValueError, match="after the issue time"):
        list(live.replay(raws, truth, 2024, causal=False))
