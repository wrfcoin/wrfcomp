"""Candidate v4 (Phase 3 arm B alone): same stores as the Phase 3 builder's B, no citizen file opened."""

from __future__ import annotations

import builtins
import os
import shutil

import numpy as np
import pytest
import xarray as xr
from test_model_citizen_arms import CITIZEN, _extract, _load, world

from wrfcomp.assimilation import phase3_qc as p3
from wrfcomp.baselines.io import UNITS, write
from wrfcomp.models import citizen_arms as arms


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    """A synthetic year (two months) of v3 stores, truth and a complete citizen extract."""
    monkeypatch.setattr(p3, "current_run_record", lambda data, year: {"digest": "synthetic-run"})
    root = tmp_path / "data"
    _, truth, issues, snapshots = world()
    (root / "truth").mkdir(parents=True)
    truth.to_netcdf(root / "truth" / "isd_conus_2024.nc")
    months = {}
    for a in issues:
        months.setdefault(str(a.init_time.values[0])[:7], []).append(a)
    for month, parts in months.items():
        store = xr.concat(parts, dim="init_time", coords="minimal", compat="identical", join="exact")
        write(store, root / "candidates" / "gefs-station-corrected-v3" / f"{month}.zarr")
    rows = {k: [] for k in ("station", "issue", "var", "value", "weight", "earned", "age_ns")}
    for i, snap in enumerate(snapshots):
        for v, var in enumerate(arms.VARIABLES):
            value, weight = snap[var]
            for c in np.flatnonzero(weight > 0):
                for k, x in zip(rows, (c, i, v, value[c], weight[c], False, 60 * 10**9), strict=True):
                    rows[k].append(x)
    cat = {
        "token": [f"CW{i:09d}" for i in range(len(CITIZEN))],
        "lat": CITIZEN[:, 0],
        "lon": CITIZEN[:, 1],
        "elevation_m": CITIZEN[:, 2],
        "stats": np.zeros((len(CITIZEN), 6), dtype=np.int64),
        **rows,
    }
    times = np.array([a.init_time.values[0] for a in issues])
    ids = truth.station_id.values.astype(str).tolist()
    return root, cat, times, ids


def _stores(entries):
    return [xr.open_zarr(m["path"], chunks=None).load() for m in entries]


def test_v4_equals_the_phase3_builders_arm_b(synthetic, tmp_path):
    root, cat, times, ids = synthetic
    _extract(root, cat, times, ids)
    full = _load("build_phase3_arms").build(root, 2024, issues=times, stations=ids)
    v4 = _load("build_v4").build(root, 2024, tmp_path / "v4", issues=times, stations=ids)
    assert [m["period"] for m in v4["v4"]] == ["2024-01", "2024-02"] and v4["n_issues"] == len(times)
    assert [m["content_sha256"] for m in v4["v4"]] == [m["content_sha256"] for m in full["arm_b"]]
    assert [m["content_sha256"] for m in v4["v3"]] == [m["content_sha256"] for m in full["arm_a"]]
    for mine, theirs in zip(_stores(v4["v4"]), _stores(full["arm_b"]), strict=True):
        xr.testing.assert_identical(mine, theirs)
        assert mine.attrs["model_id"] == arms.MODEL_OWN and mine.attrs["model_version"] == arms.VERSION
        for k in UNITS:
            assert mine[k].values.tobytes() == theirs[k].values.tobytes()
    # The arm actually changed something, so the equality is not two copies of v3.
    first = _stores(v4["v3"])[1]
    assert not first.t2m.equals(_stores(v4["v4"])[1].t2m)
    assert [d["phi"] for d in v4["issues"]] == [d["phi"] for d in full["issues"]]
    assert (tmp_path / "v4" / "build.json").is_file()


def test_v4_opens_no_citizen_path(synthetic, tmp_path, monkeypatch):
    root, cat, times, ids = synthetic
    _extract(root, cat, times, ids)  # a real extract is present and must not be touched
    expected = _load("build_phase3_arms").build(root, 2024, issues=times, stations=ids)["arm_b"]
    script = _load("build_v4")
    opened = []

    def recording(name, original):
        def wrapper(*args, **kwargs):
            opened.append(str(args[0]) if args else "")
            return original(*args, **kwargs)

        monkeypatch.setattr(*name, wrapper)

    recording((builtins, "open"), builtins.open)
    recording((os, "open"), os.open)
    recording((os, "scandir"), os.scandir)
    recording((os, "listdir"), os.listdir)
    recording((xr, "open_dataset"), xr.open_dataset)
    recording((xr, "open_zarr"), xr.open_zarr)
    citizen = root / "citizen"
    os.chmod(citizen / "conus", 0)  # any read below it fails outright
    os.chmod(citizen, 0)
    try:
        out = script.build(root, 2024, tmp_path / "v4", issues=times, stations=ids)
    finally:
        os.chmod(citizen, 0o755)
        os.chmod(citizen / "conus", 0o755)
    assert opened and not [p for p in opened if p.startswith(str(citizen))]
    assert [m["content_sha256"] for m in out["v4"]] == [m["content_sha256"] for m in expected]
    # And with no citizen data on disk at all.
    shutil.rmtree(citizen)
    again = script.build(root, 2024, tmp_path / "v4-again", issues=times, stations=ids)
    assert [m["content_sha256"] for m in again["v4"]] == [m["content_sha256"] for m in expected]


def test_v4_refuses_inputs_off_the_schedule_or_stations(synthetic, tmp_path):
    root, _, times, ids = synthetic
    script = _load("build_v4")
    out = tmp_path / "v4"
    with pytest.raises(script.BuildRefusedError, match="exactly the year's issue schedule"):
        script.build(root, 2024, out, issues=times[:-1], stations=ids)
    with pytest.raises(script.BuildRefusedError, match="exactly the year's issue schedule"):
        script.build(root, 2024, out, stations=ids)  # the registered 704 issues
    with pytest.raises(script.BuildRefusedError, match="v3 store 2024-01 is not on the registered"):
        script.build(root, 2024, out, issues=times)  # the registered 500 stations
    february = root / "candidates" / "gefs-station-corrected-v3" / "2024-02.zarr"
    february.rename(tmp_path / "aside.zarr")
    with pytest.raises(script.BuildRefusedError, match="exactly the year's issue schedule"):
        script.build(root, 2024, out, issues=times, stations=ids)
    (tmp_path / "aside.zarr").rename(february)
    truth_path = root / "truth" / "isd_conus_2024.nc"
    with xr.open_dataset(truth_path) as ds:
        fewer = ds.load().isel(station_id=slice(0, 3))
    truth_path.rename(tmp_path / "truth.nc")
    with pytest.raises(script.BuildRefusedError, match="no truth file"):
        script.build(root, 2024, out, issues=times, stations=ids)
    fewer.to_netcdf(truth_path)
    with pytest.raises(script.BuildRefusedError, match="truth file is not on the registered stations"):
        script.build(root, 2024, out, issues=times, stations=ids)
    assert not out.exists()  # every refusal came before the first write


def test_v4_schedules():
    script = _load("build_v4")
    np.testing.assert_array_equal(script.issue_times(2024), p3.issue_times(2024))
    t2023 = script.issue_times(2023)
    assert str(t2023[0])[:13] == "2023-01-01T00" and str(t2023[-1])[:13] == "2023-12-15T12"
    assert len(t2023) == 2 * 349
    with pytest.raises(KeyError):
        script.issue_times(2025)
