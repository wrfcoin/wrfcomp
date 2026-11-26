"""The live write-path check (wrfcomp.live.paths). Synthetic, no network. Runs on every platform."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from wrfcomp.live import metar, paths
from wrfcomp.live.dates import LiveYear
from wrfcomp.live.locks import file_lock
from wrfcomp.live.paths import LivePathError, check_target

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="live writes are refused on Windows")


@pytest.fixture(autouse=True)
def _no_temp_rule(monkeypatch):
    """pytest's tmp_path is under the temp directory, which live data may never use."""
    monkeypatch.setattr(paths, "TEMP_DIRS", ())


def _tree(tmp_path):
    live = tmp_path / "wrfcomp-data" / "live"
    live.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    return live, outside


@posix_only
def test_a_redirect_that_is_symlink_cannot_see_is_refused_before_a_metar_write(tmp_path, monkeypatch):
    """A Windows junction is not reported by Path.is_symlink(). Simulated here: the metar folder
    redirects outside the live tree and is_symlink() says False everywhere. The full real-path check
    still refuses, and nothing is written outside."""
    live, outside = _tree(tmp_path)
    os.symlink(outside, live / "metar")
    monkeypatch.setattr(Path, "is_symlink", lambda self: False)
    with pytest.raises(LivePathError, match="outside the live folder"):
        metar.fetch(live / "metar", ["KAAA"], 30, get=lambda url: b"[]")
    assert not any(outside.iterdir())


@posix_only
def test_check_target_resolves_the_full_path_including_the_last_component(tmp_path):
    live, outside = _tree(tmp_path)
    (live / "metar").mkdir()
    os.symlink(outside / "x.json", live / "metar" / "x.json")  # the last component redirects
    with pytest.raises(LivePathError, match="outside the live folder"):
        check_target(live, live / "metar" / "x.json")
    assert check_target(live, live / "metar" / "y.json") == live / "metar" / "y.json"
    with pytest.raises(LivePathError, match="outside the live folder"):
        check_target(live, live / ".." / "truth" / "x.json")
    for not_live in (live.parent, tmp_path, Path("/")):
        with pytest.raises(LivePathError):
            check_target(not_live, not_live / "x.json")


def test_live_writes_are_refused_on_windows(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "_platform", lambda: "win32")
    live, _ = _tree(tmp_path)
    with pytest.raises(LivePathError, match="not supported on Windows"):
        check_target(live, live / "metar" / "x.json")
    with pytest.raises(LivePathError, match="not supported on Windows"):
        metar.fetch(live / "metar", ["KAAA"], 30, get=lambda url: b"[]")
    assert not (live / "metar").exists() or not any((live / "metar").iterdir())


def _writers(live, outside):
    """Every low-level live writer, called directly as a library user would."""
    store = LiveYear(2026).io.layout(
        xr.Dataset(
            coords={
                "station_id": ["00000000001"],
                "latitude": ("station_id", [40.0]),
                "longitude": ("station_id", [-100.0]),
                "elevation_m": ("station_id", [500.0]),
            }
        ),
        [np.datetime64("2026-03-01T00")],
        [3, 6, 12, 24, 48],
        31,
        "noaa-gefs",
    )
    store = store.fillna(280.0)

    def lock(path, root):
        with file_lock(path, root=root):
            pass

    return {
        "LiveYear.write": lambda root, where: LiveYear(2026, root=root).write(store, where / "x.zarr"),
        "LiveYear.io.write": lambda root, where: LiveYear(2026, root=root).io.write(store, where / "x.zarr"),
        "LiveYear.gefs.extract": lambda root, where: LiveYear(2026, root=root).gefs.extract(
            None, "2026-03-01T00", where / "cache", None
        ),
        "metar.publish_file": lambda root, where: metar.publish_file(where / "x.json", "x\n", root=root),
        "locks.file_lock": lambda root, where: lock(where / "x.lock", root),
        "metar.fetch": lambda root, where: metar.fetch(where, ["KAAA"], 30, get=lambda url: b"[]"),
    }


@pytest.mark.parametrize(
    "writer",
    [
        "LiveYear.write",
        "LiveYear.io.write",
        "LiveYear.gefs.extract",
        "metar.publish_file",
        "locks.file_lock",
        "metar.fetch",
    ],
)
def test_each_writer_refuses_on_windows_when_called_directly(tmp_path, monkeypatch, writer):
    live, outside = _tree(tmp_path)
    monkeypatch.setattr(paths, "_platform", lambda: "win32")
    with pytest.raises(LivePathError, match="not supported on Windows"):
        _writers(live, outside)[writer](live, live / "metar")
    assert not any(live.iterdir())


@posix_only
@pytest.mark.parametrize(
    "writer",
    [
        "LiveYear.write",
        "LiveYear.io.write",
        "LiveYear.gefs.extract",
        "metar.publish_file",
        "locks.file_lock",
        "metar.fetch",
    ],
)
def test_each_writer_refuses_a_target_outside_a_live_root_when_called_directly(tmp_path, writer):
    live, outside = _tree(tmp_path)
    call = _writers(live, outside)[writer]
    with pytest.raises(LivePathError):
        call(live, outside)  # a real live root, a target outside it
    with pytest.raises(LivePathError):
        call(outside, outside)  # a root that is not a live folder
    assert not any(outside.iterdir())


def test_a_policy_without_a_root_cannot_write_through_any_private_module(tmp_path):
    live, outside = _tree(tmp_path)
    calls = _writers(live, outside)
    policy = LiveYear(2026)
    store = LiveYear(2026, root=live)  # only to get a valid dataset below
    del store
    with pytest.raises(LivePathError, match="no live root"):
        policy.gefs.extract(None, "2026-03-01T00", live / "cache", None)
    with pytest.raises(LivePathError):
        calls["LiveYear.io.write"](outside, outside)
    assert not any(live.iterdir()) and not any(outside.iterdir())


def test_every_live_entry_point_refuses_to_start_on_windows(tmp_path, monkeypatch):
    import importlib.util

    from wrfcomp.live import v4 as live

    monkeypatch.setattr(paths, "_platform", lambda: "win32")
    data = tmp_path / "wrfcomp-data"
    data.mkdir()
    with pytest.raises(live.LiveRefusedError, match="not supported on Windows"):
        live.build_issue(np.datetime64("2026-12-02T00"), data)
    with pytest.raises(live.LiveRefusedError, match="not supported on Windows"):
        live.fetch_cycle(data / "live", np.datetime64("2026-12-02T00"))
    with pytest.raises(LivePathError, match="not supported on Windows"):
        metar.fetch(data / "live" / "metar", ["KAAA"], 30, get=lambda url: b"[]")
    script = Path(__file__).resolve().parents[1] / "scripts" / "live_v4.py"
    spec = importlib.util.spec_from_file_location("live_v4_script", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("WRFCOMP_DATA_DIR", str(data))
    monkeypatch.setattr(sys, "argv", ["live_v4.py", "build", "--init", "2026-12-02T00"])
    with pytest.raises(live.LiveRefusedError, match="not supported on Windows"):
        module.main()
    assert [p.name for p in data.iterdir()] == []


@posix_only
def test_a_folder_created_by_a_concurrent_writer_is_not_an_error(tmp_path, monkeypatch):
    """The race seen in the concurrency test: two writers both see the lock folder missing; the one
    that creates it second must not fail. Simulated deterministically: the folder appears between the
    existence check and the mkdir."""
    live, _ = _tree(tmp_path)
    lock = live / "locks" / "build.lock"
    real_is_dir = Path.is_dir

    def appears_meanwhile(self):
        if self == lock.parent and not real_is_dir(self):
            os.mkdir(self)  # the other writer wins the race right after the check
            return False
        return real_is_dir(self)

    monkeypatch.setattr(Path, "is_dir", appears_meanwhile)
    with file_lock(lock, root=live):
        pass
    monkeypatch.undo()
    assert lock.parent.is_dir() and lock.is_file()
    (live / "locks" / "file").write_text("x")
    with pytest.raises(FileExistsError):
        paths.make_dir(live / "locks" / "file")  # an existing FILE is still an error
