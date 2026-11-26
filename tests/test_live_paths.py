"""The live write-path check (wrfcomp.live.paths). Synthetic, no network. Runs on every platform."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from wrfcomp.live import metar, paths
from wrfcomp.live.paths import LivePathError, check_target

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="live writes are refused on Windows")


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


def test_live_writes_are_refused_on_windows(tmp_path, monkeypatch):
    monkeypatch.setattr(paths.sys, "platform", "win32")
    live, _ = _tree(tmp_path)
    with pytest.raises(LivePathError, match="not supported on Windows"):
        check_target(live, live / "metar" / "x.json")
    with pytest.raises(LivePathError, match="not supported on Windows"):
        metar.fetch(live / "metar", ["KAAA"], 30, get=lambda url: b"[]")
    assert not (live / "metar").exists() or not any((live / "metar").iterdir())
