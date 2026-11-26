"""The one check every live write goes through.

Before any file or folder under the live tree is written (fetch files, locks, staging, issue
folders, GEFS cycles and cache), its FULL target path is resolved with ``os.path.realpath`` (every
symlink on the way, the last component included) and must lie inside ``realpath(live root)``.

Live writes are refused on Windows: directory junctions and locked files make a full-path check
unreliable there, and the daily live run is on Linux. Reading, testing and the evaluation code are
unaffected.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


class LivePathError(RuntimeError):
    """A live write target is outside the live tree, or live writes are not supported here."""


def check_target(live_root: Path, target: Path) -> Path:
    """Return ``target`` if its full real path is inside the real live root; refuse otherwise."""
    if sys.platform == "win32":
        raise LivePathError(
            "live writes are not supported on Windows (junctions and locked files make the path "
            "check unreliable); run the live builder on Linux or macOS"
        )
    root = Path(os.path.realpath(live_root))
    real = Path(os.path.realpath(target))
    if not real.is_relative_to(root):
        raise LivePathError(f"{target} resolves to {real}, outside the live folder {root}")
    return Path(target)
