"""The one check every live write goes through, called by the lowest-level writers themselves.

Every writer of the live tree (``metar.publish_file``, ``LiveYear.write``, ``locks.file_lock``,
and everything built on them) takes the live root as a required argument and calls
:func:`check_target` before it writes. It refuses unless:

* the platform is not Windows: directory junctions and locked files make a full-path check
  unreliable there, and the daily live run is on Linux. Reading, tests and the evaluation code are
  unaffected;
* the root is a real live root (:func:`check_live_root`): a folder named ``live`` that is not a
  link and holds no evaluation folder, in a data directory that is not inside the repository or a
  temp directory and is not, or is not inside, an evaluation folder;
* the FULL target path, resolved with ``os.path.realpath`` (every link on the way, the last
  component included), lies inside ``realpath(root)``.
"""

from __future__ import annotations

import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

LIVE_DIR = "live"
REPO = Path(__file__).resolve().parents[3]
# Folder names the evaluation pipeline uses under the data directory. A live root must not be one of
# them, sit inside one, or hold one.
EVALUATION_DIRS = frozenset(
    {
        "baselines",
        "bundle",
        "candidates",
        "citizen",
        "confirm-2023",
        "configs",
        "experiments",
        "gate1",
        "isd-lite",
        "parsed",
        "phase3",
        "promotion",
        "raw",
        "thresholds",
        "truth",
    }
)
TEMP_DIRS = ("/tmp", tempfile.gettempdir())


class LivePathError(RuntimeError):
    """A live write target or root is not allowed, or live writes are not supported here."""


def _platform() -> str:
    return sys.platform


def refuse_windows() -> None:
    if _platform() == "win32":
        raise LivePathError(
            "live writes are not supported on Windows (junctions and locked files make the path "
            "check unreliable); run the live builder on Linux or macOS"
        )


def check_data_dir(data: Path) -> Path:
    """The resolved data directory, refused if it is inside the repository or a temp directory, or
    is or is inside an evaluation folder (or is itself a ``live`` folder)."""
    resolved = Path(data).resolve()
    if resolved.is_relative_to(REPO) or any(resolved.is_relative_to(Path(t).resolve()) for t in TEMP_DIRS):
        raise LivePathError("live data belongs in the external data directory")
    if EVALUATION_DIRS.intersection(resolved.parts) or resolved.name == LIVE_DIR:
        raise LivePathError(f"the data directory {resolved} is or is inside an evaluation folder")
    return resolved


def check_live_root(root: Path) -> Path:
    """``root`` if it is a real live root (see the module docstring)."""
    root = Path(root)
    if root.name != LIVE_DIR:
        raise LivePathError(f"{root} is not a live folder")
    check_data_dir(root.parent)
    if root.is_symlink() or (root.exists() and not root.is_dir()):
        raise LivePathError("the live folder must be a real directory, not a link")
    if root.is_dir() and EVALUATION_DIRS.intersection(p.name for p in root.iterdir()):
        raise LivePathError("the live folder holds an evaluation folder")
    return root


def check_target(root: Path, target: Path) -> Path:
    """Approve ONE live write to ``target``: call it immediately before that write, never for a
    read or a validation (use :func:`check_path`). Refuses as :func:`check_path` does."""
    return check_path(root, target)


def check_path(root: Path, target: Path) -> Path:
    """Return ``target`` if it is an allowed live path (module docstring); refuse otherwise. For
    validation and reads: it approves no write."""
    refuse_windows()
    check_live_root(root)
    real_root = Path(os.path.realpath(root))
    real = Path(os.path.realpath(target))
    if not real.is_relative_to(real_root):
        raise LivePathError(f"{target} resolves to {real}, outside the live folder {real_root}")
    return Path(target)


@contextmanager
def write_scope(root: Path, target: Path):
    """For a write whose inner file names are not known in advance (a Zarr store, the GEFS point
    cache, a staged file linked into place): check ``target`` exactly, then run the write. Its inner
    writes must stay under ``target`` or be its own temporary files next to it; the write-audit test
    holds every writer to that, call by call."""
    check_target(root, target)
    yield Path(target)


def make_dir(path: Path, *, parents: bool = True) -> None:
    """Create ``path`` (its write was just approved). Another writer creating it at the same moment
    is not an error; anything else that is not a real folder is."""
    try:
        Path(path).mkdir(parents=parents)
    except FileExistsError:
        if not Path(path).is_dir() or Path(path).is_symlink():
            raise
