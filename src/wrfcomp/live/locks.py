"""Writer locks for the live folder, the same on Linux, macOS and Windows.

A lock is two layers: a ``threading.Lock`` per lock file (threads of one process; on Windows a
second handle of the same process is not reliably excluded by the OS lock), and an OS lock on the
lock file (other processes): ``fcntl.flock`` on POSIX, ``msvcrt.locking`` on Windows. Both are
released when the holder leaves the block or the process ends.
"""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None
    import msvcrt

_GUARD = threading.Lock()
_THREAD_LOCKS: dict[str, threading.Lock] = {}


class LockBusyError(RuntimeError):
    """Another writer holds the lock (non-blocking request)."""


def _os_lock(fd: int, blocking: bool) -> None:
    if fcntl is not None:
        fcntl.flock(fd, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        return
    os.lseek(fd, 0, os.SEEK_SET)  # pragma: no cover - Windows
    while True:  # pragma: no cover - Windows: LK_LOCK gives up after 10 s, so retry when blocking
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            return
        except OSError:
            if not blocking:
                raise
            threading.Event().wait(0.05)


def _os_unlock(fd: int) -> None:
    if fcntl is not None:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return
    os.lseek(fd, 0, os.SEEK_SET)  # pragma: no cover - Windows
    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)  # pragma: no cover - Windows


@contextmanager
def file_lock(path: Path, *, blocking: bool = False):
    """Hold the lock ``path`` (created if needed). Non-blocking by default: raises
    :class:`LockBusyError` if another thread or process holds it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    key = os.path.normcase(str(path.resolve()))
    with _GUARD:
        local = _THREAD_LOCKS.setdefault(key, threading.Lock())
    if not local.acquire(blocking=blocking):
        raise LockBusyError(path.name)
    try:
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            try:
                _os_lock(fd, blocking)
            except OSError as exc:
                raise LockBusyError(path.name) from exc
            try:
                yield
            finally:
                _os_unlock(fd)
        finally:
            os.close(fd)
    finally:
        local.release()
