"""The only place live operation lifts the evaluation code's 2025 date checks, for one live year.

The frozen model and harness code refuses every time on or after the sealed-test start
(2025-01-01). That rule protects the evaluation splits and stays in force everywhere else. Live
forecasting (owner ruling 2026-11-25) runs the same frozen code on current data, so inside
:func:`live_dates` exactly these checks see the end of the live year instead of 2025-01-01:

1. ``gefs_station_v2.year_bounds`` (``END``), used by the v2, v3 and v4 replays.
2. ``baselines.io.layout`` (the sealed-test start via ``load_config``), used by the GEFS extractor.
3. ``harness.schemas.validate_forecast_dataset`` (the same start via ``load_config``), used when a
   forecast store is checked or written.

No model file and nothing under ``configs/`` is edited; the substitute config differs from
``configs/splits.yaml`` only in ``sealed_test.start_utc``. Everything is restored on exit, also on
error. The truth parsers' cutoff is not lifted: live readings never go through them.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from datetime import UTC, datetime

import numpy as np

from wrfcomp.baselines import io as baselines_io
from wrfcomp.harness import config as harness_config
from wrfcomp.harness import schemas
from wrfcomp.models import gefs_station_v2 as v2

FIRST_LIVE_YEAR = 2025
_LOCK = threading.Lock()


class LiveRefusedError(RuntimeError):
    """A live input or request breaks a live rule; nothing was written."""


def _live_config(year: int):
    real = harness_config.load_config()
    sealed = real.splits.sealed_test.model_copy(update={"start_utc": datetime(year + 1, 1, 1, tzinfo=UTC)})
    return real.model_copy(update={"splits": real.splits.model_copy(update={"sealed_test": sealed})})


@contextmanager
def live_dates(year: int):
    """Let the frozen code run on ``year`` (2025 or later), and on nothing after it."""
    if not isinstance(year, int) or year < FIRST_LIVE_YEAR:
        raise LiveRefusedError(f"live dates are for {FIRST_LIVE_YEAR} or later, not {year!r}")
    cfg = _live_config(year)

    def load_config(path=None):
        return cfg

    end = np.datetime64(f"{year + 1:04d}-01-01")
    if not _LOCK.acquire(blocking=False):
        raise LiveRefusedError("the date checks are already lifted")
    try:
        saved = (v2.END, baselines_io.load_config, schemas.load_config)
        if saved[0] != np.datetime64("2025-01-01"):
            raise LiveRefusedError("the frozen END is not 2025-01-01")
        v2.END, baselines_io.load_config, schemas.load_config = end, load_config, load_config
        try:
            yield cfg
        finally:
            v2.END, baselines_io.load_config, schemas.load_config = saved
    finally:
        _LOCK.release()
