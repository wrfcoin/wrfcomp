"""The live-only date policy: the frozen code, run for exactly one live year, without touching it.

The frozen model and harness code refuses every time on or after the sealed-test start
(2025-01-01). That rule protects the evaluation splits. Live forecasting (owner ruling 2026-11-25)
runs the same frozen code on current data, so a :class:`LiveYear` holds its OWN private copies of
the frozen modules on the live path, loaded from the very same source files (their hashes are
recorded), and changes only those copies:

1. ``gefs_station_v2.year_bounds``: accepts the live year and refuses every other year (so a live
   2026 policy refuses 2025, and 2027);
2. ``baselines.io.layout``: reads a config whose sealed-test start is 1 January after the live year
   (the GEFS extractor's copy uses this layout);
3. ``harness.schemas.validate_forecast_dataset``: called with that config explicitly (``cfg=``).

Nothing shared is changed: no attribute of the imported ``wrfcomp`` modules, nothing in
``configs/``. Ordinary calls anywhere else in the process (another thread included) still refuse
every 2025-or-later time while a live build runs. The policy object is passed explicitly down the
live path; there is no global switch.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.util
import itertools
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from wrfcomp.baselines import gefs as shared_gefs
from wrfcomp.baselines import io as shared_io
from wrfcomp.harness import schemas
from wrfcomp.harness.config import load_config
from wrfcomp.models import citizen_arms as shared_arms
from wrfcomp.models import gefs_station_v2 as shared_v2
from wrfcomp.models import gefs_station_v3 as shared_v3

FIRST_LIVE_YEAR = 2025
_COUNTER = itertools.count()
_IMPORT_LOCK = threading.Lock()


class LiveRefusedError(RuntimeError):
    """A live input or request breaks a live rule; nothing was written."""


def _private(module, tag: str):
    """A fresh module object executed from ``module``'s own source file. It is registered only under
    a unique private name, and only while it executes (dataclasses look themselves up there)."""
    name = f"{module.__package__}._live_{tag}_{module.__name__.rsplit('.', 1)[1]}"
    spec = importlib.util.spec_from_file_location(name, module.__file__)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load a private copy of {module.__name__}")
    copy = importlib.util.module_from_spec(spec)
    with _IMPORT_LOCK:
        sys.modules[name] = copy
        try:
            spec.loader.exec_module(copy)
        finally:
            del sys.modules[name]
    return copy


def _sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class LiveYear:
    """The frozen code for one live year: ``.gefs``, ``.v2``, ``.v3``, ``.arms`` and ``.write``."""

    def __init__(self, year: int):
        if not isinstance(year, int) or isinstance(year, bool) or year < FIRST_LIVE_YEAR:
            raise LiveRefusedError(f"live years are {FIRST_LIVE_YEAR} or later, not {year!r}")
        self.year = year
        real = load_config()
        sealed = real.splits.sealed_test.model_copy(
            update={"start_utc": datetime(year + 1, 1, 1, tzinfo=UTC)}
        )
        cfg = real.model_copy(update={"splits": real.splits.model_copy(update={"sealed_test": sealed})})
        self.config = cfg
        tag = f"{year}_{next(_COUNTER)}"

        self.io = _private(shared_io, tag)
        self.io.load_config = lambda path=None: cfg
        self.io.validate_forecast_dataset = functools.partial(schemas.validate_forecast_dataset, cfg=cfg)

        self.gefs = _private(shared_gefs, tag)
        self.gefs.layout = self.io.layout

        self.v2 = _private(shared_v2, tag)
        self.v2.END = np.datetime64(f"{year + 1:04d}-01-01")
        self.v2.validate = self.io.validate
        frozen_bounds = self.v2.year_bounds

        def year_bounds(y):
            if y != year:
                raise ValueError(f"sealed forecast year: this live policy is for {year} only")
            return frozen_bounds(y)

        self.v2.year_bounds = year_bounds

        self.v3 = _private(shared_v3, tag)
        self.v3.v2 = self.v2
        self.arms = _private(shared_arms, tag)
        self.arms.v2 = self.v2

        shared = (shared_io, shared_gefs, shared_v2, shared_v3, shared_arms)
        copies = (self.io, self.gefs, self.v2, self.v3, self.arms)
        self.sources = {m.__name__: _sha256(m.__file__) for m in shared}
        if [_sha256(c.__file__) for c in copies] != list(self.sources.values()):
            raise LiveRefusedError("a private copy is not the frozen source")

    def check_store_year(self, ds) -> None:
        start = np.datetime64(f"{self.year:04d}-01-01", "ns")
        end = np.datetime64(f"{self.year + 1:04d}-01-01", "ns")
        inits = ds.init_time.values
        if inits.min() < start or (inits[:, None] + ds.lead_time.values[None, :]).max() >= end:
            raise LiveRefusedError(f"a store outside the live year {self.year}")

    def write(self, ds, path) -> None:
        """``baselines.io.write`` with the live year's config, after checking every time is in it."""
        self.check_store_year(ds)
        self.io.write(ds, path)
