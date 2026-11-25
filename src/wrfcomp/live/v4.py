"""Live candidate v4: one store per issue time T, built by the frozen v3 and v4 code from live inputs.

Layout under ``<data>/live/`` (never under the evaluation folders):

* ``gefs/<YYYYMMDDHH>.zarr``: the raw GEFS cycle at the 500 stations (``baselines.gefs.extract``,
  unchanged: same bucket, byte ranges from the index, point extraction, lapse and dewpoint);
* ``metar/<fetch time>.json``: every METAR fetch, unchanged (``wrfcomp.live.metar``);
* ``<YYYY-MM-DD>T<HH>Z/v4.zarr``, ``v4.sha256`` and ``build.json``: the store for issue T.

The model is the frozen code: ``gefs_station_v3.Replay`` (v3) then ``citizen_arms.Replay`` with no
citizen data (arm B = v4), exactly as ``scripts/build_v4.py``. Learned state (v2 regression, v3
pressure gain, v4 own-reading gain) is never stored: it is rebuilt each time by replaying every
kept live cycle of the year in order, from the first live cycle (a cold start, as the frozen replay
starts every 1 January). Each replayed issue t sees only ``metar.causal(truth, t)``: labels up to t
and readings measured at or before t; errors of earlier issues therefore enter only once their
verifying reading has been measured. The frozen code keeps no state across a calendar year and
refuses issues whose 48 h lead crosses it (30 Dec 00Z to 31 Dec 12Z); live does the same.
"""

from __future__ import annotations

import io
import json
import re
import subprocess
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from wrfcomp.baselines import gefs
from wrfcomp.baselines.io import CONFIG, write
from wrfcomp.harness.hashing import content_sha256
from wrfcomp.live import metar
from wrfcomp.live.dates import LiveRefusedError, live_dates
from wrfcomp.models import citizen_arms as arms
from wrfcomp.models import gefs_station_v2 as v2
from wrfcomp.models import gefs_station_v3 as v3

LIVE_DIR = "live"
ISSUE_HOURS = (0, 12)
HOUR = np.timedelta64(1, "h")
S3 = "https://noaa-gefs-pds.s3.amazonaws.com"


# -- places ------------------------------------------------------------------------------------


def live_root(data: Path) -> Path:
    """``<data>/live``; refused inside the repository or under /tmp."""
    data = Path(data).resolve()
    repo = CONFIG.parents[1]
    if data.is_relative_to(repo) or data.is_relative_to("/tmp"):
        raise LiveRefusedError("live data belongs in the external data directory")
    return data / LIVE_DIR


def issue_dir(root: Path, t) -> Path:
    return root / f"{pd.Timestamp(t):%Y-%m-%dT%HZ}"


def raw_path(root: Path, t) -> Path:
    return root / "gefs" / f"{pd.Timestamp(t):%Y%m%d%H}.zarr"


def stations() -> xr.Dataset:
    """The registered stations (``configs/stations_conus.csv``) in ascending id order."""
    table = pd.read_csv(CONFIG.with_name("stations_conus.csv"), dtype={"station_id": str})
    table = table.sort_values("station_id", kind="mergesort")
    out = xr.Dataset(coords={"station_id": table.station_id.to_numpy(dtype=str)})
    for key, column in (("latitude", "lat"), ("longitude", "lon"), ("elevation_m", "elevation_m")):
        out.coords[key] = ("station_id", table[column].to_numpy())
    return out


def icao_map() -> dict[str, str]:
    """ICAO id -> station id for every registered station (from NOAA's isd-history.csv)."""
    text = resources.files("wrfcomp.live").joinpath("stations_icao.csv").read_text()
    table = pd.read_csv(io.StringIO(text), dtype=str)
    ids = stations().station_id.values.astype(str).tolist()
    if table.station_id.tolist() != ids or table.icao.duplicated().any():
        raise LiveRefusedError("the ICAO table does not match the registered stations")
    return dict(zip(table.icao, table.station_id, strict=True))


def _stem_time(path: Path) -> np.datetime64:
    return np.datetime64(pd.Timestamp(path.stem[:8] + "T" + path.stem[8:]).to_datetime64(), "ns")


def check_issue(t) -> np.datetime64:
    """An issue time the frozen live model can build: 00Z or 12Z, 2025 or later, not 30 to 31 Dec."""
    ts = pd.Timestamp(t)
    if ts != ts.floor("h") or ts.hour not in ISSUE_HOURS:
        raise LiveRefusedError("live v4 issues are at 00Z and 12Z")
    t = np.datetime64(ts.to_datetime64(), "ns")
    if ts.year < 2025:
        raise LiveRefusedError("live issues are 2025 or later; earlier years are evaluation data")
    if t + v2.LEADS[-1] >= np.datetime64(f"{ts.year + 1:04d}-01-01"):
        raise LiveRefusedError("the frozen model makes no forecast whose 48 h lead crosses 1 January")
    return t


# -- GEFS --------------------------------------------------------------------------------------


class CheckedTransfer(gefs.Transfer):
    """``gefs.Transfer`` that refuses any file of another cycle and any field of another time.

    Every URL must be a file of cycle T, and every index line of a field v4 uses must carry the
    reference time T, the lead of its file and the member of its file."""

    def __init__(self, init):
        super().__init__()
        self.init = pd.Timestamp(init)
        self.prefix = f"{S3}/gefs.{self.init:%Y%m%d}/{self.init:%H}/atmos/pgrb2ap5/"

    def get(self, url, interval=None):
        name = re.fullmatch(
            rf"ge(c00|p\d\d)\.t{self.init:%H}z\.pgrb2a\.0p50\.f(\d{{3}})(\.idx)?",
            url[len(self.prefix) :] if url.startswith(self.prefix) else "",
        )
        if not name:
            raise LiveRefusedError(f"not a file of cycle {self.init:%Y-%m-%d %HZ}: {url}")
        data = super().get(url, interval)
        if name.group(3):
            self._check_index(data.decode(), name.group(1), int(name.group(2)))
        return data

    def _check_index(self, index: str, member: str, lead: int) -> None:
        wanted = {*gefs.FIELDS.values(), gefs.HEIGHT}
        step = "anl" if lead == 0 else f"{lead} hour fcst"
        ens = "ENS=low-res ctl" if member == "c00" else f"ENS=+{int(member[1:])}"
        seen = set()
        for line in index.splitlines():
            row = line.split(":")
            if len(row) < 6 or tuple(row[3:5]) not in wanted:
                continue
            if row[2] != f"d={self.init:%Y%m%d%H}" or row[5] != step or (len(row) > 6 and row[6] != ens):
                raise LiveRefusedError(f"index field time or member is not cycle T: {line}")
            seen.add(tuple(row[3:5]))
        if not seen:
            raise LiveRefusedError("index has none of the fields v4 uses")


def fetch_cycle(root: Path, t, *, transfer: CheckedTransfer | None = None) -> dict:
    """Fetch GEFS cycle ``t`` at the stations and keep it as ``gefs/<YYYYMMDDHH>.zarr``."""
    t = check_issue(t)
    dest = raw_path(root, t)
    if dest.exists():
        with xr.open_zarr(dest, chunks=None) as kept:
            return {"path": str(dest), "content_sha256": content_sha256(kept), "bytes": 0}
    transfer = CheckedTransfer(t) if transfer is None else transfer
    with live_dates(pd.Timestamp(t).year):
        raw = gefs.extract(stations(), t, root / "cache", transfer)
        write(raw, dest)
    with xr.open_zarr(dest, chunks=None) as kept:
        return {"path": str(dest), "content_sha256": content_sha256(kept), "bytes": transfer.bytes}


# -- replay ------------------------------------------------------------------------------------


def no_citizen_snapshot() -> dict:
    return {v: (np.empty(0), np.empty(0), np.empty(0, dtype=np.int64)) for v in arms.VARIABLES}


NO_PAIRS = arms.Pairs(
    citizen=np.empty(0, int), official=np.empty(0, int), distance_km=np.empty(0), cell=np.empty(0, int)
)


def replay(raws: list[xr.Dataset], truth: xr.Dataset, year: int, *, causal=True):
    """Frozen v3 then frozen v4 over ``raws`` in order. Yields (t, v3 issue, v4 issue, diagnostics).

    ``truth`` covers the year from 1 January (hourly labels); with ``causal`` each issue t is given
    only ``metar.causal(truth, t)``."""
    start, _ = v2.year_bounds(year)
    truth = truth.sel(valid_time=slice(np.datetime64(start, "ns"), None))
    v3_replay = v3.Replay(year)
    v4_replay = arms.Replay(year, truth, NO_PAIRS)
    for raw in raws:
        t = raw.init_time.values[0]
        seen = metar.causal(truth, t) if causal else truth
        a, _ = v3_replay.issue(raw, seen)
        v4_replay.truth = seen
        b, _, diag = v4_replay.issue(a, no_citizen_snapshot())
        yield t, a, b, diag


def kept_cycles(root: Path, t: np.datetime64) -> list[Path]:
    """Every kept 00Z/12Z cycle of T's year up to and including T, in order."""
    year = pd.Timestamp(t).year
    out = []
    for path in sorted((root / "gefs").glob(f"{year:04d}??????.zarr")):
        c = _stem_time(path)
        if c <= t and pd.Timestamp(c).hour in ISSUE_HOURS:
            out.append(path)
    return out


def code_state() -> dict:
    root = CONFIG.parents[1]
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"code_commit": None, "clean_tree": None}
    return {"code_commit": commit, "clean_tree": status == ""}


def build_issue(data: Path, t) -> dict:
    """Build the live v4 store for issue ``t`` from the kept cycles and METAR fetches."""
    root = live_root(data)
    t = check_issue(t)
    year = pd.Timestamp(t).year
    paths = kept_cycles(root, t)
    if not paths or raw_path(root, t) != paths[-1]:
        raise LiveRefusedError("the GEFS cycle of the issue is not kept; fetch it first")
    fetches = metar.stored(root / "metar")
    if not fetches or metar.fetch_time(fetches[-1]) < t + metar.CLOSE_AFTER:
        raise LiveRefusedError("no METAR fetch 15 minutes or more after the issue time; readings not final")
    first = _stem_time(paths[0])
    expected = np.arange(first, t + HOUR, 12 * HOUR)
    present = {_stem_time(p) for p in paths}
    missing = [str(np.datetime_as_string(c, unit="h")) for c in expected if c not in present]
    site = stations()
    reports = metar.reports_frame(fetches, icao_map())
    if len(reports):
        reports = reports[reports.obs_time < np.datetime64(f"{year + 1:04d}-01-01")]
    truth = metar.hourly_truth(reports, site, np.datetime64(f"{year:04d}-01-01T00", "ns"), t)
    raws, raw_hashes = [], []
    for path in paths:
        with xr.open_zarr(path, chunks=None) as source:
            raw = source.load()
        c = _stem_time(path)
        if not np.array_equal(raw.init_time.values, [c]):
            raise LiveRefusedError(f"kept cycle {path.name} holds another initialization")
        raws.append(raw)
        raw_hashes.append({"init": path.stem, "content_sha256": content_sha256(raw)})
    issues = []
    with live_dates(year):
        for c, _, b, diag in replay(raws, truth, year):
            issues.append({"init": diag.init, "phi": diag.phi, "own_defined": diag.own_defined})
            if c == t:
                store = b
        out_dir = issue_dir(root, t)
        dest = out_dir / "v4.zarr"
        for var in store.variables.values():
            var.encoding.clear()
        if dest.exists():
            with xr.open_zarr(dest, chunks=None) as previous:
                xr.testing.assert_identical(store, previous.load())
        else:
            write(store, dest)
    with xr.open_zarr(dest, chunks=None) as saved:
        digest = content_sha256(saved)
    record = {
        "init": str(np.datetime_as_string(t, unit="h")),
        "model_id": store.attrs["model_id"],
        "model_version": store.attrs["model_version"],
        "candidate": "v4 = v3 + the airport's own reading (frozen Phase 3 arm B), live inputs",
        "content_sha256": digest,
        "store": dest.name,
        **code_state(),
        "live_start": str(np.datetime_as_string(first, unit="h")),
        "missing_cycles": missing,
        "gefs": raw_hashes,
        "metar": metar.sha256_files(fetches),
        "truth_source": metar.API,
        "issues": issues,
    }
    (out_dir / "v4.sha256").write_text(f"{digest}  v4.zarr\n")
    (out_dir / "build.json").write_text(json.dumps(record, indent=1) + "\n")
    return record
