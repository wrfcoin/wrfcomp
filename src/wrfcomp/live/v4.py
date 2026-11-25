"""Live candidate v4: one store per issue time T, built by the frozen v3 and v4 code from live inputs.

Layout under ``<data>/live/`` (never under the evaluation folders; see :func:`live_root`):

* ``gefs/<YYYYMMDDHH>.zarr``: the raw GEFS cycle at the 500 stations, written once
  (``baselines.gefs.extract``, unchanged: same bucket, index byte ranges, point extraction, lapse and
  dewpoint);
* ``metar/<fetch time>.json``: every METAR fetch, unchanged, written once (``wrfcomp.live.metar``);
* ``<YYYY-MM-DD>T<HH>Z/``: per issue, written once: ``inputs.json`` (the pinned inputs), ``v4.zarr``,
  ``v4.sha256`` and ``build.json``.

**What each issue sees is frozen when it is first built.** ``inputs.json`` pins the METAR fetch files
(name and sha256) present at that moment and the GEFS cycle's hash. Every later replay gives that
issue exactly those fetches (and only reports measured at or before its time), so a fetch appended
later (a correction, say) can never change an earlier issue, its store, or the predictors it left in
the learned state. Later issues pin later fetches, so verification readings enter the error history
only from the first issue whose build could see them.

**The model** is the frozen code (``gefs_station_v3.Replay`` then ``citizen_arms.Replay`` with no
citizen data, exactly as ``scripts/build_v4.py``), run through a :class:`~wrfcomp.live.dates.LiveYear`
policy passed explicitly (private copies of the frozen modules; nothing shared is changed). Learned
state is never stored: it is rebuilt by replaying every kept live cycle of the year in order from the
first (a cold start, as the frozen replay starts every 1 January). The frozen code keeps no state
across a calendar year and refuses issues whose 48 h lead crosses it (30 Dec 00Z to 31 Dec 12Z).
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import subprocess
import tempfile
from importlib import resources
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import xarray as xr

from wrfcomp.baselines import gefs as shared_gefs
from wrfcomp.baselines.io import CONFIG
from wrfcomp.harness.hashing import content_sha256
from wrfcomp.live import metar
from wrfcomp.live.dates import LiveRefusedError, LiveYear

LIVE_DIR = "live"
ISSUE_HOURS = (0, 12)
HOUR = np.timedelta64(1, "h")
S3 = "https://noaa-gefs-pds.s3.amazonaws.com"
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


# -- places ------------------------------------------------------------------------------------


def default_data_dir() -> Path:
    return Path(os.environ.get("WRFCOMP_DATA_DIR") or Path.home() / "wrfcomp-data")


def live_root(data: Path | None = None) -> Path:
    """``<data>/live``, refused unless it is a safe place for live data:

    the data directory is not inside the repository or a temp directory, is not an evaluation folder
    and is not inside one; ``live`` is not a symlink and holds no evaluation folder."""
    data = Path(default_data_dir() if data is None else data)
    resolved = data.resolve()
    repo = CONFIG.parents[1].resolve()
    if resolved.is_relative_to(repo) or any(resolved.is_relative_to(Path(t).resolve()) for t in TEMP_DIRS):
        raise LiveRefusedError("live data belongs in the external data directory")
    if EVALUATION_DIRS.intersection(resolved.parts) or resolved.name == LIVE_DIR:
        raise LiveRefusedError(f"the data directory {resolved} is or is inside an evaluation folder")
    root = resolved / LIVE_DIR
    if root.is_symlink() or (root.exists() and not root.is_dir()):
        raise LiveRefusedError("the live folder must be a real directory, not a link")
    if root.is_dir() and EVALUATION_DIRS.intersection(p.name for p in root.iterdir()):
        raise LiveRefusedError("the live folder holds an evaluation folder")
    return root


def safe(root: Path, path: Path) -> Path:
    """``path`` inside ``root`` with no symlink on the way (checked before every write)."""
    path = Path(path)
    rel = path.relative_to(root)
    here = root
    for part in rel.parts:
        here = here / part
        if here.is_symlink():
            raise LiveRefusedError(f"refusing a linked path inside the live folder: {here}")
    if not path.resolve().is_relative_to(root.resolve()):
        raise LiveRefusedError(f"{path} is outside the live folder")
    return path


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
    if t + np.timedelta64(48, "h") >= np.datetime64(f"{ts.year + 1:04d}-01-01"):
        raise LiveRefusedError("the frozen model makes no forecast whose 48 h lead crosses 1 January")
    return t


# -- GEFS --------------------------------------------------------------------------------------


class CheckedTransfer(shared_gefs.Transfer):
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
        wanted = {*shared_gefs.FIELDS.values(), shared_gefs.HEIGHT}
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


def fetch_cycle(
    root: Path, t, *, transfer: CheckedTransfer | None = None, policy: LiveYear | None = None
) -> dict:
    """Fetch GEFS cycle ``t`` at the stations and keep it, once, as ``gefs/<YYYYMMDDHH>.zarr``."""
    t = check_issue(t)
    policy = LiveYear(pd.Timestamp(t).year) if policy is None else policy
    if policy.year != pd.Timestamp(t).year:
        raise LiveRefusedError("the live policy is for another year")
    dest = safe(root, raw_path(root, t))
    if dest.exists():
        with xr.open_zarr(dest, chunks=None) as kept:
            return {"path": str(dest), "content_sha256": content_sha256(kept), "bytes": 0}
    transfer = CheckedTransfer(t) if transfer is None else transfer
    raw = policy.gefs.extract(stations(), t, safe(root, root / "cache"), transfer)
    policy.write(raw, dest)
    with xr.open_zarr(dest, chunks=None) as kept:
        return {"path": str(dest), "content_sha256": content_sha256(kept), "bytes": transfer.bytes}


# -- replay ------------------------------------------------------------------------------------


def no_citizen_snapshot(models) -> dict:
    return {v: (np.empty(0), np.empty(0), np.empty(0, dtype=np.int64)) for v in models.arms.VARIABLES}


def no_pairs(models):
    return models.arms.Pairs(
        citizen=np.empty(0, int), official=np.empty(0, int), distance_km=np.empty(0), cell=np.empty(0, int)
    )


def frozen_models():
    """The shared frozen modules, for evaluation years (tests replay 2024 with these)."""
    from wrfcomp.models import citizen_arms, gefs_station_v2, gefs_station_v3

    return SimpleNamespace(v2=gefs_station_v2, v3=gefs_station_v3, arms=citizen_arms)


def replay(raws: list[xr.Dataset], truth_for, year: int, models):
    """Frozen v3 then frozen v4 over ``raws`` in order. Yields (t, v3 issue, v4 issue, diagnostics).

    ``truth_for(t)`` is the truth issue t may see (labels from 1 January up to t); ``models`` is a
    :class:`LiveYear` for live years, :func:`frozen_models` for evaluation years."""
    v3_replay = None
    for raw in raws:
        t = raw.init_time.values[0]
        seen = truth_for(t)
        metar.assert_causal(seen, t)
        if v3_replay is None:
            v3_replay = models.v3.Replay(year)
            v4_replay = models.arms.Replay(year, seen, no_pairs(models))
        a, _ = v3_replay.issue(raw, seen)
        v4_replay.truth = seen
        b, _, diag = v4_replay.issue(a, no_citizen_snapshot(models))
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_once(root: Path, path: Path, text: str) -> None:
    path = safe(root, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != text:
            raise LiveRefusedError(f"{path} exists with other content; issue files are written once")
        return
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(text)
    tmp.rename(path)


def pinned_inputs(root: Path, t: np.datetime64, raw_hash: str) -> dict:
    """The inputs issue ``t`` sees: read from its ``inputs.json``, or pinned now if it has none."""
    path = issue_dir(root, t) / "inputs.json"
    if path.exists():
        safe(root, path)
        pinned = json.loads(path.read_text())
        if pinned["gefs_sha256"] != raw_hash:
            raise LiveRefusedError(f"the kept GEFS cycle of {pinned['init']} differs from its pinned hash")
        for item in pinned["metar"]:
            file = root / "metar" / item["file"]
            if not file.is_file() or _sha256(safe(root, file)) != item["sha256"]:
                raise LiveRefusedError(f"pinned METAR fetch {item['file']} is missing or changed")
        return pinned
    fetches = metar.stored(root / "metar")
    if not any(metar.fetch_time(f) >= t + metar.CLOSE_AFTER for f in fetches):
        when = np.datetime_as_string(t, unit="h")
        raise LiveRefusedError(f"no METAR fetch 15 minutes or more after {when}; readings not final")
    pinned = {
        "init": str(np.datetime_as_string(t, unit="h")),
        "gefs_sha256": raw_hash,
        "metar": [{"file": f.name, "sha256": _sha256(safe(root, f))} for f in fetches],
        "truth_source": metar.API,
    }
    _write_once(root, path, json.dumps(pinned, indent=1) + "\n")
    return pinned


def build_issue(t, data: Path | None = None) -> dict:
    """Build the live v4 store for issue ``t``, first building (once) every earlier kept issue of the
    year that has not been built yet. Existing issues are replayed from their pinned inputs and must
    reproduce their stores exactly."""
    root = live_root(data)
    t = check_issue(t)
    year = pd.Timestamp(t).year
    policy = LiveYear(year)
    paths = kept_cycles(root, t)
    if not paths or raw_path(root, t) != paths[-1]:
        raise LiveRefusedError("the GEFS cycle of the issue is not kept; fetch it first")
    first = _stem_time(paths[0])
    present = {_stem_time(p) for p in paths}
    missing = [
        str(np.datetime_as_string(c, unit="h"))
        for c in np.arange(first, t + HOUR, 12 * HOUR)
        if c not in present
    ]
    site, icaos = stations(), icao_map()
    start = np.datetime64(f"{year:04d}-01-01T00", "ns")
    raws, pins, frames = [], {}, {}
    for path in paths:
        with xr.open_zarr(safe(root, path), chunks=None) as source:
            raw = source.load()
        c = _stem_time(path)
        if not np.array_equal(raw.init_time.values, [c]):
            raise LiveRefusedError(f"kept cycle {path.name} holds another initialization")
        raws.append(raw)
        pins[c] = pinned_inputs(root, c, content_sha256(raw))

    def truth_for(c):
        names = [item["file"] for item in pins[c]["metar"]]
        for name in names:
            if name not in frames:
                frames[name] = metar.reports_frame([root / "metar" / name], icaos)
        # Corrections: the later pinned fetch wins (reports_frame's rule, applied across files).
        parts = [frames[n].assign(_order=i) for i, n in enumerate(names) if len(frames[n])]
        reports = (
            pd.concat(parts)
            .sort_values(["station_id", "obs_time", "_order"], kind="mergesort")
            .drop_duplicates(["station_id", "obs_time"], keep="last")
            .drop(columns="_order")
            if parts
            else metar.reports_frame([], icaos)
        )
        return metar.truth_as_of(reports, site, start, c)

    result = None
    issues = []
    for c, _, b, diag in replay(raws, truth_for, year, policy):
        issues.append({"init": diag.init, "phi": diag.phi, "own_defined": diag.own_defined})
        record = _publish(root, policy, c, b, pins[c], first, missing, issues)
        if c == t:
            result = record
    return result


def _publish(root, policy, c, store, pinned, first, missing, issues) -> dict:
    out_dir = issue_dir(root, c)
    dest = safe(root, out_dir / "v4.zarr")
    for var in store.variables.values():
        var.encoding.clear()
    if dest.exists():
        with xr.open_zarr(dest, chunks=None) as previous:
            xr.testing.assert_identical(store, previous.load())
    else:
        policy.write(store, dest)
    with xr.open_zarr(dest, chunks=None) as saved:
        digest = content_sha256(saved)
    _write_once(root, out_dir / "v4.sha256", f"{digest}  v4.zarr\n")
    build = out_dir / "build.json"
    if not build.exists():
        record = {
            "init": pinned["init"],
            "model_id": store.attrs["model_id"],
            "model_version": store.attrs["model_version"],
            "candidate": "v4 = v3 + the airport's own reading (frozen Phase 3 arm B), live inputs",
            "content_sha256": digest,
            "store": dest.name,
            **code_state(),
            "frozen_sources_sha256": policy.sources,
            "live_start": str(np.datetime_as_string(first, unit="h")),
            "missing_cycles": [m for m in missing if np.datetime64(m) < c],
            "inputs": "inputs.json",
            "issues": list(issues),
        }
        _write_once(root, build, json.dumps(record, indent=1) + "\n")
    record = json.loads(safe(root, build).read_text())
    if record["content_sha256"] != digest:
        raise LiveRefusedError(f"{build} records another store")
    return record
