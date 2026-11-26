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
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from datetime import UTC, datetime
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
from wrfcomp.live.locks import LockBusyError, LockPathError, file_lock
from wrfcomp.live.paths import (
    LIVE_DIR,
    LivePathError,
    check_data_dir,
    check_live_root,
    check_target,
)

ISSUE_HOURS = (0, 12)
HOUR = np.timedelta64(1, "h")
S3 = "https://noaa-gefs-pds.s3.amazonaws.com"
# -- places ------------------------------------------------------------------------------------


def default_data_dir() -> Path:
    return Path(os.environ.get("WRFCOMP_DATA_DIR") or Path.home() / "wrfcomp-data")


def live_root(data: Path | None = None) -> Path:
    """``<data>/live``, refused unless it is a safe place for live data (``paths.check_live_root``):
    the data directory is not inside the repository or a temp directory, is not an evaluation folder
    and is not inside one; ``live`` is not a symlink and holds no evaluation folder."""
    data = Path(default_data_dir() if data is None else data)
    try:
        return check_live_root(check_data_dir(data) / LIVE_DIR)
    except LivePathError as exc:
        raise LiveRefusedError(str(exc)) from exc


def safe(root: Path, path: Path) -> Path:
    """``path`` inside ``root`` with no symlink on the way (checked before every write)."""
    path = Path(path)
    rel = path.relative_to(root)
    here = root
    for part in rel.parts:
        here = here / part
        if here.is_symlink():
            raise LiveRefusedError(f"refusing a linked path inside the live folder: {here}")
    try:
        return check_target(root, path)
    except LivePathError as exc:
        raise LiveRefusedError(str(exc)) from exc


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
    with _lock(root, f"gefs-{pd.Timestamp(t):%Y%m%d%H}"):
        dest = safe(root, raw_path(root, t))
        if dest.exists():
            with xr.open_zarr(dest, chunks=None) as kept:
                return {"path": str(dest), "content_sha256": content_sha256(kept), "bytes": 0}
        transfer = CheckedTransfer(t) if transfer is None else transfer
        cache = safe(root, root / "cache")
        cache.mkdir(parents=True, exist_ok=True)
        _no_links_below(root, cache)
        raw = policy.gefs.extract(stations(), t, cache, transfer)
        _no_links_below(root, cache)
        policy.write(raw, dest, root=root)
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


# -- writing: locks, no-replace publication ------------------------------------------------------


@contextmanager
def _lock(root: Path, name: str, *, blocking: bool = False):
    """An exclusive writer lock ``locks/<name>.lock`` (``wrfcomp.live.locks``). Non-blocking by
    default: a second writer is refused, not queued."""
    path = safe(root, root / "locks" / f"{name}.lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    safe(root, path)
    try:
        with file_lock(path, root=root, blocking=blocking):
            yield
    except LockBusyError as exc:
        raise LiveRefusedError(f"another live writer holds {path.name}") from exc
    except (LockPathError, LivePathError) as exc:
        raise LiveRefusedError(str(exc)) from exc


def _publish_once(root: Path, path: Path, text: str) -> bool:
    """Write ``path`` once (exclusive staging file, atomic no-replace link). Returns False, leaving the
    file untouched, if it already exists."""
    path = safe(root, path)
    try:
        metar.publish_file(path, text, root=root)
    except FileExistsError:
        return False
    return True


def _no_links_below(root: Path, directory: Path) -> None:
    directory = safe(root, directory)
    for here, dirs, files in os.walk(directory, followlinks=False):
        for name in dirs + files:
            if (Path(here) / name).is_symlink():
                raise LiveRefusedError(f"refusing a linked path inside the live folder: {Path(here) / name}")
            safe(root, Path(here) / name)  # the full real path, as for every write


# -- pinned inputs and the frozen history --------------------------------------------------------

AUTHORIZED_WINDOW = np.timedelta64(24, "h")


def _now() -> np.datetime64:
    return np.datetime64(datetime.now(UTC).replace(tzinfo=None), "ns")


def _iso(t) -> str:
    return str(np.datetime_as_string(np.datetime64(t, "ns"), unit="s"))


def _hour(t) -> str:
    return str(np.datetime_as_string(np.datetime64(t, "ns"), unit="h"))


def published_issues(root: Path, year: int) -> dict:
    """Every published issue of ``year`` by issue time. An issue is published only once its whole
    folder (``inputs.json``, ``v4.zarr``, ``v4.sha256``, ``build.json``) has been moved into place
    in one rename; a published issue is immutable. Staging folders (``.staging/``) are never
    published issues."""
    out = {}
    for path in sorted(root.glob(f"{year:04d}-??-??T??Z/build.json")):
        manifest = json.loads(safe(root, path.with_name("inputs.json")).read_text())
        out[np.datetime64(manifest["init"], "ns")] = manifest
    return out


def fetch_snapshot(root: Path, started: np.datetime64) -> list[tuple[np.datetime64, Path, str]]:
    """The METAR fetches available at build start, listed ONCE: (completion time, file, sha256) of
    every fetch file in place whose completion time is at or before the start. Called under the fetch
    lock (see :func:`build_issue`), so a fetch is either complete and in place before the start or
    completed after it. A file without a verified completion time (legacy names) is refused."""
    out = []
    for f in metar.stored(safe(root, root / "metar")):
        try:
            when, digest = metar.verified_fetch(safe(root, f))
        except ValueError as exc:
            raise LiveRefusedError(str(exc)) from exc
        if when <= started:
            out.append((when, f, digest))
    return out


def pin_fetches(root: Path, t: np.datetime64, started: np.datetime64, snapshot) -> dict:
    """The METAR fetches issue ``t`` may see, from the build-start snapshot: those completed by its
    cutoff, the earlier of the build start and ``t`` + 24 h. If none of them completed 15 minutes or
    more after ``t`` (a delayed first build), it is a retrospective backfill: the fetches up to the
    first one completed 15 minutes or more after ``t``, and nothing later, flagged as such."""
    cutoff = min(started, t + AUTHORIZED_WINDOW)
    final = [when for when, _, _ in snapshot if when >= t + metar.CLOSE_AFTER]
    if not final:
        raise LiveRefusedError(f"no METAR fetch 15 minutes or more after {_hour(t)}; readings not final")
    backfill = min(final) > cutoff
    if backfill:
        cutoff = min(final)
    chosen = [(f, digest) for when, f, digest in snapshot if when <= cutoff]
    return {
        "build_started": str(np.datetime_as_string(started, unit="us")),
        "fetch_cutoff": str(np.datetime_as_string(cutoff, unit="us")),
        "retrospective_backfill": bool(backfill),
        "metar": [{"file": f.name, "sha256": digest} for f, digest in chosen],
    }


def _check_manifest_inputs(root: Path, manifest: dict, raw_hash: str | None) -> None:
    if raw_hash is not None and manifest["gefs_sha256"] != raw_hash:
        raise LiveRefusedError(f"the kept GEFS cycle of {manifest['init']} differs from its pinned hash")
    for item in manifest["metar"]:
        file = root / "metar" / item["file"]
        if not file.is_file() or _sha256(safe(root, file)) != item["sha256"]:
            raise LiveRefusedError(f"pinned METAR fetch {item['file']} is missing or changed")


def build_issue(t, data: Path | None = None, *, now=None) -> dict:
    """Build the live v4 store for issue ``t``.

    Published issues are immutable: their manifest pins their METAR fetches and the ordered GEFS
    history they were replayed over, gaps included. A new issue extends the history of the latest
    published issue with the kept cycles after it, in order. Refused: an issue older than an already
    published one (a gap can never be filled in later); a pinned cycle that is missing or changed; a
    published store that does not reproduce. Each new issue is assembled in a staging folder and moved
    into place in one rename only after its store is built; a failed build publishes nothing for the
    issue it failed on and leaves no pin behind."""
    root = live_root(data)
    t = check_issue(t)
    year = pd.Timestamp(t).year
    policy = LiveYear(year)
    with _lock(root, "build"):
        # The synchronization point with fetch publication: the build start and the one list of
        # eligible fetches are taken together under the fetch lock.
        with _lock(root, metar.FETCH_LOCK.removesuffix(".lock"), blocking=True):
            started = _now() if now is None else np.datetime64(now, "ns")
            snapshot = fetch_snapshot(root, started)
        return _build(root, t, year, started, snapshot, policy)


def _build(root, t, year, started, snapshot, policy) -> dict:
    every = published_issues(root, year)
    if t not in every and any(p > t for p in every):
        raise LiveRefusedError(
            f"{_hour(t)} is older than an already published issue; a gap can never be filled in"
        )
    published = {c: m for c, m in every.items() if c <= t}
    kept = {_stem_time(p): p for p in kept_cycles(root, t)}
    if t in published:
        latest, new = t, []
    else:
        latest = max(published) if published else None
        new = [c for c in sorted(kept) if latest is None or c > latest]
        if not new or new[-1] != t:
            raise LiveRefusedError("the GEFS cycle of the issue is not kept; fetch it first")
    if latest is not None:
        chain = [np.datetime64(h["init"], "ns") for h in published[latest]["history"]] + [latest]
    else:
        chain = []
    ignored = [_hour(c) for c in sorted(kept) if latest is not None and c < latest and c not in chain]
    raws, hashes = {}, {}
    for c in chain + new:
        if c not in kept:
            raise LiveRefusedError(f"the pinned GEFS cycle {_hour(c)} is missing")
        with xr.open_zarr(safe(root, kept[c]), chunks=None) as source:
            raw = source.load()
        if not np.array_equal(raw.init_time.values, [c]):
            raise LiveRefusedError(f"kept cycle {kept[c].name} holds another initialization")
        raws[c], hashes[c] = raw, content_sha256(raw)
    for c in chain:
        _check_manifest_inputs(root, published[c], hashes[c])
        expected = [h["gefs_sha256"] for h in published[c]["history"]]
        if expected != [hashes[np.datetime64(h["init"], "ns")] for h in published[c]["history"]]:
            raise LiveRefusedError(f"the GEFS history pinned by {_hour(c)} has changed")
    order = chain + new
    manifests = dict(published)
    for k, c in enumerate(new):  # in memory only, until the issue is published
        history = order[: len(chain) + k]
        grid = np.arange(history[0], c, 12 * HOUR) if history else np.array([], "datetime64[ns]")
        manifests[c] = {
            "init": _hour(c),
            "gefs_sha256": hashes[c],
            "history": [{"init": _hour(h), "gefs_sha256": hashes[h]} for h in history],
            "missing_cycles": [_hour(g) for g in grid if g not in set(history)],
            "ignored_late_cycles": ignored,
            **pin_fetches(root, c, started, snapshot),
            "truth_source": metar.API,
        }

    site, icaos = stations(), icao_map()
    start = np.datetime64(f"{year:04d}-01-01T00", "ns")
    frames = {}

    def truth_for(c):
        names = [item["file"] for item in manifests[c]["metar"]]
        for item in manifests[c]["metar"]:
            name = item["file"]
            if name not in frames:
                path = safe(root, root / "metar" / name)
                if _sha256(path) != item["sha256"]:
                    raise LiveRefusedError(f"pinned METAR fetch {name} is missing or changed")
                frames[name] = metar.reports_frame([path], icaos)
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

    result, issues = None, []
    for c, _, b, diag in replay([raws[c] for c in order], truth_for, year, policy):
        issues.append({"init": diag.init, "phi": diag.phi, "own_defined": diag.own_defined})
        with _lock(root, _hour(c)):
            if c in published:
                record = _verify_published(root, c, b)
            else:
                record = _publish_new(root, policy, c, b, manifests[c], order[0], issues)
        if c == t:
            result = record
    return result


def _clean_store(store):
    for var in store.variables.values():
        var.encoding.clear()
    return store


def _verify_published(root, c, store) -> dict:
    """A published issue must reproduce its store exactly; nothing in its folder is written."""
    out_dir = issue_dir(root, c)
    dest = safe(root, out_dir / "v4.zarr")
    with xr.open_zarr(dest, chunks=None) as previous:
        try:
            xr.testing.assert_identical(_clean_store(store), previous.load())
        except AssertionError as exc:
            raise LiveRefusedError(f"published issue {_hour(c)} does not reproduce; refused") from exc
        digest = content_sha256(previous)
    record = json.loads(safe(root, out_dir / "build.json").read_text())
    if record["content_sha256"] != digest or (out_dir / "v4.sha256").read_text() != f"{digest}  v4.zarr\n":
        raise LiveRefusedError(f"published issue {_hour(c)} records another store")
    return record


def _publish_new(root, policy, c, store, manifest, first, issues) -> dict:
    """Assemble the whole issue folder in ``.staging/`` and move it into place in one rename. On any
    failure the staging folder is removed and nothing of the issue is published."""
    final = safe(root, issue_dir(root, c))
    if final.exists():
        raise LiveRefusedError(f"{final.name} exists but is not a published issue; remove it by hand")
    staging_root = safe(root, root / ".staging")
    staging_root.mkdir(exist_ok=True)
    staging = safe(root, Path(tempfile.mkdtemp(dir=staging_root, prefix=f"{final.name}.")))
    try:
        metar.publish_file(staging / "inputs.json", json.dumps(manifest, indent=1) + "\n", root=root)
        dest = staging / "v4.zarr"
        policy.write(_clean_store(store), dest, root=root)
        with xr.open_zarr(dest, chunks=None) as saved:
            digest = content_sha256(saved)
        metar.publish_file(staging / "v4.sha256", f"{digest}  v4.zarr\n", root=root)
        record = {
            "init": manifest["init"],
            "model_id": store.attrs["model_id"],
            "model_version": store.attrs["model_version"],
            "candidate": "v4 = v3 + the airport's own reading (frozen Phase 3 arm B), live inputs",
            "content_sha256": digest,
            "store": dest.name,
            **code_state(),
            "frozen_sources_sha256": policy.sources,
            "live_start": _hour(first),
            "missing_cycles": manifest["missing_cycles"],
            "retrospective_backfill": manifest["retrospective_backfill"],
            "inputs": "inputs.json",
            "issues": list(issues),
        }
        metar.publish_file(staging / "build.json", json.dumps(record, indent=1) + "\n", root=root)
        if final.exists() or final.is_symlink():
            raise LiveRefusedError(f"{final.name} appeared during the build; refused")
        os.rename(staging, final)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return record
