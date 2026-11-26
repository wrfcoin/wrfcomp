"""Publish a revealed live v4 issue and check a published copy against the chain (lane I2, deliverable 3).

**Where.** One gzip tar per issue, uploaded by the operator as a GitHub release asset on the public
wrfcomp repository: one release per month (tag ``live-v4-YYYY-MM``), one asset per issue
(``wrfcomp-live-v4-<YYYY-MM-DD>T<HH>Z.tar.gz``, about 2 MB). Release assets on a public repository
cost nothing, need no new account, keep a stable download URL, and stay out of the git history.
Zenodo needs a manual deposit per version, and committing stores to the repository would grow every
clone. The upload is the operator's (``scripts/live_daily.sh``); nothing here sends anything.

**When.** Only after the reveal is confirmed on chain (``receipt.json`` exists), so after init: the
commit lands at about T + 4 h 15 min.

**What.** ``<issue>/v4.zarr`` (the store), ``data.json`` and ``manifest.json`` (the exact revealed
bytes), ``receipt.json`` (heights, block hashes and times, counted leads) and ``publication.json``
(the commitment's preimage parts: nonce, timestamp, station label, forecaster; no local path, key
path, CLI path or token). The archive is deterministic: sorted members, zero times and owners.

**check-reveal** (on a published archive, for example one downloaded back from the release):

* extracts it safely (regular files and folders only, under one ``<issue>/`` folder, size and count
  caps) into a work folder under ``<live>/verify/``;
* re-hashes the store (the harness content hash) and requires it to equal ``forecast_sha256`` in
  ``data.json``, ``output_hash`` in ``manifest.json`` and ``content_sha256`` in the receipt; the store
  must be the v4 model at the receipt's init;
* re-derives the commit id from ``data.json``, nonce, station and timestamp (the chain's formula)
  and the manifest digest, and requires both to equal the receipt;
* re-reads the chain (:func:`chain_check`): both recorded blocks with the same hash and time, the
  commit block carrying a ForecastCommit and the reveal block a ForecastReveal sent by the
  forecaster, the reveal at commit + 1 .. + 3, at least 6 blocks after the reveal, and, while the
  node still serves it, the commit status reporting our reveal included at the recorded height;
* recomputes the counted leads from the block time read now.

It writes ``<live>/verify/<issue>.json`` with the result. Reads are the local node's unauthenticated
GETs, at least 3 s apart (``chain.Node``); no bearer is used.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path

import numpy as np

from wrfcomp.live import chain
from wrfcomp.live.paths import check_path, check_target, make_dir, write_scope

RELEASE_REPO = "wrfcoin/wrfcomp"
ASSET_PREFIX = "wrfcomp-live-v4-"
MAX_MEMBERS = 20_000
MAX_BYTES = 200 * 1024 * 1024
PUBLICATION_FIELDS = (
    "commit_id",
    "nonce",
    "timestamp",
    "station_id",
    "forecaster",
    "chain_id",
    "network_id",
    "init",
    "content_sha256",
    "data_sha256",
    "manifest_digest",
    "model_id",
)


class PublishRefusedError(RuntimeError):
    """A publication or a check refused rather than guess."""


def issue_name(t) -> str:
    return chain.chain_dir(Path("."), t).name


def release_tag(t) -> str:
    return f"live-v4-{issue_name(t)[:7]}"


def asset_name(t) -> str:
    return f"{ASSET_PREFIX}{issue_name(t)}.tar.gz"


def archive_path(root: Path, t) -> Path:
    return Path(root) / "publish" / asset_name(t)


def _store_hash(store: Path) -> str:
    import xarray as xr

    from wrfcomp.harness.hashing import content_sha256

    with xr.open_zarr(store, chunks=None) as ds:
        return content_sha256(ds)


def _publish_bytes(root: Path, path: Path, payload: bytes) -> None:
    """Write ``path`` once (fsynced, linked into place); an identical existing file is fine."""
    path = Path(path)
    if path.exists():
        if path.read_bytes() != payload:
            raise PublishRefusedError(f"{path.name} already exists with other content")
        return
    if not path.parent.is_dir():
        check_target(root, path.parent)
        make_dir(path.parent)
    with write_scope(root, path):
        fd, staging = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".staging")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            try:
                os.link(staging, path)
            except FileExistsError:
                if path.read_bytes() != payload:
                    raise PublishRefusedError(f"{path.name} appeared with other content") from None
        finally:
            os.unlink(staging)


def _tar_bytes(issue: str, store: Path, files: dict[str, bytes]) -> bytes:
    """A deterministic gzip tar: sorted members under ``<issue>/``, zero times, owners and gzip header."""
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as tar:

        def add_dir(name: str) -> None:
            info = tarfile.TarInfo(name)
            info.type, info.mode, info.mtime = tarfile.DIRTYPE, 0o755, 0
            tar.addfile(info)

        def add_file(name: str, payload: bytes) -> None:
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = len(payload), 0o644, 0
            tar.addfile(info, io.BytesIO(payload))

        entries: dict[str, bytes | None] = {f"{issue}/{k}": v for k, v in files.items()}
        for path in store.rglob("*"):
            if path.is_symlink():
                raise PublishRefusedError(f"the store holds a link: {path.name}")
            rel = f"{issue}/v4.zarr/{path.relative_to(store).as_posix()}"
            entries[rel] = None if path.is_dir() else path.read_bytes()
        entries[f"{issue}/v4.zarr"] = None
        add_dir(issue)
        for name in sorted(entries):
            if entries[name] is None:
                add_dir(name)
            else:
                add_file(name, entries[name])
    out = io.BytesIO()
    with gzip.GzipFile(fileobj=out, mode="wb", mtime=0, compresslevel=9, filename="") as gz:
        gz.write(raw.getvalue())
    return out.getvalue()


def pack(root: Path, t) -> dict:
    """Build ``<live>/publish/<asset>`` (and its ``.sha256``) for a revealed issue. Refuses unless the
    receipt exists and the store's content hash still equals the committed one."""
    from wrfcomp.live import v4 as live

    t = np.datetime64(t, "ns")
    out = chain.chain_dir(root, t)
    receipt = json.loads(check_path(root, out / "receipt.json").read_text())
    record = json.loads((out / "commit.json").read_text())
    store = live.issue_dir(root, t) / "v4.zarr"
    check_path(root, store)
    digest = _store_hash(store)
    if not (digest == receipt["content_sha256"] == record["content_sha256"]):
        raise PublishRefusedError("the store's content hash differs from the committed one; not publishing")
    data = (out / "data.json").read_bytes()
    man = (out / "manifest.json").read_bytes()
    if hashlib.sha256(chain.compact(json.loads(data)).encode()).hexdigest() != receipt["data_sha256"]:
        raise PublishRefusedError("data.json differs from the revealed data; not publishing")
    if chain.manifest_digest(json.loads(man)) != receipt["manifest_digest"]:
        raise PublishRefusedError("manifest.json differs from the revealed manifest; not publishing")
    publication = {k: record[k] for k in PUBLICATION_FIELDS}
    files = {
        "data.json": data,
        "manifest.json": man,
        "receipt.json": (json.dumps(receipt, indent=1, sort_keys=True) + "\n").encode(),
        "publication.json": (json.dumps(publication, indent=1, sort_keys=True) + "\n").encode(),
    }
    payload = _tar_bytes(issue_name(t), store, files)
    path = archive_path(root, t)
    _publish_bytes(root, path, payload)
    sha = hashlib.sha256(payload).hexdigest()
    _publish_bytes(root, path.with_name(path.name + ".sha256"), f"{sha}  {path.name}\n".encode())
    return {
        "archive": str(path),
        "sha256": sha,
        "bytes": len(payload),
        "tag": release_tag(t),
        "asset": path.name,
    }


def _safe_extract(archive: Path, dest: Path, issue: str) -> Path:
    """Extract ``archive`` into ``dest`` (an approved, empty work folder): regular files and folders
    only, all under ``<issue>/``, at most MAX_MEMBERS members and MAX_BYTES bytes."""
    with tarfile.open(archive, mode="r:gz") as tar:
        members = tar.getmembers()
        if len(members) > MAX_MEMBERS:
            raise PublishRefusedError("the archive has too many members")
        total = 0
        for m in members:
            parts = Path(m.name).parts
            if m.name.startswith("/") or ".." in parts or not parts or parts[0] != issue:
                raise PublishRefusedError(f"unexpected path in the archive: {m.name!r}")
            if not (m.isfile() or m.isdir()):
                raise PublishRefusedError(f"the archive holds a non-regular member: {m.name!r}")
            total += m.size
        if total > MAX_BYTES:
            raise PublishRefusedError("the archive is too large")
        tar.extractall(dest, members=members, filter="data")
    return dest / issue


def chain_check(receipt: dict, node: chain.Node) -> dict:
    """Re-read the chain for a receipt: the recorded blocks (hash, time, our transactions), the window,
    the confirmations and, while served, the commit status. Returns what was confirmed."""
    forecaster = receipt["forecaster"]
    committed, revealed = int(receipt["commit_height"]), int(receipt["reveal_height"])
    if not committed + 1 <= revealed <= committed + chain.REVEAL_WINDOW_BLOCKS:
        raise PublishRefusedError("the receipt's reveal is outside its commit's window")
    status, value = node.get(f"/transactions/commit/{receipt['commit_id']}/status")
    status_served = False
    if status == 200 and isinstance(value, dict) and value.get("status") in chain.INCLUDED_STATUSES:
        if value.get("commit_id") != receipt["commit_id"] or value.get("revealed_at") != revealed:
            raise PublishRefusedError("the chain reports our reveal at another height than the receipt")
        status_served = True
    elif not (status == 404 or (isinstance(value, dict) and value.get("status") == "not_found")):
        raise PublishRefusedError("the chain does not report our reveal as included")
    blocks = {}
    for height, kind, key in (
        (committed, "ForecastCommit", "commit"),
        (revealed, "ForecastReveal", "reveal"),
    ):
        try:
            block = chain._block(node, height)
        except chain.ChainRefusedError as exc:
            raise PublishRefusedError(str(exc).replace("; no receipt", "")) from None
        if block["hash"] != receipt[f"{key}_block_hash"]:
            raise PublishRefusedError(f"block {height} is not the recorded {key} block (another hash)")
        when = chain.format_time_ns(chain.parse_time_ns(block["timestamp"]))
        if when != receipt[f"{key}_block_time"]:
            raise PublishRefusedError(f"block {height} has another time than recorded")
        if chain._ours(block, kind, forecaster) < 1:
            raise PublishRefusedError(f"block {height} carries no {kind} from the forecaster")
        blocks[key] = block
    tip = node.tip()  # last, after the block reads
    if not chain.confirmed(tip, revealed):
        raise PublishRefusedError(f"the reveal block has {chain.successors(tip, revealed)} successors, not 6")
    leads = chain.counted_leads(
        np.datetime64(receipt["init"], "ns"), chain.parse_time_ns(blocks["commit"]["timestamp"])
    )
    if leads["counted_leads_hours"] != receipt["counted_leads_hours"]:
        raise PublishRefusedError("the counted leads recomputed from the chain differ from the receipt")
    return {"commit_status_served": status_served, "tip": tip, **leads}


def check_reveal(root: Path, archive: Path, t, node: chain.Node) -> dict:
    """Check a published archive for issue ``t`` against its own files and the chain; write
    ``<live>/verify/<issue>.json``."""
    t = np.datetime64(t, "ns")
    issue = issue_name(t)
    archive = Path(archive)
    if not archive.is_file():
        raise PublishRefusedError(f"no archive at {archive}")
    work_root = Path(root) / "verify" / ".work"
    if not work_root.is_dir():
        with write_scope(root, work_root):
            make_dir(work_root)
    with write_scope(root, work_root):
        work = Path(tempfile.mkdtemp(dir=work_root, prefix=f"{issue}."))
    try:
        with write_scope(root, work):
            folder = _safe_extract(archive, work, issue)
        result = _check_folder(folder, t, node)
    finally:
        with write_scope(root, work):
            shutil.rmtree(work, ignore_errors=True)
    result["archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    text = json.dumps(result, indent=1, sort_keys=True) + "\n"
    out = Path(root) / "verify" / f"{issue}.json"
    _publish_or_compare(root, out, text)
    return result


def _publish_or_compare(root: Path, path: Path, text: str) -> None:
    """Write once. A re-run is fine if its outcome is the same apart from what moves with time (the
    tip, and whether the node still serves the commit status)."""
    if path.exists():
        old, new = json.loads(path.read_text()), json.loads(text)
        for moving in ("tip", "commit_status_served"):
            old.pop(moving, None)
            new.pop(moving, None)
        if old != new:
            raise PublishRefusedError(f"{path.name} exists with another outcome")
        return
    _publish_bytes(root, path, text.encode())


def _check_folder(folder: Path, t: np.datetime64, node: chain.Node) -> dict:
    receipt = json.loads((folder / "receipt.json").read_text())
    publication = json.loads((folder / "publication.json").read_text())
    raw_data = (folder / "data.json").read_text()
    data, man = json.loads(raw_data), json.loads((folder / "manifest.json").read_text())
    if np.datetime64(receipt["init"], "ns") != t or np.datetime64(publication["init"], "ns") != t:
        raise PublishRefusedError("the archive is for another issue")
    for key in ("commit_id", "station_id", "forecaster", "chain_id", "network_id", "content_sha256"):
        if publication[key] != receipt[key]:
            raise PublishRefusedError(f"publication.json and receipt.json disagree on {key}")
    if receipt["chain_id"] != chain.CHAIN_ID or receipt["network_id"] != chain.NETWORK_ID:
        raise PublishRefusedError("the receipt is not for testnet 888")
    chain.check_data(data)
    if raw_data != chain.compact(data) + "\n":
        raise PublishRefusedError("data.json is not the compact revealed form")
    if hashlib.sha256(chain.compact(data).encode()).hexdigest() != receipt["data_sha256"]:
        raise PublishRefusedError("data.json differs from the revealed data")
    cid = chain.commit_id(data, publication["nonce"], publication["station_id"], publication["timestamp"])
    if cid != receipt["commit_id"]:
        raise PublishRefusedError("data.json, nonce, station and timestamp do not derive the committed id")
    if chain.manifest_digest(man) != receipt["manifest_digest"]:
        raise PublishRefusedError("manifest.json differs from the revealed manifest")
    digest = _store_hash(folder / "v4.zarr")
    if not (digest == data["forecast_sha256"] == man["output_hash"] == receipt["content_sha256"]):
        raise PublishRefusedError("the published store's content hash is not the committed hash")
    import xarray as xr

    with xr.open_zarr(folder / "v4.zarr", chunks=None) as ds:
        inits = ds["init_time"].values
        attrs = dict(ds.attrs)
    if inits.size != 1 or np.datetime64(inits[0], "ns") != t:
        raise PublishRefusedError("the published store is for another init")
    if attrs.get("model_id") != chain.V4_MODEL_ID or attrs.get("model_version") != chain.V4_MODEL_VERSION:
        raise PublishRefusedError("the published store is not the v4 model")
    on_chain = chain_check(receipt, node)
    return {
        "issue": issue_name(t),
        "commit_id": receipt["commit_id"],
        "content_sha256": digest,
        "commit_height": receipt["commit_height"],
        "reveal_height": receipt["reveal_height"],
        "commit_block_hash": receipt["commit_block_hash"],
        "reveal_block_hash": receipt["reveal_block_hash"],
        "ok": True,
        **on_chain,
    }
