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
* verifies the files (:func:`verify_files`): store hash, ``data.json`` (init, model, content hash,
  commit id), the manifest (digest, model, horizon, output hash, the harness lock) and the receipt
  (chain 888, the window, the 6-block rule);
* re-reads the chain (:func:`chain_check`): chain 888 testnet, the commit status SERVED for that
  commit id with our station and reveal height (a 404 is inconclusive, never a pass), both recorded
  blocks with the same hash, time and our transactions, 6 blocks after the reveal, twice;
* recomputes the counted and late leads from the block time read now.

``pack`` runs the same two checks before it writes anything.

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


def _lock_sha256() -> str:
    from wrfcomp.live.paths import REPO

    return hashlib.sha256((REPO / "HARNESS_LOCK.json").read_bytes()).hexdigest()


def _int(value, what: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise PublishRefusedError(f"the receipt's {what} is not a positive block height")
    return value


def verify_files(t, receipt: dict, publication: dict, data_text: str, manifest: dict, store: Path) -> str:
    """Everything a receipt claims that its own files can prove, for issue ``t``:

    * the receipt is for testnet 888, issue ``t``, with the 6-block rule and its window, and
      ``publication`` (the preimage parts) agrees with it;
    * ``data.json`` is the compact revealed form, hashes to the receipt's ``data_sha256``, names the v4
      model at ``t`` and the receipt's content hash, and with the nonce, station label and timestamp
      derives the receipt's commit id (the chain's formula);
    * the manifest hashes to the receipt's ``manifest_digest``, names the same model, the 48 h horizon,
      the content hash as ``output_hash`` and this checkout's ``HARNESS_LOCK.json`` as ``input_hash``;
    * the store is a real folder whose content hash is the committed one, the v4 model at ``t``.

    Returns the content hash."""
    import xarray as xr

    t = np.datetime64(t, "ns")
    if receipt.get("chain_id") != chain.CHAIN_ID or receipt.get("network_id") != chain.NETWORK_ID:
        raise PublishRefusedError("the receipt is not for testnet 888")
    if np.datetime64(receipt["init"], "ns") != t or np.datetime64(publication["init"], "ns") != t:
        raise PublishRefusedError("the receipt is for another issue")
    if receipt.get("confirmations_required") != chain.CONFIRMATIONS:
        raise PublishRefusedError("the receipt does not record the 6-block confirmation rule")
    committed = _int(receipt.get("commit_height"), "commit height")
    revealed = _int(receipt.get("reveal_height"), "reveal height")
    if receipt.get("reveal_window_last_block") != committed + chain.REVEAL_WINDOW_BLOCKS:
        raise PublishRefusedError("the receipt's window is not commit + 1 .. + 3")
    if not committed + 1 <= revealed <= committed + chain.REVEAL_WINDOW_BLOCKS:
        raise PublishRefusedError("the receipt's reveal is outside its commit's window")
    if receipt.get("station_id") != chain.STATION_LABEL:
        raise PublishRefusedError("the receipt names another station label")
    for key in ("commit_id", "station_id", "forecaster", "chain_id", "network_id", "content_sha256"):
        if publication.get(key) != receipt.get(key):
            raise PublishRefusedError(f"publication.json and receipt.json disagree on {key}")
    data = json.loads(data_text)
    chain.check_data(data)
    if data_text != chain.compact(data) + "\n":
        raise PublishRefusedError("data.json is not the compact revealed form")
    if hashlib.sha256(chain.compact(data).encode()).hexdigest() != receipt["data_sha256"]:
        raise PublishRefusedError("data.json differs from the revealed data")
    if np.datetime64(data["init"], "ns") != t:
        raise PublishRefusedError("data.json names another init")
    if data["model_id"] != chain.V4_MODEL_ID or data["model_version"] != chain.V4_MODEL_VERSION:
        raise PublishRefusedError("data.json names another model")
    if data["forecast_sha256"] != receipt["content_sha256"]:
        raise PublishRefusedError("data.json commits another content hash than the receipt names")
    cid = chain.commit_id(data, publication["nonce"], publication["station_id"], publication["timestamp"])
    if cid != receipt["commit_id"]:
        raise PublishRefusedError("data.json, nonce, station and timestamp do not derive the committed id")
    if chain.manifest_digest(manifest) != receipt["manifest_digest"]:
        raise PublishRefusedError("manifest.json differs from the revealed manifest")
    if manifest["model_id"] != data["model_id"] or manifest["horizon_hours"] != max(chain.LEAD_HOURS):
        raise PublishRefusedError("the manifest names another model or horizon")
    if manifest["output_hash"] != receipt["content_sha256"]:
        raise PublishRefusedError("the manifest names another output hash")
    if manifest["input_hash"] != _lock_sha256():
        raise PublishRefusedError(
            "the manifest's input_hash is not this checkout's HARNESS_LOCK.json; check out the harness "
            "version the manifest names"
        )
    store = Path(store)
    if store.is_symlink() or not store.is_dir():
        raise PublishRefusedError("the store is not a real folder")
    digest = _store_hash(store)
    if digest != receipt["content_sha256"]:
        raise PublishRefusedError("the store's content hash is not the committed hash")
    with xr.open_zarr(store, chunks=None) as ds:
        inits, attrs = ds["init_time"].values, dict(ds.attrs)
    if inits.size != 1 or np.datetime64(inits[0], "ns") != t:
        raise PublishRefusedError("the store is for another init")
    if attrs.get("model_id") != chain.V4_MODEL_ID or attrs.get("model_version") != chain.V4_MODEL_VERSION:
        raise PublishRefusedError("the store is not the v4 model")
    return digest


def local_files(root: Path, t) -> tuple[dict, dict, str, dict, Path]:
    """The issue's own receipt, preimage parts, revealed files and store, from the live folder."""
    from wrfcomp.live import v4 as live

    out = chain.chain_dir(root, t)
    receipt = json.loads(check_path(root, out / "receipt.json").read_text())
    record = json.loads(check_path(root, out / "commit.json").read_text())
    publication = {k: record[k] for k in PUBLICATION_FIELDS}
    data_text = check_path(root, out / "data.json").read_text()
    manifest = json.loads(check_path(root, out / "manifest.json").read_text())
    store = check_path(root, live.issue_dir(root, t) / "v4.zarr")
    return receipt, publication, data_text, manifest, store


def pack(root: Path, t, node: chain.Node) -> dict:
    """Build ``<live>/publish/<asset>`` (and its ``.sha256``) for a revealed issue, only after its
    files are verified (:func:`verify_files`) AND the chain confirms the reveal now
    (:func:`chain_check`: served commit status, 6 blocks after the reveal, both blocks unchanged).
    A receipt file alone never publishes anything."""
    t = np.datetime64(t, "ns")
    receipt, publication, data_text, manifest, store = local_files(root, t)
    verify_files(t, receipt, publication, data_text, manifest, store)
    chain_check(receipt, node)
    out = chain.chain_dir(root, t)
    files = {
        "data.json": data_text.encode(),
        "manifest.json": (out / "manifest.json").read_bytes(),
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


def verify_download(root: Path, t, folder: Path) -> str:
    """The downloaded asset and its ``.sha256`` (both in ``folder``): the asset's sha256 must equal the
    one the ``.sha256`` names and the one of the archive packed here. Returns it."""
    name = asset_name(t)
    asset, sums = Path(folder) / name, Path(folder) / f"{name}.sha256"
    if not asset.is_file() or not sums.is_file():
        raise PublishRefusedError("the downloaded asset or its .sha256 is missing")
    fields = sums.read_text().split()
    if len(fields) != 2 or fields[1] != name:
        raise PublishRefusedError("the .sha256 file does not name the asset")
    got = hashlib.sha256(asset.read_bytes()).hexdigest()
    local = archive_path(root, t)
    packed = hashlib.sha256(check_path(root, local).read_bytes()).hexdigest() if local.is_file() else None
    if got != fields[0] or (packed is not None and got != packed):
        raise PublishRefusedError("the downloaded asset's sha256 differs from the published or packed one")
    return got


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


def _tip_on_888(node: chain.Node) -> int:
    """The node's tip, from a /network/info that names chain 888, testnet."""
    status, info = node.get("/network/info")
    if status != 200 or not isinstance(info, dict):
        raise PublishRefusedError("the node did not answer /network/info")
    if str(info.get("chain_id")) != chain.CHAIN_ID or str(info.get("network_id")) != chain.NETWORK_ID:
        raise PublishRefusedError(
            f"the node is chain {info.get('chain_id')} {info.get('network_id')}, not 888 testnet"
        )
    tip = info.get("chain_height")
    if not isinstance(tip, int) or isinstance(tip, bool):
        raise PublishRefusedError("the node did not report its chain height")
    return tip


def chain_check(receipt: dict, node: chain.Node) -> dict:
    """Re-read the chain for a receipt; refuse unless all of it holds now:

    * the node is chain 888, testnet;
    * the commit status is SERVED for the receipt's commit id and names our station label and our
      reveal, included, at the recorded height. A 404 or not_found is never a pass: once the node
      stops serving the status, the check is inconclusive (refused), not confirmed;
    * if the node was given the reader bearer and the receipt has the commit's transaction id, that
      transaction is included at the recorded commit height (as the receipt was made);
    * both recorded blocks have the recorded hash and time and carry a ForecastCommit / ForecastReveal
      from the forecaster; the reveal is at commit + 1 .. + 3;
    * the reveal block has at least 6 successor blocks (``chain.confirmed``), from the tip read LAST;
    * everything is read twice; the two passes must agree.

    Then the counted and late leads and the window recomputed from the chain must equal the receipt."""
    forecaster = receipt["forecaster"]
    committed = _int(receipt.get("commit_height"), "commit height")
    revealed = _int(receipt.get("reveal_height"), "reveal height")
    if not committed + 1 <= revealed <= committed + chain.REVEAL_WINDOW_BLOCKS:
        raise PublishRefusedError("the receipt's reveal is outside its commit's window")

    def read() -> dict:
        status, value = node.get(f"/transactions/commit/{receipt['commit_id']}/status")
        if status == 404 or (isinstance(value, dict) and value.get("status") == "not_found"):
            raise PublishRefusedError(
                "inconclusive: the node no longer serves this commit's status, so the commitment cannot "
                "be confirmed (not a pass)"
            )
        if status != 200 or not isinstance(value, dict) or value.get("status") not in chain.INCLUDED_STATUSES:
            raise PublishRefusedError("the chain does not report our reveal as included")
        if value.get("commit_id") != receipt["commit_id"] or value.get("station_id") != receipt["station_id"]:
            raise PublishRefusedError("the chain reports another commit or station")
        if value.get("revealed_at") != revealed:
            raise PublishRefusedError("the chain reports our reveal at another height than the receipt")
        if node._token and receipt.get("commit_tx_id"):
            try:
                if chain._included_at(node, receipt["commit_tx_id"], "ForecastCommit") != committed:
                    raise PublishRefusedError("our commit transaction is in another block than recorded")
            except chain.ChainRefusedError as exc:
                raise PublishRefusedError(str(exc).replace("; no receipt", "")) from None
        seen = {}
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
            seen[key] = (block["hash"], when)
        tip = _tip_on_888(node)  # last, after the block reads
        if not chain.confirmed(tip, revealed):
            raise PublishRefusedError(
                f"the reveal block has {chain.successors(tip, revealed)} successor blocks, not "
                f"{chain.CONFIRMATIONS}"
            )
        return {"seen": seen, "tip": tip}

    first, second = read(), read()
    if first["seen"] != second["seen"]:
        raise PublishRefusedError("the chain changed between the two reads (a reorg?); not confirmed")
    commit_time = chain.parse_time_ns(second["seen"]["commit"][1])
    leads = chain.counted_leads(np.datetime64(receipt["init"], "ns"), commit_time)
    if (
        leads["counted_leads_hours"] != receipt["counted_leads_hours"]
        or leads["late_leads_hours"] != receipt["late_leads_hours"]
    ):
        raise PublishRefusedError(
            "the counted or late leads recomputed from the chain differ from the receipt"
        )
    return {"commit_status_served": True, "tip": second["tip"], **leads}


def check_reveal(root: Path, archive: Path, t, node: chain.Node) -> dict:
    """Check a published archive for issue ``t`` against its own files (:func:`verify_files`) and the
    chain (:func:`chain_check`); write ``<live>/verify/<issue>.json``."""
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
        receipt = json.loads((folder / "receipt.json").read_text())
        publication = json.loads((folder / "publication.json").read_text())
        data_text = (folder / "data.json").read_text()
        manifest = json.loads((folder / "manifest.json").read_text())
        digest = verify_files(t, receipt, publication, data_text, manifest, folder / "v4.zarr")
        on_chain = chain_check(receipt, node)
    finally:
        with write_scope(root, work):
            shutil.rmtree(work, ignore_errors=True)
    result = {
        "issue": issue,
        "commit_id": receipt["commit_id"],
        "content_sha256": digest,
        "commit_height": receipt["commit_height"],
        "reveal_height": receipt["reveal_height"],
        "commit_block_hash": receipt["commit_block_hash"],
        "reveal_block_hash": receipt["reveal_block_hash"],
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "ok": True,
        **on_chain,
    }
    text = json.dumps(result, indent=1, sort_keys=True) + "\n"
    _publish_or_compare(root, Path(root) / "verify" / f"{issue}.json", text)
    return result


def _publish_or_compare(root: Path, path: Path, text: str) -> None:
    """Write once. A re-run is fine if its outcome is the same apart from the tip."""
    if path.exists():
        old, new = json.loads(path.read_text()), json.loads(text)
        old.pop("tip", None)
        new.pop("tip", None)
        if old != new:
            raise PublishRefusedError(f"{path.name} exists with another outcome")
        return
    _publish_bytes(root, path, text.encode())
