"""Commit a published live v4 issue on testnet 888 and reveal its hash (lane I2, deliverable 2).

Follows the I1 design (stage 2). The chain stores a commit of a small data object; the reveal,
within the chain's reveal window (3 blocks), discloses that object and a manifest whose
``output_hash`` is the store's ``content_sha256``. The store itself is published off chain after
its init and checked against the hash (deliverable 3). Nothing here signs or submits: the helper
prepares the files and writes one ``submit.sh`` that the operator runs with the ``wrfcoin`` CLI.

The commit id and manifest digest follow core4 (``Transaction::compute_commit_id``,
``ForecastArtifactManifest``):

* commit id = sha3-256 of ``"{data}||{nonce}||{station_id}||{timestamp}||forecast"``, with
  ``data`` the data object as compact JSON with sorted keys (serde_json without preserve_order);
* manifest fields in the Rust declaration order; digest = sha3-256 of its compact JSON.

The data object carries strings only, so no float formatting can differ between Python and Rust.

**Timeliness.** A GEFS cycle is complete about 4 h after its init, so a commit lands after T. A lead
counts live only if its valid time is at least 15 minutes after the commit block's time
(docs/LIVE_V4.md); ``receipt.json`` lists which leads count.

**Reads** go to the local node only (``http://127.0.0.1:8556``; never wrfcoin.com or another host),
unauthenticated GETs, at least 3 s apart (under 30 a minute).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shlex
import shutil
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import xarray as xr

from wrfcomp.harness.hashing import content_sha256
from wrfcomp.live import metar
from wrfcomp.live.paths import check_path, check_target, make_dir, write_scope

CHAIN_ID = "888"
NETWORK_ID = "testnet"
NODE_URL = "http://127.0.0.1:8556"
ALLOWED_NODES = frozenset({NODE_URL})
STATION_LABEL = "wrfcomp-live-v4"
VARIABLES = ["msl", "t2m", "td2m", "u10", "v10"]
LEAD_HOURS = (1, 3, 6, 12, 24, 48)
MARGIN = np.timedelta64(15, "m")
REVEAL_WINDOW_BLOCKS = 3
INCLUDED_STATUSES = frozenset({"revealed_pending_attestation", "revealed_attested"})
POLL_SECONDS = 20
MIN_REQUEST_GAP_S = 3.0
# Keys the data object must never carry: the chain would read them as a weather reading or a paid
# forecast record (I1 design, section 1).
FORBIDDEN_KEYS = frozenset(
    {"schema", "schema_version", "weather_schema_version", "observation", "temp_c_centi", "pressure_pa"}
    | {"t2m", "td2m", "u10", "v10", "msl", "temperature", "pressure", "dewpoint", "wind_speed"}
)
_ADDRESS = re.compile(r"0x[0-9a-fA-F]{40}")
_SAFE = re.compile(r"[A-Za-z0-9_./:+-]+")


class ChainRefusedError(RuntimeError):
    """A chain step breaks a rule; nothing was written or sent."""


# -- the chain's own hashing ----------------------------------------------------------------------


def compact(obj) -> str:
    """serde_json::to_string of a parsed Value: compact, keys sorted, UTF-8 kept."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def commit_id(data: dict, nonce: str, station_id: str, timestamp: int) -> str:
    preimage = f"{compact(data)}||{nonce}||{station_id}||{int(timestamp)}||forecast"
    return hashlib.sha3_256(preimage.encode()).hexdigest()


MANIFEST_FIELDS = (
    "schema_version",
    "model_id",
    "input_hash",
    "output_hash",
    "produced_at",
    "region",
    "horizon_hours",
    "input_variables",
    "output_variables",
)


def manifest(model_id, input_hash, output_hash, produced_at, horizon_hours=48, region="conus-isd") -> dict:
    out = {
        "schema_version": 1,
        "model_id": model_id,
        "input_hash": input_hash,
        "output_hash": output_hash,
        "produced_at": int(produced_at),
        "region": region,
        "horizon_hours": int(horizon_hours),
        "input_variables": list(VARIABLES),
        "output_variables": list(VARIABLES),
    }
    validate_manifest(out)
    return out


def _hex64(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdefABCDEF" for c in value)


def validate_manifest(m: dict) -> None:
    """``ForecastArtifactManifest::validate``, rule for rule, plus the field order."""
    if tuple(m) != MANIFEST_FIELDS:
        raise ChainRefusedError("manifest fields are not in the chain's declaration order")
    if m["schema_version"] != 1:
        raise ChainRefusedError("manifest schema_version must be 1")
    if not m["model_id"] or len(m["model_id"].encode()) > 128:
        raise ChainRefusedError("manifest model_id must be 1 to 128 bytes")
    if not (_hex64(m["input_hash"]) and _hex64(m["output_hash"])):
        raise ChainRefusedError("manifest input_hash and output_hash must be 64 hex")
    if not 1 <= m["horizon_hours"] <= 720:
        raise ChainRefusedError("manifest horizon_hours must be 1 to 720")
    for key in ("input_variables", "output_variables"):
        values = m[key]
        if not values or any(a.encode() >= b.encode() for a, b in zip(values, values[1:], strict=False)):
            raise ChainRefusedError(f"manifest {key} must be non-empty, sorted and unique")


def canonical_manifest(m: dict) -> dict:
    """A manifest parsed from anywhere (core4's CLI prints it with sorted keys) in the Rust
    declaration order, refused unless it has exactly the manifest's fields."""
    if not isinstance(m, dict) or set(m) != set(MANIFEST_FIELDS):
        raise ChainRefusedError("not a manifest: wrong fields")
    return {k: m[k] for k in MANIFEST_FIELDS}


def manifest_digest(m: dict) -> str:
    validate_manifest(m)
    return hashlib.sha3_256(json.dumps(m, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def check_data(data: dict) -> None:
    if not isinstance(data, dict) or not data:
        raise ChainRefusedError("the data object must be a non-empty JSON object")
    bad = FORBIDDEN_KEYS.intersection(data)
    if bad:
        raise ChainRefusedError(f"the data object must not carry {sorted(bad)}")
    if not all(isinstance(v, str) for v in data.values()):
        raise ChainRefusedError("the data object carries strings only (no float formatting differences)")


# -- timeliness -----------------------------------------------------------------------------------


def counted_leads(init, block_time) -> dict:
    """Which leads count live: valid time at least 15 minutes after the commit block's time."""
    init, block_time = np.datetime64(init, "ns"), np.datetime64(block_time, "ns")
    counted, late = [], []
    for hours in LEAD_HOURS:
        (counted if init + np.timedelta64(hours, "h") >= block_time + MARGIN else late).append(hours)
    return {"counted_leads_hours": counted, "late_leads_hours": late}


# -- prepare ---------------------------------------------------------------------------------------


def chain_dir(root: Path, t) -> Path:
    """``<live>/chain/<YYYY-MM-DD>T<HH>Z/``: the issue folder itself is immutable once published."""
    return root / "chain" / f"{pd.Timestamp(t):%Y-%m-%dT%HZ}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_key_file(root: Path, key: Path) -> Path:
    """The throwaway forecaster key: a regular file (not a link) owned by this user, mode exactly 0600,
    in ``<live>/keys/``, which must itself be a real folder owned by this user with mode exactly 0700.
    Its contents are never opened here; the CLI reads it when the operator runs ``submit.sh``."""
    key = Path(key)
    folder = root / "keys"
    if folder.is_symlink() or not folder.is_dir():
        raise ChainRefusedError("<live>/keys/ must be a real folder (make it with: install -d -m 0700)")
    if folder.stat().st_mode & 0o7777 != 0o700 or folder.stat().st_uid != os.getuid():
        raise ChainRefusedError("<live>/keys/ must be owned by you with mode exactly 0700")
    if key.is_symlink() or not key.is_file():
        raise ChainRefusedError("the signer key must be a regular file, not a link")
    if key.parent.resolve() != folder.resolve():
        raise ChainRefusedError("the throwaway forecaster key must live in <live>/keys/")
    if key.stat().st_mode & 0o7777 != 0o600 or key.stat().st_uid != os.getuid():
        raise ChainRefusedError("the signer key file must be owned by you with mode exactly 0600")
    check_path(root, key)
    return key


# The published v4 identity a live commit may carry (citizen_arms.MODEL_OWN / VERSION).
V4_MODEL_ID = "gefs-station-corrected-own"
V4_MODEL_VERSION = "1"


def _published_identity(root: Path, t) -> dict:
    """Read and cross-check a published issue: the store's single init and model attributes, its
    build record and its pinned inputs must all agree with the requested issue and the v4 identity."""
    from wrfcomp.live import v4 as live

    issue = live.issue_dir(root, t)
    build_path, inputs_path, store_path = issue / "build.json", issue / "inputs.json", issue / "v4.zarr"
    if not build_path.is_file() or not inputs_path.is_file() or not store_path.is_dir():
        raise ChainRefusedError("the issue is not published; build it first")
    build = json.loads(build_path.read_text())
    inputs = json.loads(inputs_path.read_text())
    want = str(np.datetime_as_string(np.datetime64(t, "h"), unit="h"))
    exact = np.datetime64(t, "ns")
    sha_path = issue / "v4.sha256"
    with xr.open_zarr(store_path, chunks=None) as store:
        digest = content_sha256(store)
        inits = store.init_time.values
        attrs = dict(store.attrs)
    if digest != build.get("content_sha256"):
        raise ChainRefusedError("the published store does not match its build.json hash")
    if inits.size != 1 or np.datetime64(inits[0], "ns") != exact:
        raise ChainRefusedError("the published store is not the requested issue (its init differs)")
    if not sha_path.is_file() or sha_path.read_text() != f"{digest}  v4.zarr\n":
        raise ChainRefusedError("the published v4.sha256 does not match the store")
    if build.get("init") != want or inputs.get("init") != want:
        raise ChainRefusedError("the build or inputs record names another issue")
    if attrs.get("model_id") != V4_MODEL_ID or attrs.get("model_version") != V4_MODEL_VERSION:
        raise ChainRefusedError("the published store is not candidate v4")
    if build.get("model_id") != attrs.get("model_id") or build.get("model_version") != attrs.get(
        "model_version"
    ):
        raise ChainRefusedError("the build record's model differs from the store's")
    return {
        "init": want,
        "content_sha256": digest,
        "model_id": attrs["model_id"],
        "model_version": attrs["model_version"],
        "inputs_sha256": _sha256_file(inputs_path),
        "build_started": inputs["build_started"],
    }


def prepare(
    root: Path,
    t,
    *,
    forecaster: str,
    signer_key: Path,
    cli: Path,
    cli_sha256: str,
    repo: Path,
    allow_12z: bool = False,
    now: datetime | None = None,
    nonce: str | None = None,
) -> dict:
    """Write ``<live>/chain/<issue>/`` (once, as a whole): data.json, manifest.json, commit.json and
    submit.sh are staged together and moved into place in one rename, so an interrupted prepare
    leaves nothing behind and can simply be run again."""
    from wrfcomp.live import v4 as live

    t = live.check_issue(t)
    if pd.Timestamp(t).hour == 12 and not allow_12z:
        raise ChainRefusedError("only 00Z issues are committed (12Z needs --allow-12z)")
    ident = _published_identity(root, t)
    if not _ADDRESS.fullmatch(forecaster or ""):
        raise ChainRefusedError("--forecaster must be 0x followed by 40 hex")
    _check_key_file(root, signer_key)
    cli = Path(cli)
    if not cli.is_file() or _sha256_file(cli) != cli_sha256:
        raise ChainRefusedError("the CLI binary is missing or its sha256 differs from --cli-sha256")
    out = chain_dir(root, t)
    if out.exists() or out.is_symlink():
        raise ChainRefusedError(f"{out.name} is already prepared; one commit per issue")
    now = now or datetime.now(UTC)
    timestamp = int(now.timestamp())
    nonce = nonce or secrets.token_hex(32)
    data = {
        "forecast_sha256": ident["content_sha256"],
        "init": ident["init"],
        "inputs_sha256": ident["inputs_sha256"],
        "model_id": ident["model_id"],
        "model_version": ident["model_version"],
        "wrfcomp_commit": "v1",
    }
    check_data(data)
    produced = int(np.datetime64(ident["build_started"], "s").astype(int))
    lock_sha = _sha256_file(Path(repo) / "HARNESS_LOCK.json")
    man = manifest(
        ident["model_id"], lock_sha, ident["content_sha256"], produced, horizon_hours=max(LEAD_HOURS)
    )
    record = {
        "chain_id": CHAIN_ID,
        "network_id": NETWORK_ID,
        "node_url": NODE_URL,
        "station_id": STATION_LABEL,
        "forecaster": forecaster,
        "nonce": nonce,
        "timestamp": timestamp,
        "commit_id": commit_id(data, nonce, STATION_LABEL, timestamp),
        "data_sha256": hashlib.sha256(compact(data).encode()).hexdigest(),
        "manifest_digest": manifest_digest(man),
        "init": ident["init"],
        "content_sha256": ident["content_sha256"],
        "model_id": ident["model_id"],
        "signer_key": str(signer_key),
        "cli": str(cli),
        "cli_sha256": cli_sha256,
        "prepared_at": now.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "if_committed_now": counted_leads(t, np.datetime64(now.astimezone(UTC).replace(tzinfo=None), "ns")),
    }
    files = {
        "data.json": compact(data) + "\n",
        "manifest.json": json.dumps(man, separators=(",", ":")) + "\n",
        "commit.json": json.dumps(record, indent=1) + "\n",
        "submit.sh": submit_script(out, record, repo),
    }
    staging_root = out.parent / ".staging"
    with write_scope(root, staging_root):
        make_dir(staging_root)
    with write_scope(root, staging_root):
        staging = Path(tempfile.mkdtemp(dir=staging_root, prefix=f"{out.name}."))
    try:
        for name, text in files.items():
            metar.publish_file(staging / name, text, root=root)
        check_target(root, out)
        if out.exists() or out.is_symlink():
            raise ChainRefusedError(f"{out.name} appeared during prepare; refused")
        os.rename(staging, out)
    except BaseException:
        with write_scope(root, staging):
            shutil.rmtree(staging, ignore_errors=True)
        raise
    return record


def submit_script(out: Path, record: dict, repo: Path) -> str:
    """The one script the operator runs. Every value in it was validated; all are shell-quoted."""
    for key in ("forecaster", "nonce", "station_id", "chain_id", "network_id", "commit_id"):
        if not _SAFE.fullmatch(str(record[key])):
            raise ChainRefusedError(f"unsafe characters in {key}")
    q = shlex.quote
    cont = " \\\n    "
    common = cont.join(
        [
            f"--forecaster {q(record['forecaster'])}",
            f"--station-id {q(record['station_id'])}",
            '--data-file "$DIR/data.json"',
            f"--nonce {q(record['nonce'])}",
            f"--timestamp {int(record['timestamp'])}",
        ]
    )
    signing = cont.join(
        [
            f"--signer-key {q(record['signer_key'])}",
            f"--chain-id {q(record['chain_id'])}",
            f"--network-id {q(record['network_id'])}",
        ]
    )
    commit_cmd = f'"$CLI" --output json forecast commit{cont}{common}{cont}{signing}{cont}--node-url "$NODE"'
    reveal_cmd = (
        f'"$CLI" --output json forecast reveal{cont}{common}{cont}'
        '--manifest-file "$DIR/manifest.json"' + cont + '--node-url "$NODE"'
    )
    reveal_check = reveal_cmd.replace(" reveal", " reveal --validate-only", 1)
    lines = [
        "#!/usr/bin/env bash",
        f"# submit.sh for live v4 issue {record['init']}: testnet {record['chain_id']} only.",
        "# Written by scripts/live_commit.py prepare.   bash submit.sh          the first run",
        "#                                               bash submit.sh resume   after an interruption:",
        "#   it continues from whatever stage was saved and never broadcasts the commit twice.",
        "# The CLI uses the operator bearer (WRFCOIN_OPERATOR_TOKEN) from this shell; the helper steps run",
        "# without it, and nothing prints it. One run at a time; the commit is never broadcast twice.",
        "set -euo pipefail",
        "umask 077",
        f"DIR={q(str(out))}",
        f"REPO={q(str(Path(repo).resolve()))}",
        f"CLI={q(record['cli'])}",
        f"NODE={q(record['node_url'])}",
        'MODE="${1:-run}"',
        'case "$MODE" in run|resume) ;; *) echo "usage: bash submit.sh [resume]"; exit 2;; esac',
        'cd "$REPO"',
        "py() {",
        "    env -u WRFCOIN_OPERATOR_TOKEN -u WRFCOIN_READER_TOKEN \\",
        '        uv run --frozen python -P scripts/live_commit.py "$@" --dir "$DIR"',
        "}",
        'if ! mkdir "$DIR/active" 2>/dev/null; then',
        '    echo "another submit.sh is running for this issue"',
        '    echo "(if a run was killed: check nothing is running, then remove $DIR/active)"',
        "    exit 1",
        "fi",
        "trap 'rmdir \"$DIR/active\"' EXIT",
        "",
        'echo "checking the CLI binary, the key and the node"',
        "py check-cli",
        "py check-key",
        "py check-node",
        'if [ -e "$DIR/commit.claim" ] && [ "$MODE" = run ]; then',
        '    echo "a commit was already attempted here; it is never sent twice"',
        '    echo "to finish it:  bash submit.sh resume"',
        "    exit 1",
        "fi",
        'if [ ! -e "$DIR/commit.claim" ]; then',
        '    echo "validating the commit offline (nothing is sent)"',
        f'    {commit_cmd.replace(" commit", " commit --validate-only", 1)} > "$DIR/validate.json"',
        "    py check-validate",
        "    py check-key",
        '    mkdir "$DIR/commit.claim"',
        '    echo "committing"',
        f'    {commit_cmd} > "$DIR/commit_result.json.part"',
        '    mv "$DIR/commit_result.json.part" "$DIR/commit_result.json"',
        "    py check-submitted --step commit",
        '    mkdir "$DIR/commit.sent"',
        "fi",
        f'echo "waiting for committed_pending_reveal (one read every {POLL_SECONDS} s)"',
        "py wait-pending",
        'if [ ! -e "$DIR/reveal.sent" ]; then',
        "    NEXT=send",
        '    if [ -e "$DIR/reveal.claim" ]; then',
        '        echo "a reveal was attempted before; asking the chain whether it was included"',
        '        NEXT="$(py reconcile-reveal)"',
        "    fi",
        '    if [ "$NEXT" != included ]; then',
        '        echo "checking the reveal against the prepared commitment"',
        "        py check-reveal-inputs",
        f'        {reveal_check} > "$DIR/reveal_validate.json"',
        "        py check-reveal-validate",
        '        NEXT="$(py check-window)"',
        "    fi",
        '    if [ "$NEXT" = send ]; then',
        '        mkdir -p "$DIR/reveal.claim"',
        '        echo "revealing the hash and manifest"',
        f'        if {reveal_cmd} > "$DIR/reveal_result.json.part"; then',
        '            mv "$DIR/reveal_result.json.part" "$DIR/reveal_result.json"',
        "            py check-submitted --step reveal",
        '            mkdir "$DIR/reveal.sent"',
        "        else",
        '            echo "the reveal send failed (a resend may be refused as a duplicate);"',
        '            echo "the chain decides: waiting for our reveal to be included"',
        "        fi",
        "    else",
        '        echo "our earlier reveal is already included; nothing is sent"',
        "    fi",
        "fi",
        'echo "waiting for the reveal to be included inside the reveal window"',
        "py wait-revealed",
        "py record",
        'echo "done: $DIR/receipt.json"',
    ]
    return "\n".join(lines) + "\n"


# -- the steps submit.sh runs ----------------------------------------------------------------------


class Node:
    """Unauthenticated GETs to the local node only, at least MIN_REQUEST_GAP_S apart."""

    def __init__(self, url: str | None = None, *, get=None, sleep=time.sleep, clock=time.monotonic):
        url = NODE_URL if url is None else url
        if url not in ALLOWED_NODES:
            raise ChainRefusedError(f"reads go to the local node only ({NODE_URL}), not {url}")
        parts = urlsplit(url)
        if parts.hostname != "127.0.0.1" or parts.scheme != "http":
            raise ChainRefusedError("reads go to the local node only")
        self.url, self._get, self.sleep, self.clock, self.last = (
            url,
            get or self._http_get,
            sleep,
            clock,
            None,
        )

    @staticmethod
    def _http_get(url: str) -> tuple[int, bytes]:
        try:
            with urlopen(Request(url, headers={"Accept": "application/json"}), timeout=20) as response:
                return response.status, response.read()
        except Exception as exc:  # noqa: BLE001 - HTTPError carries a status; others are failures
            status = getattr(exc, "code", None)
            if status is None:
                raise
            return status, exc.read() if hasattr(exc, "read") else b""

    def get(self, path: str):
        if self.last is not None:
            wait = MIN_REQUEST_GAP_S - (self.clock() - self.last)
            if wait > 0:
                self.sleep(wait)
        self.last = self.clock()
        status, body = self._get(self.url + path)
        value = json.loads(body) if body else None
        if isinstance(value, dict) and value.get("ok") is True and "data" in value:
            value = value["data"]
        return status, value

    def tip(self) -> int:
        status, info = self.get("/network/info")
        if status != 200 or not isinstance(info, dict) or not isinstance(info.get("chain_height"), int):
            raise ChainRefusedError("the node did not report its chain height")
        return info["chain_height"]


def _live(out: Path) -> Path:
    return Path(out).parent.parent


def _load(out: Path, name: str):
    return json.loads((Path(out) / name).read_text())


def _save(out: Path, name: str, value) -> None:
    """Write a checkpoint once. An identical existing checkpoint is fine (a resumed step); a
    different one is refused."""
    text = json.dumps(value, indent=1) + "\n"
    path = Path(out) / name
    try:
        metar.publish_file(path, text, root=_live(out))
    except FileExistsError:
        if path.read_text() != text:
            raise ChainRefusedError(f"{name} already exists with other content") from None


def _cli_data(out: Path, name: str) -> dict:
    envelope = _load(out, name)
    if not isinstance(envelope, dict) or envelope.get("ok") is not True:
        raise ChainRefusedError(f"{name}: the CLI reported a failure")
    return envelope["data"]


def check_cli(out: Path) -> None:
    record = _load(out, "commit.json")
    if _sha256_file(Path(record["cli"])) != record["cli_sha256"]:
        raise ChainRefusedError("the CLI binary's sha256 differs from the one recorded at prepare")


def check_key(out: Path) -> None:
    """Recheck the key folder (0700) and file (0600) right before the CLI signs."""
    _check_key_file(_live(out), Path(_load(out, "commit.json")["signer_key"]))


def check_node(out: Path, node: Node) -> dict:
    record = _load(out, "commit.json")
    status, info = node.get("/network/info")
    if status != 200 or not isinstance(info, dict):
        raise ChainRefusedError("the node did not answer /network/info")
    if str(info.get("chain_id")) != record["chain_id"] or str(info.get("network_id")) != record["network_id"]:
        raise ChainRefusedError(
            f"the node is chain {info.get('chain_id')} {info.get('network_id')}, not {record['chain_id']} "
            f"{record['network_id']}; refusing"
        )
    return info


def check_validate(out: Path) -> None:
    record, data = _load(out, "commit.json"), _cli_data(out, "validate.json")
    body = data.get("body", {})
    if (
        data.get("validated") is not True
        or data.get("submitted") is not False
        or data.get("kind") != "commit"
    ):
        raise ChainRefusedError("validate.json is not an offline commit validation")
    if data.get("commit_id") != record["commit_id"] or body.get("commit_hash") != record["commit_id"]:
        raise ChainRefusedError("the CLI's commit id differs from the helper's; refusing")
    if body.get("forecaster") != record["forecaster"] or body.get("station_id") != record["station_id"]:
        raise ChainRefusedError("the validated body names another forecaster or station")


def check_submitted(out: Path, step: str) -> dict:
    record = _load(out, "commit.json")
    data = _cli_data(out, f"{step}_result.json")
    if data.get("commit_id") != record["commit_id"] or not data.get("tx_id"):
        raise ChainRefusedError(f"{step}: the CLI result names another commit or no transaction")
    return data


def reveal_window(pending: dict) -> tuple[int, int]:
    """The blocks a reveal may land in: commit height + 1 .. the earliest of commit height + 3 and
    the chain's own reported commit and reveal expiries (inclusive)."""
    committed = int(pending["committed_at"])
    last = min(
        committed + REVEAL_WINDOW_BLOCKS, int(pending["commit_expires_at"]), int(pending["reveal_expires_at"])
    )
    if last < committed + 1:
        raise ChainRefusedError("the chain reports no block a reveal could land in")
    return committed + 1, last


def _deadline(pending: dict) -> int:
    return reveal_window(pending)[1]


def _check_pending(record: dict, value: dict) -> dict:
    """A pending commit is OURS and carries the chain's heights (CommitRecord, commit_reveal.rs)."""
    if value.get("commit_id") != record["commit_id"]:
        raise ChainRefusedError("the node's pending commit is another commit; not revealing")
    if value.get("station_id") != record["station_id"] or value.get("contribution_type") != "forecast":
        raise ChainRefusedError("the node's pending commit names another station or type; not revealing")
    if value.get("status") != "committed_pending_reveal":
        raise ChainRefusedError("the saved commit status is not committed_pending_reveal")
    heights = [value.get(k) for k in ("committed_at", "commit_expires_at", "reveal_expires_at")]
    if not all(isinstance(h, int) and not isinstance(h, bool) and h >= 0 for h in heights):
        raise ChainRefusedError("the node's pending commit has no valid heights; not revealing")
    reveal_window(value)
    return value


def _check_revealed(record: dict, pending: dict, value: dict) -> dict:
    """OUR reveal, included (a revealed_* status), at a height inside the window."""
    if value.get("commit_id") != record["commit_id"] or value.get("station_id") != record["station_id"]:
        raise ChainRefusedError("the node reports another reveal; no receipt")
    if value.get("status") not in INCLUDED_STATUSES:
        raise ChainRefusedError("the reveal is not reported as included; no receipt")
    first, last = reveal_window(pending)
    height = value.get("revealed_at")
    if not isinstance(height, int) or isinstance(height, bool) or not first <= height <= last:
        raise ChainRefusedError(
            f"the reveal was not included inside the window {first}..{last} (reported {height!r}); no receipt"
        )
    return value


def wait_pending(
    out: Path, node: Node, *, timeout_s: float = 600.0, sleep=time.sleep, clock=time.monotonic
) -> dict:
    """Poll the commit status until OUR commit is ``committed_pending_reveal``; save it."""
    record = _load(out, "commit.json")
    if (Path(out) / "status_pending.json").exists():  # resumed: reuse the saved, rechecked checkpoint
        return _check_pending(record, _load(out, "status_pending.json"))
    start, start_tip = clock(), node.tip()
    while True:
        status, value = node.get(f"/transactions/commit/{record['commit_id']}/status")
        if status == 200 and isinstance(value, dict):
            if value.get("status") == "committed_pending_reveal":
                pending = _check_pending(record, value)
                _save(out, "status_pending.json", pending)
                return pending
            if value.get("status") not in (None, "not_found"):
                raise ChainRefusedError(f"unexpected commit status {value.get('status')!r}; not revealing")
        if clock() - start > timeout_s or node.tip() > start_tip + 2 * REVEAL_WINDOW_BLOCKS:
            raise ChainRefusedError(
                "the commit is not on chain: OPERATOR DECISION needed. It is never sent again "
                "automatically; check the node, then decide whether to prepare a new commit."
            )
        sleep(POLL_SECONDS)


def check_reveal_inputs(out: Path) -> None:
    """Rederive everything the reveal will carry from the exact files it will send."""
    record = _load(out, "commit.json")
    raw = (Path(out) / "data.json").read_text()
    data = json.loads(raw)
    check_data(data)
    if (
        raw != compact(data) + "\n"
        or hashlib.sha256(compact(data).encode()).hexdigest() != record["data_sha256"]
    ):
        raise ChainRefusedError("data.json differs from the prepared data object; not revealing")
    if commit_id(data, record["nonce"], record["station_id"], record["timestamp"]) != record["commit_id"]:
        raise ChainRefusedError("data.json no longer derives the committed id; not revealing")
    man = json.loads((Path(out) / "manifest.json").read_text())
    if manifest_digest(man) != record["manifest_digest"]:
        raise ChainRefusedError("manifest.json differs from the prepared manifest; not revealing")
    if man["output_hash"] != data["forecast_sha256"] or man["model_id"] != data["model_id"]:
        raise ChainRefusedError("the manifest does not describe the committed store; not revealing")


def check_reveal_validate(out: Path) -> None:
    """The CLI's offline reveal body must be exactly the prepared reveal."""
    record = _load(out, "commit.json")
    data = _cli_data(out, "reveal_validate.json")
    body = data.get("body", {})
    expected = {
        "forecaster": record["forecaster"],
        "station_id": record["station_id"],
        "payload": json.loads((Path(out) / "data.json").read_text()),
        "nonce": record["nonce"],
        "timestamp": record["timestamp"],
        "commit_hash": record["commit_id"],
        "manifest": json.loads((Path(out) / "manifest.json").read_text()),
        "verification_receipt": None,
    }
    if (
        data.get("validated") is not True
        or data.get("submitted") is not False
        or data.get("kind") != "reveal"
    ):
        raise ChainRefusedError("reveal_validate.json is not an offline reveal validation")
    if data.get("commit_id") != record["commit_id"] or body != expected:
        raise ChainRefusedError("the CLI's reveal body differs from the prepared reveal; not revealing")
    if manifest_digest(canonical_manifest(body["manifest"])) != record["manifest_digest"]:
        raise ChainRefusedError("the CLI's reveal manifest differs from the prepared one; not revealing")


def _final_block(node: Node, height: int) -> str:
    status, block = node.get(f"/chain/block/{height}")
    if status != 200 or not isinstance(block, dict) or block.get("height") != height:
        raise ChainRefusedError(f"the node did not return block {height}")
    if not (isinstance(block.get("hash"), str) and _hex64(block["hash"])):
        raise ChainRefusedError(f"block {height} has no final hash")
    return block["hash"]


def check_window(out: Path, node: Node) -> str:
    """Broadcast the reveal only if the next block can carry it: the node's current view still has
    OUR commit pending at the saved height, the node is not behind that height, the commit block is
    final, and the next block is inside the window (both bounds). Returns "send", or "included" if
    our earlier reveal has meanwhile been included (then nothing is sent)."""
    record = _load(out, "commit.json")
    pending = _check_pending(record, _load(out, "status_pending.json"))
    first, last = reveal_window(pending)
    committed = int(pending["committed_at"])
    status, now = node.get(f"/transactions/commit/{record['commit_id']}/status")
    if status == 200 and isinstance(now, dict) and now.get("status") in INCLUDED_STATUSES:
        _check_revealed(record, pending, now)
        return "included"
    if status != 200 or not isinstance(now, dict) or now.get("status") != "committed_pending_reveal":
        raise ChainRefusedError("the node does not report our commit as pending; not revealing")
    if int(_check_pending(record, now)["committed_at"]) != committed:
        raise ChainRefusedError("the node reports our commit at another height; not revealing")
    tip = node.tip()
    if tip < committed:
        raise ChainRefusedError(f"the node is behind the commit (tip {tip} < {committed}); not revealing")
    _final_block(node, committed)
    if not first <= tip + 1 <= last:
        raise ChainRefusedError(
            f"the next block {tip + 1} is outside the window {first}..{last}; not revealing"
        )
    return "send"


def reconcile_reveal(out: Path, node: Node) -> str:
    """A reveal was attempted but not confirmed sent (interrupted around the send): ask the chain.
    "included" if our reveal is there; "resend" if our commit is still pending (sending again is safe:
    the chain refuses a duplicate reveal); otherwise refused for the operator to decide."""
    record = _load(out, "commit.json")
    status, value = node.get(f"/transactions/commit/{record['commit_id']}/status")
    if status == 200 and isinstance(value, dict):
        if value.get("status") in INCLUDED_STATUSES and value.get("commit_id") == record["commit_id"]:
            return "included"
        if value.get("status") == "committed_pending_reveal":
            _check_pending(record, value)
            return "resend"
    raise ChainRefusedError(
        "the chain shows neither our pending commit nor our reveal: OPERATOR DECISION needed"
    )


def _inclusion(record: dict, pending: dict, value: dict, node: Node) -> dict:
    """The evidence for OUR included reveal, read from the node now: the reveal height inside the
    window, and the final hashes of the commit block and of the reveal block."""
    value = _check_revealed(record, pending, value)
    return {
        "commit_id": record["commit_id"],
        "station_id": record["station_id"],
        "status": value["status"],
        "committed_at": int(pending["committed_at"]),
        "revealed_at": int(value["revealed_at"]),
        "commit_block_hash": _final_block(node, int(pending["committed_at"])),
        "reveal_block_hash": _final_block(node, int(value["revealed_at"])),
    }


PINNED = ("commit_id", "station_id", "committed_at", "revealed_at", "commit_block_hash", "reveal_block_hash")


def _same_inclusion(saved: dict, fresh: dict) -> bool:
    return all(saved.get(k) == fresh.get(k) for k in PINNED)


def wait_revealed(
    out: Path, node: Node, *, timeout_s: float = 900.0, sleep=time.sleep, clock=time.monotonic
) -> dict:
    """Poll until the node reports OUR reveal included inside the window, with final commit and
    reveal blocks; pin that evidence once. Always read from the node: on a resumed run the fresh
    evidence must match the pinned one (same heights and block hashes), or there is no receipt."""
    record = _load(out, "commit.json")
    pending = _check_pending(record, _load(out, "status_pending.json"))
    deadline = _deadline(pending)
    pinned_path = Path(out) / "status_revealed.json"
    start = clock()
    while True:
        status, value = node.get(f"/transactions/commit/{record['commit_id']}/status")
        if status == 200 and isinstance(value, dict) and value.get("status") in INCLUDED_STATUSES:
            evidence = _inclusion(record, pending, value, node)
            if pinned_path.exists():
                if not _same_inclusion(_load(out, "status_revealed.json"), evidence):
                    raise ChainRefusedError("the node's inclusion differs from the pinned one; no receipt")
                return evidence
            _save(out, "status_revealed.json", {k: evidence[k] for k in PINNED})
            return evidence
        if pinned_path.exists():
            raise ChainRefusedError("the node no longer reports the pinned inclusion; no receipt")
        if node.tip() > deadline:
            raise ChainRefusedError(
                f"the reveal window closed at block {deadline} without our reveal; no receipt"
            )
        if clock() - start > timeout_s:
            raise ChainRefusedError("the reveal was not seen in time; no receipt")
        sleep(POLL_SECONDS)


_RFC3339 = re.compile(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(?:\.(\d{1,9}))?(Z|[+-]\d\d:\d\d)")


def parse_time_ns(text: str) -> np.datetime64:
    """An RFC 3339 time to UTC nanoseconds, exactly (no rounding through datetime's microseconds)."""
    found = _RFC3339.fullmatch(str(text))
    if not found:
        raise ChainRefusedError(f"not an RFC 3339 time: {text!r}")
    base, fraction, zone = found.groups()
    t = np.datetime64(base, "ns") + np.timedelta64(int((fraction or "0").ljust(9, "0")), "ns")
    if zone != "Z":
        sign = 1 if zone[0] == "+" else -1
        t -= sign * (np.timedelta64(int(zone[1:3]), "h") + np.timedelta64(int(zone[4:6]), "m"))
    return t


def format_time_ns(t: np.datetime64) -> str:
    return str(np.datetime_as_string(np.datetime64(t, "ns"), unit="ns")) + "Z"


def record_receipt(out: Path, node: Node) -> dict:
    """Write receipt.json, only after re-reading OUR reveal's inclusion from the node now: heights
    inside the window, and the same commit and reveal block hashes as pinned when it was confirmed."""
    record = _load(out, "commit.json")
    pending = _check_pending(record, _load(out, "status_pending.json"))
    pinned = _load(out, "status_revealed.json")
    status, value = node.get(f"/transactions/commit/{record['commit_id']}/status")
    if status != 200 or not isinstance(value, dict):
        raise ChainRefusedError("the node does not report our reveal now; no receipt")
    evidence = _inclusion(record, pending, value, node)
    if not _same_inclusion(pinned, evidence):
        raise ChainRefusedError("the node's inclusion differs from the pinned one; no receipt")
    committed, deadline = int(pending["committed_at"]), _deadline(pending)
    block_status, block = node.get(f"/chain/block/{committed}")
    if (
        block_status != 200
        or not isinstance(block, dict)
        or block.get("hash") != evidence["commit_block_hash"]
    ):
        raise ChainRefusedError(f"block {committed} changed; no receipt")
    block_time = parse_time_ns(block["timestamp"])

    def tx(step):
        try:
            return check_submitted(out, step)["tx_id"]
        except FileNotFoundError:
            return None  # interrupted around the broadcast; the chain status is the evidence

    receipt = {
        "commit_id": record["commit_id"],
        "commit_tx_id": tx("commit"),
        "reveal_tx_id": tx("reveal"),
        "commit_height": committed,
        "reveal_height": evidence["revealed_at"],
        "reveal_window_last_block": deadline,
        "commit_block_hash": evidence["commit_block_hash"],
        "reveal_block_hash": evidence["reveal_block_hash"],
        "commit_block_time": format_time_ns(block_time),
        "reveal_status": evidence["status"],
        "chain_id": record["chain_id"],
        "network_id": record["network_id"],
        "station_id": record["station_id"],
        "forecaster": record["forecaster"],
        "init": record["init"],
        "content_sha256": record["content_sha256"],
        "manifest_digest": record["manifest_digest"],
        "rule": "a lead counts only if its valid time is at least 15 minutes after the commit block time",
        **counted_leads(np.datetime64(record["init"], "ns"), block_time),
    }
    _save(out, "receipt.json", receipt)
    return receipt
