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
import re
import secrets
import shlex
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
from wrfcomp.live.paths import check_path

CHAIN_ID = "888"
NETWORK_ID = "testnet"
NODE_URL = "http://127.0.0.1:8556"
ALLOWED_NODES = frozenset({NODE_URL})
STATION_LABEL = "wrfcomp-live-v4"
VARIABLES = ["msl", "t2m", "td2m", "u10", "v10"]
LEAD_HOURS = (1, 3, 6, 12, 24, 48)
MARGIN = np.timedelta64(15, "m")
REVEAL_WINDOW_BLOCKS = 3
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
    """The throwaway forecaster key: a regular file (not a link) under ``<live>/keys/``, mode 0600.
    Its contents are never opened here; the CLI reads it when the operator runs ``submit.sh``."""
    key = Path(key)
    if key.is_symlink() or not key.is_file():
        raise ChainRefusedError("the signer key must be a regular file, not a link")
    if key.parent.resolve() != (root / "keys").resolve():
        raise ChainRefusedError("the throwaway forecaster key must live in <live>/keys/")
    if key.stat().st_mode & 0o777 != 0o600:
        raise ChainRefusedError("the signer key file must have mode 0600")
    check_path(root, key)
    return key


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
    """Write ``<live>/chain/<issue>/`` (once): data.json, manifest.json, commit.json, submit.sh."""
    from wrfcomp.live import v4 as live

    t = live.check_issue(t)
    if pd.Timestamp(t).hour == 12 and not allow_12z:
        raise ChainRefusedError("only 00Z issues are committed (12Z needs --allow-12z)")
    issue = live.issue_dir(root, t)
    build_path, inputs_path, store_path = issue / "build.json", issue / "inputs.json", issue / "v4.zarr"
    if not build_path.is_file() or not inputs_path.is_file() or not store_path.is_dir():
        raise ChainRefusedError("the issue is not published; build it first")
    build = json.loads(build_path.read_text())
    with xr.open_zarr(store_path, chunks=None) as store:
        digest = content_sha256(store)
    if digest != build["content_sha256"]:
        raise ChainRefusedError("the published store does not match its build.json hash")
    if not _ADDRESS.fullmatch(forecaster or ""):
        raise ChainRefusedError("--forecaster must be 0x followed by 40 hex")
    _check_key_file(root, signer_key)
    cli = Path(cli)
    if not cli.is_file() or _sha256_file(cli) != cli_sha256:
        raise ChainRefusedError("the CLI binary is missing or its sha256 differs from --cli-sha256")
    out = chain_dir(root, t)
    if (out / "commit.json").exists():
        raise ChainRefusedError(f"{out.name} is already prepared; one commit per issue")
    now = now or datetime.now(UTC)
    timestamp = int(now.timestamp())
    nonce = nonce or secrets.token_hex(32)
    lock_sha = _sha256_file(Path(repo) / "HARNESS_LOCK.json")
    data = {
        "forecast_sha256": digest,
        "init": build["init"],
        "inputs_sha256": _sha256_file(inputs_path),
        "model_id": build["model_id"],
        "model_version": build["model_version"],
        "wrfcomp_commit": "v1",
    }
    check_data(data)
    produced = int(np.datetime64(json.loads(inputs_path.read_text())["build_started"], "s").astype(int))
    man = manifest(build["model_id"], lock_sha, digest, produced, horizon_hours=max(LEAD_HOURS))
    record = {
        "chain_id": CHAIN_ID,
        "network_id": NETWORK_ID,
        "node_url": NODE_URL,
        "station_id": STATION_LABEL,
        "forecaster": forecaster,
        "nonce": nonce,
        "timestamp": timestamp,
        "commit_id": commit_id(data, nonce, STATION_LABEL, timestamp),
        "manifest_digest": manifest_digest(man),
        "init": build["init"],
        "content_sha256": digest,
        "signer_key": str(signer_key),
        "cli": str(cli),
        "cli_sha256": cli_sha256,
        "prepared_at": now.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "if_committed_now": counted_leads(t, np.datetime64(now.astimezone(UTC).replace(tzinfo=None), "ns")),
    }
    for name, value in (
        ("data.json", compact(data) + "\n"),
        ("manifest.json", json.dumps(man, separators=(",", ":")) + "\n"),
        ("commit.json", json.dumps(record, indent=1) + "\n"),
        ("submit.sh", submit_script(out, record, repo)),
    ):
        metar.publish_file(out / name, value, root=root)
    return record


def submit_script(out: Path, record: dict, repo: Path) -> str:
    """The one script the operator runs. Every value in it was validated; all are shell-quoted."""
    for key in ("forecaster", "nonce", "station_id", "chain_id", "network_id", "commit_id"):
        if not _SAFE.fullmatch(str(record[key])):
            raise ChainRefusedError(f"unsafe characters in {key}")
    q = shlex.quote
    common = " \\\n    ".join(
        [
            f"--forecaster {q(record['forecaster'])}",
            f"--station-id {q(record['station_id'])}",
            '--data-file "$DIR/data.json"',
            f"--nonce {q(record['nonce'])}",
            f"--timestamp {int(record['timestamp'])}",
        ]
    )
    signing = " \\\n    ".join(
        [
            f"--signer-key {q(record['signer_key'])}",
            f"--chain-id {q(record['chain_id'])}",
            f"--network-id {q(record['network_id'])}",
        ]
    )
    chain_name = f"{record['chain_id']} {record['network_id']}"
    lines = [
        "#!/usr/bin/env bash",
        f"# submit.sh for live v4 issue {record['init']}: testnet {record['chain_id']} only.",
        "# Written by scripts/live_commit.py prepare. Run it once:  bash submit.sh",
        "# It needs the operator bearer in this shell (WRFCOIN_OPERATOR_TOKEN) and never prints it.",
        "# Steps: check the CLI binary; check the node; validate the commit offline; commit;",
        "# wait for committed_pending_reveal; reveal (hash and manifest only); write receipt.json.",
        "set -euo pipefail",
        "umask 077",
        f"DIR={q(str(out))}",
        f"REPO={q(str(Path(repo).resolve()))}",
        f"CLI={q(record['cli'])}",
        f"NODE={q(record['node_url'])}",
        'cd "$REPO"',
        'py() { uv run --frozen python -P scripts/live_commit.py "$@" --dir "$DIR"; }',
        'if [ -e "$DIR/commit_result.json" ] || [ -e "$DIR/commit_result.json.part" ]; then',
        '    echo "a commit was already attempted here: check its status before trying again"; exit 1',
        "fi",
        "",
        'echo "1/7 checking the CLI binary"',
        "py check-cli",
        f'echo "2/7 checking the node is chain {chain_name}"',
        "py check-node",
        'echo "3/7 validating the commit (nothing is sent)"',
        f'"$CLI" --output json forecast commit --validate-only \\\n    {common} \\\n    {signing} \\\n'
        '    --node-url "$NODE" > "$DIR/validate.json"',
        "py check-validate",
        'echo "4/7 committing"',
        f'"$CLI" --output json forecast commit \\\n    {common} \\\n    {signing} \\\n'
        '    --node-url "$NODE" > "$DIR/commit_result.json.part"',
        'mv "$DIR/commit_result.json.part" "$DIR/commit_result.json"',
        "py check-submitted --step commit",
        f'echo "5/7 waiting for committed_pending_reveal (one read every {POLL_SECONDS} s)"',
        "py wait-pending",
        'echo "6/7 revealing the hash and manifest"',
        f'"$CLI" --output json forecast reveal \\\n    {common} \\\n'
        '    --manifest-file "$DIR/manifest.json" --node-url "$NODE" > "$DIR/reveal_result.json.part"',
        'mv "$DIR/reveal_result.json.part" "$DIR/reveal_result.json"',
        "py check-submitted --step reveal",
        'echo "7/7 writing receipt.json"',
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


def _load(out: Path, name: str):
    return json.loads((Path(out) / name).read_text())


def _cli_data(out: Path, name: str) -> dict:
    envelope = _load(out, name)
    if not isinstance(envelope, dict) or envelope.get("ok") is not True:
        raise ChainRefusedError(f"{name}: the CLI reported a failure")
    return envelope["data"]


def check_cli(out: Path) -> None:
    record = _load(out, "commit.json")
    if _sha256_file(Path(record["cli"])) != record["cli_sha256"]:
        raise ChainRefusedError("the CLI binary's sha256 differs from the one recorded at prepare")


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
    if data.get("validated") is not True or data.get("submitted") is not False:
        raise ChainRefusedError("validate.json is not an offline validation")
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


def wait_pending(
    out: Path, node: Node, *, timeout_s: float = 600.0, sleep=time.sleep, clock=time.monotonic
) -> dict:
    """Poll the commit status until ``committed_pending_reveal``; save it as status_pending.json."""
    record = _load(out, "commit.json")
    start = clock()
    while True:
        status, value = node.get(f"/transactions/commit/{record['commit_id']}/status")
        if status == 200 and isinstance(value, dict):
            if value.get("status") == "committed_pending_reveal":
                metar.publish_file(
                    Path(out) / "status_pending.json", json.dumps(value, indent=1) + "\n", root=_live(out)
                )
                return value
            if value.get("status") not in (None, "not_found"):
                raise ChainRefusedError(f"unexpected commit status {value.get('status')!r}; not revealing")
        if clock() - start > timeout_s:
            raise ChainRefusedError(
                "the commit did not reach committed_pending_reveal in time; not revealing"
            )
        sleep(POLL_SECONDS)


def _live(out: Path) -> Path:
    return Path(out).parent.parent


def record_receipt(out: Path, node: Node) -> dict:
    """Write receipt.json: commit and reveal transactions, the commit block, and the leads that count."""
    record = _load(out, "commit.json")
    pending = _load(out, "status_pending.json")
    commit, reveal = check_submitted(out, "commit"), check_submitted(out, "reveal")
    height = int(pending["committed_at"])  # a block height (CommitRecord.committed_at)
    status, block = node.get(f"/chain/block/{height}")
    if status != 200 or not isinstance(block, dict) or int(block.get("height", -1)) != height:
        raise ChainRefusedError(f"the node did not return block {height}")
    block_time = np.datetime64(
        datetime.fromisoformat(block["timestamp"]).astimezone(UTC).replace(tzinfo=None), "ns"
    )
    status, after = node.get(f"/transactions/commit/{record['commit_id']}/status")
    receipt = {
        "commit_id": record["commit_id"],
        "commit_tx_id": commit["tx_id"],
        "reveal_tx_id": reveal["tx_id"],
        "commit_height": height,
        "commit_block_hash": block.get("hash"),
        "commit_block_time": str(np.datetime_as_string(block_time, unit="s")) + "Z",
        "reveal_status": (after or {}).get("status") if status == 200 else None,
        "revealed_at": (after or {}).get("revealed_at") if status == 200 else None,
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
    metar.publish_file(Path(out) / "receipt.json", json.dumps(receipt, indent=1) + "\n", root=_live(out))
    return receipt
