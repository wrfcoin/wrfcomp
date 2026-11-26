"""Commit/reveal of a live v4 issue (wrfcomp.live.chain, scripts/live_commit.py). Synthetic only.

No network, no real CLI, no key material: a fake ``wrfcoin`` CLI and a fake node on a random local
port stand in for them in the end-to-end test of the generated submit.sh."""

from __future__ import annotations

import builtins
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from test_live_v4 import H, make_live_world

from wrfcomp.live import chain, paths
from wrfcomp.live import v4 as live
from wrfcomp.live.dates import LiveYear

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="live writes are refused on Windows")
REPO = Path(__file__).resolve().parents[1]
FORECASTER = "0x" + "1f" * 20


# -- the chain's hashing ---------------------------------------------------------------------------


def test_commit_id_is_sha3_of_the_compact_sorted_preimage():
    data = {"b": "2", "a": "1", "c": "é/x"}
    expected = hashlib.sha3_256(
        '{"a":"1","b":"2","c":"é/x"}||n0||wrfcomp-live-v4||1796169600||forecast'.encode()
    )
    assert chain.commit_id(data, "n0", "wrfcomp-live-v4", 1796169600) == expected.hexdigest()
    assert chain.compact(data) == '{"a":"1","b":"2","c":"é/x"}'  # sorted, compact, UTF-8, "/" kept


def test_manifest_keeps_the_rust_field_order_and_digest():
    m = chain.manifest("gefs-station-corrected-own", "a" * 64, "b" * 64, 1796169600)
    assert tuple(m) == chain.MANIFEST_FIELDS
    text = (
        '{"schema_version":1,"model_id":"gefs-station-corrected-own","input_hash":"'
        + "a" * 64
        + '","output_hash":"'
        + "b" * 64
        + '","produced_at":1796169600,"region":"conus-isd",'
        '"horizon_hours":48,"input_variables":["msl","t2m","td2m","u10","v10"],'
        '"output_variables":["msl","t2m","td2m","u10","v10"]}'
    )
    assert chain.manifest_digest(m) == hashlib.sha3_256(text.encode()).hexdigest()


@pytest.mark.parametrize(
    "change, message",
    [
        ({"schema_version": 2}, "schema_version"),
        ({"model_id": ""}, "model_id"),
        ({"model_id": "x" * 129}, "model_id"),
        ({"input_hash": "z" * 64}, "64 hex"),
        ({"output_hash": "a" * 63}, "64 hex"),
        ({"horizon_hours": 0}, "horizon_hours"),
        ({"horizon_hours": 721}, "horizon_hours"),
        ({"input_variables": []}, "sorted and unique"),
        ({"output_variables": ["t2m", "msl"]}, "sorted and unique"),
        ({"output_variables": ["msl", "msl"]}, "sorted and unique"),
    ],
)
def test_manifest_validation_mirrors_the_chain(change, message):
    m = chain.manifest("m", "a" * 64, "b" * 64, 1)
    m.update(change)
    with pytest.raises(chain.ChainRefusedError, match=message):
        chain.validate_manifest(m)
    reordered = dict(reversed(list(chain.manifest("m", "a" * 64, "b" * 64, 1).items())))
    with pytest.raises(chain.ChainRefusedError, match="declaration order"):
        chain.validate_manifest(reordered)


@pytest.mark.parametrize("bad", [{"schema": "x"}, {"temp_c_centi": "1"}, {"t2m": "1"}, {"a": 1.5}, {}])
def test_the_data_object_never_looks_like_a_reading_or_a_paid_forecast(bad):
    with pytest.raises(chain.ChainRefusedError):
        chain.check_data(bad)


def test_a_lead_counts_only_if_valid_15_minutes_after_the_commit_block():
    t = np.datetime64("2026-12-02T00", "ns")
    on_edge = chain.counted_leads(t, t + 6 * H - np.timedelta64(15, "m"))
    assert on_edge == {"counted_leads_hours": [6, 12, 24, 48], "late_leads_hours": [1, 3]}
    past_edge = chain.counted_leads(t, t + 6 * H - np.timedelta64(15, "m") + np.timedelta64(1, "s"))
    assert past_edge["counted_leads_hours"] == [12, 24, 48]


@pytest.mark.parametrize("url", ["https://wrfcoin.com", "http://127.0.0.1:9999", "http://10.0.0.1:8556"])
def test_reads_go_to_the_ws_node_only(url):
    with pytest.raises(chain.ChainRefusedError, match="local node"):
        chain.Node(url)


def test_reads_are_spaced_at_least_three_seconds_apart():
    slept, now = [], [0.0]
    node = chain.Node(get=lambda url: (200, b"{}"), sleep=lambda s: slept.append(s), clock=lambda: now[0])
    node.get("/a")
    now[0] = 1.0
    node.get("/b")
    assert slept == [pytest.approx(2.0)]


# -- prepare -----------------------------------------------------------------------------------------


@pytest.fixture
def published(tmp_path, monkeypatch):
    """A published synthetic live world (10 days, 4 stations) plus a key file and a CLI file.

    The key file holds no real key (64 zero hex digits); nothing here signs with it."""
    data = tmp_path / "wrfcomp-data"
    root, raws, last = make_live_world(data, monkeypatch)
    live.build_issue(last, data)
    keys = root / "keys"
    keys.mkdir()
    keys.chmod(0o700)
    key = keys / "forecaster.key"
    key.write_text("0" * 64)
    key.chmod(0o600)
    state = tmp_path / "state"
    state.mkdir()
    cli = tmp_path / "bin" / "wrfcoin"
    cli.parent.mkdir()
    cli.write_text(FAKE_CLI.format(python=sys.executable, state=str(state)))  # fixed: its hash is pinned
    cli.chmod(0o700)
    monkeypatch.setenv("WRFCOMP_DATA_DIR", str(data))
    return data, root, last, key, cli


def _prepare(root, t, key, cli, **kw):
    return chain.prepare(
        root,
        t,
        forecaster=kw.pop("forecaster", FORECASTER),
        signer_key=key,
        cli=cli,
        cli_sha256=kw.pop("cli_sha256", chain._sha256_file(cli)),
        repo=REPO,
        now=kw.pop("now", datetime(2026, 3, 10, 4, 20, tzinfo=UTC)),
        **kw,
    )


def test_prepare_writes_the_commit_files_once_and_never_opens_the_key(published, monkeypatch):
    data, root, last, key, cli = published
    t = last - 12 * H  # the 00Z issue
    opened = []
    with pytest.MonkeyPatch.context() as spy_patch:
        for module, name in ((builtins, "open"), (os, "open")):
            original = getattr(module, name)

            def spy(file, *args, _original=original, **kwargs):
                opened.append(os.fspath(file) if isinstance(file, (str, os.PathLike)) else file)
                return _original(file, *args, **kwargs)

            spy_patch.setattr(module, name, spy)
        record = _prepare(root, t, key, cli)
    assert str(key) not in opened and os.fspath(key) not in opened
    out = chain.chain_dir(root, t)
    assert sorted(p.name for p in out.iterdir()) == ["commit.json", "data.json", "manifest.json", "submit.sh"]
    assert not any((out.parent / ".staging").iterdir())
    data_obj = json.loads((out / "data.json").read_text())
    build = json.loads((live.issue_dir(root, t) / "build.json").read_text())
    assert data_obj["forecast_sha256"] == build["content_sha256"] == record["content_sha256"]
    assert record["commit_id"] == chain.commit_id(
        data_obj, record["nonce"], chain.STATION_LABEL, record["timestamp"]
    )
    manifest = json.loads((out / "manifest.json").read_text())
    assert tuple(manifest) == chain.MANIFEST_FIELDS and manifest["output_hash"] == build["content_sha256"]
    assert record["chain_id"] == "888" and record["network_id"] == "testnet"
    assert record["node_url"] == "http://127.0.0.1:8556"
    with pytest.raises(chain.ChainRefusedError, match="already prepared"):
        _prepare(root, t, key, cli)


def _expect_refusal(published, match, **kw):
    data, root, last, key, cli = published
    with pytest.raises(chain.ChainRefusedError, match=match):
        _prepare(root, kw.pop("t", last - 12 * H), kw.pop("key", key), kw.pop("cli", cli), **kw)


def test_prepare_refuses_12z_without_the_flag(published):
    data, root, last, key, cli = published
    _expect_refusal(published, "only 00Z", t=last)
    assert _prepare(root, last, key, cli, allow_12z=True)["init"] == str(
        np.datetime_as_string(last, unit="h")
    )


def test_prepare_refuses_a_bad_forecaster_or_cli(published):
    _expect_refusal(published, "0x followed by 40 hex", forecaster="0x123")
    _expect_refusal(published, "sha256 differs", cli_sha256="0" * 64)


def test_prepare_refuses_a_key_that_is_not_a_private_0600_file_in_keys(published, tmp_path):
    data, root, last, key, cli = published
    key.chmod(0o644)
    _expect_refusal(published, "mode exactly 0600")
    key.chmod(0o600)
    elsewhere = root / "forecaster.key"
    shutil.copy(key, elsewhere)
    elsewhere.chmod(0o600)
    _expect_refusal(published, "live in <live>/keys/", key=elsewhere)
    link = root / "keys" / "linked.key"
    os.symlink(key, link)
    _expect_refusal(published, "not a link", key=link)


def test_prepare_refuses_an_unpublished_or_changed_issue(published):
    data, root, last, key, cli = published
    _expect_refusal(published, "not published", t=last + 12 * H)
    build = live.issue_dir(root, last - 12 * H) / "build.json"
    record = json.loads(build.read_text())
    build.chmod(0o600)
    build.write_text(json.dumps({**record, "content_sha256": "0" * 64}))
    _expect_refusal(published, "does not match its build.json")


def test_submit_script_shell_quotes_and_never_echoes_the_token(published):
    data, root, last, key, cli = published
    _prepare(root, last - 12 * H, key, cli)
    script = (chain.chain_dir(root, last - 12 * H) / "submit.sh").read_text()
    assert "set -euo pipefail" in script and "set -x" not in script
    assert "WRFCOIN_OPERATOR_TOKEN" in script and "$WRFCOIN_OPERATOR_TOKEN" not in script
    assert "--chain-id 888" in script and "--network-id testnet" in script
    reveal = script.split("forecast reveal", 1)[1].split("mv ", 1)[0]
    assert "--signer-key" not in reveal and "--receipt-file" not in script
    assert subprocess.run(["bash", "-n", "-c", script], check=False).returncode == 0


# -- submit.sh end to end, with a fake CLI and a fake node --------------------------------------------


FAKE_CLI = r"""#!{python}
# A stand-in for the wrfcoin CLI: same flags and JSON envelope; logs every call; sends nothing.
import hashlib, json, os, sys, time
from pathlib import Path

state = Path({state!r})
args = sys.argv[1:]
with (state / "cli.log").open("a") as fh:
    fh.write(json.dumps(args) + "\n")
assert args[:2] == ["--output", "json"] and args[2] == "forecast", args
kind = args[3]
opts = {{}}
i = 4
while i < len(args):
    if args[i] == "--validate-only":
        opts["validate_only"] = True
        i += 1
    else:
        opts[args[i].lstrip("-")] = args[i + 1]
        i += 2
data = json.loads(Path(opts["data-file"]).read_text())
compact = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
pre = f"{{compact}}||{{opts['nonce']}}||{{opts['station-id']}}||{{opts['timestamp']}}||forecast"
cid = hashlib.sha3_256(pre.encode()).hexdigest()
opt_file = state / "cli_options.json"
behaviour = json.loads(opt_file.read_text()) if opt_file.exists() else {{}}
if behaviour.get("wrong_id"):
    cid = "0" * 64
body = {{"forecaster": opts["forecaster"], "station_id": opts["station-id"], "commit_hash": cid}}
if kind == "reveal":
    body.update(payload=data, nonce=opts["nonce"], timestamp=int(opts["timestamp"]),
                manifest=json.loads(Path(opts["manifest-file"]).read_text()), verification_receipt=None)
if opts.get("validate_only"):
    crash = behaviour.get("crash_on_reveal_check") and not (state / "crashed_reveal").exists()
    if kind == "reveal" and crash:
        (state / "crashed_reveal").write_text("1")
        sys.exit(8)
    out = {{"validated": True, "kind": kind, "submitted": False, "commit_id": cid, "body": body}}
else:
    if kind == "commit":
        time.sleep(behaviour.get("commit_delay", 0))
    once = state / f"killed_{{kind}}"
    if behaviour.get("kill_before_send") == kind and not once.exists():
        once.write_text("1")
        os.kill(os.getppid(), 9)  # the script dies between its claim and the send
        sys.exit(9)
    if kind == "reveal" and (state / "reveal").exists():
        error = {{"code": "rejected", "message": "duplicate reveal", "hint": None}}
        refused = {{"ok": False, "command": "forecast", "exit_code": 1, "error": error}}
        print(json.dumps(refused, sort_keys=True))
        sys.exit(1)
    (state / kind).write_text("broadcast")
    with (state / "sent.log").open("a") as fh:
        fh.write(kind + "\n")
    if behaviour.get("kill_after_send") == kind and not once.exists():
        once.write_text("1")
        os.kill(os.getppid(), 9)  # sent; the script dies before it records that
        sys.exit(9)
    if kind == "commit" and behaviour.get("crash_after_commit") and not (state / "crashed").exists():
        (state / "crashed").write_text("1")
        sys.exit(9)  # the broadcast happened; the result never reached the script
    out = {{"commit_id": cid, "tx_id": f"tx-{{kind}}-1", "result": {{"id": f"tx-{{kind}}-1"}}}}
print(json.dumps({{"ok": True, "command": "forecast", "exit_code": 0, "data": out}}, sort_keys=True))
"""

UV_SHIM = r"""#!{python}
# Stands in for `uv run --frozen python -P <script> ...`: run the helper against the fake node.
import os, runpy, sys
from pathlib import Path
args = sys.argv[1:]
assert args[:5] == ["run", "--frozen", "python", "-P", "scripts/live_commit.py"], args
with (Path({state!r}) / "helper_env.log").open("a") as fh:
    fh.write(repr(sorted(k for k in os.environ if "TOKEN" in k)) + "\n")
from wrfcomp.live import chain, paths
from wrfcomp.live.dates import LiveYear
chain.NODE_URL = {node!r}
chain.ALLOWED_NODES = frozenset({{{node!r}}})
chain.MIN_REQUEST_GAP_S = 0.0
chain.POLL_SECONDS = 0
paths.TEMP_DIRS = ()
sys.argv = ["scripts/live_commit.py", *args[5:]]
runpy.run_path("scripts/live_commit.py", run_name="__main__")
"""

COMMIT_HEIGHT = 40000


class FakeNode:
    """A tiny chain. The tip advances one block per status read; the commit lands at 40000 with the
    3-block reveal window (last block 40003); the reveal lands per ``reveal``: "next" (the next block),
    "last" (the last eligible block), "never", "late" (one block after the window) or "other" (reported
    for another commit)."""

    def __init__(self, state, init, *, chain_id="888", reveal="next", block_hash="ab" * 32, expiries=(10, 3)):
        self.state, self.init, self.chain_id, self.reveal, self.expiries = (
            state,
            init,
            chain_id,
            reveal,
            expiries,
        )
        when = pd.Timestamp(init + 4 * H + np.timedelta64(15, "m")).tz_localize("UTC")
        self.block_time = when.isoformat()
        self.block_hash, self.tip, self.requests, self.revealed_at = block_hash, COMMIT_HEIGHT - 1, [], None
        self.other_hashes, self.reads, self.lost = {}, 0, False
        node = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                node.requests.append(self.path)
                status, body = node.answer(self.path)
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(body).encode())

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def answer(self, path):
        deadline = COMMIT_HEIGHT + min(3, *self.expiries)
        if path == "/network/info":
            return 200, {"chain_id": self.chain_id, "network_id": "testnet", "chain_height": self.tip}
        if path.startswith("/transactions/commit/"):
            cid = path.split("/")[3]
            if self.lost:
                return 404, {"commit_id": cid, "status": "not_found"}
            self.reads += 1
            self.tip = COMMIT_HEIGHT - 1 + self.reads // 2
            if (self.state / "reveal").exists() and self.revealed_at is None and self.reveal != "never":
                heights = {
                    "next": self.tip + 1,  # a reveal sent at tip T lands at T + 1 at the earliest
                    "last": deadline,
                    "late": deadline + 1,
                    "other": self.tip + 1,
                    "beyond_cap": COMMIT_HEIGHT + 4,
                    "lag": min(self.tip + 2, deadline),
                }
                self.revealed_at = heights[self.reveal]
            if self.revealed_at is not None and self.tip >= min(
                self.revealed_at, COMMIT_HEIGHT + max(self.expiries)
            ):
                return 200, {
                    "commit_id": "f" * 64 if self.reveal == "other" else cid,
                    "station_id": "wrfcomp-live-v4",
                    "status": "revealed_pending_attestation",
                    "revealed_at": self.revealed_at,
                }
            if (self.state / "commit").exists():
                return 200, {
                    "commit_id": cid,
                    "station_id": "wrfcomp-live-v4",
                    "contribution_type": "forecast",
                    "status": "committed_pending_reveal",
                    "committed_at": COMMIT_HEIGHT,
                    "commit_expires_at": COMMIT_HEIGHT + self.expiries[0],
                    "reveal_expires_at": COMMIT_HEIGHT + self.expiries[1],
                }
            return 404, {"commit_id": cid, "status": "not_found"}
        if path.startswith("/chain/block/"):
            height = int(path.rsplit("/", 1)[1])
            if height > self.tip:
                return 404, {}
            block_hash = (
                self.block_hash if height == COMMIT_HEIGHT else self.other_hashes.get(height, "cd" * 32)
            )
            data = {"height": height, "hash": block_hash, "timestamp": self.block_time}
            return 200, {"ok": True, "data": data}
        return 404, {}


def _node(tmp_path, published, **kw):
    state = tmp_path / "state"
    state.mkdir(exist_ok=True)
    return FakeNode(state, published[2] - 12 * H, **kw)


def _run_submit(published, tmp_path, *, mode=None, node=None, **cli_kw):
    data, root, last, key, _ = published
    t = last - 12 * H
    state = tmp_path / "state"
    state.mkdir(exist_ok=True)
    cli = tmp_path / "bin" / "wrfcoin"
    (state / "cli_options.json").write_text(json.dumps(cli_kw))
    node = node or FakeNode(state, t)
    shim_dir = tmp_path / "shim"
    shim_dir.mkdir(exist_ok=True)
    (shim_dir / "uv").write_text(UV_SHIM.format(python=sys.executable, node=node.url, state=str(state)))
    (shim_dir / "uv").chmod(0o700)
    if not chain.chain_dir(root, t).exists():
        _prepare(root, t, key, cli)
    env = {
        "PATH": f"{shim_dir}{os.pathsep}{os.environ['PATH']}",
        "WRFCOMP_DATA_DIR": str(data),
        "WRFCOIN_OPERATOR_TOKEN": "token-must-never-appear",
        "HOME": os.environ.get("HOME", str(tmp_path)),
        "PYTHONPATH": str(REPO / "src"),
    }
    argv = ["bash", str(chain.chain_dir(root, t) / "submit.sh"), *([mode] if mode else [])]
    result = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=300)
    return result, _calls(state), chain.chain_dir(root, t), node, t


def _calls(state):
    log = state / "cli.log"
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


def _broadcasts(calls, kind):
    return [c for c in calls if c[3] == kind and "--validate-only" not in c]


def _sent(state, kind):
    """Broadcasts that actually happened (the fake CLI logs them; a refused duplicate is not one)."""
    log = state / "sent.log"
    return log.read_text().splitlines().count(kind) if log.exists() else 0


def _after_kill(out):
    """The operator's step after a killed run (docs): check nothing runs, remove the active claim."""
    if (out / "active").exists():
        (out / "active").rmdir()


def _no_token_anywhere(result, out, state):
    texts = [result.stdout, result.stderr]
    texts += [p.read_text(errors="replace") for p in out.parent.parent.rglob("*") if p.is_file()]
    assert not any("token-must-never-appear" in t for t in texts)
    log = state / "helper_env.log"
    helper_envs = log.read_text().splitlines() if log.exists() else []
    assert helper_envs and all(line == "[]" for line in helper_envs)  # helper steps run without the bearer


def test_submit_sh_commits_waits_reveals_and_writes_the_receipt(published, tmp_path):
    result, calls, out, node, t = _run_submit(published, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    kinds = [(c[3], "--validate-only" in c) for c in calls]
    assert kinds == [("commit", True), ("commit", False), ("reveal", True), ("reveal", False)]
    for c in calls[:2]:
        assert c[c.index("--chain-id") + 1] == "888" and c[c.index("--network-id") + 1] == "testnet"
    assert all("--signer-key" not in c for c in calls[2:]) and all("--receipt-file" not in c for c in calls)
    receipt = json.loads((out / "receipt.json").read_text())
    record = json.loads((out / "commit.json").read_text())
    assert receipt["commit_id"] == record["commit_id"] and receipt["commit_height"] == COMMIT_HEIGHT
    assert (
        COMMIT_HEIGHT <= receipt["reveal_height"] <= receipt["reveal_window_last_block"] == COMMIT_HEIGHT + 3
    )
    assert receipt["commit_tx_id"] == "tx-commit-1" and receipt["reveal_tx_id"] == "tx-reveal-1"
    assert receipt["counted_leads_hours"] == [6, 12, 24, 48] and receipt["late_leads_hours"] == [1, 3]
    assert all(not p.startswith("/transactions/forecast") for p in node.requests)  # only reads
    _no_token_anywhere(result, out, tmp_path / "state")
    again, calls_again, *_ = _run_submit(published, tmp_path, node=node)
    node.server.shutdown()
    assert again.returncode != 0 and "never sent twice" in again.stdout
    assert len(calls_again) == len(calls)


@pytest.mark.parametrize("reveal, ok", [("last", True), ("never", False), ("late", False), ("other", False)])
def test_a_receipt_only_for_our_reveal_included_inside_the_window(published, tmp_path, reveal, ok):
    """On the last eligible block: a receipt. Still pending when the window closes, reported after it,
    or reported for another commit: no receipt, and a non-zero exit."""
    node = _node(tmp_path, published, reveal=reveal)
    result, calls, out, *_ = _run_submit(published, tmp_path, node=node)
    node.server.shutdown()
    assert len(_broadcasts(calls, "reveal")) == 1
    if ok:
        assert result.returncode == 0, result.stderr
        assert json.loads((out / "receipt.json").read_text())["reveal_height"] == COMMIT_HEIGHT + 3
    else:
        assert result.returncode != 0 and "no receipt" in result.stderr
        assert not (out / "receipt.json").exists()
    _no_token_anywhere(result, out, tmp_path / "state")


def test_no_reveal_is_broadcast_once_the_window_has_closed(published, tmp_path):
    node = _node(tmp_path, published)
    real = node.answer

    def tip_runs_ahead(path):
        status, body = real(path)
        if path == "/network/info" and (node.state / "commit").exists():
            body = {**body, "chain_height": COMMIT_HEIGHT + 3}  # the next block is past the window
        return status, body

    node.answer = tip_runs_ahead
    result, calls, out, *_ = _run_submit(published, tmp_path, node=node)
    node.server.shutdown()
    assert result.returncode != 0 and "outside the window" in result.stderr
    assert _broadcasts(calls, "reveal") == [] and not (out / "receipt.json").exists()


def test_a_wrong_commit_id_in_the_pending_status_stops_the_reveal(published, tmp_path):
    node = _node(tmp_path, published)
    real = node.answer

    def wrong_pending(path):
        status, body = real(path)
        if isinstance(body, dict) and body.get("status") == "committed_pending_reveal":
            body = {**body, "commit_id": "e" * 64}
        return status, body

    node.answer = wrong_pending
    result, calls, *_ = _run_submit(published, tmp_path, node=node)
    node.server.shutdown()
    assert result.returncode != 0 and "another commit" in result.stderr
    assert _broadcasts(calls, "reveal") == []


def test_a_changed_manifest_means_zero_reveal_submissions(published, tmp_path):
    data, root, last, key, cli = published
    t = last - 12 * H
    _prepare(root, t, key, cli)
    path = chain.chain_dir(root, t) / "manifest.json"
    other = json.loads(path.read_text())
    other["produced_at"] += 1  # structurally valid, same commit id, a different manifest
    path.write_text(json.dumps(other, separators=(",", ":")) + "\n")
    result, calls, out, node, _ = _run_submit(published, tmp_path)
    node.server.shutdown()
    assert result.returncode != 0 and "manifest.json differs" in result.stderr
    assert [c for c in calls if c[3] == "reveal"] == []
    assert len(_broadcasts(calls, "commit")) == 1


def test_the_cli_reveal_body_must_equal_the_prepared_reveal(published):
    data, root, last, key, cli = published
    t = last - 12 * H
    _prepare(root, t, key, cli)
    out = chain.chain_dir(root, t)
    record = json.loads((out / "commit.json").read_text())
    body = {
        "forecaster": record["forecaster"],
        "station_id": record["station_id"],
        "payload": json.loads((out / "data.json").read_text()),
        "nonce": record["nonce"],
        "timestamp": record["timestamp"],
        "commit_hash": record["commit_id"],
        "manifest": json.loads((out / "manifest.json").read_text()),
        "verification_receipt": None,
    }
    valid = {"validated": True, "submitted": False, "kind": "reveal", "commit_id": record["commit_id"]}
    good = {"ok": True, "data": {**valid, "body": body}}
    (out / "reveal_validate.json").write_text(json.dumps(good))
    chain.check_reveal_validate(out)
    changes = (
        {"nonce": "x"},
        {"manifest": {**body["manifest"], "region": "elsewhere"}},
        {"verification_receipt": {}},
    )
    for change in changes:
        bad = json.loads(json.dumps(good))
        bad["data"]["body"].update(change)
        (out / "reveal_validate.json").write_text(json.dumps(bad))
        with pytest.raises(chain.ChainRefusedError, match="reveal body differs"):
            chain.check_reveal_validate(out)


def test_two_concurrent_submit_runs_broadcast_one_commit(published, tmp_path):
    data, root, last, key, cli = published
    t = last - 12 * H
    node = _node(tmp_path, published)
    _prepare(root, t, key, cli)
    results = []

    def run():
        results.append(_run_submit(published, tmp_path, node=node, commit_delay=2)[0])

    first = threading.Thread(target=run)
    first.start()
    for _ in range(400):
        if (chain.chain_dir(root, t) / "commit.claim").exists():
            break
        threading.Event().wait(0.05)
    second = threading.Thread(target=run)
    second.start()
    first.join(300)
    second.join(300)
    node.server.shutdown()
    assert sorted(r.returncode for r in results) == [0, 1], [r.stdout + r.stderr for r in results]
    assert any("another submit.sh is running" in r.stdout for r in results)
    assert len(_broadcasts(_calls(node.state), "commit")) == 1


def test_resume_after_an_interrupted_commit_never_rebroadcasts_it(published, tmp_path):
    node = _node(tmp_path, published)
    first, calls, out, *_ = _run_submit(published, tmp_path, node=node, crash_after_commit=True)
    assert (
        first.returncode != 0
        and (out / "commit.claim").is_dir()
        and not (out / "commit_result.json").exists()
    )
    plain, calls, *_ = _run_submit(published, tmp_path, node=node, crash_after_commit=True)
    assert plain.returncode != 0 and "never sent twice" in plain.stdout
    resumed, calls, *_ = _run_submit(published, tmp_path, mode="resume", node=node, crash_after_commit=True)
    node.server.shutdown()
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert len(_broadcasts(calls, "commit")) == 1 and len(_broadcasts(calls, "reveal")) == 1
    receipt = json.loads((out / "receipt.json").read_text())
    assert receipt["commit_tx_id"] is None and receipt["reveal_tx_id"] == "tx-reveal-1"
    _no_token_anywhere(resumed, out, node.state)


def test_resume_from_the_prepared_stage_runs_the_whole_submission(published, tmp_path):
    data, root, last, key, cli = published
    _prepare(root, last - 12 * H, key, cli)
    result, calls, out, node, _ = _run_submit(published, tmp_path, mode="resume")
    node.server.shutdown()
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(_broadcasts(calls, "commit")) == 1 and len(_broadcasts(calls, "reveal")) == 1
    assert (out / "receipt.json").is_file()


def test_submit_sh_stops_before_committing_when_the_cli_disagrees_on_the_commit_id(published, tmp_path):
    result, calls, out, node, _ = _run_submit(published, tmp_path, wrong_id=True)
    node.server.shutdown()
    assert result.returncode != 0 and "differs from the helper" in result.stderr
    assert [("--validate-only" in c) for c in calls] == [True]
    assert not (out / "commit.claim").exists()
    _no_token_anywhere(result, out, tmp_path / "state")


def test_submit_sh_stops_before_anything_on_another_chain(published, tmp_path):
    node = _node(tmp_path, published, chain_id="777")
    result, calls, *_ = _run_submit(published, tmp_path, node=node)
    node.server.shutdown()
    assert result.returncode != 0 and "refusing" in result.stderr
    assert calls == []


def test_submit_sh_rechecks_the_key_folder_before_signing(published, tmp_path):
    data, root, last, key, cli = published
    _prepare(root, last - 12 * H, key, cli)
    (root / "keys").chmod(0o755)
    result, calls, out, node, _ = _run_submit(published, tmp_path)
    node.server.shutdown()
    (root / "keys").chmod(0o700)
    assert result.returncode != 0 and "mode exactly 0700" in result.stderr and calls == []


def test_a_pending_block_hash_gives_no_receipt(published, tmp_path):
    node = _node(tmp_path, published, block_hash="pending")
    result, calls, out, *_ = _run_submit(published, tmp_path, node=node)
    node.server.shutdown()
    assert result.returncode != 0 and "no final hash" in result.stderr and not (out / "receipt.json").exists()


def test_wait_pending_refuses_an_unexpected_status_and_times_out(published):
    data, root, last, key, cli = published
    _prepare(root, last - 12 * H, key, cli)
    out = chain.chain_dir(root, last - 12 * H)

    def node_with(status_answer):
        tips = iter(range(100, 10_000))

        def get(url):
            if url.endswith("/network/info"):
                return 200, json.dumps({"chain_height": next(tips)}).encode()
            return status_answer

        return chain.Node(get=get, sleep=lambda s: None)

    with pytest.raises(chain.ChainRefusedError, match="unexpected commit status"):
        chain.wait_pending(out, node_with((200, b'{"status": "revealed_attested"}')), sleep=lambda s: None)
    with pytest.raises(chain.ChainRefusedError, match="OPERATOR DECISION"):
        chain.wait_pending(out, node_with((404, b'{"status": "not_found"}')), sleep=lambda s: None)
    assert not (out / "status_pending.json").exists()


def test_the_commit_script_refuses_to_start_on_windows(monkeypatch):
    import runpy

    monkeypatch.setattr(paths, "_platform", lambda: "win32")
    monkeypatch.setattr(sys, "argv", ["live_commit.py", "check-cli", "--dir", "x"])
    with pytest.raises(live.LiveRefusedError, match="not supported on Windows"):
        runpy.run_path(str(REPO / "scripts" / "live_commit.py"), run_name="__main__")


# -- binding the metadata to the published store ------------------------------------------------------


def _store_and_records(root, t):
    issue = live.issue_dir(root, t)
    return issue / "v4.zarr", issue / "build.json", issue / "inputs.json"


def _rewrite(path, change):
    record = json.loads(path.read_text())
    record.update(change)
    path.chmod(0o600)
    path.write_text(json.dumps(record))


@pytest.mark.parametrize(
    "where, change, message",
    [
        ("build", {"init": "2026-03-09T00"}, "another issue"),
        ("inputs", {"init": "2026-03-09T00"}, "another issue"),
        ("build", {"model_id": "gefs-station-corrected"}, "model differs"),
        ("build", {"model_version": "2"}, "model differs"),
    ],
)
def test_prepare_refuses_a_record_that_disagrees_with_the_store(published, where, change, message):
    data, root, last, key, cli = published
    _, build, inputs = _store_and_records(root, last - 12 * H)
    _rewrite(build if where == "build" else inputs, change)
    _expect_refusal(published, message)


@pytest.mark.parametrize("attrs", [{"model_version": "2"}, {"model_id": "x"}])
def test_prepare_refuses_a_store_whose_own_model_attributes_are_not_v4(published, attrs):
    """content_sha256 deliberately ignores attributes: same hash, another model. Refused on the attrs,
    also when the build record was changed to agree with them."""
    import zarr

    data, root, last, key, cli = published
    store, build, _ = _store_and_records(root, last - 12 * H)
    zarr.open_group(str(store), mode="r+").attrs.update(attrs)
    xr.open_zarr(store, chunks=None).close()
    _rewrite(build, attrs)
    _expect_refusal(published, "not candidate v4")


def test_prepare_refuses_another_issues_store_with_a_matching_hash(published):
    data, root, last, key, cli = published
    t = last - 12 * H
    other_store, other_build, _ = _store_and_records(root, t - 12 * H)
    store, build, _ = _store_and_records(root, t)
    shutil.rmtree(store)
    shutil.copytree(other_store, store)
    _rewrite(build, {"content_sha256": json.loads(other_build.read_text())["content_sha256"]})
    _expect_refusal(published, "not the requested issue")


def test_prepare_refuses_a_key_folder_that_is_not_exactly_0700(published):
    data, root, last, key, cli = published
    for mode in (0o755, 0o750, 0o701):
        (root / "keys").chmod(mode)
        _expect_refusal(published, "mode exactly 0700")
    (root / "keys").chmod(0o700)
    key.chmod(0o640)
    _expect_refusal(published, "mode exactly 0600")


def test_an_interrupted_prepare_leaves_nothing_and_can_run_again(published):
    data, root, last, key, cli = published
    t = last - 12 * H
    original = chain.metar.publish_file

    def fail_on_script(path, text, *, root):
        if Path(path).name == "submit.sh":
            raise OSError("power cut")
        return original(path, text, root=root)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(chain.metar, "publish_file", fail_on_script)
        with pytest.raises(OSError, match="power cut"):
            _prepare(root, t, key, cli)
    out = chain.chain_dir(root, t)
    assert not out.exists() and not any((out.parent / ".staging").iterdir())
    _prepare(root, t, key, cli)
    assert (out / "submit.sh").is_file()


def test_exact_nanosecond_timeliness_survives_the_receipt():
    t = np.datetime64("2026-12-02T00", "ns")
    edge = t + 6 * H - np.timedelta64(15, "m")
    for offset_ns, counted in ((0, True), (1, False), (1000, False)):
        text = pd.Timestamp(edge + np.timedelta64(offset_ns, "ns")).tz_localize("UTC").isoformat()
        block = chain.parse_time_ns(text)
        assert block == edge + np.timedelta64(offset_ns, "ns")
        leads = chain.counted_leads(t, block)
        assert (6 in leads["counted_leads_hours"]) is counted
        stored = chain.format_time_ns(block)
        assert chain.counted_leads(t, chain.parse_time_ns(stored)) == leads  # recomputable from the receipt
    shifted = chain.parse_time_ns("2026-12-02T07:45:00.000000001+02:00")
    assert shifted == np.datetime64("2026-12-02T05:45:00.000000001", "ns")
    with pytest.raises(chain.ChainRefusedError):
        chain.parse_time_ns("2026-12-02 05:45:00")


# -- round 2 regressions ---------------------------------------------------------------------------------


def test_a_reveal_body_in_core4s_sorted_key_order_is_accepted(published):
    """core4 prints the reveal body (manifest included) from a serde_json Value: keys sorted. That is
    the same manifest; only a different one is refused."""
    data, root, last, key, cli = published
    t = last - 12 * H
    _prepare(root, t, key, cli)
    out = chain.chain_dir(root, t)
    record = json.loads((out / "commit.json").read_text())
    body = {
        "forecaster": record["forecaster"],
        "station_id": record["station_id"],
        "payload": json.loads((out / "data.json").read_text()),
        "nonce": record["nonce"],
        "timestamp": record["timestamp"],
        "commit_hash": record["commit_id"],
        "manifest": json.loads((out / "manifest.json").read_text()),
        "verification_receipt": None,
    }
    valid = {"validated": True, "submitted": False, "kind": "reveal", "commit_id": record["commit_id"]}
    (out / "reveal_validate.json").write_text(
        json.dumps({"ok": True, "data": {**valid, "body": body}}, sort_keys=True)
    )
    assert list(json.loads((out / "reveal_validate.json").read_text())["data"]["body"]["manifest"]) == sorted(
        chain.MANIFEST_FIELDS
    )
    chain.check_reveal_validate(out)
    with pytest.raises(chain.ChainRefusedError, match="wrong fields"):
        chain.canonical_manifest({**body["manifest"], "extra": 1})


def _move_store_init(store, build, sha, new_init):
    ds = xr.open_zarr(store, chunks=None).load()
    ds = ds.assign_coords(init_time=[np.datetime64(new_init, "ns")])
    for var in ds.variables.values():
        var.encoding.clear()
    shutil.rmtree(store)
    LiveYear(2026, root=store.parent.parent).write(ds, store)
    digest = chain.content_sha256(xr.open_zarr(store, chunks=None))
    _rewrite(build, {"content_sha256": digest})
    sha.chmod(0o600)
    sha.write_text(f"{digest}  v4.zarr\n")


def test_prepare_refuses_a_store_at_another_instant_of_the_same_hour(published):
    data, root, last, key, cli = published
    t = last - 12 * H
    store, build, _ = _store_and_records(root, t)
    _move_store_init(store, build, store.with_name("v4.sha256"), t + np.timedelta64(30, "m"))
    _expect_refusal(published, "its init differs")


def test_prepare_refuses_a_published_hash_file_that_disagrees(published):
    data, root, last, key, cli = published
    sha = live.issue_dir(root, last - 12 * H) / "v4.sha256"
    sha.chmod(0o600)
    sha.write_text("0" * 64 + "  v4.zarr\n")
    _expect_refusal(published, "v4.sha256 does not match")


def test_the_reveal_window_is_commit_plus_one_to_plus_three_even_with_later_expiries():
    pending = {"committed_at": 100, "commit_expires_at": 106, "reveal_expires_at": 112}
    assert chain.reveal_window(pending) == (101, 103)
    assert chain.reveal_window({**pending, "reveal_expires_at": 102}) == (101, 102)
    with pytest.raises(chain.ChainRefusedError):
        chain.reveal_window({**pending, "reveal_expires_at": 100})


def _checkpoints(out, record, pending, revealed):
    (out / "status_pending.json").write_text(json.dumps(pending))
    (out / "status_revealed.json").write_text(json.dumps(revealed))


@pytest.mark.parametrize(
    "change, message",
    [
        ({"revealed_at": 104}, "not included inside the window"),
        ({"revealed_at": 100}, "not included inside the window"),
        ({"commit_id": "e" * 64}, "another reveal"),
        ({"station_id": "somewhere"}, "another reveal"),
        ({"status": "committed_pending_reveal"}, "not reported as included"),
    ],
)
def test_record_validates_the_saved_evidence_itself(published, change, message):
    data, root, last, key, cli = published
    t = last - 12 * H
    _prepare(root, t, key, cli)
    out = chain.chain_dir(root, t)
    record = json.loads((out / "commit.json").read_text())
    base = {"commit_id": record["commit_id"], "station_id": record["station_id"]}
    pending = {
        **base,
        "contribution_type": "forecast",
        "status": "committed_pending_reveal",
        "committed_at": 100,
        "commit_expires_at": 106,
        "reveal_expires_at": 112,
    }
    revealed = {**base, "status": "revealed_pending_attestation", "revealed_at": 103, **change}
    pinned = {
        **base,
        "committed_at": 100,
        "revealed_at": 103,
        "commit_block_hash": "ab" * 32,
        "reveal_block_hash": "cd" * 32,
    }
    _checkpoints(out, record, pending, pinned)

    def get(url):
        if "/chain/block/" in url:
            height = int(url.rsplit("/", 1)[1])
            digest = ("ab" if height == 100 else "cd") * 32
            block = {"height": height, "hash": digest, "timestamp": "2026-03-10T04:15:00Z"}
            return 200, json.dumps({"ok": True, "data": block}).encode()
        return 200, json.dumps(revealed).encode()

    node = chain.Node(get=get, sleep=lambda s: None)
    with pytest.raises(chain.ChainRefusedError, match=message):
        chain.record_receipt(out, node)
    assert not (out / "receipt.json").exists()


def test_a_reveal_beyond_commit_plus_three_gets_no_receipt_end_to_end(published, tmp_path):
    node = _node(tmp_path, published, reveal="beyond_cap", expiries=(6, 12))
    result, calls, out, *_ = _run_submit(published, tmp_path, node=node)
    node.server.shutdown()
    assert result.returncode != 0 and "no receipt" in result.stderr and not (out / "receipt.json").exists()


def test_resume_from_the_included_stage(published, tmp_path):
    """Interrupted after the commit was included (status_pending.json saved), before the reveal."""
    node = _node(tmp_path, published)
    first, calls, out, *_ = _run_submit(published, tmp_path, node=node, crash_on_reveal_check=True)
    assert (
        first.returncode != 0
        and (out / "status_pending.json").exists()
        and not (out / "reveal.claim").exists()
    )
    resumed, calls, *_ = _run_submit(
        published, tmp_path, mode="resume", node=node, crash_on_reveal_check=True
    )
    node.server.shutdown()
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert len(_broadcasts(calls, "commit")) == 1 and len(_broadcasts(calls, "reveal")) == 1
    assert (out / "receipt.json").is_file()


def test_resume_after_the_node_recovers_and_again_after_the_receipt(published, tmp_path):
    """The commit block had no final hash yet: nothing is revealed. After the node recovers, resume
    reveals and writes the receipt with one commit and one reveal broadcast; a further resume changes
    nothing."""
    node = _node(tmp_path, published, block_hash="pending")
    first, calls, out, *_ = _run_submit(published, tmp_path, node=node)
    assert first.returncode != 0 and "no final hash" in first.stderr and _sent(node.state, "reveal") == 0
    node.block_hash = "ab" * 32
    resumed, calls, *_ = _run_submit(published, tmp_path, mode="resume", node=node)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert _sent(node.state, "commit") == 1 and _sent(node.state, "reveal") == 1
    receipt = (out / "receipt.json").read_text()
    again, *_ = _run_submit(published, tmp_path, mode="resume", node=node)
    node.server.shutdown()
    assert again.returncode == 0, again.stdout + again.stderr
    assert _sent(node.state, "commit") == 1 and _sent(node.state, "reveal") == 1
    assert (out / "receipt.json").read_text() == receipt


# -- round 3 regressions ---------------------------------------------------------------------------------


def test_no_reveal_when_the_node_is_behind_the_commit(published, tmp_path):
    """A stale or wrong node: its tip is below the saved commit height. Nothing is broadcast."""
    node = _node(tmp_path, published)
    real = node.answer

    def behind(path):
        status, body = real(path)
        if path == "/network/info" and (node.state / "commit").exists():
            body = {**body, "chain_height": COMMIT_HEIGHT - 1}
        return status, body

    node.answer = behind
    result, calls, out, *_ = _run_submit(published, tmp_path, node=node)
    node.server.shutdown()
    assert result.returncode != 0 and "behind the commit" in result.stderr
    assert _sent(node.state, "reveal") == 0 and not (out / "receipt.json").exists()


def test_no_reveal_when_the_node_reports_the_commit_at_another_height(published, tmp_path):
    node = _node(tmp_path, published)
    real = node.answer
    seen = {"n": 0}

    def moved(path):
        status, body = real(path)
        if isinstance(body, dict) and body.get("status") == "committed_pending_reveal":
            seen["n"] += 1
            if seen["n"] > 1:  # the saved checkpoint says 40000; the node's view now differs
                body = {**body, "committed_at": COMMIT_HEIGHT - 5}
        return status, body

    node.answer = moved
    result, calls, *_ = _run_submit(published, tmp_path, node=node)
    node.server.shutdown()
    assert result.returncode != 0 and "another height" in result.stderr and _sent(node.state, "reveal") == 0


def test_resume_after_a_kill_between_the_reveal_claim_and_the_send(published, tmp_path):
    """reveal.claim exists, nothing was sent: resume asks the chain (commit still pending), sends the
    reveal, and finishes with one commit and one reveal broadcast."""
    node = _node(tmp_path, published, expiries=(10, 10))
    first, calls, out, *_ = _run_submit(published, tmp_path, node=node, kill_before_send="reveal")
    assert first.returncode != 0 and (out / "reveal.claim").is_dir() and not (out / "reveal.sent").exists()
    assert _sent(node.state, "reveal") == 0
    _after_kill(out)
    resumed, *_ = _run_submit(published, tmp_path, mode="resume", node=node, kill_before_send="reveal")
    node.server.shutdown()
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert "asking the chain" in resumed.stdout
    assert _sent(node.state, "commit") == 1 and _sent(node.state, "reveal") == 1
    assert (out / "receipt.json").is_file() and (out / "reveal.sent").is_dir()


def test_resume_after_a_kill_right_after_the_reveal_send_does_not_send_again(published, tmp_path):
    """Sent, but the script died before recording it: the chain shows our reveal; no second send."""
    node = _node(tmp_path, published, expiries=(10, 10))
    first, calls, out, *_ = _run_submit(published, tmp_path, node=node, kill_after_send="reveal")
    assert first.returncode != 0 and not (out / "reveal.sent").exists() and _sent(node.state, "reveal") == 1
    _after_kill(out)
    resumed, calls, *_ = _run_submit(published, tmp_path, mode="resume", node=node, kill_after_send="reveal")
    node.server.shutdown()
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert _sent(node.state, "reveal") == 1 and (out / "receipt.json").is_file()


def test_a_resend_refused_as_a_duplicate_is_settled_by_the_chain(published, tmp_path):
    """Sent and killed before recording; the chain has not shown the reveal yet, so resume sends it
    again; the chain refuses the duplicate; the inclusion it then reports decides."""
    node = _node(tmp_path, published, reveal="lag", expiries=(10, 10))
    first, calls, out, *_ = _run_submit(published, tmp_path, node=node, kill_after_send="reveal")
    assert first.returncode != 0 and _sent(node.state, "reveal") == 1
    _after_kill(out)
    node.revealed_at = None  # the reveal is not visible yet when resume asks
    resumed, calls, *_ = _run_submit(published, tmp_path, mode="resume", node=node, kill_after_send="reveal")
    node.server.shutdown()
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert "a resend may be refused as a duplicate" in resumed.stdout
    assert _sent(node.state, "reveal") == 1 and (out / "receipt.json").is_file()


def test_a_commit_interrupted_before_its_send_is_never_sent_by_resume(published, tmp_path):
    """commit.claim exists, nothing was sent: resume never rebroadcasts the commit; it reports that
    the operator must decide."""
    node = _node(tmp_path, published)
    first, calls, out, *_ = _run_submit(published, tmp_path, node=node, kill_before_send="commit")
    assert first.returncode != 0 and (out / "commit.claim").is_dir() and _sent(node.state, "commit") == 0
    _after_kill(out)
    resumed, *_ = _run_submit(published, tmp_path, mode="resume", node=node, kill_before_send="commit")
    node.server.shutdown()
    assert resumed.returncode != 0 and "OPERATOR DECISION" in resumed.stderr
    assert _sent(node.state, "commit") == 0 and not (out / "receipt.json").exists()


@pytest.mark.parametrize(
    "damage", ["replacement reveal block", "replacement commit block", "commitment lost"]
)
def test_resume_rechecks_pinned_inclusion_before_any_receipt(published, tmp_path, damage):
    """Interrupted after the inclusion was pinned, before the receipt. On resume the node shows a
    replacement block, or no longer has the commitment: no receipt."""
    node = _node(tmp_path, published)
    done, calls, out, *_ = _run_submit(published, tmp_path, node=node)
    assert done.returncode == 0, done.stderr
    pinned = json.loads((out / "status_revealed.json").read_text())
    (out / "receipt.json").unlink()  # as if the run stopped right before writing it
    if damage == "replacement reveal block":
        node.other_hashes[pinned["revealed_at"]] = "ee" * 32
    elif damage == "replacement commit block":
        node.block_hash = "ee" * 32
    else:
        node.lost = True
    resumed, *_ = _run_submit(published, tmp_path, mode="resume", node=node)
    node.server.shutdown()
    assert resumed.returncode != 0 and "no receipt" in resumed.stderr
    assert not (out / "receipt.json").exists()
    assert _sent(node.state, "commit") == 1 and _sent(node.state, "reveal") == 1


def test_only_the_included_statuses_core4_defines_count():
    record = {"commit_id": "a" * 64, "station_id": "wrfcomp-live-v4"}
    pending = {"committed_at": 100, "commit_expires_at": 106, "reveal_expires_at": 112}
    for status in ("revealed_pending_attestation", "revealed_attested"):
        chain._check_revealed(record, pending, {**record, "status": status, "revealed_at": 101})
    with pytest.raises(chain.ChainRefusedError, match="not reported as included"):
        chain._check_revealed(record, pending, {**record, "status": "revealed_something", "revealed_at": 101})
