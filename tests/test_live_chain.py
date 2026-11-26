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
from test_live_v4 import H, make_live_world

from wrfcomp.live import chain, paths
from wrfcomp.live import v4 as live

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
    key = keys / "forecaster.key"
    key.write_text("0" * 64)
    key.chmod(0o600)
    cli = tmp_path / "bin" / "wrfcoin"
    cli.parent.mkdir()
    cli.write_text("#!/bin/sh\nexit 1\n")
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
    _expect_refusal(published, "mode 0600")
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


FAKE_CLI = r'''#!{python}
"""A stand-in for the wrfcoin CLI: same flags and JSON envelope; logs every call; sends nothing."""
import hashlib, json, sys
from pathlib import Path

state = Path({state!r})
args = sys.argv[1:]
log = state / "cli.log"
with log.open("a") as fh:
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
if {wrong_id}:
    cid = "0" * 64
body = {{"forecaster": opts["forecaster"], "station_id": opts["station-id"], "commit_hash": cid}}
if opts.get("validate_only"):
    data_out = {{"validated": True, "kind": kind, "submitted": False, "commit_id": cid, "body": body}}
else:
    (state / kind).write_text("done")
    data_out = {{"commit_id": cid, "tx_id": f"tx-{{kind}}-1", "result": {{"id": f"tx-{{kind}}-1"}}}}
print(json.dumps({{"ok": True, "command": "forecast", "exit_code": 0, "data": data_out}}))
'''

UV_SHIM = r'''#!{python}
"""Stands in for `uv run --frozen python -P <script> ...`: run the helper with the fake node."""
import runpy, sys
args = sys.argv[1:]
assert args[:5] == ["run", "--frozen", "python", "-P", "scripts/live_commit.py"], args
from wrfcomp.live import chain, paths
chain.NODE_URL = {node!r}
chain.ALLOWED_NODES = frozenset({{{node!r}}})
chain.MIN_REQUEST_GAP_S = 0.0
paths.TEMP_DIRS = ()
sys.argv = ["scripts/live_commit.py", *args[5:]]
runpy.run_path("scripts/live_commit.py", run_name="__main__")
'''


class FakeNode:
    def __init__(self, state, init, *, chain_id="888"):
        self.state, self.init, self.chain_id = state, init, chain_id
        self.requests = []
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
        if path == "/network/info":
            return 200, {"chain_id": self.chain_id, "network_id": "testnet"}
        if path.startswith("/transactions/commit/"):
            cid = path.split("/")[3]
            if (self.state / "reveal").exists():
                return 200, {"commit_id": cid, "status": "revealed_pending_attestation", "revealed_at": 40002}
            if (self.state / "commit").exists():
                return 200, {"commit_id": cid, "status": "committed_pending_reveal", "committed_at": 40000}
            return 404, {"commit_id": cid, "status": "not_found"}
        if path == "/chain/block/40000":
            when = pd.Timestamp(self.init + 4 * H + np.timedelta64(15, "m")).tz_localize("UTC").isoformat()
            return 200, {"ok": True, "data": {"height": 40000, "hash": "ab" * 32, "timestamp": when}}
        return 404, {}


def _run_submit(published, tmp_path, *, wrong_id=False, chain_id="888"):
    data, root, last, key, _ = published
    t = last - 12 * H
    state = tmp_path / "state"
    state.mkdir(exist_ok=True)
    cli = tmp_path / "bin" / "wrfcoin"
    cli.write_text(FAKE_CLI.format(python=sys.executable, state=str(state), wrong_id=wrong_id))
    cli.chmod(0o700)
    node = FakeNode(state, t, chain_id=chain_id)
    shim_dir = tmp_path / "shim"
    shim_dir.mkdir(exist_ok=True)
    (shim_dir / "uv").write_text(UV_SHIM.format(python=sys.executable, node=node.url))
    (shim_dir / "uv").chmod(0o700)
    if not (chain.chain_dir(root, t) / "commit.json").exists():
        _prepare(root, t, key, cli)
    env = {
        "PATH": f"{shim_dir}{os.pathsep}{os.environ['PATH']}",
        "WRFCOMP_DATA_DIR": str(data),
        "WRFCOIN_OPERATOR_TOKEN": "token-must-never-appear",
        "HOME": os.environ.get("HOME", str(tmp_path)),
        "PYTHONPATH": str(REPO / "src"),
    }
    result = subprocess.run(
        ["bash", str(chain.chain_dir(root, t) / "submit.sh")],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    node.server.shutdown()
    calls = (
        [json.loads(line) for line in (state / "cli.log").read_text().splitlines()]
        if (state / "cli.log").exists()
        else []
    )
    return result, calls, chain.chain_dir(root, t), node, t


def test_submit_sh_commits_waits_reveals_and_writes_the_receipt(published, tmp_path):
    result, calls, out, node, t = _run_submit(published, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    kinds = [(c[3], "--validate-only" in c) for c in calls]
    assert kinds == [("commit", True), ("commit", False), ("reveal", False)]
    for c in calls[:2]:
        assert c[c.index("--chain-id") + 1] == "888" and c[c.index("--network-id") + 1] == "testnet"
    assert "--signer-key" not in calls[2] and "--receipt-file" not in calls[2]
    receipt = json.loads((out / "receipt.json").read_text())
    record = json.loads((out / "commit.json").read_text())
    assert receipt["commit_id"] == record["commit_id"] and receipt["commit_height"] == 40000
    assert receipt["commit_tx_id"] == "tx-commit-1" and receipt["reveal_tx_id"] == "tx-reveal-1"
    assert receipt["counted_leads_hours"] == [6, 12, 24, 48] and receipt["late_leads_hours"] == [1, 3]
    assert receipt["reveal_status"] == "revealed_pending_attestation"
    assert all(not p.startswith("/transactions/forecast") for p in node.requests)  # only reads
    everything = (
        result.stdout
        + result.stderr
        + "".join(
            p.read_text()
            for p in out.parent.parent.rglob("*")
            if p.is_file() and p.suffix in (".json", ".sh", ".log")
        )
    )
    assert "token-must-never-appear" not in everything
    # A second run refuses before any CLI call.
    again, calls_again, *_ = _run_submit(published, tmp_path)
    assert again.returncode != 0 and "already attempted" in again.stdout
    assert len(calls_again) == len(calls)


def test_submit_sh_stops_before_committing_when_the_cli_disagrees_on_the_commit_id(published, tmp_path):
    result, calls, out, *_ = _run_submit(published, tmp_path, wrong_id=True)
    assert result.returncode != 0 and "differs from the helper" in result.stderr
    assert [("--validate-only" in c) for c in calls] == [True]
    assert not (out / "commit_result.json").exists()


def test_submit_sh_stops_before_anything_on_another_chain(published, tmp_path):
    result, calls, out, *_ = _run_submit(published, tmp_path, chain_id="777")
    assert result.returncode != 0 and "refusing" in result.stderr
    assert calls == []


def test_wait_pending_refuses_an_unexpected_status_and_times_out(published):
    data, root, last, key, cli = published
    _prepare(root, last - 12 * H, key, cli)
    out = chain.chain_dir(root, last - 12 * H)
    answers = iter([(200, b'{"status": "revealed_attested"}')])
    node = chain.Node(get=lambda url: next(answers), sleep=lambda s: None)
    with pytest.raises(chain.ChainRefusedError, match="unexpected commit status"):
        chain.wait_pending(out, node, sleep=lambda s: None)
    clock = iter(range(0, 10_000, 100))
    node = chain.Node(get=lambda url: (404, b'{"status": "not_found"}'), sleep=lambda s: None)
    with pytest.raises(chain.ChainRefusedError, match="in time"):
        chain.wait_pending(out, node, timeout_s=250, sleep=lambda s: None, clock=lambda: next(clock))
    assert not (out / "status_pending.json").exists()


def test_the_commit_script_refuses_to_start_on_windows(monkeypatch):
    import runpy

    monkeypatch.setattr(paths, "_platform", lambda: "win32")
    monkeypatch.setattr(sys, "argv", ["live_commit.py", "check-cli", "--dir", "x"])
    with pytest.raises(live.LiveRefusedError, match="not supported on Windows"):
        runpy.run_path(str(REPO / "scripts" / "live_commit.py"), run_name="__main__")
