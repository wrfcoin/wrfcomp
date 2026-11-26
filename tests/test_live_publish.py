"""Deliverables 3-5 of lane I2: publish a revealed issue, check-reveal, the ledger reading on-chain
commits as receipts, and the daily run. Synthetic data, a fake CLI, a fake node and a fake gh only."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from test_live_chain import (
    COMMIT_HEIGHT,
    FAKE_CLI,
    FORECASTER,
    REPO,
    FakeNode,
    H,
    _published_world,  # noqa: F401 - the module fixture the revealed world is copied from
    _run_submit,
)
from test_live_v4 import make_live_world_patches

from wrfcomp.live import chain, ledger, publish
from wrfcomp.live import v4 as live
from wrfcomp.pofs_shadow import Commitment, EpochEvidence

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="live writes are refused on Windows")


def _drop(data: Path) -> None:
    for path in sorted(data.rglob("*"), reverse=True):
        if path.is_dir() and not path.is_symlink():
            path.chmod(0o700)
    shutil.rmtree(data, ignore_errors=True)


@pytest.fixture(scope="module")
def _revealed_world(_published_world, tmp_path_factory):  # noqa: F811
    """One real submit.sh run (fake CLI, fake node) to the receipt, once per module."""
    template, last = _published_world
    base = tmp_path_factory.mktemp("revealed")
    data = base / "wrfcomp-data"
    shutil.copytree(template, data, symlinks=True)
    with pytest.MonkeyPatch.context() as mp:
        make_live_world_patches(mp)
        mp.setenv("WRFCOMP_DATA_DIR", str(data))
        root = live.live_root(data)
        keys = root / "keys"
        keys.mkdir()
        keys.chmod(0o700)
        key = keys / "forecaster.key"
        key.write_text("0" * 64)
        key.chmod(0o600)
        state = base / "state"
        state.mkdir()
        cli = base / "bin" / "wrfcoin"
        cli.parent.mkdir()
        cli.write_text(FAKE_CLI.format(python=sys.executable, state=str(state)))
        cli.chmod(0o700)
        result, calls, out, node, t = _run_submit((data, root, last, key, cli), base)
        node.server.shutdown()
        assert result.returncode == 0, result.stdout + result.stderr
    return data, t


@pytest.fixture
def revealed(_revealed_world, tmp_path, monkeypatch):
    template, t = _revealed_world
    data = tmp_path / "wrfcomp-data"
    shutil.copytree(template, data, symlinks=True)
    make_live_world_patches(monkeypatch)
    monkeypatch.setenv("WRFCOMP_DATA_DIR", str(data))
    root = live.live_root(data)
    receipt = json.loads((chain.chain_dir(root, t) / "receipt.json").read_text())
    yield root, t, receipt
    _drop(data)


def _chain(tmp_path, t, receipt):
    """A fake node holding the same chain the receipt was recorded on, well past 6 confirmations."""
    state = tmp_path / "node_state"
    state.mkdir(exist_ok=True)
    (state / "commit").write_text("broadcast")
    (state / "reveal").write_text("broadcast")
    node = FakeNode(state, t)
    node.revealed_at = receipt["reveal_height"]
    node.reads = 60  # the tip moves to about commit + 30 on the first status read
    return node


def _reader(node):
    """chain.Node for the fake node, without the 3 s spacing (the spacing has its own test)."""
    import unittest.mock

    return unittest.mock.patch.multiple(chain, ALLOWED_NODES=frozenset({node.url}), MIN_REQUEST_GAP_S=0.0)


def _node(node):
    return chain.Node(node.url, sleep=lambda s: None)


# -- pack ---------------------------------------------------------------------------------------------------


def test_pack_writes_one_deterministic_archive_with_no_local_detail(revealed, tmp_path):
    root, t, receipt = revealed
    first = publish.pack(root, t)
    again = publish.pack(root, t)  # idempotent
    assert first == again and first["asset"] == f"wrfcomp-live-v4-{publish.issue_name(t)}.tar.gz"
    assert first["tag"] == f"live-v4-{publish.issue_name(t)[:7]}"
    archive = Path(first["archive"])
    payload = archive.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == first["sha256"]
    assert archive.with_name(archive.name + ".sha256").read_text() == f"{first['sha256']}  {archive.name}\n"
    with tarfile.open(archive) as tar:
        names = tar.getnames()
        members = tar.getmembers()
        publication = json.loads(tar.extractfile(f"{publish.issue_name(t)}/publication.json").read())
    issue = publish.issue_name(t)
    assert {
        f"{issue}/{n}" for n in ("data.json", "manifest.json", "receipt.json", "publication.json")
    } <= set(names)
    assert any(n.startswith(f"{issue}/v4.zarr/") for n in names) and names == sorted(names)
    assert all(m.mtime == 0 and m.uid == 0 and m.gid == 0 for m in members)
    assert (
        set(publication) == set(publish.PUBLICATION_FIELDS)
        and publication["commit_id"] == receipt["commit_id"]
    )
    assert (
        str(tmp_path).encode() not in payload
        and b"signer_key" not in payload
        and b"forecaster.key" not in payload
    )
    # the same issue packed from another copy of the same world gives the same bytes
    other = tmp_path / "other"
    shutil.copytree(root.parent, other / "wrfcomp-data", symlinks=True)
    shutil.rmtree(other / "wrfcomp-data" / "live" / "publish")
    assert publish.pack(live.live_root(other / "wrfcomp-data"), t)["sha256"] == first["sha256"]
    _drop(other)


def test_pack_refuses_without_a_receipt_or_with_another_hash(revealed):
    root, t, receipt = revealed
    path = chain.chain_dir(root, t) / "receipt.json"
    path.chmod(0o600)
    path.write_text(json.dumps({**receipt, "content_sha256": "0" * 64}))
    with pytest.raises(publish.PublishRefusedError, match="differs from the committed one"):
        publish.pack(root, t)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        publish.pack(root, t)
    assert not (root / "publish").exists()


# -- check-reveal --------------------------------------------------------------------------------------------


def test_check_reveal_accepts_the_published_archive_and_records_it(revealed, tmp_path):
    root, t, receipt = revealed
    archive = Path(publish.pack(root, t)["archive"])
    node = _chain(tmp_path, t, receipt)
    with _reader(node):
        result = publish.check_reveal(root, archive, t, _node(node))
        again = publish.check_reveal(root, archive, t, _node(node))  # re-running is fine
    node.server.shutdown()
    assert result["ok"] and result["commit_status_served"] and again["ok"]
    assert result["content_sha256"] == receipt["content_sha256"]
    assert result["counted_leads_hours"] == receipt["counted_leads_hours"]
    recorded = json.loads((root / "verify" / f"{publish.issue_name(t)}.json").read_text())
    assert (
        recorded["commit_id"] == receipt["commit_id"]
        and recorded["archive_sha256"] == result["archive_sha256"]
    )
    assert not any((root / "verify" / ".work").iterdir())  # the work folder is cleaned
    reads = [p for p in node.requests]
    assert all(not p.startswith("/transactions/") or p.startswith("/transactions/commit/") for p in reads)
    assert node.authorization is None  # no bearer


def _repack(root, t, archive, tmp_path, change):
    """Extract ``archive``, apply ``change(folder)``, and pack it again the publisher's way."""
    issue = publish.issue_name(t)
    work = tmp_path / "repack"
    with tarfile.open(archive) as tar:
        tar.extractall(work, filter="data")
    folder = work / issue
    change(folder)
    files = {
        n: (folder / n).read_bytes()
        for n in ("data.json", "manifest.json", "receipt.json", "publication.json")
    }
    out = tmp_path / "tampered.tar.gz"
    out.write_bytes(publish._tar_bytes(issue, folder / "v4.zarr", files))
    return out


def _edit_json(folder, name, **changes):
    path = folder / name
    value = json.loads(path.read_text())
    value.update(changes)
    text = chain.compact(value) + "\n" if name == "data.json" else json.dumps(value)
    path.write_text(text)


def _other_store(folder):
    store = folder / "v4.zarr"
    ds = xr.open_zarr(store, chunks=None).load()
    ds["t2m"] = ds["t2m"] + 0.5
    for var in ds.variables.values():
        var.encoding.clear()
    shutil.rmtree(store)
    ds.to_zarr(store, mode="w-", consolidated=True, zarr_format=2)  # as baselines.io writes stores


@pytest.mark.parametrize(
    "change, message",
    [
        (_other_store, "content hash is not the committed hash"),
        (lambda f: _edit_json(f, "data.json", model_version="2"), "differs from the revealed data"),
        (lambda f: _edit_json(f, "publication.json", nonce="0" * 32), "do not derive the committed id"),
        (lambda f: _edit_json(f, "manifest.json", produced_at=1), "differs from the revealed manifest"),
        (lambda f: _edit_json(f, "receipt.json", commit_block_hash="ee" * 32), "another hash"),
        (lambda f: _edit_json(f, "receipt.json", reveal_height=COMMIT_HEIGHT + 3), "another height"),
        (lambda f: _edit_json(f, "publication.json", forecaster="0x" + "22" * 20), "disagree on forecaster"),
    ],
)
def test_check_reveal_refuses_a_tampered_archive(revealed, tmp_path, change, message):
    root, t, receipt = revealed
    archive = Path(publish.pack(root, t)["archive"])
    tampered = _repack(root, t, archive, tmp_path, change)
    node = _chain(tmp_path, t, receipt)
    with _reader(node), pytest.raises(publish.PublishRefusedError, match=message):
        publish.check_reveal(root, tampered, t, _node(node))
    node.server.shutdown()
    assert not (root / "verify" / f"{publish.issue_name(t)}.json").exists()


@pytest.mark.parametrize("kind", ["absolute", "parent", "link", "other issue"])
def test_check_reveal_refuses_unsafe_archive_members(revealed, tmp_path, kind):
    root, t, receipt = revealed
    issue = publish.issue_name(t)
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w:gz") as tar:
        info = tarfile.TarInfo(
            {
                "absolute": "/etc/x",
                "parent": f"{issue}/../../x",
                "link": f"{issue}/data.json",
                "other issue": "2026-01-01T00Z/data.json",
            }[kind]
        )
        if kind == "link":
            info.type, info.linkname = tarfile.SYMTYPE, "/etc/passwd"
            tar.addfile(info)
        else:
            info.size = 2
            tar.addfile(info, io.BytesIO(b"{}"))
    archive = tmp_path / "unsafe.tar.gz"
    archive.write_bytes(raw.getvalue())
    node = _chain(tmp_path, t, receipt)
    with _reader(node), pytest.raises(publish.PublishRefusedError, match="unexpected path|non-regular"):
        publish.check_reveal(root, archive, t, _node(node))
    node.server.shutdown()


@pytest.mark.parametrize(
    "damage, message, ok",
    [
        ("replaced reveal block", "another hash", False),
        ("replaced commit block", "another hash", False),
        ("no commit tx from the forecaster", "carries no ForecastCommit", False),
        ("status pruned", None, True),
        ("too few blocks after the reveal", "successors, not 6", False),
    ],
)
def test_check_reveal_rereads_the_chain(revealed, tmp_path, damage, message, ok):
    root, t, receipt = revealed
    archive = Path(publish.pack(root, t)["archive"])
    node = _chain(tmp_path, t, receipt)
    if damage == "replaced reveal block":
        node.other_hashes[receipt["reveal_height"]] = "ee" * 32
    elif damage == "replaced commit block":
        node.block_hash = "ee" * 32
    elif damage == "no commit tx from the forecaster":
        node.commit_at = COMMIT_HEIGHT - 1  # the block at the recorded height carries no commit of ours
        node.other_hashes[COMMIT_HEIGHT] = receipt["commit_block_hash"]
    elif damage == "status pruned":
        node.lost = True  # the node no longer serves the commit status (pruned); blocks still check out
        node.tip = COMMIT_HEIGHT + 30
    else:
        node.reads = 2 * (receipt["reveal_height"] - COMMIT_HEIGHT + 1) + 8  # 5 blocks after the reveal
    with _reader(node):
        if ok:
            result = publish.check_reveal(root, archive, t, _node(node))
            assert result["ok"] and result["commit_status_served"] is False
        else:
            with pytest.raises(publish.PublishRefusedError, match=message):
                publish.check_reveal(root, archive, t, _node(node))
    node.server.shutdown()


# -- the ledger ----------------------------------------------------------------------------------------------


def test_the_ledger_deadline_is_exactly_the_per_lead_rule():
    t = np.datetime64("2026-12-02T00", "ns")
    for lead in chain.LEAD_HOURS:
        edge = t + np.timedelta64(lead, "h") - chain.MARGIN
        for offset in (-1, 0, 1):
            block = edge + np.timedelta64(offset, "ns")
            counted = lead in chain.counted_leads(t, block)["counted_leads_hours"]
            late = bool(block >= ledger.lead_deadline(t, lead))  # pofs_shadow: received_at >= init_time
            assert counted != late, (lead, offset)


def test_the_ledger_builds_pofs_evidence_from_a_receipt_rechecked_on_chain(revealed, tmp_path):
    root, t, receipt = revealed
    node = _chain(tmp_path, t, receipt)
    with _reader(node):
        evidence = ledger.issue_evidence(root, t, _node(node))
    node.server.shutdown()
    assert set(evidence) == set(chain.LEAD_HOURS)
    inputs = json.loads((live.issue_dir(root, t) / "inputs.json").read_text())
    for lead, ev in evidence.items():
        assert isinstance(ev, EpochEvidence) and ev.gefs_digest == inputs["gefs_sha256"]
        assert ev.init_time == ledger.lead_deadline(t, lead)
        assert dict(ev.commitments) == {
            FORECASTER: Commitment(
                receipt["content_sha256"], chain.parse_time_ns(receipt["commit_block_time"])
            )
        }
    rows = ledger.summary(evidence)
    assert [r["lead_hours"] for r in rows if r["timely"]] == receipt["counted_leads_hours"]
    assert [r["lead_hours"] for r in rows if not r["timely"]] == receipt["late_leads_hours"]
    assert all(
        not p.startswith("/transactions/") or p.startswith("/transactions/commit/") for p in node.requests
    )


def test_the_ledger_never_trusts_a_receipt_the_chain_does_not_confirm(revealed, tmp_path):
    root, t, receipt = revealed
    node = _chain(tmp_path, t, receipt)
    node.other_hashes[receipt["reveal_height"]] = "ee" * 32
    with _reader(node), pytest.raises(publish.PublishRefusedError, match="another hash"):
        ledger.issue_evidence(root, t, _node(node))
    node.server.shutdown()


def test_two_receipts_for_one_wallet_in_one_issue_are_refused(revealed):
    root, t, receipt = revealed
    with pytest.raises(ValueError, match="two receipts"):
        ledger.lead_evidence(t, 6, [receipt, receipt], "a" * 64)


def test_ledger_reads_are_spaced_under_30_a_minute():
    assert chain.MIN_REQUEST_GAP_S >= 2.0  # 60 / 2 = 30 a minute at most; the gap is 3 s (20 a minute)


# -- the daily run -------------------------------------------------------------------------------------------


DAILY_UV = r"""#!{python}
# Stands in for `uv run --frozen python -P <script> ...`: network steps are logged and skipped; the
# other steps run for real against the fake node.
import json, os, runpy, sys
from pathlib import Path
args = sys.argv[1:]
assert args[:4] == ["run", "--frozen", "python", "-P"], args
script, rest = args[4], args[5:]
with (Path({state!r}) / "daily.log").open("a") as fh:
    fh.write(json.dumps([script, *rest]) + "\n")
assert not any("TOKEN" in k for k in os.environ), "a helper step saw a bearer"
if script == "scripts/live_v4.py" and rest[0] in ("fetch-metar", "fetch-gefs"):
    assert "--network" in rest
    print(json.dumps({{"bytes": 0}}))
    sys.exit(0)
from wrfcomp.live import chain, paths
chain.NODE_URL = {node!r}
chain.ALLOWED_NODES = frozenset({{{node!r}}})
chain.MIN_REQUEST_GAP_S = 0.0
paths.TEMP_DIRS = ()
if script == "scripts/live_v4.py":
    sys.path.insert(0, "tests")
    import pytest
    from test_live_v4 import make_live_world_patches
    make_live_world_patches(pytest.MonkeyPatch())
sys.argv = [script, *rest]
runpy.run_path(script, run_name="__main__")
"""

FAKE_GH = r"""#!{python}
# Stands in for gh: releases are folders under the state folder; every call is logged.
import json, shutil, sys
from pathlib import Path
state = Path({state!r})
args = sys.argv[1:]
with (state / "gh.log").open("a") as fh:
    fh.write(json.dumps(args) + "\n")
assert args[0] == "release" and "--repo" in args and args[args.index("--repo") + 1] == "wrfcoin/wrfcomp", args
verb, tag = args[1], args[2]
release = state / "releases" / tag
if verb == "view":
    if not release.is_dir():
        sys.exit(1)
    if "--json" in args:
        for p in sorted(release.iterdir()):
            print(p.name)
elif verb == "create":
    release.mkdir(parents=True)
elif verb == "upload":
    for f in args[3:args.index("--repo")]:
        shutil.copy(f, release / Path(f).name)
elif verb == "download":
    pattern, dest = args[args.index("--pattern") + 1], Path(args[args.index("--dir") + 1])
    shutil.copy(release / pattern, dest / pattern)
else:
    sys.exit(f"unexpected gh call {{args}}")
"""


def _daily(root, t, tmp_path, node, cycle="00", extra_env=None):
    state = tmp_path / "daily_state"
    state.mkdir(exist_ok=True)
    shim = tmp_path / "daily_shim"
    shim.mkdir(exist_ok=True)
    (shim / "uv").write_text(DAILY_UV.format(python=sys.executable, state=str(state), node=node.url))
    (shim / "gh").write_text(FAKE_GH.format(python=sys.executable, state=str(state)))
    (shim / "uv").chmod(0o700)
    (shim / "gh").chmod(0o700)
    env = {
        "PATH": f"{shim}{os.pathsep}{os.environ['PATH']}",
        "WRFCOMP_DATA_DIR": str(root.parent),
        "HOME": os.environ.get("HOME", str(tmp_path)),
        "PYTHONPATH": str(REPO / "src"),
        "WRFCOMP_FORECASTER": FORECASTER,
        "WRFCOIN_CLI": "/nonexistent/wrfcoin",
        "WRFCOIN_CLI_SHA256": "0" * 64,
        **(extra_env or {}),
    }
    day = str(np.datetime_as_string(t, unit="D"))
    result = subprocess.run(
        ["bash", str(REPO / "scripts" / "live_daily.sh"), cycle, day],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        cwd=tmp_path,
    )
    log = state / "daily.log"
    steps = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    gh_log = state / "gh.log"
    gh = [json.loads(line) for line in gh_log.read_text().splitlines()] if gh_log.exists() else []
    return result, steps, gh, state


def test_the_daily_run_publishes_and_verifies_once_and_can_run_again(revealed, tmp_path):
    root, t, receipt = revealed
    node = _chain(tmp_path, t, receipt)
    first, steps, gh, state = _daily(root, t, tmp_path, node, extra_env={"WRFCOIN_OPERATOR_TOKEN": "x"})
    assert first.returncode == 0, first.stdout + first.stderr
    order = [s[1] if s[0] != "scripts/live_commit.py" else "prepare" for s in steps]
    assert order == ["fetch-metar", "fetch-gefs", "build", "names", "pack", "check-reveal"]  # receipt exists
    verbs = [c[1] for c in gh]
    assert verbs == ["view", "create", "view", "upload", "download"]
    asset = publish.asset_name(t)
    assert sorted(p.name for p in (state / "releases" / publish.release_tag(t)).iterdir()) == [
        asset,
        asset + ".sha256",
    ]
    assert json.loads((root / "verify" / f"{publish.issue_name(t)}.json").read_text())["ok"]
    again, steps, gh, _ = _daily(root, t, tmp_path, node)
    node.server.shutdown()
    assert again.returncode == 0, again.stdout + again.stderr
    assert [c[1] for c in gh][5:] == ["view", "view", "download"]  # nothing created or uploaded again


def test_the_12z_run_only_fetches_and_builds(revealed, tmp_path):
    root, t, receipt = revealed
    node = _chain(tmp_path, t, receipt)
    result, steps, gh, _ = _daily(root, t + 12 * H, tmp_path, node, cycle="12")
    node.server.shutdown()
    assert [s[1] for s in steps][:2] == ["fetch-metar", "fetch-gefs"] and gh == []
    assert all(s[0] == "scripts/live_v4.py" for s in steps)


@pytest.mark.parametrize("claimed, expected", [(False, []), (True, ["resume"])])
def test_the_daily_run_runs_or_resumes_submit_sh(revealed, tmp_path, claimed, expected):
    root, t, receipt = revealed
    out = chain.chain_dir(root, t)
    (out / "receipt.json").unlink()
    if not claimed:
        for name in ("commit.claim", "commit.sent", "reveal.claim", "reveal.sent"):
            if (out / name).is_dir():
                (out / name).rmdir()
    stub = out / "submit.sh"
    stub.chmod(0o700)
    log = tmp_path / "submit_args"
    stub.write_text(f'#!/usr/bin/env bash\nprintf "%s\\n" "$@" > {log}\nexit 7\n')
    node = _chain(tmp_path, t, receipt)
    result, steps, gh, _ = _daily(root, t, tmp_path, node)
    node.server.shutdown()
    assert result.returncode == 7 and gh == []  # nothing is published without a receipt
    assert log.read_text().split() == expected


def test_the_daily_script_refuses_bad_arguments(tmp_path):
    for argv in (["06"], ["00", "2026-13"], []):
        result = subprocess.run(
            ["bash", str(REPO / "scripts" / "live_daily.sh"), *argv],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 2
