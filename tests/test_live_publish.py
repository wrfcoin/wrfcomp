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


def _chain(tmp_path, t, receipt, **kw):
    """A fake node holding the same chain the receipt was recorded on, well past 6 confirmations, with
    optional damage: ``chain_id``, ``lost`` (status not served), ``reveal_hash`` / ``commit_hash``
    (replaced blocks), ``no_commit_tx``, ``tip_from`` (the tip this many blocks after the reveal),
    ``station`` (status names another station), ``served_commit`` (status served only for that
    commit id), ``flip_reveal_hash_on_second_pass``."""
    state = tmp_path / "node_state"
    state.mkdir(exist_ok=True)
    (state / "commit").write_text("broadcast")
    (state / "reveal").write_text("broadcast")
    node = FakeNode(state, t, chain_id=kw.get("chain_id", "888"))
    revealed = receipt["reveal_height"]
    node.revealed_at = revealed
    node.reads = 60  # the tip moves to about commit + 30 on the first status read
    node.lost = kw.get("lost", False)
    if "reveal_hash" in kw:
        node.other_hashes[revealed] = kw["reveal_hash"]
    if "commit_hash" in kw:
        node.block_hash = kw["commit_hash"]
    if kw.get("no_commit_tx"):
        node.commit_at = COMMIT_HEIGHT - 1  # the block at the recorded height carries no commit of ours
        node.other_hashes[COMMIT_HEIGHT] = receipt["commit_block_hash"]
    if "tip_from" in kw:  # after the first status read, tip = revealed + tip_from
        node.reads = 2 * (revealed + kw["tip_from"] - COMMIT_HEIGHT + 1) - 1
    real, seen = node.answer, {"status": 0}

    def answer(path):
        if path.startswith("/transactions/commit/"):
            seen["status"] += 1
            cid = path.split("/")[3]
            if kw.get("served_commit") and cid != kw["served_commit"]:
                return 404, {"commit_id": cid, "status": "not_found"}
            if kw.get("flip_reveal_hash_on_second_pass") and seen["status"] == 2:
                node.other_hashes[revealed] = "ee" * 32
        status, body = real(path)
        if "station" in kw and isinstance(body, dict) and "station_id" in body:
            body = {**body, "station_id": kw["station"]}
        return status, body

    node.answer = answer
    return node


def _reader(node):
    """chain.Node for the fake node, without the 3 s spacing (the spacing has its own test)."""
    import unittest.mock

    return unittest.mock.patch.multiple(chain, ALLOWED_NODES=frozenset({node.url}), MIN_REQUEST_GAP_S=0.0)


def _node(node):
    return chain.Node(node.url, sleep=lambda s: None)


def _pack(root, t, tmp_path, receipt, **node_kw):
    node = _chain(tmp_path, t, receipt, **node_kw)
    try:
        with _reader(node):
            return publish.pack(root, t, _node(node))
    finally:
        node.server.shutdown()


# -- pack ---------------------------------------------------------------------------------------------------


def test_pack_writes_one_deterministic_archive_with_no_local_detail(revealed, tmp_path):
    root, t, receipt = revealed
    first = _pack(root, t, tmp_path, receipt)
    again = _pack(root, t, tmp_path, receipt)  # idempotent
    assert first == again and first["asset"] == f"wrfcomp-live-v4-{publish.issue_name(t)}.tar.gz"
    assert first["tag"] == f"live-v4-{publish.issue_name(t)[:7]}"
    archive = Path(first["archive"])
    payload = archive.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == first["sha256"]
    assert archive.with_name(archive.name + ".sha256").read_text() == f"{first['sha256']}  {archive.name}\n"
    issue = publish.issue_name(t)
    with tarfile.open(archive) as tar:
        names = tar.getnames()
        members = tar.getmembers()
        publication = json.loads(tar.extractfile(f"{issue}/publication.json").read())
    wanted = {f"{issue}/{n}" for n in ("data.json", "manifest.json", "receipt.json", "publication.json")}
    assert wanted <= set(names)
    assert any(n.startswith(f"{issue}/v4.zarr/") for n in names) and names == sorted(names)
    assert all(m.mtime == 0 and m.uid == 0 and m.gid == 0 for m in members)
    assert set(publication) == set(publish.PUBLICATION_FIELDS)
    assert publication["commit_id"] == receipt["commit_id"]
    for private in (str(tmp_path).encode(), b"signer_key", b"forecaster.key"):
        assert private not in payload
    other = tmp_path / "other"  # the same issue packed from another copy gives the same bytes
    shutil.copytree(root.parent, other / "wrfcomp-data", symlinks=True)
    shutil.rmtree(other / "wrfcomp-data" / "live" / "publish")
    assert _pack(live.live_root(other / "wrfcomp-data"), t, tmp_path, receipt)["sha256"] == first["sha256"]
    _drop(other)


def _rewrite_receipt(root, t, **changes):
    path = chain.chain_dir(root, t) / "receipt.json"
    path.chmod(0o600)
    path.write_text(json.dumps({**json.loads(path.read_text()), **changes}))


@pytest.mark.parametrize(
    "change, node_kw, message",
    [
        ({"content_sha256": "0" * 64}, {}, "differs from the revealed data|publication.json and receipt"),
        ({"confirmations_required": 0}, {}, "6-block confirmation rule"),
        ({"chain_id": "777"}, {}, "not for testnet 888"),
        ({"reveal_height": True}, {}, "not a positive block height"),
        ({}, {"tip_from": 5}, "successor blocks, not 6"),
        ({}, {"reveal_hash": "ee" * 32}, "another hash"),
        ({}, {"lost": True}, "inconclusive"),
        ({}, {"chain_id": "777"}, "not 888 testnet"),
    ],
)
def test_pack_publishes_nothing_unless_files_and_chain_confirm_the_reveal_now(
    revealed, tmp_path, change, node_kw, message
):
    """Finding 1: a receipt file alone never publishes: an unconfirmed, reorged or unserved reveal, a
    wrong chain or a doctored receipt is refused before any archive exists."""
    root, t, receipt = revealed
    if change:
        _rewrite_receipt(root, t, **change)
    with pytest.raises(publish.PublishRefusedError, match=message):
        _pack(root, t, tmp_path, receipt, **node_kw)
    assert not (root / "publish").exists()


def test_pack_refuses_without_a_receipt(revealed, tmp_path):
    root, t, receipt = revealed
    (chain.chain_dir(root, t) / "receipt.json").unlink()
    with pytest.raises(FileNotFoundError):
        _pack(root, t, tmp_path, receipt)


# -- check-reveal --------------------------------------------------------------------------------------------


def test_check_reveal_accepts_the_published_archive_and_records_it(revealed, tmp_path):
    root, t, receipt = revealed
    archive = Path(_pack(root, t, tmp_path, receipt)["archive"])
    node = _chain(tmp_path, t, receipt)
    with _reader(node):
        result = publish.check_reveal(root, archive, t, _node(node))
        again = publish.check_reveal(root, archive, t, _node(node))  # re-running is fine
    node.server.shutdown()
    assert result["ok"] and result["commit_status_served"] and again["ok"]
    assert result["content_sha256"] == receipt["content_sha256"]
    assert result["counted_leads_hours"] == receipt["counted_leads_hours"]
    recorded = json.loads((root / "verify" / f"{publish.issue_name(t)}.json").read_text())
    assert recorded["commit_id"] == receipt["commit_id"]
    assert recorded["archive_sha256"] == result["archive_sha256"]
    assert not any((root / "verify" / ".work").iterdir())  # the work folder is cleaned
    assert all(
        not p.startswith("/transactions/") or p.startswith("/transactions/commit/") for p in node.requests
    )
    assert node.authorization is None  # no bearer
    status_reads = [p for p in node.requests if p.startswith("/transactions/commit/")]
    assert len(status_reads) == 4  # two passes per check, two checks


def _repack(t, archive, tmp_path, change):
    """Extract ``archive``, apply ``change(folder)``, and pack it again the publisher's way."""
    issue = publish.issue_name(t)
    work = tmp_path / "repack"
    with tarfile.open(archive) as tar:
        tar.extractall(work, filter="data")
    folder = work / issue
    change(folder)
    names = ("data.json", "manifest.json", "receipt.json", "publication.json")
    files = {n: (folder / n).read_bytes() for n in names}
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


def _consistent(folder, **data_changes):
    """Change data.json AND rewrite the receipt and publication to match it (a consistent forgery)."""
    _edit_json(folder, "data.json", **data_changes)
    data = json.loads((folder / "data.json").read_text())
    publication = json.loads((folder / "publication.json").read_text())
    cid = chain.commit_id(data, publication["nonce"], publication["station_id"], publication["timestamp"])
    sha = hashlib.sha256(chain.compact(data).encode()).hexdigest()
    _edit_json(folder, "receipt.json", data_sha256=sha, commit_id=cid)
    _edit_json(folder, "publication.json", commit_id=cid)


def _manifest(folder, **changes):
    """Change the manifest AND the receipt's digest to match it."""
    _edit_json(folder, "manifest.json", **changes)
    man = json.loads((folder / "manifest.json").read_text())
    _edit_json(folder, "receipt.json", manifest_digest=chain.manifest_digest(man))


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
        (lambda f: _edit_json(f, "receipt.json", late_leads_hours=[1]), "counted or late leads"),
        (lambda f: _edit_json(f, "receipt.json", reveal_height=-1), "not a positive block height"),
        (lambda f: _consistent(f, init="2026-03-11T00"), "names another init"),
        (lambda f: _consistent(f, model_id="other"), "names another model"),
        (lambda f: _manifest(f, horizon_hours=24), "another model or horizon"),
        (lambda f: _manifest(f, input_hash="0" * 64), "HARNESS_LOCK.json"),
    ],
)
def test_check_reveal_refuses_a_tampered_archive(revealed, tmp_path, change, message):
    root, t, receipt = revealed
    archive = Path(_pack(root, t, tmp_path, receipt)["archive"])
    tampered = _repack(t, archive, tmp_path, change)
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
        name = {
            "absolute": "/etc/x",
            "parent": f"{issue}/../../x",
            "link": f"{issue}/data.json",
            "other issue": "2026-01-01T00Z/data.json",
        }[kind]
        info = tarfile.TarInfo(name)
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
    "node_kw, message",
    [
        ({"reveal_hash": "ee" * 32}, "another hash"),
        ({"commit_hash": "ee" * 32}, "another hash"),
        ({"no_commit_tx": True}, "carries no ForecastCommit"),
        ({"lost": True}, "inconclusive"),
        ({"tip_from": 5}, "successor blocks, not 6"),
        ({"chain_id": "777"}, "not 888 testnet"),
        ({"station": "somewhere"}, "another commit or station"),
        ({"flip_reveal_hash_on_second_pass": True}, "another hash"),
    ],
)
def test_check_reveal_rereads_the_chain(revealed, tmp_path, node_kw, message):
    """Finding 2: a 404 / not_found status is never a pass (inconclusive); the node must be chain 888
    testnet; the station label must match; the two passes must agree."""
    root, t, receipt = revealed
    archive = Path(_pack(root, t, tmp_path, receipt)["archive"])
    node = _chain(tmp_path, t, receipt, **node_kw)
    with _reader(node), pytest.raises(publish.PublishRefusedError, match=message):
        publish.check_reveal(root, archive, t, _node(node))
    node.server.shutdown()


def test_a_fabricated_commit_on_real_blocks_is_refused(revealed, tmp_path):
    """Finding 2 reproduction: an archive for a never-submitted commit id reusing the wallet's real
    block hashes and times. The node does not serve that commit's status: refused, not a pass."""
    root, t, receipt = revealed
    archive = Path(_pack(root, t, tmp_path, receipt)["archive"])
    forged = _repack(t, archive, tmp_path, lambda f: _consistent(f, inputs_sha256="0" * 64))
    node = _chain(tmp_path, t, receipt, served_commit=receipt["commit_id"])
    with _reader(node), pytest.raises(publish.PublishRefusedError, match="inconclusive"):
        publish.check_reveal(root, forged, t, _node(node))
    node.server.shutdown()


def test_verify_download_requires_the_sha256_and_the_packed_bytes(revealed, tmp_path):
    root, t, receipt = revealed
    packed = _pack(root, t, tmp_path, receipt)
    folder = tmp_path / "download"
    folder.mkdir()
    archive = Path(packed["archive"])
    shutil.copy(archive, folder / archive.name)
    with pytest.raises(publish.PublishRefusedError, match="missing"):
        publish.verify_download(root, t, folder)
    shutil.copy(archive.with_name(archive.name + ".sha256"), folder / (archive.name + ".sha256"))
    assert publish.verify_download(root, t, folder) == packed["sha256"]
    (folder / archive.name).write_bytes(archive.read_bytes() + b"x")
    with pytest.raises(publish.PublishRefusedError, match="sha256 differs"):
        publish.verify_download(root, t, folder)


# -- the ledger ----------------------------------------------------------------------------------------------


def test_the_lead_adjusted_time_is_exactly_the_per_lead_rule():
    t = np.datetime64("2026-12-02T00", "ns")
    for lead in chain.LEAD_HOURS:
        edge = t + np.timedelta64(lead, "h") - chain.MARGIN
        for offset in (-1, 0, 1):
            block = edge + np.timedelta64(offset, "ns")
            counted = lead in chain.counted_leads(t, block)["counted_leads_hours"]
            late = bool(ledger.lead_received_at(block, lead) >= t)  # pofs_shadow: received_at >= init_time
            assert counted != late, (lead, offset)


def test_the_ledger_builds_pofs_evidence_from_a_verified_receipt(revealed, tmp_path):
    root, t, receipt = revealed
    node = _chain(tmp_path, t, receipt)
    with _reader(node):
        ev = ledger.issue_evidence(root, t, _node(node))
    node.server.shutdown()
    inputs = json.loads((live.issue_dir(root, t) / "inputs.json").read_text())
    block_time = chain.parse_time_ns(receipt["commit_block_time"])
    counted = receipt["counted_leads_hours"]
    assert isinstance(ev, EpochEvidence) and ev.gefs_digest == inputs["gefs_sha256"]
    assert ev.init_time == t  # pofs_shadow's contract: the period's init
    assert dict(ev.commitments) == {
        FORECASTER: Commitment(receipt["content_sha256"], ledger.lead_received_at(block_time, min(counted)))
    }
    assert ev.commitments[FORECASTER].received_at < ev.init_time  # timely over the counted leads
    rows = ledger.summary(ledger.per_lead(receipt, inputs["gefs_sha256"]))
    assert [r["lead_hours"] for r in rows if r["timely"]] == counted
    assert [r["lead_hours"] for r in rows if not r["timely"]] == receipt["late_leads_hours"]


@pytest.mark.parametrize(
    "damage, message",
    [
        ("content hash", "disagree on content_sha256|commits another content hash"),
        ("status 404", "inconclusive"),
        ("replaced block", "another hash"),
        ("store", "content hash is not the committed hash"),
    ],
)
def test_the_ledger_never_builds_evidence_from_an_unverified_receipt(revealed, tmp_path, damage, message):
    """Finding 3: a receipt naming another content hash, a commitment the node no longer serves, a
    replaced block or a changed store gives no evidence at all."""
    root, t, receipt = revealed
    node_kw = {}
    if damage == "content hash":
        _rewrite_receipt(root, t, content_sha256="aa" * 32)
    elif damage == "status 404":
        node_kw = {"lost": True}
    elif damage == "replaced block":
        node_kw = {"reveal_hash": "ee" * 32}
    else:
        _other_store(live.issue_dir(root, t))
    node = _chain(tmp_path, t, receipt, **node_kw)
    with _reader(node), pytest.raises(publish.PublishRefusedError, match=message):
        ledger.issue_evidence(root, t, _node(node))
    node.server.shutdown()


def _scorecards(t, gefs, digest, leads):
    """The committed January 2024 GEFS reference re-dated to issue ``t`` and cut to ``leads``, and a
    candidate scorecard for ``digest`` over the same leads that passes every gate (as in the
    pofs_shadow tests)."""
    from datetime import datetime

    from wrfcomp.pofs_shadow import load_scorecard

    ref = load_scorecard(REPO / "experiments" / "gefs_2024-01.json").model_dump()
    when = datetime.fromisoformat(str(np.datetime_as_string(t, unit="s")))
    ref.update(period_start=when, period_end=when, forecast_sha256=gefs, leads_scored=list(leads))
    for per_lead_rows in ref["metrics"].values():
        for key in [k for k in per_lead_rows if k != "all" and int(k) not in leads]:
            del per_lead_rows[key]
    reference = load_scorecard_from(ref)
    candidate = json.loads(json.dumps(ref, default=str))
    candidate.update(model_id="wrfcomp-live-v4", forecast_sha256=digest, baseline_id="noaa-gefs")
    candidate.update(baseline_sha256=gefs, passed_gate=True)
    passing = dict(
        crpss_vs_baseline=0.25,
        crpss_vs_baseline_ecdf=0.24,
        spread_skill_ratio=1.0,
        dm_pvalue=0.01,
        twcrps_p95=0.1,
        baseline_twcrps_p95=0.2,
    )
    for var, rows in candidate["metrics"].items():
        candidate["gates"][var] = dict.fromkeys(candidate["gates"][var], True)
        rows["all"].update(passing)
        for h in leads:
            rows[str(h)]["loss"] *= 0.75
    return reference, load_scorecard_from(candidate)


def load_scorecard_from(value):
    """Validate a scorecard dict exactly as ``pofs_shadow.load_scorecard`` validates a file."""
    from wrfcomp.harness.schemas import Scorecard

    return Scorecard.model_validate_json(json.dumps(value, default=str))


def test_settlement_end_to_end_pays_a_verified_timely_issue_and_nothing_else(revealed, tmp_path):
    """Finding 3: the evidence settles through pofs_shadow.settle_from_scorecards. A verified receipt
    is paid for a scorecard over the leads it counts; a scorecard that also scores a lead the commit
    was too late for (3 h) is 'late' and unpaid; a forged commitment (another digest) is
    'non_reveal_or_mismatch'; an unverified receipt gives no evidence at all (previous test)."""
    from wrfcomp.pofs_shadow import ShadowLedgerParams, settle_from_scorecards

    root, t, receipt = revealed
    counted = receipt["counted_leads_hours"]
    assert counted == [6, 12, 24, 48] and 3 in receipt["late_leads_hours"]
    node = _chain(tmp_path, t, receipt)
    with _reader(node):
        timely = ledger.issue_evidence(root, t, _node(node))  # over the counted leads
        with_late = ledger.issue_evidence(root, t, _node(node), leads=[3, *counted])
    node.server.shutdown()

    def settle(leads, ev, digest=receipt["content_sha256"]):
        ref, card = _scorecards(t, ev.gefs_digest, digest, leads)
        rows = settle_from_scorecards(
            {"gefs": ref, FORECASTER: card},
            baseline_key="gefs",
            pool_amount=550,
            params=ShadowLedgerParams(alpha_slash=0.5),
            stakes=100,
            evidence=ev,
        )
        return {p.miner_id: p for p in rows.payouts}[FORECASTER]

    paid = settle(counted, timely)
    assert paid.reward == 550 and paid.status not in ("late", "uncommitted", "gefs_gate"), paid
    late = settle([3, *counted], with_late)
    assert late.status == "late" and late.reward == 0
    forged = settle(counted, timely, digest="bb" * 32)
    assert forged.status == "non_reveal_or_mismatch" and forged.reward == 0


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
# Stands in for gh: releases are folders under the state folder; every call is logged. It fails the
# verb named in state/gh_fail, corrupts downloads if state/corrupt_download exists, and asserts it
# never sees a node bearer.
import json, os, shutil, sys
from pathlib import Path
state = Path({state!r})
args = sys.argv[1:]
with (state / "gh.log").open("a") as fh:
    fh.write(json.dumps(args) + "\n")
assert not [k for k in os.environ if k.startswith("WRFCOIN_") and k.endswith("TOKEN")], "gh saw a node bearer"
assert args[0] == "release" and "--repo" in args and args[args.index("--repo") + 1] == "wrfcoin/wrfcomp", args
verb = args[1]
fail = state / "gh_fail"
if fail.exists() and fail.read_text().strip() == verb:
    sys.exit("gh: simulated failure")
releases = state / "releases"
if verb == "list":
    for p in sorted(releases.iterdir()) if releases.is_dir() else []:
        print(p.name)
    sys.exit(0)
tag = args[2]
release = releases / tag
if verb == "view":
    if not release.is_dir():
        sys.exit(1)
    for p in sorted(release.iterdir()):
        print(p.name)
elif verb == "create":
    release.mkdir(parents=True)
elif verb == "upload":
    for f in args[3:args.index("--repo")]:
        shutil.copy(f, release / Path(f).name)
elif verb == "download":
    dest = Path(args[args.index("--dir") + 1])
    patterns = [args[i + 1] for i, a in enumerate(args) if a == "--pattern"]
    for pattern in patterns:
        shutil.copy(release / pattern, dest / pattern)
        if (state / "corrupt_download").exists() and not pattern.endswith(".sha256"):
            with (dest / pattern).open("ab") as fh:
                fh.write(b"x")
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


TOKENS = {"WRFCOIN_OPERATOR_TOKEN": "x", "WRFCOIN_READER_TOKEN": "y"}


def test_the_daily_run_publishes_and_verifies_once_and_can_run_again(revealed, tmp_path):
    root, t, receipt = revealed
    node = _chain(tmp_path, t, receipt)
    first, steps, gh, state = _daily(root, t, tmp_path, node, extra_env=TOKENS)
    assert first.returncode == 0, first.stdout + first.stderr
    order = [s[1] if s[0] != "scripts/live_commit.py" else "prepare" for s in steps]
    expected = ["fetch-metar", "fetch-gefs", "build", "names", "pack", "verify-download", "check-reveal"]
    assert order == expected  # the receipt exists: no prepare, no submit.sh
    assert [c[1] for c in gh] == ["list", "create", "view", "upload", "download"]
    download = next(c for c in gh if c[1] == "download")
    asset = publish.asset_name(t)
    assert download.count("--pattern") == 2 and asset + ".sha256" in download
    assert sorted(p.name for p in (state / "releases" / publish.release_tag(t)).iterdir()) == [
        asset,
        asset + ".sha256",
    ]
    assert json.loads((root / "verify" / f"{publish.issue_name(t)}.json").read_text())["ok"]
    again, steps, gh, _ = _daily(root, t, tmp_path, node, extra_env=TOKENS)
    node.server.shutdown()
    assert again.returncode == 0, again.stdout + again.stderr
    assert [c[1] for c in gh][5:] == ["list", "view", "download"]  # nothing created or uploaded again


@pytest.mark.parametrize("verb", ["list", "view", "create"])
def test_the_daily_run_stops_on_any_gh_failure_before_uploading(revealed, tmp_path, verb):
    """Finding 4: a failing gh call stops the run; it never falls through to an upload."""
    root, t, receipt = revealed
    node = _chain(tmp_path, t, receipt)
    state = tmp_path / "daily_state"
    state.mkdir()
    (state / "gh_fail").write_text(verb)
    result, steps, gh, _ = _daily(root, t, tmp_path, node, extra_env=TOKENS)
    node.server.shutdown()
    assert result.returncode != 0 and "upload" not in [c[1] for c in gh]
    assert gh[-1][1] == verb


def test_the_daily_run_refuses_a_release_holding_only_one_of_the_two_assets(revealed, tmp_path):
    root, t, receipt = revealed
    node = _chain(tmp_path, t, receipt)
    state = tmp_path / "daily_state"
    release = state / "releases" / publish.release_tag(t)
    release.mkdir(parents=True)
    (release / publish.asset_name(t)).write_bytes(b"an earlier upload")
    result, steps, gh, _ = _daily(root, t, tmp_path, node)
    node.server.shutdown()
    assert result.returncode != 0 and "OPERATOR DECISION" in result.stderr
    assert "upload" not in [c[1] for c in gh] and "download" not in [c[1] for c in gh]


def test_the_daily_run_refuses_a_download_that_differs_from_the_sha256(revealed, tmp_path):
    root, t, receipt = revealed
    node = _chain(tmp_path, t, receipt)
    state = tmp_path / "daily_state"
    state.mkdir()
    (state / "corrupt_download").write_text("1")
    result, steps, gh, _ = _daily(root, t, tmp_path, node)
    node.server.shutdown()
    assert result.returncode != 0 and "sha256 differs" in result.stderr
    assert [s[1] for s in steps][-1] == "verify-download"  # check-reveal never ran
    assert not (root / "verify" / f"{publish.issue_name(t)}.json").exists()


def test_the_daily_run_publishes_nothing_while_the_reveal_is_not_final(revealed, tmp_path):
    """Finding 1 end to end: a receipt exists but the chain shows only 5 blocks after the reveal: pack
    refuses and no gh call happens."""
    root, t, receipt = revealed
    node = _chain(tmp_path, t, receipt, tip_from=5)
    result, steps, gh, _ = _daily(root, t, tmp_path, node)
    node.server.shutdown()
    assert result.returncode != 0 and "successor blocks, not 6" in result.stderr
    assert gh == [] and not (root / "publish").exists()


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
