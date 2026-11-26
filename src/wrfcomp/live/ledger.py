"""On-chain commits as ``pofs_shadow`` receipts (lane I2, deliverable 4; the I1 design).

``pofs_shadow`` settles from ``EpochEvidence``: a map of wallet to ``Commitment(digest, received_at)``,
and the rule "a receipt at or after ``init_time`` is late". This module builds that evidence from
live ``receipt.json`` files, each re-checked on the chain first (``publish.chain_check``: the recorded
blocks, hashes, times, our transactions, the window, 6 confirmations), so a receipt file alone is
never trusted:

* wallet = the forecaster address; digest = the store's content hash (the harness hash the evaluator
  uses for a forecast); received_at = the commit block's time, read from the chain, to the nanosecond;
* the live rule is per lead (owner ruling): lead L of issue T counts only if T + L is at least 15
  minutes after the commit block time. One ``EpochEvidence`` per (issue, lead) with
  ``init_time = T + L - 15 min + 1 ns`` makes the existing test exactly that rule: received_at
  >= T + L - 15 min + 1 ns (late) if and only if the block time is after T + L - 15 min. Nothing in
  ``pofs_shadow`` or ``configs/`` changes;
* ``gefs_digest`` is the content hash of the raw GEFS cycle the issue was built from (pinned in the
  issue's ``inputs.json``).

Reads go through ``chain.Node``: the local node only, unauthenticated GETs at least 3 s apart (at
most 20 a minute; the limit is 30), never wrfcoin.com or any other host, no proxy, no redirect.
About 4 GETs per receipt (status, two blocks, tip).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from wrfcomp.live import chain, publish
from wrfcomp.live.paths import check_path
from wrfcomp.pofs_shadow import Commitment, EpochEvidence

ONE_NS = np.timedelta64(1, "ns")


def lead_deadline(init, lead_hours: int) -> np.datetime64:
    """The ``init_time`` for lead ``lead_hours`` of issue ``init``: a commit block at or after it is
    late, exactly as ``chain.counted_leads`` decides (T + L >= block + 15 min counts)."""
    return np.datetime64(init, "ns") + np.timedelta64(int(lead_hours), "h") - chain.MARGIN + ONE_NS


def verified_receipt(root: Path, t, node: chain.Node) -> dict:
    """The issue's receipt, re-checked on the chain now; refused if any check fails."""
    t = np.datetime64(t, "ns")
    path = check_path(root, chain.chain_dir(root, t) / "receipt.json")
    receipt = json.loads(path.read_text())
    if np.datetime64(receipt["init"], "ns") != t:
        raise publish.PublishRefusedError("receipt.json is for another issue")
    if receipt["chain_id"] != chain.CHAIN_ID or receipt["network_id"] != chain.NETWORK_ID:
        raise publish.PublishRefusedError("the receipt is not for testnet 888")
    publish.chain_check(receipt, node)
    return receipt


def gefs_digest(root: Path, t) -> str:
    from wrfcomp.live import v4 as live

    inputs = check_path(root, live.issue_dir(root, t) / "inputs.json")
    return json.loads(inputs.read_text())["gefs_sha256"]


def commitment(receipt: dict) -> tuple[str, Commitment]:
    return receipt["forecaster"], Commitment(
        digest=receipt["content_sha256"], received_at=chain.parse_time_ns(receipt["commit_block_time"])
    )


def lead_evidence(t, lead_hours: int, receipts: list[dict], gefs: str) -> EpochEvidence:
    """``EpochEvidence`` for one lead of issue ``t`` from re-checked receipts of that issue."""
    t = np.datetime64(t, "ns")
    commitments = {}
    for receipt in receipts:
        if np.datetime64(receipt["init"], "ns") != t:
            raise ValueError("a receipt for another issue")
        wallet, c = commitment(receipt)
        if wallet in commitments:
            raise ValueError(f"two receipts for wallet {wallet} in one issue")
        commitments[wallet] = c
    return EpochEvidence(init_time=lead_deadline(t, lead_hours), gefs_digest=gefs, commitments=commitments)


def issue_evidence(root: Path, t, node: chain.Node) -> dict[int, EpochEvidence]:
    """One ``EpochEvidence`` per lead for our live issue ``t``, from its re-checked receipt."""
    receipt = verified_receipt(root, t, node)
    gefs = gefs_digest(root, t)
    return {lead: lead_evidence(t, lead, [receipt], gefs) for lead in chain.LEAD_HOURS}


def summary(evidence: dict[int, EpochEvidence]) -> list[dict]:
    """Per lead and wallet: the digest, the commit time, the deadline and whether it is timely."""
    rows = []
    for lead, ev in sorted(evidence.items()):
        for wallet, c in sorted(ev.commitments.items()):
            rows.append(
                {
                    "lead_hours": lead,
                    "wallet": wallet,
                    "digest": c.digest,
                    "received_at": chain.format_time_ns(c.received_at),
                    "late_from": chain.format_time_ns(ev.init_time),
                    "timely": bool(c.received_at < ev.init_time),  # pofs_shadow's own test, negated
                }
            )
    return rows
