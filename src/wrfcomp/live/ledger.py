"""On-chain commits as ``pofs_shadow`` receipts (lane I2, deliverable 4; the I1 design).

``pofs_shadow`` settles from ``EpochEvidence(init_time, gefs_digest, commitments)``, where
``commitments`` maps a wallet to ``Commitment(digest, received_at)``. Its contract:
``init_time`` is the epoch's first init (it must equal the reference scorecard's ``period_start``),
and a commitment is late when ``received_at >= init_time``. This module builds that evidence for
our live issues, from receipts that are verified first, never trusted as files:

* **Verified.** The issue's own files (``publish.verify_files``) prove the receipt's content hash
  is the one ``data.json`` commits to, that ``data.json`` derives the receipt's commit id, and that
  the local store hashes to it. The chain (``publish.chain_check``) then confirms the commitment now:
  chain 888, the commit status SERVED for that commit id (a 404 is never a pass), both blocks
  unchanged, 6 blocks after the reveal, read twice.
* **Mapping.** wallet = the forecaster address; digest = the store's content hash (the hash the
  evaluator puts in ``forecast_sha256``); ``gefs_digest`` = the raw GEFS cycle's content hash pinned in
  the issue's ``inputs.json``; ``init_time`` = the issue's init T, as ``pofs_shadow`` requires.
* **Per lead.** The live rule (owner ruling) is that lead L counts only if T + L is at least 15
  minutes after the commit block time. ``pofs_shadow`` pays a scorecard only over ALL the leads it
  scores, with one receipt time, so the commit time is adjusted for the shortest scored lead:
  ``received_at = commit block time + 15 min - min(leads) - 1 ns``.
  Then ``received_at < T`` (timely) if and only if every scored lead counts
  (``test_the_lead_adjusted_time_is_exactly_the_per_lead_rule``). A live issue is therefore settled
  with a scorecard over its counted leads (6, 12, 24 and 48 h at a commit near T + 4 h 15 min); a
  scorecard that includes a lead which does not count is late as a whole. ``pofs_shadow`` itself is
  unchanged.

Reads go through ``chain.Node``: the local node only, unauthenticated GETs at least 3 s apart (at
most 20 a minute; the limit is 30), never wrfcoin.com or any other host, no proxy, no redirect.
About 10 GETs per receipt (two passes of status, two blocks and the tip).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from wrfcomp.live import chain, publish
from wrfcomp.live.paths import check_path
from wrfcomp.pofs_shadow import Commitment, EpochEvidence

ONE_NS = np.timedelta64(1, "ns")


def lead_received_at(block_time, lead_hours: int) -> np.datetime64:
    """The commit time as lead ``lead_hours`` sees it: timely (``< T``) exactly when the lead counts."""
    return np.datetime64(block_time, "ns") + chain.MARGIN - np.timedelta64(int(lead_hours), "h") - ONE_NS


def verified_receipt(root: Path, t, node: chain.Node) -> dict:
    """The issue's receipt, verified against its own files and store, then on the chain, now."""
    t = np.datetime64(t, "ns")
    receipt, publication, data_text, manifest, store = publish.local_files(root, t)
    publish.verify_files(t, receipt, publication, data_text, manifest, store)
    publish.chain_check(receipt, node)
    return receipt


def gefs_digest(root: Path, t) -> str:
    from wrfcomp.live import v4 as live

    inputs = check_path(root, live.issue_dir(root, t) / "inputs.json")
    return json.loads(inputs.read_text())["gefs_sha256"]


def lead_evidence(t, leads, receipts: list[dict], gefs: str) -> EpochEvidence:
    """``EpochEvidence`` for a scorecard over ``leads`` (hours; one int for one lead) of issue ``t``,
    from verified receipts of that issue."""
    t = np.datetime64(t, "ns")
    leads = [int(leads)] if np.isscalar(leads) else [int(h) for h in leads]
    if not leads or not set(leads) <= set(chain.LEAD_HOURS):
        raise ValueError("leads must be some of the live leads")
    commitments = {}
    for receipt in receipts:
        if np.datetime64(receipt["init"], "ns") != t:
            raise ValueError("a receipt for another issue")
        wallet = receipt["forecaster"]
        if wallet in commitments:
            raise ValueError(f"two receipts for wallet {wallet} in one issue")
        block_time = chain.parse_time_ns(receipt["commit_block_time"])
        commitments[wallet] = Commitment(receipt["content_sha256"], lead_received_at(block_time, min(leads)))
    return EpochEvidence(init_time=t, gefs_digest=gefs, commitments=commitments)


def issue_evidence(root: Path, t, node: chain.Node, leads=None) -> EpochEvidence:
    """``EpochEvidence`` for our live issue ``t`` and a scorecard over ``leads`` (default: the leads
    the verified receipt counts), from its verified receipt."""
    receipt = verified_receipt(root, t, node)
    leads = receipt["counted_leads_hours"] if leads is None else leads
    return lead_evidence(t, leads, [receipt], gefs_digest(root, t))


def per_lead(receipt: dict, gefs: str) -> dict[int, EpochEvidence]:
    """One ``EpochEvidence`` per single lead (diagnostics; a payment needs a full scorecard)."""
    return {lead: lead_evidence(receipt["init"], lead, [receipt], gefs) for lead in chain.LEAD_HOURS}


def summary(evidence: dict[int, EpochEvidence]) -> list[dict]:
    """Per lead and wallet: the digest, the lead-adjusted commit time and whether it is timely."""
    rows = []
    for lead, ev in sorted(evidence.items()):
        for wallet, c in sorted(ev.commitments.items()):
            rows.append(
                {
                    "lead_hours": lead,
                    "wallet": wallet,
                    "digest": c.digest,
                    "received_at_for_lead": chain.format_time_ns(c.received_at),
                    "init_time": chain.format_time_ns(ev.init_time),
                    "timely": bool(c.received_at < ev.init_time),  # pofs_shadow's own test, negated
                }
            )
    return rows
