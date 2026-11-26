"""Publish a revealed live v4 issue, check a published copy, and read its on-chain commit as a ledger
receipt (I2 deliverables 3 and 4). Each step takes --init T (e.g. 2026-12-02T00).

  names         prints "<release tag> <asset name> <archive path>" (no I/O beyond the path check)
  pack          writes <live>/publish/<asset> and <asset>.sha256 once the reveal is confirmed
  check-reveal  --archive <file>: re-hashes the published store against data.json, the manifest and
                the receipt, re-derives the commit id, and re-reads the chain; writes
                <live>/verify/<issue>.json
  ledger        re-checks the receipt on the chain and prints the pofs_shadow evidence per lead
                (wallet, digest, commit time, deadline, timely)

Nothing here uploads, signs or submits: the upload is the operator's (scripts/live_daily.sh). Reads
go to the local node only (http://127.0.0.1:8556), unauthenticated, at least 3 s apart, never
through a proxy, never following a redirect.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from wrfcomp.live import chain, paths, publish


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("names", "pack", "ledger"):
        sub.add_parser(name).add_argument("--init", required=True)
    check = sub.add_parser("check-reveal")
    check.add_argument("--init", required=True)
    check.add_argument("--archive", required=True, type=Path)
    args = parser.parse_args()
    paths.refuse_windows()
    root = paths.live_root()
    t = np.datetime64(args.init, "ns")
    if args.command == "names":
        print(publish.release_tag(t), publish.asset_name(t), publish.archive_path(root, t))
    elif args.command == "pack":
        print(json.dumps(publish.pack(root, t)))
    elif args.command == "check-reveal":
        result = publish.check_reveal(root, args.archive, t, chain.Node())
        keys = ("issue", "ok", "content_sha256", "commit_status_served", "counted_leads_hours")
        print(json.dumps({k: result[k] for k in keys}))
    else:
        from wrfcomp.live import ledger

        print(json.dumps(ledger.summary(ledger.issue_evidence(root, t, chain.Node())), indent=1))


if __name__ == "__main__":
    main()
