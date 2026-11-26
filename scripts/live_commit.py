"""Commit a published live v4 issue on testnet 888 and reveal its hash (I1 design, stage 2).

  prepare --init T --forecaster 0x... --signer-key <live>/keys/forecaster.key \\
          --cli /path/to/wrfcoin --cli-sha256 <sha256>
      writes <live>/chain/<issue>/ (data.json, manifest.json, commit.json, submit.sh), once.
      Nothing is signed, sent or read from the network. The key file is never opened.

The operator then runs ``bash <live>/chain/<issue>/submit.sh``, which calls the wrfcoin CLI and
these steps (each takes --dir <live>/chain/<issue>):

  check-cli | check-key | check-node | check-validate | check-submitted --step commit|reveal |
  wait-pending | check-reveal-inputs | check-reveal-validate | check-window | wait-revealed | record

The throwaway forecaster key is made once with the CLI (never a producer, oracle or station key),
in a folder made first with mode 0700:
  install -d -m 0700 <live>/keys
  wrfcoin --output json session-key keygen --out <live>/keys/forecaster.key
Its "session_key_address" is the --forecaster address.

The data directory is $WRFCOMP_DATA_DIR (default ~/wrfcomp-data). Testnet 888 only. Live writes are
refused on Windows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from wrfcomp.live import chain
from wrfcomp.live import v4 as live
from wrfcomp.live.paths import check_path

REPO = Path(__file__).resolve().parents[1]


def _chain_dir(root: Path, value: str) -> Path:
    out = Path(value)
    if out.parent.name != "chain" or out.parent.parent.resolve() != root.resolve():
        raise chain.ChainRefusedError("--dir must be <live>/chain/<issue>/")
    check_path(root, out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--init", required=True)
    prep.add_argument("--forecaster", required=True)
    prep.add_argument("--signer-key", required=True, type=Path)
    prep.add_argument("--cli", required=True, type=Path)
    prep.add_argument("--cli-sha256", required=True)
    prep.add_argument("--allow-12z", action="store_true")
    for name in (
        "check-cli",
        "check-key",
        "check-node",
        "check-validate",
        "wait-pending",
        "check-reveal-inputs",
        "check-reveal-validate",
        "check-window",
        "wait-revealed",
        "record",
    ):
        sub.add_parser(name).add_argument("--dir", required=True)
    submitted = sub.add_parser("check-submitted")
    submitted.add_argument("--step", required=True, choices=("commit", "reveal"))
    submitted.add_argument("--dir", required=True)
    args = parser.parse_args()
    live.refuse_windows_entry()
    root = live.live_root()
    if args.command == "prepare":
        out = chain.prepare(
            root,
            np.datetime64(args.init),
            forecaster=args.forecaster,
            signer_key=args.signer_key,
            cli=args.cli,
            cli_sha256=args.cli_sha256,
            repo=REPO,
            allow_12z=args.allow_12z,
        )
        print(json.dumps({k: out[k] for k in ("init", "commit_id", "station_id", "if_committed_now")}))
        print(f"next: bash {chain.chain_dir(root, np.datetime64(args.init)) / 'submit.sh'}")
        return
    out = _chain_dir(root, args.dir)
    if args.command == "check-cli":
        chain.check_cli(out)
    elif args.command == "check-key":
        chain.check_key(out)
    elif args.command == "check-reveal-inputs":
        chain.check_reveal_inputs(out)
    elif args.command == "check-reveal-validate":
        chain.check_reveal_validate(out)
    elif args.command == "check-window":
        print(json.dumps({"tip": chain.check_window(out, chain.Node())}))
    elif args.command == "wait-revealed":
        print(json.dumps({"revealed_at": chain.wait_revealed(out, chain.Node())["revealed_at"]}))
    elif args.command == "check-node":
        info = chain.check_node(out, chain.Node())
        print(json.dumps({"chain_id": info.get("chain_id"), "network_id": info.get("network_id")}))
    elif args.command == "check-validate":
        chain.check_validate(out)
    elif args.command == "check-submitted":
        print(json.dumps({"tx_id": chain.check_submitted(out, args.step)["tx_id"]}))
    elif args.command == "wait-pending":
        print(json.dumps({"status": chain.wait_pending(out, chain.Node())["status"]}))
    elif args.command == "record":
        receipt = chain.record_receipt(out, chain.Node())
        print(json.dumps({k: receipt[k] for k in ("commit_height", "reveal_height", "counted_leads_hours")}))


if __name__ == "__main__":
    main()
