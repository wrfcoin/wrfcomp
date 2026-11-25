"""Live candidate v4 for one issue time T (00Z or 12Z), on current data, under ``<data>/live/``.

Steps (each refuses rather than guess; see ``wrfcomp.live``):

  fetch-gefs   --init T --network   GEFS cycle T at the 500 stations (~184 MB; S3 byte ranges)
  fetch-metar  --hours N --network  the last N hours of METAR/SPECI for the 500 stations (< 1 MB)
  build        --init T             the v4 store, its content hash and build.json

Network steps need ``--network``: nothing is downloaded without it. ``fetch-gefs`` needs the
``grib`` extra (``uv sync --extra grib``). Live data is for live operation only: it is never
scored as an evaluation result or mixed into training, thresholds or references.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from wrfcomp.live import metar
from wrfcomp.live import v4 as live


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("WRFCOMP_DATA_DIR") or Path.home() / "wrfcomp-data"),
    )
    sub = parser.add_subparsers(dest="step", required=True)
    gefs = sub.add_parser("fetch-gefs")
    gefs.add_argument("--init", required=True, help="issue time, e.g. 2026-12-02T00")
    gefs.add_argument("--network", action="store_true", help="allow the download")
    obs = sub.add_parser("fetch-metar")
    obs.add_argument("--hours", type=int, default=30)
    obs.add_argument("--network", action="store_true", help="allow the download")
    build = sub.add_parser("build")
    build.add_argument("--init", required=True)
    args = parser.parse_args()
    root = live.live_root(args.data_dir)
    if args.step in ("fetch-gefs", "fetch-metar") and not args.network:
        parser.error("this step downloads data; pass --network once the download is approved")
    if args.step == "fetch-gefs":
        out = live.fetch_cycle(root, np.datetime64(args.init))
    elif args.step == "fetch-metar":
        path, received = metar.fetch(root / "metar", sorted(live.icao_map()), args.hours)
        out = {"path": str(path), "bytes": received}
    else:
        record = live.build_issue(args.data_dir, np.datetime64(args.init))
        out = {k: record[k] for k in ("init", "model_id", "content_sha256", "live_start", "missing_cycles")}
    print(json.dumps(out))


if __name__ == "__main__":
    main()
