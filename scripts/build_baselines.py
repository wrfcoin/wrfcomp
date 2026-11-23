"""Build one init in the configured external data directory."""

import argparse
import os
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from wrfcomp.baselines import climatology, gefs, persistence
from wrfcomp.baselines.io import CONFIG, boundaries, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=["gefs", "climatology", "persistence"])
    parser.add_argument("init", help="UTC timestamp")
    parser.add_argument("--truth", type=Path, help="hourly NetCDF or Zarr under the data directory")
    args = parser.parse_args()
    init = pd.Timestamp(args.init)
    if init.tz is not None:
        init = init.tz_convert("UTC").tz_localize(None)
    if init != init.floor("h") or init.hour not in (0, 6, 12, 18):
        parser.error("choose a synoptic initialization")
    if init + pd.Timedelta(hours=48) >= boundaries()[1]:
        parser.error("forecast overlaps the sealed period")
    data = Path(os.environ.get("WRFCOMP_DATA_DIR", Path.home() / "wrfcomp-data")).resolve()
    if data.is_relative_to("/tmp") or data.is_relative_to(CONFIG.parents[1]):
        parser.error("use the external data directory")
    dest = data / "baselines" / args.model / f"{init:%Y%m%d%H}.zarr"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        parser.error(f"output already exists: {dest}")
    start = time.monotonic()
    tx = gefs.Transfer()
    if args.model == "gefs":
        table = pd.read_csv(CONFIG.with_name("stations_conus.csv"), dtype={"station_id": str})
        stations = xr.Dataset(coords={"station_id": table.station_id.to_numpy(dtype=str)})
        for key, column in (("latitude", "lat"), ("longitude", "lon"), ("elevation_m", "elevation_m")):
            stations.coords[key] = ("station_id", table[column].to_numpy())
        result = gefs.extract(stations, init, data / "baselines" / "points", tx)
    else:
        if args.truth is None or not args.truth.resolve().is_relative_to(data):
            parser.error("--truth must identify a truth store in the data directory")
        model = climatology if args.model == "climatology" else persistence
        opener = xr.open_zarr if args.truth.is_dir() else xr.open_dataset
        with opener(args.truth) as truth:
            # Refuse an archive containing sealed observations before loading any weather values.
            if truth.valid_time.values.max() >= boundaries()[1]:
                parser.error("sealed truth archive")
            result = model.forecast(truth, [np.datetime64(init)], (1, 3, 6, 12, 24, 48))
    write(result, dest)
    print(
        f"{dest} bytes={tx.bytes} seconds={time.monotonic() - start:.3f} "
        f"peak_rss_KiB={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}"
    )


if __name__ == "__main__":
    main()
