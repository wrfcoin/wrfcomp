"""Extract station points from NOAA's 0.5-degree GEFS A files."""

import hashlib
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock
from time import sleep
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from .io import STATION, UNITS, layout


def _require_eccodes():
    try:
        import eccodes as ec
    except ImportError as exc:
        raise ImportError(
            "NOAA GEFS GRIB2 decoding requires the optional 'grib' extra "
            "(eccodes and eccodeslib). Install it with `uv sync --extra grib` "
            "or `pip install 'wrfcomp[grib]'`. On Windows, use WSL2 for NOAA GEFS downloads."
        ) from exc
    return ec

LEADS = (3, 6, 12, 24, 48)
FIELDS = {
    "t2m": ("TMP", "2 m above ground"),
    "rh": ("RH", "2 m above ground"),
    "u10": ("UGRD", "10 m above ground"),
    "v10": ("VGRD", "10 m above ground"),
    "msl": ("PRMSL", "mean sea level"),
}
HEIGHT = ("HGT", "surface")


def index_ranges(index):
    rows = [line.split(":") for line in index.splitlines() if line.strip()]
    ranges = {}
    for i, row in enumerate(rows):
        start = int(row[1])
        end = int(rows[i + 1][1]) - 1 if i + 1 < len(rows) else None
        if end is not None and end < start:
            raise ValueError("non-increasing GRIB offsets")
        key = tuple(row[3:5])
        if key in (*FIELDS.values(), HEIGHT):
            if key in ranges:
                raise ValueError(f"ambiguous GRIB field {key}")
            ranges[key] = start, end
    return ranges


class Transfer:
    def __init__(self):
        self.bytes = 0
        self.lock = Lock()

    def get(self, url, interval=None):
        headers = {}
        if interval:
            first, last = interval
            headers["Range"] = f"bytes={first}-{'' if last is None else last}"
        for attempt in range(4):
            try:
                with urlopen(Request(url, headers=headers), timeout=45) as response:
                    length = int(response.headers["Content-Length"])
                    if interval:
                        match = re.fullmatch(
                            r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", "")
                        )
                        if response.status != 206 or not match:
                            raise ValueError("server did not honor the byte range")
                        a, b, total = map(int, match.groups())
                        if a != first or b != (total - 1 if last is None else last) or length != b - a + 1:
                            raise ValueError("unexpected GRIB range")
                    if length > 8_000_000:
                        raise ValueError("oversized index or single-field response")
                    data = response.read(length)
                    with self.lock:
                        self.bytes += len(data)
                    if len(data) != length:
                        raise OSError("truncated NOAA response")
                    return data
            except HTTPError as exc:
                if exc.code not in (429, 500, 502, 503, 504) or attempt == 3:
                    raise
            except (URLError, TimeoutError, OSError):
                if attempt == 3:
                    raise
            sleep(2**attempt)


def bilinear(grid, x, y):
    """Fractional grid indices, periodic longitude and bounded latitude."""
    if np.any((y < 0) | (y > grid.shape[0] - 1)):
        raise ValueError("station outside GEFS latitude bounds")
    x = np.asarray(x) % grid.shape[1]
    i = x.astype(int)
    j = np.minimum(np.asarray(y).astype(int), grid.shape[0] - 2)
    dx, dy = x - i, y - j
    right = (i + 1) % grid.shape[1]
    return (1 - dy) * ((1 - dx) * grid[j, i] + dx * grid[j, right]) + dy * (
        (1 - dx) * grid[j + 1, i] + dx * grid[j + 1, right]
    )


def points(data, stations):
    ec = _require_eccodes()
    grib = ec.codes_new_from_message(data)
    try:

        def get(key):
            return ec.codes_get(grib, key)

        if get("gridType") != "regular_ll" or get("scanningMode") != 0:
            raise ValueError("unsupported GEFS grid or scan direction")
        grid = ec.codes_get_values(grib).reshape(get("Nj"), get("Ni"))
        x = (stations.longitude.values - get("longitudeOfFirstGridPointInDegrees")) / get(
            "iDirectionIncrementInDegrees"
        )
        y = (get("latitudeOfFirstGridPointInDegrees") - stations.latitude.values) / get(
            "jDirectionIncrementInDegrees"
        )
        return bilinear(grid, x, y)
    finally:
        ec.codes_release(grib)


def lapse(temp, orography, elevation):
    return temp - 0.0065 * (elevation - orography)


def extract(stations, init, cache, transfer):
    out = layout(stations, [init], LEADS, 31, "noaa-gefs")
    t = pd.Timestamp(out.init_time.values[0])
    if t != t.floor("h") or t.hour not in (0, 6, 12, 18):
        raise ValueError("expected a GEFS cycle")
    root = f"https://noaa-gefs-pds.s3.amazonaws.com/gefs.{t:%Y%m%d}/{t:%H}/atmos/pgrb2ap5"

    def url(member, lead):
        return f"{root}/ge{member}.t{t:%H}z.pgrb2a.0p50.f{lead:03}"

    signature = hashlib.sha256()
    for key in ("station_id", *STATION):
        signature.update(stations[key].values.astype("U").tobytes())
    cache = Path(cache) / "gefs-v1" / signature.hexdigest() / t.strftime("%Y%m%d%H")
    cache.mkdir(parents=True, exist_ok=True)

    def atomic_points(path, values):
        with NamedTemporaryFile(dir=cache, delete=False) as f:
            try:
                np.save(f, values, allow_pickle=False)
                f.close()
                os.replace(f.name, path)
            finally:
                Path(f.name).unlink(missing_ok=True)

    height_file = cache / "height.npy"
    if not height_file.exists():
        base = url("c00", 0)
        ranges = index_ranges(transfer.get(base + ".idx").decode())
        atomic_points(height_file, points(transfer.get(base, ranges[HEIGHT]), stations))
    height = np.load(height_file)

    def one(job):
        j, m = job
        member = f"p{m:02}" if m else "c00"
        lead = LEADS[j]
        path = cache / f"{member}-{lead}.npy"
        if not path.exists():
            base = url(member, lead)
            ranges = index_ranges(transfer.get(base + ".idx").decode())
            fields = {k: points(transfer.get(base, ranges[field]), stations) for k, field in FIELDS.items()}
            c = fields["t2m"] - 273.15
            rh = fields.pop("rh")
            if np.any(~np.isfinite(rh) | (rh <= 0)):
                raise ValueError("relative humidity outside dewpoint formula domain")
            gamma = np.log(rh / 100) + 17.625 * c / (243.04 + c)
            fields["t2m"] = lapse(fields["t2m"], height, stations.elevation_m.values)
            fields["td2m"] = np.minimum(273.15 + 243.04 * gamma / (17.625 - gamma), fields["t2m"])
            atomic_points(path, np.stack([fields[k] for k in UNITS]).astype("f4"))
        return j, m, np.load(path)

    with ThreadPoolExecutor(max_workers=8) as pool:
        for j, m, values in pool.map(one, ((j, m) for j in range(len(LEADS)) for m in range(31))):
            for name, values_at_stations in zip(UNITS, values, strict=True):
                out[name].values[0, j, :, m] = values_at_stations
    return out
