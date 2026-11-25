"""Live airport readings: METAR and SPECI from the aviationweather.gov Data API, as an hourly truth.

Source: ``https://aviationweather.gov/api/data/metar`` (NOAA Aviation Weather Center, free, no key),
JSON, the 500 registered stations by ICAO id. Each fetch is stored unchanged, append-only, as
``live/metar/<fetch time>.json``; the hourly truth is always rebuilt from the stored fetches, so it
can be recomputed by anyone who has them.

The 2024 truth came from NOAA ISD (global-hourly). Differences and how they are handled:

* Values are decoded from the raw report (``rawOb``), not from the API's derived fields, so the
  rounding is the report's own: temperature and dewpoint in tenths of a degree C from the remarks
  T-group when present (as ISD's ``TMP``/``DEW`` for ASOS), else whole degrees from the body.
* Pressure: sea-level pressure from the remarks ``SLPppp`` group (tenths of hPa, as ISD's ``SLP``).
  The altimeter setting is never used; no SLP (or ``SLPNO``) gives NaN, as ISD does.
* Wind: direction in degrees and speed in knots, converted to m/s rounded to 0.1 m/s as ISD stores
  it (5 kt -> 2.6 m/s); calm (``00000KT``) -> (0, 0); variable (``VRB``) or missing direction -> NaN
  (ISD direction 999). Gusts are ignored. Winds not in knots -> NaN.
* Dewpoint above temperature in the same report: the dewpoint is discarded before hourly matching,
  as ``harness.truth.parse_isd_csv`` does per ISD row.
* Report types: METAR and SPECI only (ISD ``FM-15``/``FM-16``). The ISD loader also keeps ``SY-MT``
  rows and falls back to ``FM-12`` synoptic rows for a station-year with no METAR; neither exists in
  this feed, so a station that sends no METAR has no live truth.
* Quality control: ISD's NCEI flags do not exist in real time. Live-only plausibility limits are
  applied instead (temperature and dewpoint -80..60 deg C, sea-level pressure 850..1090 hPa, wind at
  most 90 m/s, ISD's own wind limit); a value outside them is NaN. ISD's loader has no temperature or
  pressure limits of this kind; it relies on the NCEI flags.
* Corrections: two reports of one station at one exact time: the later fetch, then the later position
  in it, wins, among the fetches an issue is allowed to see (see ``wrfcomp.live.v4``: each issue's
  fetches are pinned when it is first built).
* The hourly rule is the harness rule (``harness.truth.match_station_to_hours``): the report nearest
  the hour within 15 minutes, the earlier one on a tie, ``X_offset_min`` = measurement minus label.
  For an issue at T, every report measured after T (to the exact second) is dropped BEFORE that
  selection (:func:`truth_as_of`), so a later report can never take a slot from an earlier one.

Live truth is never written under the evaluation folders, never mixed with ISD truth and never
scored as a 2023/2024 result.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import xarray as xr

from wrfcomp.harness.truth import OFFSET_FILL_VALUE, match_station_to_hours

API = "https://aviationweather.gov/api/data/metar"
USER_AGENT = "wrfcomp-live/0.1"
BATCH = 100
MAX_HOURS = 48
VARIABLES = ("t2m", "td2m", "u10", "v10", "msl")
KNOT = 0.514444
CLOSE_AFTER = np.timedelta64(15, "m")
MAX_WIND_MS = 90.0  # ISD's limit (900 tenths of m/s)
SLP_RANGE_HPA = (850.0, 1090.0)
TEMP_RANGE_C = (-80.0, 60.0)

_TIME = re.compile(r"^(?:METAR |SPECI )?(?:COR )?[A-Z0-9]{4} (\d{2})(\d{2})(\d{2})Z\b")
_WIND = re.compile(r"\b(\d{3}|VRB)(\d{2,3})(?:G\d{2,3})?(KT|MPS|KMH)\b")
_BODY_TEMP = re.compile(r"\s(M?\d{2})/(M?\d{2})?(?=\s|$)")
_T_GROUP = re.compile(r"\sT([01])(\d{3})(?:([01])(\d{3}))?(?=\s|$)")
_SLP = re.compile(r"\sSLP(\d{3})(?=\s|$)")


def _whole(text: str) -> float:
    return -float(text[1:]) if text.startswith("M") else float(text)


def _in(value: float, bounds: tuple[float, float]) -> float:
    return value if bounds[0] <= value <= bounds[1] else math.nan


def decode(raw: str) -> dict[str, float]:
    """Decode one raw METAR/SPECI into contract units (K, K, m/s, m/s, Pa); NaN where absent."""
    body, _, remarks = raw.partition(" RMK ")
    out = dict.fromkeys(VARIABLES, math.nan)
    temp = dew = math.nan
    found = _BODY_TEMP.search(" " + body)
    if found:
        temp = _whole(found.group(1))
        dew = _whole(found.group(2)) if found.group(2) else math.nan
    group = _T_GROUP.search(" " + remarks) if remarks else None
    if group:
        temp = (-1 if group.group(1) == "1" else 1) * int(group.group(2)) / 10
        if group.group(3):
            dew = (-1 if group.group(3) == "1" else 1) * int(group.group(4)) / 10
    out["t2m"] = _in(temp, TEMP_RANGE_C) + 273.15
    out["td2m"] = _in(dew, TEMP_RANGE_C) + 273.15
    if out["td2m"] > out["t2m"]:  # as parse_isd_csv: a same-report dewpoint above temperature
        out["td2m"] = math.nan
    slp = _SLP.search(" " + remarks) if remarks else None
    if slp:
        tenths = int(slp.group(1))
        hpa = (10000 + tenths if tenths < 500 else 9000 + tenths) / 10
        out["msl"] = _in(hpa, SLP_RANGE_HPA) * 100.0
    wind = _WIND.search(body)
    if wind and wind.group(3) == "KT":
        speed = round(int(wind.group(2)) * KNOT, 1)
        if speed > MAX_WIND_MS:
            pass
        elif speed == 0 and wind.group(1) in ("000", "VRB"):
            out["u10"] = out["v10"] = 0.0
        elif wind.group(1) != "VRB" and 1 <= int(wind.group(1)) <= 360:
            rad = math.radians(int(wind.group(1)) % 360)
            u, v = -speed * math.sin(rad), -speed * math.cos(rad)
            out["u10"] = 0.0 if abs(u) < 1e-12 else u
            out["v10"] = 0.0 if abs(v) < 1e-12 else v
    return out


def report_time(record: dict) -> np.datetime64:
    """The measurement time: the API's ``obsTime``, refused unless the report's own DDHHMMZ agrees."""
    t = np.datetime64(datetime.fromtimestamp(int(record["obsTime"]), UTC).replace(tzinfo=None), "ns")
    stamp = _TIME.match(record["rawOb"])
    ts = pd.Timestamp(t)
    if not stamp or (int(stamp.group(1)), int(stamp.group(2)), int(stamp.group(3))) != (
        ts.day,
        ts.hour,
        ts.minute,
    ):
        raise ValueError(f"report time does not match obsTime: {record['rawOb'][:24]!r}")
    return t


# -- fetch -------------------------------------------------------------------------------------


def _http_get(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept-Encoding": "gzip"})
    with urlopen(request, timeout=60) as response:
        payload = response.read()
        if response.headers.get("Content-Encoding") == "gzip":
            payload = gzip.decompress(payload)
        return payload


def fetch(
    directory: Path,
    icaos: list[str],
    hours: int,
    *,
    now: datetime | None = None,
    get: Callable[[str], bytes] = _http_get,
) -> tuple[Path, int]:
    """Fetch the last ``hours`` of reports for ``icaos`` and store them as one append-only file.

    Returns the file and the payload bytes received (decompressed size is not counted twice)."""
    if not 1 <= hours <= MAX_HOURS:
        raise ValueError(f"hours must be 1..{MAX_HOURS}")
    now = datetime.now(UTC) if now is None else now
    stamp = now.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    reports, received = [], 0
    for i in range(0, len(icaos), BATCH):
        query = urlencode({"ids": ",".join(icaos[i : i + BATCH]), "format": "json", "hours": hours})
        payload = get(f"{API}?{query}")
        received += len(payload)
        batch = json.loads(payload) if payload.strip() else []
        if not isinstance(batch, list):
            raise ValueError("unexpected METAR API response")
        reports.extend(batch)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{stamp}.json"
    if path.exists():
        raise FileExistsError(path)
    body = {"source": API, "fetched_at": stamp, "hours": hours, "reports": reports}
    tmp = path.with_suffix(".part")
    tmp.write_text(json.dumps(body, sort_keys=True) + "\n")
    tmp.rename(path)
    return path, received


def fetch_time(path: Path) -> np.datetime64:
    return np.datetime64(datetime.strptime(path.stem, "%Y%m%dT%H%M%SZ"), "ns")


def stored(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.json"))


# -- hourly truth ------------------------------------------------------------------------------


def reports_frame(paths: list[Path], icao_to_station: dict[str, str]) -> pd.DataFrame:
    """Every decoded report of the registered stations, in fetch order, corrections last."""
    rows = []
    for order, path in enumerate(paths):
        for position, record in enumerate(json.loads(path.read_text())["reports"]):
            station = icao_to_station.get(str(record.get("icaoId", "")))
            if station is None or not record.get("rawOb") or "obsTime" not in record:
                continue
            rows.append(
                {"station_id": station, "obs_time": report_time(record), "order": (order, position)}
                | decode(record["rawOb"])
            )
    if not rows:
        return pd.DataFrame(columns=["station_id", "obs_time", *VARIABLES])
    frame = pd.DataFrame(rows).sort_values(["station_id", "obs_time", "order"], kind="mergesort")
    frame = frame.drop_duplicates(["station_id", "obs_time"], keep="last")
    return frame.drop(columns="order").reset_index(drop=True)


def hourly_truth(
    reports: pd.DataFrame, stations: xr.Dataset, start: np.datetime64, end: np.datetime64
) -> xr.Dataset:
    """Hourly truth for labels ``start..end`` in the harness layout (values and ``X_offset_min``)."""
    labels = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="h")
    ids = stations.station_id.values.astype(str)
    data = {v: np.full((len(labels), len(ids)), np.nan, np.float32) for v in VARIABLES}
    offsets = {v: np.full((len(labels), len(ids)), OFFSET_FILL_VALUE, np.int16) for v in VARIABLES}
    groups = dict(tuple(reports.groupby("station_id"))) if len(reports) else {}
    for s, station in enumerate(ids):
        frame = groups.get(station)
        if frame is None:
            continue
        frame = frame[["obs_time", *VARIABLES]].astype({v: np.float32 for v in VARIABLES})
        values, offs = match_station_to_hours(frame.reset_index(drop=True), labels, 15)
        for v in VARIABLES:
            data[v][:, s], offsets[v][:, s] = values[v], offs[v]
    truth = xr.Dataset(coords={"valid_time": labels.values.astype("datetime64[ns]"), **dict(stations.coords)})
    for v in VARIABLES:
        truth[v] = (("valid_time", "station_id"), data[v])
        truth[v + "_offset_min"] = (("valid_time", "station_id"), offsets[v])
    return truth


def truth_as_of(reports: pd.DataFrame, stations: xr.Dataset, start: np.datetime64, t) -> xr.Dataset:
    """The hourly truth an issue at ``t`` may see: only reports measured at or before ``t`` (exact
    time, seconds included) compete for the hourly slots, labels ``start..t``."""
    t = np.datetime64(t, "ns")
    if len(reports):
        reports = reports[reports.obs_time.values.astype("datetime64[ns]") <= t]
    truth = hourly_truth(reports, stations, start, t)
    assert_causal(truth, t)
    return truth


def causal(truth: xr.Dataset, t: np.datetime64) -> xr.Dataset:
    """The truth an issue at ``t`` may see: labels up to ``t``, values measured at or before ``t``."""
    t = np.datetime64(t, "ns")
    out = truth.sel(valid_time=slice(None, t)).copy(deep=True)
    minutes = ((t - out.valid_time.values) / np.timedelta64(1, "m"))[:, None]
    for v in VARIABLES:
        offset = out[v + "_offset_min"].values
        late = (offset != OFFSET_FILL_VALUE) & (offset > minutes)
        out[v].values[late] = np.nan
        out[v + "_offset_min"].values[late] = OFFSET_FILL_VALUE
    assert_causal(out, t)
    return out


def assert_causal(truth: xr.Dataset, t: np.datetime64) -> None:
    """Refuse any label after ``t`` and any finite value measured after ``t``."""
    t = np.datetime64(t, "ns")
    labels = truth.valid_time.values
    if labels.size and labels.max() > t:
        raise ValueError("truth has a label after the issue time")
    for v in VARIABLES:
        values, offset = truth[v].values, truth[v + "_offset_min"].values.astype(np.int64)
        measured = labels[:, None] + offset * np.timedelta64(1, "m")
        if (np.isfinite(values) & (measured > t)).any():
            raise ValueError(f"truth has a {v} reading measured after the issue time")


def sha256_files(paths: list[Path]) -> list[dict]:
    return [{"file": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]
