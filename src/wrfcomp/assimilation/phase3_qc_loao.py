"""Official-free citizen QC driver for Phase 3 Design 2 Track B (leave-out-airports).

This module never opens ``isd_conus_<year>.nc`` and accepts no official observation series.
It imports the shipped pure QC gates (:mod:`wrfcomp.assimilation.qc`), trust transitions
(:mod:`wrfcomp.assimilation.trust`), and the trial runner's asynchronous buddy-alignment
helpers, evaluating citizen observations strictly against citizen-only spatial buddies and
raw GEFS backgrounds adjusted from the nearest static station coordinates.

Design 1's frozen ``phase3_qc.QC_CODE`` files are never modified; this module carries its
own provenance namespace (``phase3-extract-loao.v1``) hashing this driver, its CLI runner,
the full frozen ``phase3_qc.code_digest()``, the static station table, citizen tiles, and
raw GEFS inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from wrfcomp.harness.metrics import evaluation_init_times

from . import phase3_qc as p3
from .qc import (
    Flag,
    background_gate,
    cells,
    citizen_trust_median,
    distances_km,
    spatial_series,
    static_gate,
    temporal_gate,
)
from .trust import StationState, advance_state

EXTRACT_FORMAT = "phase3-extract-loao.v1"
THERMAL = p3.THERMAL
NEIGHBOUR_KM = p3.NEIGHBOUR_KM
WINDOW_MIN = p3.WINDOW_MIN
LAST_ISSUE = {2023: "2023-12-15T12", **p3.YEAR_END}
_REPO = Path(__file__).resolve().parents[3]
STATIONS_CSV = _REPO / "configs" / "stations_conus.csv"
LOAO_QC_FILES = (
    "src/wrfcomp/assimilation/phase3_qc_loao.py",
    "scripts/run_phase3_qc_loao.py",
)


@dataclass(frozen=True)
class StaticSites:
    """Static coordinates and elevations from the station table only (zero official readings)."""

    station_id: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    height: np.ndarray


def issue_times(year: int) -> np.ndarray:
    """The 00Z and 12Z issue times of the year's registered schedule."""
    if year in p3.YEAR_END:
        return p3.issue_times(year)
    if year not in LAST_ISSUE:
        raise ValueError(f"unsupported year {year}; expected one of {sorted(LAST_ISSUE)}")
    return evaluation_init_times(np.datetime64(f"{year}-01-01T00"), np.datetime64(LAST_ISSUE[year]), [0, 12])


def load_static_sites(stations_csv: Path = STATIONS_CSV) -> StaticSites:
    """Read static station ids, latitudes, longitudes, and elevations (registered schema, no obs)."""
    df = pd.read_csv(stations_csv, dtype={"station_id": str})
    required = ("station_id", "lat", "lon", "elevation_m")
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"station table missing required columns: {missing}")
    return StaticSites(
        station_id=df["station_id"].to_numpy(dtype=str),
        lat=df["lat"].to_numpy(dtype=np.float64),
        lon=df["lon"].to_numpy(dtype=np.float64),
        height=df["elevation_m"].to_numpy(dtype=np.float64),
    )


def code_digest(repo: Path = _REPO) -> str:
    """Hash this driver, its CLI script, and the entire frozen Design 1 QC code scope."""
    own = [(name, p3.sha256_file(repo / name)) for name in LOAO_QC_FILES]
    return p3.digest_of("qc-loao-code", own, p3.code_digest())


def load_loao_inputs(
    data_dir: Path, year: int, *, stations_csv: Path = STATIONS_CSV
) -> tuple[object, StaticSites, p3.ForecastArchive]:
    """Load trial helpers, static site coordinates, and raw GEFS cycles. Never opens ISD truth."""
    start = np.datetime64(f"{year}-01-01T00", "ns")
    issues = issue_times(year)
    trial = p3.load_trial_module(start)
    sites = load_static_sites(stations_csv)
    forecasts = p3.load_forecasts(
        Path(data_dir) / "baselines" / "gefs", sites.station_id, start, issues[-1], trial
    )
    cycles = np.unique(forecasts.issued)
    if len(cycles) != len(issues) or not np.array_equal(cycles, issues.astype("datetime64[ns]")):
        raise p3.IncompleteError("the GEFS cycles present are not exactly the year's issue times")
    return trial, sites, forecasts


def run_record(
    year: int,
    tiles_manifest: dict,
    forecasts: p3.ForecastArchive,
    *,
    stations_csv: Path = STATIONS_CSV,
    repo: Path = _REPO,
) -> dict:
    """Provenance record binding every code, coordinate, tile, and GEFS input to LOAO QC weights."""
    record = {
        "format": EXTRACT_FORMAT,
        "tiles_digest": tiles_manifest["digest"],
        "tiles_content_digest": p3.digest_of(sorted(tiles_manifest["parts"].items())),
        "code_digest": code_digest(repo),
        "stations_sha256": p3.sha256_file(stations_csv),
        "gefs_digest": p3.forecasts_digest(forecasts),
        "issues": len(issue_times(year)),
        "window_minutes": WINDOW_MIN,
        "neighbour_km": NEIGHBOUR_KM,
    }
    record["digest"] = p3.digest_of("qc-loao", year, record)
    return record


def current_run_record(
    data_dir: Path, year: int, *, stations_csv: Path = STATIONS_CSV, repo: Path = _REPO
) -> dict:
    """Recompute the LOAO QC provenance record from the files currently on disk."""
    manifest = p3.verify_tiles(Path(data_dir) / "citizen" / "conus" / f"{year}-tiles")
    _, _, forecasts = load_loao_inputs(data_dir, year, stations_csv=stations_csv)
    return run_record(year, manifest, forecasts, stations_csv=stations_csv, repo=repo)


def _raw_background(
    station: p3.StationData,
    var: str,
    choice: np.ndarray,
    nearest: int,
    sites: StaticSites,
    forecasts: p3.ForecastArchive,
    trial,
) -> np.ndarray:
    """Raw GEFS ensemble mean at nearest static site, lapse-adjusted to citizen height. No official bias."""
    cfg = trial.CFG
    bg = np.where(choice >= 0, forecasts.means[var][choice, nearest], np.nan)
    lapse = cfg.lapse_k_per_km * (sites.height[nearest] - station.height) / 1000.0
    return bg + lapse


def station_source_loao(
    station: p3.StationData,
    sites: StaticSites,
    forecasts: p3.ForecastArchive,
    trial,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """First pass without official observations: static/temporal gates and raw-GEFS z-scores.

    Failed reports are masked to NaN so a newer failed report supersedes an older good report
    during asynchronous buddy alignment.
    """
    cfg = trial.CFG
    distances = distances_km(station.lat, station.lon, sites.lat, sites.lon)
    nearest = int(np.argmin(distances))
    streams: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for var in THERMAL:
        full = station.series(var)
        rows = np.flatnonzero(np.isfinite(full))
        if not rows.size:
            continue
        when, values = station.times[rows], full[rows]
        choice = forecasts.choose(when)
        bg = _raw_background(station, var, choice, nearest, sites, forecasts, trial)
        z = (values - bg) / trial.SIGMA
        t2m = station.t2m
        eligible = np.ones(len(rows), dtype=bool)
        state = StationState()
        for k, row in enumerate(rows):
            ns = int(when[k].astype(np.int64))
            v = float(values[k])
            flag = static_gate(var, v, station.height, {"t2m": t2m[row]}, cfg)
            flag |= temporal_gate(var, v, ns, state, cfg, reference=np.nan)
            if flag:
                eligible[k] = False
            trial.record_recent(state, ns, v)
        streams[var] = (when, np.where(eligible, values, np.nan), np.where(eligible, z, np.nan))
    return streams


def station_weights_loao(
    station: p3.StationData,
    neighbours: list[tuple[p3.StationData, dict]],
    sites: StaticSites,
    forecasts: p3.ForecastArchive,
    trial,
    sun: np.ndarray | None = None,
) -> dict[str, p3.ChannelResult]:
    """Second pass without official observations, thermal channels only.

    Uses asynchronous 15-minute as-of buddy alignment (``trial.align``), shipped
    ``spatial_series`` / ``citizen_trust_median`` (requiring >= 3 citizen buddies within 25 km
    with >= 1 in a different 0.1-degree cell from ``station``), raw GEFS background
    (``learn_bias=False``), and explicit ``buddy_fresh=np.isfinite(medians[j])``.
    """
    cfg, sigma = trial.CFG, trial.SIGMA
    dist = distances_km(station.lat, station.lon, sites.lat, sites.lon)
    nearest = int(np.argmin(dist))
    n_buddies = len(neighbours)
    is_official = np.zeros(n_buddies, dtype=bool)
    if n_buddies:
        lat = np.array([n.lat for n, _ in neighbours], dtype=np.float64)
        lon = np.array([n.lon for n, _ in neighbours], dtype=np.float64)
        external = cells(lat, lon) != cells(station.lat, station.lon)
    else:
        external = np.zeros(0, dtype=bool)
    day_all = sun if sun is not None else p3._sun(station.times, station.lat, station.lon, cfg)
    out: dict[str, p3.ChannelResult] = {}
    for var in THERMAL:
        full = station.series(var)
        rows = np.flatnonzero(np.isfinite(full))
        if not rows.size:
            continue
        when, values = station.times[rows], full[rows].astype(np.float32)
        adjusted = np.full((len(when), n_buddies), np.nan)
        votes = adjusted.copy()
        for col, (other, streams) in enumerate(neighbours):
            stream = streams.get(var)
            if stream is None:
                continue
            adjusted[:, col], votes[:, col] = trial.align(stream, when)
            adjusted[:, col] += cfg.lapse_k_per_km * (other.height - station.height) / 1000.0
        choice = forecasts.choose(when)
        background = _raw_background(station, var, choice, nearest, sites, forecasts, trial)
        spatial_flags, _ = spatial_series(values, adjusted, is_official, external, sigma, cfg)
        medians = citizen_trust_median(adjusted, is_official, external, cfg)
        buddy_r, _ = trial.trailing_correlation(when, values, medians)
        t2m = station.t2m
        weights = np.zeros(len(rows), dtype=np.float64)
        earned = np.zeros(len(rows), dtype=bool)
        copied = np.zeros(len(rows), dtype=bool)
        state = StationState()
        for j, row in enumerate(rows):
            value = float(values[j])
            ns = int(when[j].astype(np.int64))
            flag = static_gate(var, value, station.height, {"t2m": t2m[row]}, cfg)
            flag |= temporal_gate(var, value, ns, state, cfg, reference=np.nan)
            flag |= Flag(int(spatial_flags[j]))
            innovation = value - background[j]
            day = bool(day_all[row])
            if np.isfinite(innovation):
                bg_flag, huber = background_gate(innovation / sigma, votes[j], cfg, independent=external)
                flag |= bg_flag
            else:
                huber = 0.0
            hard = bool(flag & (Flag.STATIC | Flag.TEMPORAL))
            state = advance_state(
                state,
                time_ns=ns,
                value=value,
                buddy_r=buddy_r[j],
                buddy_fresh=bool(np.isfinite(medians[j])),
                ref_r=None,
                ref_rmse=np.nan,
                ref_count=0,
                ref_hours=0,
                learn_bias=False,
                wind_drift=False,
                verify_temperature=False,
                reference_time_ns=None,
                reference=np.nan,
                innovation=innovation,
                day=day,
                solar_sensitive=True,
                huber=huber,
                sigma_bg=sigma,
                official=False,
                hard_fail=hard,
                cfg=cfg,
            )
            weights[j] = state.weight
        out[var] = p3.ChannelResult(when, values, weights, np.isfinite(values - background), earned, copied)
    return out
