"""Year-scale driver for the shipped station quality control (Phase 3, `docs/PHASE3_DESIGN.md`).

This module adds no QC rule. It runs the gates of :mod:`wrfcomp.assimilation.qc`, the state
transitions of :mod:`wrfcomp.assimilation.trust` and the matching helpers of the July trial runner
(``scripts/run_qc_trial.py``) in the same order as that runner, for air temperature and dewpoint,
and returns the final weight of every report. The wind channels are read only for the copycat
screen, whose evidence on any channel withdraws the temperature credential.

Everything a report's weight depends on was measured at or before that report: the station's own
earlier reports, neighbours' reports no later than it, official readings measured no later than it,
and forecasts issued strictly before it.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .qc import (
    Flag,
    background_gate,
    cells,
    citizen_trust_median,
    distances_km,
    official_hour_bias,
    solar_elevation,
    spatial_series,
    static_gate,
    temporal_gate,
)
from .trust import StationState, advance_state, suspected_copy, veto_recent_copy

THERMAL = ("t2m", "td2m")
YEAR_END = {2024: "2024-12-17T12"}  # the last issue of each year that may be built
EXTRACT_FORMAT = "phase3-extract.v4"
NEIGHBOUR_KM = 50.0  # docs/PHASE3_DESIGN.md, rule 4
WINDOW_MIN = 15  # rule 3
_REPO = Path(__file__).resolve().parents[3]
# The code a QC result depends on. A change to any of these files makes earlier results stale.
QC_CODE = (
    "scripts/run_phase3_qc.py",
    "scripts/run_qc_trial.py",
    "src/wrfcomp/assimilation/phase3_qc.py",
    "src/wrfcomp/assimilation/madis_archive.py",
    "src/wrfcomp/assimilation/qc.py",
    "src/wrfcomp/assimilation/trust.py",
    "src/wrfcomp/assimilation/tokens.py",
)


class IncompleteError(RuntimeError):
    """A product is missing, partial, or made from other inputs than the ones present."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_of(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()


def code_digest() -> str:
    return digest_of([(name, sha256_file(_REPO / name)) for name in QC_CODE])


def read_manifest(path: Path, what: str) -> dict:
    if not Path(path).is_file():
        raise IncompleteError(f"{what}: no manifest; the step has not completed")
    return json.loads(Path(path).read_text())


def verify_tiles(tiles_dir: Path) -> dict:
    """The tile manifest, after checking that every tile-month file it lists is present and unchanged
    and that there is no file it does not list. A tile set with a missing or altered part is refused."""
    manifest = read_manifest(Path(tiles_dir) / "manifest.json", "tiles")
    present = {str(p.relative_to(tiles_dir)) for p in Path(tiles_dir).glob("*/*.npz")}
    listed = manifest["parts"]
    if present != set(listed):
        missing, extra = len(set(listed) - present), len(present - set(listed))
        raise IncompleteError(
            f"tiles: {missing} tile-month files are missing and {extra} are not in the manifest"
        )
    changed = [name for name, digest in listed.items() if sha256_file(Path(tiles_dir) / name) != digest]
    if changed:
        raise IncompleteError(f"tiles: {len(changed)} tile-month files are not the ones the manifest lists")
    return manifest


def forecasts_digest(forecasts) -> str:
    """Content of the GEFS inputs as the QC reads them: every cycle's times and ensemble means."""
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(forecasts.issued.astype("datetime64[ns]").astype(np.int64)).tobytes())
    digest.update(np.ascontiguousarray(forecasts.valid.astype("datetime64[ns]").astype(np.int64)).tobytes())
    for name in sorted(forecasts.means):
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(forecasts.means[name], dtype=np.float64).tobytes())
    return digest.hexdigest()


def load_year_inputs(data_dir: Path, year: int):
    """Truth and GEFS for one year's QC, as the QC reads them: (trial module, truth, forecasts)."""
    start = np.datetime64(f"{year}-01-01T00", "ns")
    issues = issue_times(year)
    trial = load_trial_module(start)
    with xr.open_dataset(Path(data_dir) / "truth" / f"isd_conus_{year}.nc") as ds:
        # No label before 1 January, and no reading measured before it.
        truth = within_year(ds.sel(valid_time=slice(start, issues[-1])).load(), start)
    forecasts = load_forecasts(
        Path(data_dir) / "baselines" / "gefs", truth.station_id.values, start, issues[-1], trial
    )
    cycles = np.unique(forecasts.issued)
    if len(cycles) != len(issues) or not np.array_equal(cycles, issues.astype("datetime64[ns]")):
        raise IncompleteError("the GEFS cycles present are not exactly the year's issue times")
    return trial, truth, forecasts


def run_record(data_dir: Path, year: int, tiles_manifest: dict, forecasts) -> dict:
    """What a QC run is made from. Its digest names the run."""
    record = {
        "tiles_digest": tiles_manifest["digest"],
        "tiles_content_digest": digest_of(sorted(tiles_manifest["parts"].items())),
        "code_digest": code_digest(),
        "truth_sha256": sha256_file(Path(data_dir) / "truth" / f"isd_conus_{year}.nc"),
        "gefs_digest": forecasts_digest(forecasts),
        "issues": len(issue_times(year)),
        "window_minutes": WINDOW_MIN,
        "neighbour_km": NEIGHBOUR_KM,
    }
    record["digest"] = digest_of("qc", year, record)
    return record


def current_run_record(data_dir: Path, year: int) -> dict:
    """The record a QC run would have if it were made now from the files present (tiles verified)."""
    manifest = verify_tiles(Path(data_dir) / "citizen" / "conus" / f"{year}-tiles")
    _, _, forecasts = load_year_inputs(data_dir, year)
    return run_record(data_dir, year, manifest, forecasts)


WIND = ("u10", "v10", "wind_speed")
HALF_HOUR_NS = 1_800_000_000_000
_TRIAL_PATH = Path(__file__).resolve().parents[3] / "scripts" / "run_qc_trial.py"


def issue_times(year: int) -> np.ndarray:
    """The 00Z and 12Z issue times of the year's schedule."""
    from wrfcomp.harness.metrics import evaluation_init_times

    return evaluation_init_times(np.datetime64(f"{year}-01-01T00"), np.datetime64(YEAR_END[year]), [0, 12])


def load_trial_module(origin: np.datetime64):
    """The July trial runner, loaded as a module, with its time origin set to ``origin``.

    The origin only shifts the zero of an hours axis whose range is used; no rule depends on it.
    """
    spec = importlib.util.spec_from_file_location("wrfcomp_qc_trial_runner", _TRIAL_PATH)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load the QC trial runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    module.START = np.datetime64(origin, "ns")
    return module


# ----------------------------------------------------------------------------- forecasts


@dataclass(frozen=True)
class ForecastArchive:
    issued: np.ndarray  # datetime64[ns], one entry per (cycle, lead)
    valid: np.ndarray
    means: dict  # variable -> (entry, official station) ensemble mean
    grid_start_ns: int
    on_grid: np.ndarray  # choice at k * 30 min
    inside: np.ndarray  # choice strictly inside (k * 30 min, (k + 1) * 30 min)

    def as_tuple(self):
        return self.issued, self.valid, self.means

    def choose(self, times: np.ndarray) -> np.ndarray:
        """Same result as the trial's ``forecast_indices``, by table.

        Issue times, valid times, their midpoints and the three-hour limits all fall on whole half
        hours, so the choice is constant strictly inside each half hour.
        """
        t = np.asarray(times, dtype="datetime64[ns]").astype(np.int64) - self.grid_start_ns
        k = t // HALF_HOUR_NS
        out = np.full(t.shape, -1, dtype=int)
        ok = (k >= 0) & (k < len(self.on_grid))
        exact = t[ok] % HALF_HOUR_NS == 0
        out[ok] = np.where(exact, self.on_grid[k[ok]], self.inside[k[ok]])
        return out


def load_forecasts(gefs_dir: Path, station_ids, first, last, trial) -> ForecastArchive:
    """Ensemble means of every GEFS cycle from ``first`` to ``last`` (inclusive), as the trial reads them."""
    first, last = np.datetime64(first, "ns"), np.datetime64(last, "ns")
    issued, valid = [], []
    values: dict[str, list] = {var: [] for var in (*trial.DIRECT, "wind_speed")}
    for path in sorted(gefs_dir.glob("*.zarr")):
        try:
            cycle = pd.to_datetime(path.stem, format="%Y%m%d%H").to_datetime64()
        except ValueError:
            continue
        if not first <= cycle <= last:
            continue
        with xr.open_zarr(path, chunks=None, consolidated=True) as ds:
            if ds.sizes["init_time"] != 1 or ds.init_time.values[0] != cycle:
                raise ValueError("unexpected GEFS cycle")
            block = ds.sel(station_id=station_ids).isel(init_time=0)
            for var in trial.DIRECT:
                raw = block[var].transpose("lead_time", "station_id", "ens_member").values
                values[var].append(raw.mean(axis=2, dtype=float))
                if var == "u10":
                    u = raw
                if var == "v10":
                    values["wind_speed"].append(np.hypot(u, raw).mean(axis=2, dtype=float))
            issued.extend([cycle] * block.sizes["lead_time"])
            valid.extend(cycle + block.lead_time.values)
    if not issued:
        raise ValueError("no GEFS cycles in the requested range")
    return archive_from_arrays(
        np.array(issued), np.array(valid), {k: np.concatenate(v) for k, v in values.items()}, trial
    )


def archive_from_arrays(issued, valid, means, trial) -> ForecastArchive:
    """Wrap forecast entries (as the trial's ``forecast_archive`` returns them) with the choice table."""
    issued_a = np.asarray(issued, dtype="datetime64[ns]")
    valid_a = np.asarray(valid, dtype="datetime64[ns]")
    half = np.timedelta64(30, "m")
    start = issued_a.min().astype("datetime64[D]").astype("datetime64[ns]") - np.timedelta64(1, "D")
    stop = valid_a.max() + np.timedelta64(4, "h")
    grid = np.arange(start, stop, half)
    return ForecastArchive(
        issued=issued_a,
        valid=valid_a,
        means=means,
        grid_start_ns=int(start.astype(np.int64)),
        on_grid=trial.forecast_indices(issued_a, valid_a, grid),
        inside=trial.forecast_indices(issued_a, valid_a, grid + np.timedelta64(1, "s")),
    )


# ----------------------------------------------------------------------------- stations


@dataclass
class StationData:
    """One citizen station: stored position and its reports in time order (datetime64[ns])."""

    lat: float
    lon: float
    height: float
    times: np.ndarray
    t2m: np.ndarray
    td2m: np.ndarray
    wind_speed: np.ndarray
    wind_dir: np.ndarray

    def __post_init__(self):
        # numpy float64 scalars, as in the trial runner: a Python float next to a float32 official
        # elevation would be computed in float32.
        self.lat, self.lon, self.height = (np.float64(x) for x in (self.lat, self.lon, self.height))

    def series(self, var: str) -> np.ndarray:
        if var in THERMAL:
            return getattr(self, var).astype(float)
        speed, direction = self.wind_speed.astype(float), self.wind_dir.astype(float)
        if var == "wind_speed":
            return speed
        with np.errstate(invalid="ignore"):
            ok = np.isfinite(speed) & np.isfinite(direction) & (direction >= 0) & (direction <= 360)
        rad = np.radians(np.where(ok, direction, 0.0))
        comp = -speed * (np.sin(rad) if var == "u10" else np.cos(rad))
        comp = np.where(comp == 0.0, 0.0, comp)
        return np.where(ok, comp, np.nan).astype(np.float32).astype(float)


@dataclass(frozen=True)
class Official:
    lat: np.ndarray
    lon: np.ndarray
    height: np.ndarray
    series: dict  # variable -> list over stations of (measured, value, z)


def within_year(truth: xr.Dataset, start, variables=("t2m", "td2m", "u10", "v10", "msl")) -> xr.Dataset:
    """Official truth for one year's state: no label before ``start`` and no reading measured before it.

    A label at 00Z on 1 January with a negative offset was measured in the previous year. Its value
    is removed, so it cannot enter January state.
    """
    start = np.datetime64(start, "ns")
    out = truth.sel(valid_time=slice(start, None)).copy(deep=True)
    labels = out.valid_time.values.astype("datetime64[ns]")
    for var in variables:
        if var not in out:
            continue
        offset = out[var + "_offset_min"].values.astype(np.int64)
        measured = labels[:, None].astype(np.int64) + offset * 60 * 10**9
        early = measured < start.astype(np.int64)
        values = out[var].values
        values[early] = np.nan
    return out


def official_from_truth(truth: xr.Dataset, forecasts: ForecastArchive, trial) -> Official:
    return Official(
        lat=truth.latitude.values,
        lon=truth.longitude.values,
        height=truth.elevation_m.values,
        series=trial.official_series(truth, forecasts.as_tuple()),
    )


def _sun(times, lat, lon, cfg) -> np.ndarray:
    return np.array([solar_elevation(t, lat, lon) > cfg.day_degrees for t in times], dtype=bool)


def _background(station, var, when, choice, distances, nearest, official, forecasts, trial, *, first_pass):
    """Bias-corrected background at the station's height. The order of the additions is the trial's."""
    cfg = trial.CFG
    bg = np.where(choice >= 0, forecasts.means[var][choice, nearest], np.nan)
    lapse = cfg.lapse_k_per_km * (official.height[nearest] - station.height) / 1000
    measured, _, official_z = official.series[var][nearest]
    correction = official_hour_bias(measured, official_z * trial.SIGMA, when)
    if distances[nearest] > cfg.official_bias_radius_km:
        correction[:] = np.nan
    if first_pass:
        bg += correction + lapse
    else:
        bg += lapse
        bg += correction
    return bg


def station_source(station: StationData, official: Official, forecasts: ForecastArchive, trial, sun=None):
    """First pass (the trial's ``citizen_sources`` for one station).

    Returns the station's eligible thermal streams, for its neighbours' spatial checks, and the
    times at which any channel looked like a copy of the official station.
    """
    cfg = trial.CFG
    distances = distances_km(station.lat, station.lon, official.lat, official.lon)
    nearest = int(np.argmin(distances))
    near_official = np.flatnonzero(distances <= cfg.radius_km)
    streams, faults = {}, []
    for var in WIND:
        full = station.series(var)
        rows = np.flatnonzero(np.isfinite(full))
        if not rows.size:
            continue
        when, values = station.times[rows], full[rows]
        _, _, r, rms, count, span = trial.official_evidence(official.series[var][nearest], when, values)
        copycat = suspected_copy(r, rms, trial.SIGMA, cfg, thermal=False, count=count, hours=span)
        faults.extend(when[copycat].astype(np.int64).tolist())
    for var in THERMAL:
        full = station.series(var)
        rows = np.flatnonzero(np.isfinite(full))
        if not rows.size:
            continue
        when, values = station.times[rows], full[rows]
        reference, _, r, rms, count, span = trial.official_evidence(
            official.series[var][nearest], when, values
        )
        copycat = suspected_copy(r, rms, trial.SIGMA, cfg, thermal=True, count=count, hours=span)
        faults.extend(when[copycat].astype(np.int64).tolist())
        choice = forecasts.choose(when)
        bg = _background(
            station, var, when, choice, distances, nearest, official, forecasts, trial, first_pass=True
        )
        z = (values - bg) / trial.SIGMA
        storm = trial.official_storm_support(
            when, values, near_official, official.height, station.height, official.series[var]
        )
        day = (sun if sun is not None else _sun(station.times, station.lat, station.lon, cfg))[rows]
        t2m = station.t2m
        biases = [0.0, 0.0]
        eligible = ~copycat
        state = StationState()
        close = distances[nearest] <= cfg.radius_km
        for k, row in enumerate(rows):
            ns = int(when[k].astype(np.int64))
            v = float(values[k])
            flag = static_gate(var, v, station.height, {"t2m": t2m[row]}, cfg)
            flag |= temporal_gate(var, v, ns, state, cfg, reference=reference[k] if close else np.nan)
            if k and eligible[k - 1] and flag == Flag.TEMPORAL and storm[k]:
                flag = Flag(0)
            if flag:
                eligible[k] = False
            if np.isfinite(z[k]):
                regime = int(day[k])
                z[k] -= biases[regime] / trial.SIGMA
                if eligible[k] and abs(z[k]) <= cfg.gross_cut:
                    biases[regime] += cfg.ema_alpha * z[k] * trial.SIGMA
            trial.record_recent(state, ns, v)
        streams[var] = (when, np.where(eligible, values, np.nan), np.where(eligible, z, np.nan))
    return streams, np.unique(np.asarray(faults, dtype=np.int64))


@dataclass
class ChannelResult:
    when: np.ndarray  # datetime64[ns]
    values: np.ndarray  # float64
    weights: np.ndarray  # final QC weight after each report
    scored: np.ndarray  # True where the report had a background (the trial's weight sample)
    earned: np.ndarray  # temperature credential held after the report (t2m channel)
    copycat: np.ndarray


def station_weights(
    station: StationData,
    neighbours: list[tuple[StationData, dict]],
    copy_times: np.ndarray,
    official: Official,
    forecasts: ForecastArchive,
    trial,
    sun=None,
) -> dict[str, ChannelResult]:
    """Second pass (the trial's ``aggregate_station`` for one station), thermal channels only.

    ``neighbours`` are the other citizen stations within the buddy radius, each with the streams
    its own first pass produced. ``copy_times`` are this station's fault times from its first pass.
    """
    cfg, sigma = trial.CFG, trial.SIGMA
    dist = distances_km(station.lat, station.lon, official.lat, official.lon)
    nearest = int(np.argmin(dist))
    buddies = np.flatnonzero(dist <= cfg.radius_km)
    close = dist[nearest] <= cfg.radius_km
    external = np.ones(len(buddies), dtype=bool)
    if neighbours:
        lat = np.array([n.lat for n, _ in neighbours])
        lon = np.array([n.lon for n, _ in neighbours])
        external = np.r_[external, cells(lat, lon) != cells(station.lat, station.lon)]
    is_official = np.arange(len(external)) < len(buddies)
    copy_times = np.asarray(copy_times, dtype=np.int64)
    day_all = sun if sun is not None else _sun(station.times, station.lat, station.lon, cfg)
    out = {}
    for var in THERMAL:
        full = station.series(var)
        rows = np.flatnonzero(np.isfinite(full))
        if not rows.size:
            continue
        # The trial's second pass works on the stored float32 values (its first pass on float64).
        when, values = station.times[rows], full[rows].astype(np.float32)
        last_copy = np.searchsorted(copy_times, when.astype(np.int64), side="right") - 1
        adjusted = np.full((len(when), len(external)), np.nan)
        votes = adjusted.copy()
        for col, (other, streams) in enumerate(neighbours, start=len(buddies)):
            stream = streams.get(var)
            if stream is None:
                continue
            adjusted[:, col], votes[:, col] = trial.align(stream, when)
            adjusted[:, col] += cfg.lapse_k_per_km * (other.height - station.height) / 1000
        reference, ref_times, ref_r, ref_rmse, ref_count, ref_hours = trial.official_evidence(
            official.series[var][nearest], when, values
        )
        for col, buddy in enumerate(buddies):
            adjusted[:, col], votes[:, col] = trial.align(official.series[var][buddy], when)
            adjusted[:, col] += cfg.lapse_k_per_km * (official.height[buddy] - station.height) / 1000
        choice = forecasts.choose(when)
        background = _background(
            station, var, when, choice, dist, nearest, official, forecasts, trial, first_pass=False
        )
        spatial_flags, _ = spatial_series(values, adjusted, is_official, external, sigma, cfg)
        medians = citizen_trust_median(adjusted, is_official, external, cfg)
        buddy_r, _ = trial.trailing_correlation(when, values, medians)
        storm = trial.official_storm_support(
            when, values, buddies, official.height, station.height, official.series[var]
        )
        t2m = station.t2m
        weights = np.zeros(len(rows))
        earned = np.zeros(len(rows), dtype=bool)
        copied = np.zeros(len(rows), dtype=bool)
        state = StationState()
        for j, row in enumerate(rows):
            value = float(values[j])
            ns = int(when[j].astype(np.int64))
            flag = static_gate(var, value, station.height, {"t2m": t2m[row]}, cfg)
            flag |= temporal_gate(var, value, ns, state, cfg, reference=reference[j] if close else np.nan)
            rescued = flag == Flag.TEMPORAL and state.last_usable and storm[j]
            if rescued:
                flag = Flag(0)
            flag |= Flag(int(spatial_flags[j]))
            innovation = value - background[j]
            day = bool(day_all[row])
            bias = state.bias_day if day else state.bias_night
            if np.isfinite(innovation):
                bg_flag, huber = background_gate(
                    (innovation - (bias or 0)) / sigma, votes[j], cfg, independent=external
                )
                if rescued and huber == 0:
                    huber = cfg.huber_cut / abs((innovation - (bias or 0)) / sigma)
                flag |= bg_flag
            else:
                huber = 0
            hard = bool(flag & (Flag.STATIC | Flag.TEMPORAL))
            state = advance_state(
                state,
                time_ns=ns,
                value=value,
                buddy_r=buddy_r[j],
                buddy_fresh=np.isfinite(medians[j]),
                ref_r=ref_r[j] if np.isfinite(ref_r[j]) else None,
                ref_rmse=ref_rmse[j],
                ref_count=ref_count[j],
                ref_hours=ref_hours[j],
                wind_drift=False,
                verify_temperature=var == "t2m" and close,
                reference_time_ns=int(ref_times[j]) if ref_times[j] >= 0 else None,
                reference=reference[j],
                innovation=innovation,
                day=day,
                solar_sensitive=True,
                huber=huber,
                sigma_bg=sigma,
                official=False,
                hard_fail=hard,
                cfg=cfg,
            )
            if last_copy[j] >= 0:
                state = veto_recent_copy(state, ns, int(copy_times[last_copy[j]]), cfg)
            weights[j] = state.weight
            earned[j] = state.earned
            copied[j] = state.copycat
        out[var] = ChannelResult(when, values, weights, np.isfinite(values - background), earned, copied)
    return out


def issue_snapshot(result: ChannelResult, issues: np.ndarray, window_minutes: int = 15):
    """Rule 3 of the design, for each issue time T: the latest report measured in [T - window, T]
    whose final QC weight is above zero, or nothing.

    A report with weight zero is not a report that counts, so it does not hide an earlier one that
    does. A report after T is never read.

    Returns (has_report, value, weight, earned, age in nanoseconds). The age is exact: it is never
    rounded, because it decides which of one token's rows is the newest.
    """
    issues = np.asarray(issues, dtype="datetime64[ns]")
    usable = np.flatnonzero(result.weights > 0)
    when = result.when[usable]
    ix = np.searchsorted(when, issues, side="right") - 1
    ok = ix >= 0
    safe = np.where(ok, ix, 0)
    if not usable.size:
        none = np.zeros(len(issues), dtype=bool)
        return (
            none,
            np.full(len(issues), np.nan),
            np.zeros(len(issues)),
            none.copy(),
            np.zeros(len(issues), np.int64),
        )
    age_ns = (issues - when[safe]).astype(np.int64)
    ok &= (age_ns >= 0) & (age_ns <= window_minutes * 60 * 10**9)
    row = usable[np.where(ok, ix, 0)]
    return (
        ok,
        np.where(ok, result.values[row], np.nan),
        np.where(ok, result.weights[row], 0.0),
        np.where(ok, result.earned[row], False),
        np.where(ok, age_ns, 0),
    )
