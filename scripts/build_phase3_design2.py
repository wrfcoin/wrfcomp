"""Build Phase 3 Design 2 forecasts (Track A arm C2; Track B 5-fold LOAO arms B_LOAO and C2_LOAO).

Builds forecasts only; never scores. Keeps all Design 1 frozen modules and scored stores
strictly untouched and read-only. Supports ``--stop-after-month YYYY-MM`` and ``--out-dir``
for isolated-scratch pre-fetch hash-parity replays (Stage 2a).

Outputs under ``--out-dir`` (default ``<data-dir>/candidates/`` and ``<data-dir>/phase3-design2/<year>/``):
- Track A: ``gefs-station-corrected-citizen-d2/<year>-MM.zarr`` + ``covered_stations_a.csv``
- Track B: ``gefs-loao-official/<year>-MM.zarr``, ``gefs-loao-citizen/<year>-MM.zarr``,
  ``folds.csv``, ``covered_stations_loao.csv``, and optional post-freeze aggregate relay audit.
- Official station ids and aggregate diagnostics only; never prints or writes citizen ids or positions.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from wrfcomp.assimilation import phase3_qc as p3
from wrfcomp.assimilation import phase3_qc_loao as loao_qc
from wrfcomp.assimilation.madis_archive import (
    ARRIVAL_POLICY,
    VALUE_COLUMNS,
    arrived_by_next_issue_ns,
    latest_version_rows,
)
from wrfcomp.models import citizen_arms as arms
from wrfcomp.models import citizen_design2 as d2
from wrfcomp.models import gefs_station_v2 as v2

STOP_FREE_BYTES = 40 * 10**9
MIN_COVERED_STATIONS = 50


def _load_frozen_builder():
    path = Path(__file__).resolve().with_name("build_phase3_arms.py")
    spec = importlib.util.spec_from_file_location("wrfcomp_build_phase3_arms_d2", path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load scripts/build_phase3_arms.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


phase3_frozen = _load_frozen_builder()
BuildRefusedError = phase3_frozen.BuildRefusedError


def check_free_space(data_dir: Path, min_free_bytes: int = STOP_FREE_BYTES) -> None:
    if min_free_bytes <= 0:
        return
    free = shutil.disk_usage(data_dir).free
    if free < min_free_bytes:
        raise BuildRefusedError(
            f"free space {free / 1e9:.1f} GB is below {min_free_bytes / 1e9:.1f} GB stop line"
        )


def _is_sha256_hex(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdefABCDEF" for c in value)


def _paths_overlap(a: Path, b: Path) -> bool:
    ra, rb = a.resolve(), b.resolve()
    return ra == rb or ra in rb.parents or rb in ra.parents


def _require_isolated_partial_out(data: Path, out_root: Path | None, stop_after_month: str | None) -> None:
    if stop_after_month is None:
        return
    if out_root is None:
        raise BuildRefusedError("--stop-after-month requires an isolated --out-dir")
    if _paths_overlap(out_root, data / "candidates") or _paths_overlap(out_root, data / "phase3-design2"):
        raise BuildRefusedError(
            "partial replay --out-dir must not overlap live candidates/ or phase3-design2/"
        )


def load_loao_extract(
    path: Path, year: int, issues: np.ndarray, stations: list[str]
) -> tuple[xr.Dataset, Callable[[int], dict]]:
    if not path.is_file():
        raise BuildRefusedError("there is no LOAO citizen extract for the year")
    with xr.open_dataset(path, engine="h5netcdf") as ds:
        ds = ds.load()
    a = ds.attrs
    if a.get("format") != loao_qc.EXTRACT_FORMAT or a.get("arrival_policy") != ARRIVAL_POLICY:
        raise BuildRefusedError("the LOAO extract was not made under phase3-extract-loao.v1 and Amendment 1")
    if int(a.get("complete", 0)) != 1 or int(a.get("year", 0)) != year or not a.get("run_digest"):
        raise BuildRefusedError("the LOAO extract is not marked as a complete QC run of this year")
    if int(a.get("official_observations_used", 1)) != 0:
        raise BuildRefusedError("the LOAO extract must not use official observations")
    issue_time = ds.issue_time.values.astype(np.int64).astype("datetime64[s]").astype("datetime64[ns]")
    if not np.array_equal(issue_time, np.asarray(issues, dtype="datetime64[ns]")):
        raise BuildRefusedError("the LOAO extract is not on the year's issue schedule")
    if ds.official_station_id.values.astype(str).tolist() != list(stations):
        raise BuildRefusedError("the LOAO extract was not made for the registered official stations")
    n = ds.sizes["station"]
    order = np.argsort(ds.row_issue.values, kind="stable")
    issue = ds.row_issue.values[order]
    bounds = np.searchsorted(issue, np.arange(ds.sizes["issue"] + 1))
    names = ("row_station", "row_var", "row_value", "row_weight", "row_age_ns")
    columns = {k: ds[k].values[order] for k in names}

    def snapshot(i: int) -> dict:
        rows = slice(bounds[i], bounds[i + 1])
        out = {}
        for v, var in enumerate(d2.VARIABLES):
            value, weight, age = np.full(n, np.nan), np.zeros(n), np.zeros(n, dtype=np.int64)
            pick = columns["row_var"][rows] == v
            station = columns["row_station"][rows][pick]
            value[station] = columns["row_value"][rows][pick]
            weight[station] = columns["row_weight"][rows][pick]
            age[station] = columns["row_age_ns"][rows][pick]
            out[var] = (value, weight, age)
        return out

    return ds, snapshot


def verify_loao_extract_is_current(
    data: Path,
    year: int,
    extract_path: Path,
    extract: xr.Dataset,
    *,
    stations_csv: Path = loao_qc.STATIONS_CSV,
) -> None:
    summary = extract_path.with_name(f"issue_extract_loao_{year}.summary.json")
    if not summary.is_file() or json.loads(summary.read_text()).get("extract_sha256") != p3.sha256_file(
        extract_path
    ):
        raise BuildRefusedError("the LOAO extract does not match its published summary")
    try:
        current = loao_qc.current_run_record(data, year, stations_csv=stations_csv)
    except p3.IncompleteError as exc:
        raise BuildRefusedError(f"LOAO QC inputs behind the extract are not intact: {exc}") from None
    if current["digest"] != extract.attrs.get("run_digest"):
        raise BuildRefusedError(
            "the LOAO extract is stale relative to current tiles, stations, GEFS, or code"
        )


def _gefs_cycle_path(data: Path, t: np.datetime64) -> Path:
    stamp = pd.Timestamp(t).strftime("%Y%m%d%H")
    return data / "baselines" / "gefs" / f"{stamp}.zarr"


def _load_gefs_cycle(data: Path, t: np.datetime64, stations: list[str]) -> xr.Dataset:
    path = _gefs_cycle_path(data, t)
    if not path.is_dir():
        raise BuildRefusedError(f"missing raw GEFS cycle for {pd.Timestamp(t).isoformat()}")
    with xr.open_zarr(path, chunks=None, consolidated=True) as ds:
        if ds.sizes["init_time"] != 1 or ds.init_time.values[0] != t:
            raise BuildRefusedError(f"unexpected GEFS init_time in {path.name}")
        return ds.sel(station_id=stations).load()


def _gefs_12h_means(ds: xr.Dataset) -> dict[str, np.ndarray]:
    slot = ds.sel(lead_time=np.timedelta64(12, "h")).isel(init_time=0)
    return {
        var: slot[var].transpose("station_id", "ens_member").values.astype(np.float64).mean(axis=-1)
        for var in ("t2m", "td2m", "u10", "v10")
    }


def _collect_track_b_contributions(
    year: int,
    issues: np.ndarray,
    full_height: np.ndarray,
    pairs_loao: d2.OrientedPairs,
    c_height: np.ndarray,
    gefs_prev_means: list[dict[str, np.ndarray] | None],
    snapshots: list[dict],
    *,
    token=None,
) -> list[tuple[int, int, str, int, np.datetime64, float]]:
    """Deterministically collect ``(c_idx, s_idx, var, lead_h, issue_t, W_cs)`` for post-freeze audit."""
    tok = None if token is None else np.unique(np.asarray(token), return_inverse=True)[1]
    order = np.argsort(pairs_loao.official, kind="stable")
    by_official = (
        np.split(order, np.flatnonzero(np.diff(pairs_loao.official[order])) + 1) if order.size else []
    )
    pair_lapse = (
        d2.LAPSE_K_PER_KM * (full_height[pairs_loao.official] - c_height[pairs_loao.citizen]) / 1000.0
        if pairs_loao.citizen.size
        else np.empty(0, dtype=np.float64)
    )
    history: dict[str, list[tuple[np.datetime64, np.ndarray]]] = {v: [] for v in d2.VARIABLES}
    records: list[tuple[int, int, str, int, np.datetime64, float]] = []
    for t, prev_mean, snap in zip(issues, gefs_prev_means, snapshots, strict=True):
        for var in d2.VARIABLES:
            hist = history[var]
            while hist and hist[0][0] < t - (d2.USUAL_DAYS + 2) * d2.DAY:
                hist.pop(0)
            if not pairs_loao.citizen.size or prev_mean is None:
                hist.append((t, np.full(pairs_loao.citizen.size, np.nan)))
                continue
            value, weight, *rest = snap[var]
            value, weight = np.asarray(value, float), np.asarray(weight, float)
            age = np.asarray(rest[0]).astype(np.int64) if rest else np.zeros(len(value), dtype=np.int64)
            g_adj = prev_mean[var][pairs_loao.official] + pair_lapse
            y, w = value[pairs_loao.citizen], weight[pairs_loao.citizen]
            with np.errstate(invalid="ignore"):
                eligible = np.isfinite(y) & (w > 0.0) & np.isfinite(g_adj)
            diff = np.where(eligible, y - g_adj, np.nan)
            earlier = [
                d
                for it, d in hist
                if t - d2.USUAL_DAYS * d2.DAY <= it <= t - d2.DAY
                and (t - it) % d2.DAY == np.timedelta64(0, "ns")
            ]
            if earlier:
                stack = np.stack(earlier)
                cnt = np.isfinite(stack).sum(axis=0)
                usual = np.full(pairs_loao.citizen.size, np.nan)
                enough = cnt >= d2.USUAL_MIN
                if enough.any():
                    usual[enough] = np.nanmedian(stack[:, enough], axis=0)
                dep = np.clip(diff - usual, -d2.CLIP_K, d2.CLIP_K)
                w_eff = d2.temper_trust_weight(w)
                u10, v10 = np.asarray(prev_mean["u10"], float), np.asarray(prev_mean["v10"], float)
                for rows in by_official:
                    use_base = rows[np.isfinite(dep[rows])]
                    if use_base.size < d2.MIN_NEIGHBOURS:
                        continue
                    s_idx = int(pairs_loao.official[use_base[0]])
                    for lead_h in d2.LEAD_HOURS:
                        px, py = d2.upwind_target_km(
                            float(u10[s_idx]), float(v10[s_idx]), lead_h, d2.RADIUS_KM
                        )
                        geo_w = d2.upwind_gaussian_weight(
                            pairs_loao.rx_km[use_base],
                            pairs_loao.ry_km[use_base],
                            pairs_loao.distance_km[use_base],
                            px,
                            py,
                            d2.DISTANCE_SCALE_KM,
                        )
                        weights = w_eff[use_base] * geo_w
                        keep = d2._one_per_token(tok, pairs_loao.citizen, use_base, age, weights)
                        use, weights = use_base[keep], weights[keep]
                        if (
                            use.size < d2.MIN_NEIGHBOURS
                            or np.unique(pairs_loao.cell[use]).size < d2.MIN_CELLS
                        ):
                            continue
                        for row_idx, wt in zip(use, weights, strict=True):
                            records.append(
                                (int(pairs_loao.citizen[row_idx]), s_idx, var, int(lead_h), t, float(wt))
                            )
            hist.append((t, diff))
    return records


def load_full_resolution_streams_from_tiles(
    tiles_dir: Path,
    year: int,
    extract: xr.Dataset,
    needed_indices: set[int],
) -> dict[tuple[int, str], tuple[np.ndarray, np.ndarray]]:
    """Load retained full-resolution thermal streams for ``needed_indices`` from ``<year>-tiles/``."""
    loao_qc.verify_tiles(tiles_dir)
    start_ns = int(np.datetime64(f"{year}-01-01T00", "ns").astype(np.int64))
    key_to_idx: dict[tuple, int] = {}
    tiles_needed: set[str] = set()
    tokens = extract.station_token.values
    lats = extract.lat.values
    lons = extract.lon.values
    heights = extract.elevation_m.values
    for idx in sorted(needed_indices):
        key = (bytes(tokens[idx]), float(lats[idx]), float(lons[idx]), float(heights[idx]))
        key_to_idx[key] = idx
        ti, tj = int(np.floor(float(lats[idx]))), int(np.floor(float(lons[idx])))
        tiles_needed.add(f"{ti:+03d}_{tj:+04d}")
    chunks: dict[int, list] = {idx: [] for idx in needed_indices}
    for tile in sorted(tiles_needed):
        for part in sorted((tiles_dir / tile).glob("*.npz")):
            with np.load(part) as z:
                keys = list(
                    zip(
                        z["ident_token"].tolist(),
                        z["ident_lat"].tolist(),
                        z["ident_lon"].tolist(),
                        z["ident_elevation_m"].tolist(),
                        strict=True,
                    )
                )
                local_to_idx = np.array([key_to_idx.get(k, -1) for k in keys], dtype=np.int64)
                if not (local_to_idx >= 0).any():
                    continue
                gid = local_to_idx[z["row_ident"]]
                obs, rec = z["obs_time"], z["received_time"]
                keep = np.flatnonzero((gid >= 0) & (obs >= start_ns) & arrived_by_next_issue_ns(obs, rec))
                if not keep.size:
                    continue
                gid = gid[keep]
                arrays = [obs[keep], rec[keep], *(z[c][keep] for c in VALUE_COLUMNS)]
                cuts = np.flatnonzero(np.r_[True, gid[1:] != gid[:-1], True])
                for a, b in zip(cuts[:-1], cuts[1:], strict=True):
                    chunks[int(gid[a])].append([x[a:b] for x in arrays])
    streams: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}
    for idx, parts in chunks.items():
        if not parts:
            continue
        cols = [np.concatenate([c[i] for c in parts]) for i in range(2 + len(VALUE_COLUMNS))]
        pick = latest_version_rows(np.zeros(len(cols[0]), dtype=np.int8), cols[0], cols[1], cols[2:])
        when = cols[0][pick].astype("datetime64[ns]")
        for v_i, var in enumerate(d2.VARIABLES):
            vals = cols[2 + v_i][pick].astype(np.float64)
            ok = np.isfinite(vals)
            streams[(idx, var)] = (when[ok], vals[ok])
    return streams


def build_track_a(
    data: Path,
    year: int,
    *,
    out_root: Path | None = None,
    stop_after_month: str | None = None,
    issues=None,
    stations=None,
    verify_extract: bool = True,
    min_covered: int = MIN_COVERED_STATIONS,
    reference_covered_sha256: str | None = None,
    min_free_bytes: int = STOP_FREE_BYTES,
) -> dict:
    """Build Track A arm C2 (and optionally stop after ``stop_after_month`` in isolated scratch)."""
    check_free_space(data, min_free_bytes)
    _require_isolated_partial_out(data, out_root, stop_after_month)
    start, _ = v2.year_bounds(year)
    start = np.datetime64(start, "ns")
    issues = loao_qc.issue_times(year) if issues is None else np.asarray(issues, dtype="datetime64[ns]")
    stations = phase3_frozen.registered_stations() if stations is None else list(stations)
    extract_path = data / "citizen" / "conus" / f"issue_extract_{year}.nc"
    extract, snapshot = phase3_frozen.load_extract(extract_path, year, issues, stations)
    if verify_extract:
        if year == 2024:
            phase3_frozen.verify_extract_is_current(data, year, extract_path, extract)
        else:
            raise BuildRefusedError(
                f"standard extract provenance verification for {year} is not yet implemented "
                "(deferred until Stage 2 after manager GO)"
            )
    months = sorted((data / "candidates" / phase3_frozen.A_DIR).glob(f"{year}-??.zarr"))
    seen = []
    for path in months:
        with xr.open_zarr(path, chunks=None) as source:
            if source.station_id.values.astype(str).tolist() != stations:
                raise BuildRefusedError(f"v3 store {path.stem} is not on the registered stations")
            seen.append(source.init_time.values.astype("datetime64[ns]"))
    if not seen or not np.array_equal(np.concatenate(seen), issues):
        raise BuildRefusedError("the v3 stores are not exactly the year's issue schedule")
    with xr.open_dataset(data / "truth" / f"isd_conus_{year}.nc") as archive:
        truth = archive.sel(valid_time=slice(start, None)).load()
    if truth.station_id.values.astype(str).tolist() != stations:
        raise BuildRefusedError("the truth file is not on the registered stations")
    pairs = d2.oriented_citizen_pairs(
        extract.lat.values,
        extract.lon.values,
        extract.elevation_m.values,
        truth.latitude.values,
        truth.longitude.values,
        truth.elevation_m.values,
        exclude_same_cell=False,
    )
    replay = d2.TrackAReplay(
        year, truth, pairs, flagged=extract.ever_copycat.values, token=extract.station_token.values
    )
    cand_dir = (
        (out_root / d2.MODEL_CITIZEN_D2)
        if out_root is not None
        else (data / "candidates" / d2.MODEL_CITIZEN_D2)
    )
    meta_dir = (
        (out_root / "meta_a") if out_root is not None else (data / "phase3-design2" / f"{year}" / "track_a")
    )
    meta_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "year": year,
        "track": "A",
        "stop_after_month": stop_after_month,
        "extract_sha256": p3.sha256_file(extract_path),
        "extract_run_digest": str(extract.attrs["run_digest"]),
        "arrival_policy": ARRIVAL_POLICY,
        "arm_c2": [],
        "issues": [],
    }
    i = 0
    prev_gefs_means: dict[str, np.ndarray] | None = None
    prev_t: np.datetime64 | None = None
    for path in months:
        if stop_after_month is not None and path.stem > stop_after_month:
            break
        with xr.open_zarr(path, chunks=None) as source:
            a_month = source.load()
        parts_c2 = []
        for t in a_month.init_time.values:
            wind_prev = (
                (prev_gefs_means["u10"], prev_gefs_means["v10"])
                if prev_gefs_means is not None and prev_t == t - d2.PRIOR_LEAD_HOURS * d2.HOUR
                else None
            )
            _, c2_ds, diag = replay.issue(a_month.sel(init_time=[t]), snapshot(i), wind_prev=wind_prev)
            parts_c2.append(c2_ds)
            out["issues"].append(asdict(diag))
            cur_gefs = _load_gefs_cycle(data, t, stations)
            prev_gefs_means = _gefs_12h_means(cur_gefs)
            prev_t = t
            i += 1
        out["arm_c2"].append(phase3_frozen.publish(parts_c2, cand_dir, path.stem))
        print(
            json.dumps({"track": "A", "month": path.stem, "c2": out["arm_c2"][-1]["content_sha256"]}),
            flush=True,
        )
    if stop_after_month is None:
        covered = arms.covered_stations(truth.station_id.values, replay.citizen_defined_count, len(issues))
        text = "station_id\n" + "".join(f"{s}\n" for s in covered)
        (meta_dir / "covered_stations_a.csv").write_text(text)
        cov_sha = hashlib.sha256(text.encode()).hexdigest()
        stop_reasons = []
        if len(covered) < min_covered:
            stop_reasons.append(f"covered_count {len(covered)} < {min_covered}")
        ref_sha: str | None = None
        d1_parity: bool | None = None
        if year == 2024 or reference_covered_sha256 is not None:
            if reference_covered_sha256 is not None:
                raw_ref: object = reference_covered_sha256
                missing_file = False
            else:
                d1_build = data / "phase3" / "2024" / "build.json"
                missing_file = not d1_build.is_file()
                raw_ref = None if missing_file else json.loads(d1_build.read_text()).get("covered_sha256")
            if missing_file:
                d1_parity = False
                stop_reasons.append(
                    "missing frozen Design 1 phase3/2024/build.json to verify Track A COVERED parity"
                )
            elif not _is_sha256_hex(raw_ref):
                d1_parity = False
                stop_reasons.append(
                    "missing or invalid covered_sha256 in frozen Design 1 phase3/2024/build.json"
                )
            else:
                ref_sha = str(raw_ref)
                d1_parity = cov_sha == ref_sha
                if not d1_parity:
                    stop_reasons.append(
                        "Track A 2024 COVERED sha256 differs from frozen Design 1 COVERED list"
                    )
        out.update(
            build_kind="full_year",
            n_issues=i,
            covered_count=len(covered),
            covered_sha256=cov_sha,
            design1_covered_sha256=ref_sha,
            design1_covered_parity=d1_parity,
            scoring_ready=not stop_reasons,
            stop_reasons=stop_reasons,
        )
        (meta_dir / "build.json").write_text(json.dumps(out, indent=1) + "\n")
        if stop_reasons:
            raise BuildRefusedError("scoring handoff blocked: " + "; ".join(stop_reasons))
    else:
        if not out["arm_c2"]:
            raise BuildRefusedError("partial Track A replay produced zero monthly stores")
        out.update(build_kind="partial_replay", n_issues=i, scoring_ready=False)
        (meta_dir / "build.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def build_track_b(
    data: Path,
    year: int,
    *,
    out_root: Path | None = None,
    stop_after_month: str | None = None,
    issues=None,
    stations=None,
    stations_csv: Path = loao_qc.STATIONS_CSV,
    verify_extract: bool = True,
    run_relay_audit: bool = False,
    min_covered: int = MIN_COVERED_STATIONS,
    min_free_bytes: int = STOP_FREE_BYTES,
) -> dict:
    """Build Track B 5-fold outer-excluded arms B_LOAO and C2_LOAO (plus optional post-freeze audit)."""
    check_free_space(data, min_free_bytes)
    _require_isolated_partial_out(data, out_root, stop_after_month)
    start, _ = v2.year_bounds(year)
    start = np.datetime64(start, "ns")
    issues = loao_qc.issue_times(year) if issues is None else np.asarray(issues, dtype="datetime64[ns]")
    stations = phase3_frozen.registered_stations() if stations is None else list(stations)
    extract_path = data / "citizen" / "conus" / f"issue_extract_loao_{year}.nc"
    extract, snapshot = load_loao_extract(extract_path, year, issues, stations)
    if verify_extract:
        verify_loao_extract_is_current(data, year, extract_path, extract, stations_csv=stations_csv)
    with xr.open_dataset(data / "truth" / f"isd_conus_{year}.nc") as archive:
        truth = archive.sel(valid_time=slice(start, None)).load()
    if truth.station_id.values.astype(str).tolist() != stations:
        raise BuildRefusedError("the truth file is not on the registered stations")
    folds = d2.assign_spatial_folds(
        truth.station_id.values, truth.latitude.values, truth.longitude.values, n_folds=d2.N_FOLDS
    )
    pairs_loao = d2.oriented_citizen_pairs(
        extract.lat.values,
        extract.lon.values,
        extract.elevation_m.values,
        truth.latitude.values,
        truth.longitude.values,
        truth.elevation_m.values,
        exclude_same_cell=True,
    )
    replay = d2.TrackBReplay(
        year, truth, folds, pairs_loao, extract.elevation_m.values, token=extract.station_token.values
    )
    b_dir = (
        (out_root / d2.MODEL_LOAO_BASE)
        if out_root is not None
        else (data / "candidates" / d2.MODEL_LOAO_BASE)
    )
    c_dir = (
        (out_root / d2.MODEL_LOAO_CITIZEN)
        if out_root is not None
        else (data / "candidates" / d2.MODEL_LOAO_CITIZEN)
    )
    meta_dir = (
        (out_root / "meta_b") if out_root is not None else (data / "phase3-design2" / f"{year}" / "track_b")
    )
    meta_dir.mkdir(parents=True, exist_ok=True)
    folds_text = "station_id,fold\n" + "".join(
        f"{sid},{int(f)}\n" for sid, f in zip(stations, folds, strict=True)
    )
    (meta_dir / "folds.csv").write_text(folds_text)
    out = {
        "year": year,
        "track": "B",
        "stop_after_month": stop_after_month,
        "extract_sha256": p3.sha256_file(extract_path),
        "extract_run_digest": str(extract.attrs["run_digest"]),
        "folds_sha256": hashlib.sha256(folds_text.encode()).hexdigest(),
        "arrival_policy": ARRIVAL_POLICY,
        "arm_b_loao": [],
        "arm_c2_loao": [],
        "issues": [],
    }
    # Group issues by YYYY-MM
    by_month: dict[str, list[tuple[int, np.datetime64]]] = {}
    for idx, t in enumerate(issues):
        m = str(np.datetime_as_string(t, unit="M"))
        by_month.setdefault(m, []).append((idx, t))
    prev_gefs: xr.Dataset | None = None
    prev_means: dict[str, np.ndarray] | None = None
    prev_t: np.datetime64 | None = None
    all_prev_means: list[dict[str, np.ndarray] | None] = []
    all_snaps: list[dict] = []
    used_issues: list[np.datetime64] = []
    for month, entries in sorted(by_month.items()):
        if stop_after_month is not None and month > stop_after_month:
            break
        parts_b, parts_c = [], []
        for idx, t in entries:
            cur_gefs = _load_gefs_cycle(data, t, stations)
            has_prev = prev_gefs is not None and prev_t == t - d2.PRIOR_LEAD_HOURS * d2.HOUR
            g_base = d2.make_g_base(cur_gefs, prev_gefs if has_prev else None)
            snap = snapshot(idx)
            p_means = prev_means if has_prev else None
            b_ds, c_ds, f_diags = replay.issue(g_base, p_means, snap)
            parts_b.append(b_ds)
            parts_c.append(c_ds)
            out["issues"].append({"init": f_diags[0].init, "folds": [asdict(fd) for fd in f_diags]})
            all_prev_means.append(p_means)
            all_snaps.append(snap)
            used_issues.append(t)
            prev_gefs = cur_gefs
            prev_means = _gefs_12h_means(cur_gefs)
            prev_t = t
        out["arm_b_loao"].append(phase3_frozen.publish(parts_b, b_dir, month))
        out["arm_c2_loao"].append(phase3_frozen.publish(parts_c, c_dir, month))
        print(
            json.dumps(
                {
                    "track": "B",
                    "month": month,
                    "b_loao": out["arm_b_loao"][-1]["content_sha256"],
                    "c2_loao": out["arm_c2_loao"][-1]["content_sha256"],
                }
            ),
            flush=True,
        )
    if stop_after_month is None:
        cov_counts = d2.track_b_covered_counts(
            year,
            np.asarray(used_issues, dtype="datetime64[ns]"),
            truth.latitude.values,
            truth.longitude.values,
            truth.elevation_m.values,
            pairs_loao,
            extract.elevation_m.values,
            all_prev_means,
            all_snaps,
            token=extract.station_token.values,
        )
        covered = arms.covered_stations(truth.station_id.values, cov_counts, len(issues))
        cov_text = "station_id\n" + "".join(f"{s}\n" for s in covered)
        (meta_dir / "covered_stations_loao.csv").write_text(cov_text)
        stop_reasons = []
        if len(covered) < min_covered:
            stop_reasons.append(f"covered_loao_count {len(covered)} < {min_covered}")
        out.update(
            build_kind="full_year",
            n_issues=len(used_issues),
            covered_loao_count=len(covered),
            covered_loao_sha256=hashlib.sha256(cov_text.encode()).hexdigest(),
            scoring_ready=not stop_reasons,
            stop_reasons=stop_reasons,
        )
        if run_relay_audit:
            # Post-freeze aggregate relay diagnostics ONLY (after stores and coverage are written).
            contrib = _collect_track_b_contributions(
                year,
                np.asarray(used_issues, dtype="datetime64[ns]"),
                truth.elevation_m.values,
                pairs_loao,
                extract.elevation_m.values,
                all_prev_means,
                all_snaps,
                token=extract.station_token.values,
            )
            needed_c = {c for c, *_ in contrib}
            tiles_dir = data / "citizen" / "conus" / f"{year}-tiles"
            streams = load_full_resolution_streams_from_tiles(tiles_dir, year, extract, needed_c)
            trial = p3.load_trial_module(start)
            forecasts = p3.load_forecasts(
                data / "baselines" / "gefs", truth.station_id.values, start, issues[-1], trial
            )
            off_series = trial.official_series(p3.within_year(truth, start), forecasts.as_tuple())
            std_path = data / "citizen" / "conus" / f"issue_extract_{year}.nc"
            std_flagged = None
            if std_path.is_file():
                with xr.open_dataset(std_path, engine="h5netcdf") as sds:
                    copy_tokens = set(sds.station_token.values[sds.ever_copycat.values].tolist())
                std_flagged = np.array(
                    [tok in copy_tokens for tok in extract.station_token.values], dtype=bool
                )
            cov_idx = {stations.index(s) for s in covered}
            audit = d2.target_paired_relay_audit(
                contrib, streams, off_series, cov_idx, trial, standard_ever_copycat=std_flagged
            )
            (meta_dir / "relay_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
            out["relay_audit"] = audit
        (meta_dir / "build.json").write_text(json.dumps(out, indent=1) + "\n")
        if stop_reasons:
            raise BuildRefusedError("scoring handoff blocked: " + "; ".join(stop_reasons))
    else:
        if not out["arm_b_loao"] or not out["arm_c2_loao"]:
            raise BuildRefusedError("partial Track B replay produced zero monthly stores")
        out.update(build_kind="partial_replay", n_issues=len(used_issues), scoring_ready=False)
        (meta_dir / "build.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--year", type=int, default=2024, choices=sorted(loao_qc.LAST_ISSUE))
    parser.add_argument("--track", choices=("all", "A", "B"), default="all")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("WRFCOMP_DATA_DIR") or Path.home() / "wrfcomp-data"),
    )
    parser.add_argument("--out-dir", type=Path, default=None, help="isolated output directory for replays")
    parser.add_argument("--stop-after-month", type=str, default=None, help="stop after YYYY-MM (replay)")
    parser.add_argument("--relay-audit", action="store_true", help="run post-freeze aggregate relay audit")
    args = parser.parse_args()
    data = args.data_dir.resolve()
    out_root = None if args.out_dir is None else args.out_dir.resolve()
    if args.track in ("all", "A"):
        build_track_a(data, args.year, out_root=out_root, stop_after_month=args.stop_after_month)
    if args.track in ("all", "B"):
        build_track_b(
            data,
            args.year,
            out_root=out_root,
            stop_after_month=args.stop_after_month,
            run_relay_audit=args.relay_audit,
        )


if __name__ == "__main__":
    main()
