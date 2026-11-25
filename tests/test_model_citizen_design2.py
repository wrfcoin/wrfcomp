"""Synthetic acceptance tests for Phase 3 Design 2 (Track A, Track B 5-fold LOAO, relay audit,
pre-fetch replay wrappers). All observations are seeded synthetic fixtures."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from wrfcomp.assimilation import phase3_qc as p3
from wrfcomp.assimilation import phase3_qc_loao as loao_qc
from wrfcomp.assimilation.qc import cells, distances_km
from wrfcomp.baselines.io import UNITS, layout, write
from wrfcomp.models import citizen_arms as arms
from wrfcomp.models import citizen_design2 as d2

H = np.timedelta64(1, "h")
M = np.timedelta64(1, "m")
D = np.timedelta64(1, "D")
START = np.datetime64("2024-01-01", "ns")
LEADS = [1, 3, 6, 12, 24, 48]
GEFS_LEADS = [3, 6, 12, 24, 48]
CHANGED = [0, 1, 2]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_d1_test():
    spec = importlib.util.spec_from_file_location(
        "test_model_citizen_arms_ref", Path(__file__).with_name("test_model_citizen_arms.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------- §6 tests 1–3 (geometry & weights)


def test_calm_wind_matches_design1_haversine_to_machine_precision():
    """§6 test 1: at u10 = v10 = 0, w_geo equals Design 1's haversine weight to atol = 1e-15."""
    s_lat, s_lon = 40.05, -100.05
    c_lats = np.array([40.0, 40.1, 40.3, 39.7, 40.05])
    c_lons = np.array([-100.0, -100.4, -99.6, -100.2, -100.05])
    rx, ry, d_gc = d2.haversine_displacement_km(s_lat, s_lon, c_lats, c_lons)
    px, py = d2.upwind_target_km(0.0, 0.0, 3.0, d2.RADIUS_KM)
    assert (px, py) == (0.0, 0.0)
    w_d2 = d2.upwind_gaussian_weight(rx, ry, d_gc, px, py, d2.DISTANCE_SCALE_KM)
    w_d1 = np.exp(-np.square(d_gc / arms.DISTANCE_SCALE_KM) / 2.0)
    np.testing.assert_allclose(w_d2, w_d1, rtol=0.0, atol=1e-15)
    # Also verify raw norm equality rx^2 + ry^2 == d_gc^2.
    np.testing.assert_allclose(np.hypot(rx, ry), d_gc, rtol=0.0, atol=1e-12)


def test_upwind_direction_sign_cap_and_jan1_fallback():
    """§6 test 2: westerly wind weights west > east, caps at R = 50 km, and Jan 1 fallback is calm."""
    s_lat, s_lon = 40.0, -100.0
    # Symmetric west (upwind under u10 > 0) and east (downwind) neighbours ~30 km away.
    c_lats = np.array([40.0, 40.0])
    c_lons = np.array([-100.35, -99.65])
    rx, ry, d_gc = d2.haversine_displacement_km(s_lat, s_lon, c_lats, c_lons)
    assert rx[0] < 0 and rx[1] > 0
    px, py = d2.upwind_target_km(5.0, 0.0, 1.0, d2.RADIUS_KM)
    # v_tr = 1.5 * 5 = 7.5 m/s -> 27.0 km upwind (west, negative x)
    assert px == pytest.approx(-27.0) and py == 0.0
    w = d2.upwind_gaussian_weight(rx, ry, d_gc, px, py, d2.DISTANCE_SCALE_KM)
    assert w[0] > w[1] * 5.0
    # Strong wind caps at R = 50 km.
    px_cap, py_cap = d2.upwind_target_km(20.0, 0.0, 6.0, d2.RADIUS_KM)
    assert px_cap == pytest.approx(-50.0) and py_cap == 0.0
    # Non-finite wind falls back cleanly to calm (0, 0).
    assert d2.upwind_target_km(np.nan, 0.0, 3.0, d2.RADIUS_KM) == (0.0, 0.0)


def test_sqrt_tempering_and_covered_parity_with_design1():
    """§6 test 3: sqrt(w) tempering values and identical Track A COVERED set to Design 1."""
    w = np.array([0.0, 0.04, 0.25, 1.0])
    np.testing.assert_allclose(d2.temper_trust_weight(w), [0.0, 0.2, 0.5, 1.0], atol=1e-15)
    # Check Track A COVERED parity + byte-identical B on Design 1's own synthetic world.
    d1 = _load_d1_test()
    stations, truth, issues, snapshots = d1.world()
    p1 = arms.neighbour_pairs(
        *d1.CITIZEN.T, truth.latitude.values, truth.longitude.values, truth.elevation_m.values
    )
    p2 = d2.oriented_citizen_pairs(
        *d1.CITIZEN.T, truth.latitude.values, truth.longitude.values, truth.elevation_m.values
    )
    r1 = arms.Replay(2024, truth, p1)
    r2 = d2.TrackAReplay(2024, truth, p2)
    for a, snap in zip(issues, snapshots, strict=True):
        b1, _, _ = r1.issue(a, snap)
        b2, c2, _ = r2.issue(a, snap, wind_prev=(np.full(4, 6.0), np.zeros(4)))
        xr.testing.assert_identical(b1, b2)
        # Byte identity outside the 6 cells.
        for k in UNITS:
            keep = [j for j in range(len(LEADS)) if k not in d2.VARIABLES or j not in CHANGED]
            assert c2[k].values[:, keep].tobytes() == a[k].values[:, keep].tobytes()
    assert arms.covered_stations(truth.station_id.values, r2.citizen_defined_count, len(issues)) == (
        arms.covered_stations(truth.station_id.values, r1.citizen_defined_count, len(issues))
    )


# ----------------------------------------------------------------------------- §6 test 5 (Track B LOAO)


def _track_b_world(*, mutate_s_shift: float = 0.0, mutate_j_shift: float = 0.0, days: int = 38):
    """Multi-week synthetic world for Track B end-to-end non-vacuous mutation & positive control.

    Geometry (10 airports -> 5 balanced folds of 2 stations each):
    - Fold contains held-out airport s (index 0 at 40.0, -100.0).
    - Neighbouring in-fold training airport j (index 1 at 40.0, -99.06 -> ~80 km east of s),
      plus 8 additional airports distributed so every fold has neighbouring in-fold airports within
      150 km and every airport has >= 3 distinct citizen tokens across >= 2 cells outside cell(s)
      within 50 km, including a contributor c at ~45 km from s and a citizen buddy b at ~60 km from s.
    """
    rng = np.random.default_rng(2026)
    # 10 airports spaced ~80 km apart in a 2x5 lattice around (40.0, -100.0)
    airport_coords = [
        (40.00, -100.00, 500.0),  # 0: target s
        (40.00, -99.06, 510.0),  # 1: ~80 km east of s (in-fold neighbour j)
        (40.00, -98.12, 490.0),  # 2
        (40.72, -100.00, 520.0),  # 3: ~80 km north of s
        (40.72, -99.06, 505.0),  # 4
        (40.72, -98.12, 495.0),  # 5
        (41.44, -100.00, 530.0),  # 6
        (41.44, -99.06, 515.0),  # 7
        (41.44, -98.12, 500.0),  # 8
        (39.28, -99.53, 480.0),  # 9
    ]
    n_air = len(airport_coords)
    sids = [f"{i + 1:011d}" for i in range(n_air)]
    lats = np.array([c[0] for c in airport_coords], dtype=np.float64)
    lons = np.array([c[1] for c in airport_coords], dtype=np.float64)
    elevs = np.array([c[2] for c in airport_coords], dtype=np.float64)
    folds = d2.assign_spatial_folds(sids, lats, lons, n_folds=5)
    target_s = 0
    fold_s = int(folds[target_s])
    # Pick an in-fold training airport j within 150 km of s (prefer index 1 at ~80 km)
    d_from_s = distances_km(lats[target_s], lons[target_s], lats, lons)
    in_near = [k for k in range(n_air) if folds[k] != fold_s and d_from_s[k] <= 150.0]
    train_j = 1 if 1 in in_near else in_near[0]
    assert folds[train_j] != fold_s and 50.0 < d_from_s[train_j] <= 150.0

    # Build citizen stations around every airport: 4 distinct tokens in >= 2 cells outside cell(k)
    # within 50 km, plus for airport 0 an explicit contributor c at ~45 km west and buddy b at ~60 km west.
    cit_rows = []
    cit_tokens = []
    tok_id = 0
    for k in range(n_air):
        # 4 citizen stations around airport k, all outside cell(k) and within 45 km
        offsets = [(-0.2, -0.2), (-0.2, 0.2), (0.2, -0.2), (0.2, 0.2)]
        for dlat, dlon in offsets:
            cl, cn = round(lats[k] + dlat, 1), round(lons[k] + dlon, 1)
            if cells(cl, cn) == cells(lats[k], lons[k]):
                cl = round(cl + 0.2, 1)
            cit_rows.append((cl, cn, round(elevs[k] / 10.0) * 10.0))
            cit_tokens.append(f"CW{tok_id:09d}")
            tok_id += 1
    # Explicit Round 1 #2 / Round 2 #3 geometry near target s (40.0, -100.0):
    # contributor c at ~45 km west (40.0, -100.5 -> ~42.6 km),
    # buddy b at ~60 km west (40.0, -100.7 -> ~59.6 km),
    # plus 2 extra buddies near c (40.1, -100.5 and 39.9, -100.5) so spatial checks are active on c and b.
    extra_geom = [
        (40.0, -100.5, 500.0),  # ~42.6 km west of s (contributor c)
        (40.0, -100.7, 500.0),  # ~59.6 km west of s (buddy b, 17 km from c)
        (40.1, -100.5, 500.0),  # buddy near c
        (39.9, -100.5, 500.0),  # buddy near c
    ]
    for cl, cn, ch in extra_geom:
        cit_rows.append((cl, cn, ch))
        cit_tokens.append(f"CW{tok_id:09d}")
        tok_id += 1
    cit_geo = np.array(cit_rows, dtype=np.float64)
    n_cit = len(cit_geo)

    inits = np.arange(START, START + days * D, 12 * H)
    labels = np.arange(START, START + (days + 3) * D, H)
    hours = np.arange(len(labels), dtype=np.float64)
    stations_ds = xr.Dataset(
        coords={
            "station_id": sids,
            "latitude": ("station_id", lats.astype(np.float32)),
            "longitude": ("station_id", lons.astype(np.float32)),
            "elevation_m": ("station_id", elevs.astype(np.float32)),
        }
    )
    # Planted large-scale synoptic error (shared across airports -> positive phi_off)
    # PLUS a mesoscale approaching anomaly (seen by citizens 3 h before arrival -> positive gamma_loao!).
    diurnal = 282.0 + 5.0 * np.sin(2.0 * np.pi * hours / 24.0)
    synoptic_err = 2.2 * np.sin(2.0 * np.pi * hours / 67.0)
    meso_err = 2.8 * np.cos(2.0 * np.pi * hours / 29.0)
    truth = xr.Dataset(coords={"valid_time": labels, **dict(stations_ds.coords)})
    for k_var in UNITS:
        scale = 100.0 if k_var == "msl" else 1.0
        centre = 100000.0 if k_var == "msl" else (275.0 if k_var == "td2m" else 282.0)
        base = centre + scale * (diurnal - 282.0)[:, None] + np.zeros((1, n_air))
        if k_var in d2.VARIABLES:
            # Synoptic + mesoscale signals both appear in verifying truth with 3 h propagation lag
            sig = np.zeros((len(labels), n_air))
            sig[3:, :] = 0.6 * synoptic_err[:-3, None] + 0.7 * meso_err[:-3, None]
            base = base + sig + rng.normal(0.0, 0.15, base.shape)
            # Constant regional bias (+1.5 K) so bias_off is also non-zero
            base = base + 1.5
        else:
            base = base + scale * rng.normal(0.0, 0.2, base.shape)
        truth[k_var] = (("valid_time", "station_id"), base.astype(np.float32))
        truth[k_var + "_offset_min"] = xr.full_like(truth[k_var], -5, dtype=np.int16)
    truth["td2m"].values[:] = np.minimum(truth["td2m"].values, truth["t2m"].values)

    if mutate_s_shift != 0.0:
        truth["t2m"].values[:, target_s] += mutate_s_shift
        truth["td2m"].values[:, target_s] += mutate_s_shift
    if mutate_j_shift != 0.0:
        wave = mutate_j_shift * (0.5 + np.sin(2.0 * np.pi * hours / 41.0))
        truth["t2m"].values[:, train_j] += wave.astype(np.float32)
        truth["td2m"].values[:, train_j] += wave.astype(np.float32)

    # Build raw GEFS cycles (5 leads: 3, 6, 12, 24, 48 h; westerly u10 = +6 m/s)
    gefs_cycles = []
    for t in inits:
        g = layout(stations_ds, [t], GEFS_LEADS, 31, "noaa-gefs")
        i0 = int((t - START) / H)
        for k_var in UNITS:
            scale = 100.0 if k_var == "msl" else 1.0
            centre = 100000.0 if k_var == "msl" else (275.0 if k_var == "td2m" else 282.0)
            for j_l, lead in enumerate(GEFS_LEADS):
                if k_var == "u10":
                    lvl = np.full(n_air, 6.0)
                elif k_var == "v10":
                    lvl = np.zeros(n_air)
                else:
                    lvl = centre + scale * (diurnal[i0 + lead] - 282.0) + np.zeros(n_air)
                g[k_var].values[0, j_l] = (lvl[:, None] + scale * rng.normal(0.0, 0.8, (n_air, 31))).astype(
                    np.float32
                )
        g["td2m"].values[:] = np.minimum(g["td2m"].values, g["t2m"].values)
        gefs_cycles.append(g)

    # Run FRESH official-free citizen QC (phase3_qc_loao) on synthetic full-res citizen streams!
    trial = p3.load_trial_module(START)
    sites = loao_qc.StaticSites(np.array(sids), lats, lons, elevs)
    # Build ForecastArchive from gefs_cycles
    f_iss, f_val = [], []
    f_means = {var: [] for var in (*trial.DIRECT, "wind_speed")}
    for g in gefs_cycles:
        cyc = g.init_time.values[0]
        for lead in GEFS_LEADS:
            f_iss.append(cyc)
            f_val.append(cyc + lead * H)
        blk = g.isel(init_time=0)
        for var in trial.DIRECT:
            raw = blk[var].transpose("lead_time", "station_id", "ens_member").values
            f_means[var].append(raw.mean(axis=2, dtype=float))
            if var == "u10":
                u_arr = raw
            if var == "v10":
                f_means["wind_speed"].append(np.hypot(u_arr, raw).mean(axis=2, dtype=float))
    fc_archive = p3.archive_from_arrays(
        np.array(f_iss), np.array(f_val), {k: np.concatenate(v) for k, v in f_means.items()}, trial
    )
    # Hourly citizen reports at minute -5 before each 12-h issue (fast, crisp, exercises official-free QC)
    cit_times = (inits - 5 * M).astype("datetime64[ns]")
    cit_hours = ((cit_times - START) / H).astype(int)
    cit_stations = []
    for c_idx in range(n_cit):
        cl, cn, ch = cit_geo[c_idx]
        # Citizens observe the approaching mesoscale error + steady site offset (never target s's truth!)
        t2m = (
            282.0
            + 5.0 * np.sin(2.0 * np.pi * cit_hours / 24.0)
            + 1.5
            + 0.3 * (c_idx % 5 - 2)
            + meso_err[np.clip(cit_hours, 0, len(meso_err) - 1)]
            + rng.normal(0.0, 0.1, len(cit_times))
        )
        td2m = t2m - 7.0 + rng.normal(0.0, 0.1, len(cit_times))
        zeros = np.full(len(cit_times), np.nan, dtype=np.float32)
        cit_stations.append(
            p3.StationData(
                cl, cn, ch, cit_times, t2m.astype(np.float32), td2m.astype(np.float32), zeros, zeros
            )
        )
    sources = [loao_qc.station_source_loao(st, sites, fc_archive, trial) for st in cit_stations]
    snapshots = [
        {v: (np.full(n_cit, np.nan), np.zeros(n_cit), np.zeros(n_cit, np.int64)) for v in d2.VARIABLES}
        for _ in inits
    ]
    for c_idx, st in enumerate(cit_stations):
        d_bud = distances_km(st.lat, st.lon, cit_geo[:, 0], cit_geo[:, 1])
        near = [j for j in range(n_cit) if j != c_idx and d_bud[j] <= trial.CFG.radius_km]
        res = loao_qc.station_weights_loao(
            st, [(cit_stations[j], sources[j]) for j in near], sites, fc_archive, trial
        )
        for var in d2.VARIABLES:
            has, val, wt, _, age = p3.issue_snapshot(res[var], inits, loao_qc.WINDOW_MIN)
            for i_t in np.flatnonzero(has):
                snapshots[i_t][var][0][c_idx] = val[i_t]
                snapshots[i_t][var][1][c_idx] = wt[i_t]
                snapshots[i_t][var][2][c_idx] = age[i_t]

    pairs_loao = d2.oriented_citizen_pairs(
        cit_geo[:, 0], cit_geo[:, 1], cit_geo[:, 2], lats, lons, elevs, exclude_same_cell=True
    )
    replay = d2.TrackBReplay(2024, truth, folds, pairs_loao, cit_geo[:, 2], token=cit_tokens)
    b_list, c_list, diag_list, prev_means_list = [], [], [], []
    prev_g = None
    for g, snap in zip(gefs_cycles, snapshots, strict=True):
        g_base = d2.make_g_base(g, prev_g)
        p_means = None
        if prev_g is not None:
            slot = prev_g.sel(lead_time=np.timedelta64(12, "h")).isel(init_time=0)
            p_means = {
                v: slot[v].transpose("station_id", "ens_member").values.astype(float).mean(axis=-1)
                for v in ("t2m", "td2m", "u10", "v10")
            }
        b_ds, c_ds, f_diags = replay.issue(g_base, p_means, snap)
        b_list.append(b_ds)
        c_list.append(c_ds)
        diag_list.append(f_diags)
        prev_means_list.append(p_means)
        prev_g = g
    cov_counts = d2.track_b_covered_counts(
        2024,
        inits,
        lats,
        lons,
        elevs,
        pairs_loao,
        cit_geo[:, 2],
        prev_means_list,
        snapshots,
        token=cit_tokens,
    )
    covered = arms.covered_stations(truth.station_id.values, cov_counts, len(inits))
    np.testing.assert_array_equal(replay.citizen_defined_count, cov_counts)
    return {
        "target_s": target_s,
        "fold_s": fold_s,
        "train_j": train_j,
        "folds": folds,
        "pairs_loao": pairs_loao,
        "truth": truth,
        "gefs_cycles": gefs_cycles,
        "snapshots": snapshots,
        "prev_means_list": prev_means_list,
        "replay": replay,
        "b": b_list,
        "c": c_list,
        "diags": diag_list,
        "covered": covered,
    }


def test_track_b_non_vacuous_mutation_and_positive_control():
    """§6 test 5: non-vacuous Track B fold isolation + live in-fold positive control."""
    base = _track_b_world(mutate_s_shift=0.0, mutate_j_shift=0.0)
    mut_s = _track_b_world(mutate_s_shift=15.0, mutate_j_shift=0.0)
    mut_j = _track_b_world(mutate_s_shift=0.0, mutate_j_shift=12.0)

    s = base["target_s"]
    f_s = base["fold_s"]
    sid = str(base["truth"].station_id.values[s])

    # 1. Non-vacuity assertions: citizens and official neighbours MUST actually move s's forecast!
    assert sid in base["covered"]
    last_diag = base["diags"][-1][f_s]
    assert last_diag.phi_off["t2m"][1] > 0.0
    assert last_diag.gamma_loao["t2m"][1] > 0.0
    g_last = d2.make_g_base(base["gefs_cycles"][-1], base["gefs_cycles"][-2])
    b_last_s = base["b"][-1]["t2m"].values[0, 1, s]
    c_last_s = base["c"][-1]["t2m"].values[0, 1, s]
    g_last_s = g_last["t2m"].values[0, 1, s]
    assert not np.array_equal(b_last_s, g_last_s)
    assert not np.array_equal(c_last_s, b_last_s)  # citizens strictly move s's forecast!

    # Check target-independent cell exclusion: no pair shares cell(s).
    p = base["pairs_loao"]
    s_cell = cells(base["truth"].latitude.values[s], base["truth"].longitude.values[s])
    assert not np.any(p.cell[p.official == s] == s_cell)

    # 2. Negative control: mutating all s observations leaves fold f_s training predictors, gains,
    # coverage, and output bytes identical!
    assert mut_s["covered"] == base["covered"]
    for d0, d1 in zip(base["diags"], mut_s["diags"], strict=True):
        assert d0[f_s] == d1[f_s]
    h0 = list(base["replay"].fold_replays[f_s].history)
    h1 = list(mut_s["replay"].fold_replays[f_s].history)
    assert len(h0) == len(h1) and len(h0) > 0
    for rec0, rec1 in zip(h0, h1, strict=True):
        for var in d2.VARIABLES:
            np.testing.assert_array_equal(rec0.mean_g_bias_in[var], rec1.mean_g_bias_in[var])
            np.testing.assert_array_equal(rec0.off_in[var], rec1.off_in[var])
            np.testing.assert_array_equal(rec0.cit_in[var], rec1.cit_in[var])
            np.testing.assert_array_equal(rec0.mean_b_in[var], rec1.mean_b_in[var])
    out_mask = np.flatnonzero(base["folds"] == f_s)
    for b0, b1, c0, c1 in zip(base["b"], mut_s["b"], base["c"], mut_s["c"], strict=True):
        for var in UNITS:
            assert b0[var].values[:, :, out_mask, :].tobytes() == b1[var].values[:, :, out_mask, :].tobytes()
            assert c0[var].values[:, :, out_mask, :].tobytes() == c1[var].values[:, :, out_mask, :].tobytes()

    # 3. Positive control: mutating in-fold training O(j) moves fold f_s regression gains (phi_off /
    # gamma_loao, not just spread gain g_loao) and changes s's forecast!
    d_base = base["diags"][-1][f_s]
    d_mut = mut_j["diags"][-1][f_s]
    assert d_mut.phi_off != d_base.phi_off or d_mut.gamma_loao != d_base.gamma_loao
    assert (
        mut_j["b"][-1]["t2m"].values[0, 1, s].tobytes() != base["b"][-1]["t2m"].values[0, 1, s].tobytes()
        or mut_j["c"][-1]["t2m"].values[0, 1, s].tobytes() != base["c"][-1]["t2m"].values[0, 1, s].tobytes()
    )

    # 4. Byte identity outside the 6 cells & 100% finite outputs (zero-shift fallbacks on cold start).
    for g0, b0, c0 in zip(base["gefs_cycles"], base["b"], base["c"], strict=True):
        for var in UNITS:
            assert np.isfinite(b0[var].values).all() and np.isfinite(c0[var].values).all()
            if var not in d2.VARIABLES:
                assert b0[var].values[:, 1:].tobytes() == g0[var].values.tobytes()
                assert c0[var].values[:, 1:].tobytes() == g0[var].values.tobytes()
            else:
                # Leads 12, 24, 48 h (indices 3, 4, 5 in 6-lead store == indices 2, 3, 4 in 5-lead GEFS)
                assert b0[var].values[:, 3:].tobytes() == g0[var].values[:, 2:].tobytes()
                assert c0[var].values[:, 3:].tobytes() == g0[var].values[:, 2:].tobytes()


def test_zero_citizen_dewpoint_shift_preserves_capped_control_bytes():
    """When control B caps td2m <= t2m and citizen td2m shift is zero while citizen t2m shift is
    positive, C2.td2m must equal finished B.td2m byte-for-byte on both Track A and Track B."""
    # --- Track A ---
    d1 = _load_d1_test()
    _, truth, issues, snapshots = d1.world()
    p2 = d2.oriented_citizen_pairs(
        *d1.CITIZEN.T, truth.latitude.values, truth.longitude.values, truth.elevation_m.values
    )
    r_a = d2.TrackAReplay(2024, truth, p2)
    for a, snap in zip(issues[:-1], snapshots[:-1], strict=True):
        r_a.issue(a, snap, wind_prev=(np.full(4, 6.0), np.zeros(4)))
    a_last = issues[-1].copy(deep=True)
    # Force uncapped control td2m > t2m at station 0, lead 3 h (index 1) so B caps td2m to t2m.
    a_last["t2m"].values[0, 1, 0, :] = np.float32(280.0)
    a_last["td2m"].values[0, 1, 0, :] = np.float32(283.0)
    snap_a = {
        "t2m": (snapshots[-1]["t2m"][0] + 6.0, snapshots[-1]["t2m"][1]),
        "td2m": (np.full_like(snapshots[-1]["td2m"][0], np.nan), np.zeros_like(snapshots[-1]["td2m"][1])),
    }
    b_a, c2_a, diag_a = r_a.issue(a_last, snap_a, wind_prev=(np.full(4, 6.0), np.zeros(4)))
    assert diag_a.gamma["t2m"][1] > 0.0
    assert np.all(b_a["td2m"].values[0, 1, 0] == b_a["t2m"].values[0, 1, 0])
    assert np.all(c2_a["t2m"].values[0, 1, 0] > b_a["t2m"].values[0, 1, 0] + 0.1)
    assert c2_a["td2m"].values[0, 1, 0].tobytes() == b_a["td2m"].values[0, 1, 0].tobytes()

    # --- Track B ---
    world = _track_b_world(days=38)
    s = world["target_s"]
    f_s = world["fold_s"]
    g_base = d2.make_g_base(world["gefs_cycles"][-1], world["gefs_cycles"][-2]).copy(deep=True)
    g_base["t2m"].values[0, 1, s, :] = np.float32(280.0)
    g_base["td2m"].values[0, 1, s, :] = np.float32(285.0)
    next_t = g_base.init_time.values[0] + 12 * H
    g_base = g_base.assign_coords(init_time=[next_t])
    snap_prev = world["snapshots"][-1]
    snap_b = {
        "t2m": (snap_prev["t2m"][0] + 6.0, snap_prev["t2m"][1], snap_prev["t2m"][2]),
        "td2m": (
            np.full_like(snap_prev["td2m"][0], np.nan),
            np.zeros_like(snap_prev["td2m"][1]),
            snap_prev["td2m"][2],
        ),
    }
    b_b, c2_b, f_diags = world["replay"].issue(g_base, world["prev_means_list"][-1], snap_b)
    assert f_diags[f_s].gamma_loao["t2m"][1] > 0.0
    assert np.all(b_b["td2m"].values[0, 1, s] == b_b["t2m"].values[0, 1, s])
    assert np.all(c2_b["t2m"].values[0, 1, s] > b_b["t2m"].values[0, 1, s] + 0.1)
    assert c2_b["td2m"].values[0, 1, s].tobytes() == b_b["td2m"].values[0, 1, s].tobytes()


def test_track_b_end_to_end_post_T_truth_and_late_arrival_citizen_reports(tmp_path, monkeypatch):
    """End-to-end Track B causality & Amendment 1 late-arrival regression (partition -> LOAO QC ->
    merge -> build_track_b + relay audit + default min_covered stop + zero token/coord leaks)."""
    import re

    from wrfcomp.assimilation.madis_archive import ReportTable, save_table

    qc_std = _load_script("run_phase3_qc")
    qc_loao = _load_script("run_phase3_qc_loao")
    builder = _load_script("build_phase3_design2")

    days = 26  # 52 issues in 2024-01 (covers USUAL_MIN=7 + active gains)
    day_strs = [str(np.datetime_as_string(START + d * D, unit="D")) for d in range(days)]
    monkeypatch.setattr(qc_std, "expected_days", lambda year: list(day_strs))
    inits = np.arange(START, START + days * D, 12 * H)
    monkeypatch.setattr(loao_qc, "issue_times", lambda year: inits)
    t_cut = START + 20 * D  # 2024-01-21T00Z

    airport_coords = [
        (40.00, -100.00, 500.0),
        (40.00, -99.06, 510.0),
        (40.00, -98.12, 490.0),
        (40.72, -100.00, 520.0),
        (40.72, -99.06, 505.0),
        (40.72, -98.12, 495.0),
        (41.44, -100.00, 530.0),
        (41.44, -99.06, 515.0),
        (41.44, -98.12, 500.0),
        (39.28, -99.53, 480.0),
    ]
    n_air = len(airport_coords)
    sids = [f"{i + 1:011d}" for i in range(n_air)]
    lats = np.array([c[0] for c in airport_coords], dtype=np.float64)
    lons = np.array([c[1] for c in airport_coords], dtype=np.float64)
    elevs = np.array([c[2] for c in airport_coords], dtype=np.float64)
    stations_csv = tmp_path / "stations_10.csv"
    stations_csv.write_text(
        "station_id,lat,lon,elevation_m\n"
        + "".join(f"{sid},{la},{lo},{el}\n" for sid, la, lo, el in zip(sids, lats, lons, elevs, strict=True))
    )

    cit_rows, cit_tokens = [], []
    tok_id = 0
    for k in range(n_air):
        for dlat, dlon in ((-0.2, -0.2), (-0.2, 0.2), (0.2, -0.2), (0.2, 0.2)):
            cl, cn = round(lats[k] + dlat, 1), round(lons[k] + dlon, 1)
            if cells(cl, cn) == cells(lats[k], lons[k]):
                cl = round(cl + 0.2, 1)
            cit_rows.append((cl, cn, round(elevs[k] / 10.0) * 10.0))
            cit_tokens.append(f"CW{tok_id:09d}")
            tok_id += 1
    cit_geo = np.array(cit_rows, dtype=np.float64)

    labels = np.arange(START, START + (days + 3) * D, H)
    hours = np.arange(len(labels), dtype=np.float64)
    stations_ds = xr.Dataset(
        coords={
            "station_id": sids,
            "latitude": ("station_id", lats.astype(np.float32)),
            "longitude": ("station_id", lons.astype(np.float32)),
            "elevation_m": ("station_id", elevs.astype(np.float32)),
        }
    )

    def run_pipeline(root: Path, *, perturb: bool) -> dict:
        rng = np.random.default_rng(77)
        diurnal = 282.0 + 5.0 * np.sin(2.0 * np.pi * hours / 24.0)
        syn = 2.2 * np.sin(2.0 * np.pi * hours / 67.0)
        meso = 2.8 * np.cos(2.0 * np.pi * hours / 29.0)
        truth = xr.Dataset(coords={"valid_time": labels, **dict(stations_ds.coords)})
        for k_var in UNITS:
            scale = 100.0 if k_var == "msl" else 1.0
            centre = 100000.0 if k_var == "msl" else (275.0 if k_var == "td2m" else 282.0)
            base = centre + scale * (diurnal - 282.0)[:, None] + np.zeros((1, n_air))
            if k_var in d2.VARIABLES:
                sig = np.zeros((len(labels), n_air))
                sig[3:, :] = 0.6 * syn[:-3, None] + 0.7 * meso[:-3, None]
                base = base + sig + 1.5 + rng.normal(0.0, 0.12, base.shape)
                if perturb:
                    # Perturb strictly post-T_cut truth (valid_time > T_cut)
                    post = labels > t_cut
                    base[post, :] += 10.0 * np.sin(2.0 * np.pi * hours[post, None] / 19.0) + 6.0
            else:
                base = base + scale * rng.normal(0.0, 0.2, base.shape)
            truth[k_var] = (("valid_time", "station_id"), base.astype(np.float32))
            truth[k_var + "_offset_min"] = xr.full_like(truth[k_var], -5, dtype=np.int16)
        truth["td2m"].values[:] = np.minimum(truth["td2m"].values, truth["t2m"].values)
        (root / "truth").mkdir(parents=True)
        truth.to_netcdf(root / "truth" / "isd_conus_2024.nc")

        # Write raw GEFS cycles (identical between base and perturbed)
        for t in inits:
            g = layout(stations_ds, [t], GEFS_LEADS, 31, "noaa-gefs")
            i0 = int((t - START) / H)
            for k_var in UNITS:
                scale = 100.0 if k_var == "msl" else 1.0
                centre = 100000.0 if k_var == "msl" else (275.0 if k_var == "td2m" else 282.0)
                for j_l, lead in enumerate(GEFS_LEADS):
                    if k_var == "u10":
                        lvl = np.full(n_air, 6.0)
                    elif k_var == "v10":
                        lvl = np.zeros(n_air)
                    else:
                        lvl = centre + scale * (diurnal[i0 + lead] - 282.0) + np.zeros(n_air)
                    g[k_var].values[0, j_l] = (
                        lvl[:, None] + scale * rng.normal(0.0, 0.6, (n_air, 31))
                    ).astype(np.float32)
            g["td2m"].values[:] = np.minimum(g["td2m"].values, g["t2m"].values)
            stamp = np.datetime_as_string(t, unit="h").replace("-", "").replace("T", "")
            write(g, root / "baselines" / "gefs" / f"{stamp}.zarr")

        # Write daily MADIS citizen NetCDFs (with Amendment 1 received_time)
        citizen = root / "citizen"
        for d_idx, day in enumerate(day_strs):
            day_start = START + d_idx * D
            obs_list, rec_list, tok_list, la_list, lo_list, el_list, t_list, td_list = (
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
            )
            # Two on-time reports per day: 11:50Z (before 12Z) and 23:50Z (before next day 00Z)
            for slot_h in (11, 23):
                obs_t = day_start + slot_h * H + 50 * M
                rec_t = obs_t + 30 * np.timedelta64(1, "s")
                hr = int((obs_t - START) / H)
                for c_idx, (cl, cn, ch) in enumerate(cit_geo):
                    val = (
                        282.0
                        + 5.0 * np.sin(2.0 * np.pi * hr / 24.0)
                        + 1.5
                        + 0.25 * (c_idx % 5 - 2)
                        + meso[np.clip(hr, 0, len(meso) - 1)]
                        + rng.normal(0.0, 0.08)
                    )
                    if perturb and obs_t > t_cut:
                        val += 8.0
                    obs_list.append(obs_t)
                    rec_list.append(rec_t)
                    tok_list.append(cit_tokens[c_idx])
                    la_list.append(cl)
                    lo_list.append(cn)
                    el_list.append(ch)
                    t_list.append(val)
                    td_list.append(val - 7.0)
            n_rows = len(obs_list)
            tbl = ReportTable(
                station_id=np.array(tok_list),
                obs_time=np.array(obs_list, dtype="datetime64[ns]"),
                received_time=np.array(rec_list, dtype="datetime64[ns]"),
                lat=np.array(la_list, dtype=np.float32),
                lon=np.array(lo_list, dtype=np.float32),
                elevation_m=np.array(el_list, dtype=np.float32),
                t2m=np.array(t_list, dtype=np.float32),
                td2m=np.array(td_list, dtype=np.float32),
                wind_speed=np.full(n_rows, 2.0, dtype=np.float32),
                wind_dir=np.full(n_rows, 90.0, dtype=np.float32),
            )
            save_table(tbl, citizen / "conus" / "2024" / f"{day}.nc", attrs={"day": day, "hours_missing": ""})

        manifest = qc_std.partition(citizen, 2024)
        if perturb:
            # Inject a late-arriving pre-T_cut citizen report (obs = T_cut - 5 min, received = T_cut + 60 s)
            # directly into every partitioned tile .npz to exercise run_phase3_qc_loao's and
            # load_full_resolution_streams_from_tiles's independent Amendment 1 arrival gates.
            late_obs_ns = int((t_cut - 5 * M).astype("datetime64[ns]").astype(np.int64))
            late_rec_ns = int((t_cut + 60 * np.timedelta64(1, "s")).astype("datetime64[ns]").astype(np.int64))
            tiles_dir = citizen / "conus" / "2024-tiles"
            for rel in list(manifest["parts"]):
                part_path = tiles_dir / rel
                with np.load(part_path) as z:
                    d = {k: z[k] for k in z.files}
                n_id = len(d["ident_token"])
                extra_ident = np.arange(n_id, dtype=np.int32)
                d["row_ident"] = np.concatenate([d["row_ident"], extra_ident])
                d["obs_time"] = np.concatenate([d["obs_time"], np.full(n_id, late_obs_ns, dtype=np.int64)])
                d["received_time"] = np.concatenate(
                    [d["received_time"], np.full(n_id, late_rec_ns, dtype=np.int64)]
                )
                d["t2m"] = np.concatenate([d["t2m"], np.full(n_id, 315.0, dtype=np.float32)])
                d["td2m"] = np.concatenate([d["td2m"], np.full(n_id, 305.0, dtype=np.float32)])
                d["wind_speed"] = np.concatenate([d["wind_speed"], np.full(n_id, 2.0, dtype=np.float32)])
                d["wind_dir"] = np.concatenate([d["wind_dir"], np.full(n_id, 90.0, dtype=np.float32)])
                order = np.lexsort((d["obs_time"], d["row_ident"]))
                for col in (
                    "row_ident",
                    "obs_time",
                    "received_time",
                    "t2m",
                    "td2m",
                    "wind_speed",
                    "wind_dir",
                ):
                    d[col] = d[col][order]
                np.savez_compressed(part_path, **d)
                manifest["parts"][rel] = p3.sha256_file(part_path)
            (tiles_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
        trial, sites, fc = loao_qc.load_loao_inputs(root, 2024, stations_csv=stations_csv)
        rec = loao_qc.run_record(2024, manifest, fc, stations_csv=stations_csv)
        qc_out = citizen / "conus" / "2024-qc-loao"
        qc_out.mkdir(parents=True)
        (qc_out / "run.json").write_text(json.dumps(rec, indent=1) + "\n")
        qc_loao._G.update(
            trial=trial,
            sites=sites,
            forecasts=fc,
            issues=inits,
            tiles=citizen / "conus" / "2024-tiles",
            out=qc_out,
            digest=rec["digest"],
            start=START,
        )
        for ordinal, name in enumerate(manifest["tiles"]):
            qc_loao.run_tile((ordinal, name))
        qc_loao.merge(root, 2024, stations_csv=stations_csv, min_free_bytes=0)

        return builder.build_track_b(
            root,
            2024,
            issues=inits,
            stations=sids,
            stations_csv=stations_csv,
            verify_extract=True,
            run_relay_audit=True,
            min_covered=1,
            min_free_bytes=0,
        )

    base_out = run_pipeline(tmp_path / "base", perturb=False)
    pert_out = run_pipeline(tmp_path / "pert", perturb=True)
    assert base_out["build_kind"] == "full_year" and base_out["scoring_ready"] is True
    assert pert_out["build_kind"] == "full_year" and pert_out["scoring_ready"] is True
    assert base_out["covered_loao_count"] == n_air

    with (
        xr.open_zarr(tmp_path / "base" / "candidates" / d2.MODEL_LOAO_BASE / "2024-01.zarr") as b0,
        xr.open_zarr(tmp_path / "pert" / "candidates" / d2.MODEL_LOAO_BASE / "2024-01.zarr") as b1,
        xr.open_zarr(tmp_path / "base" / "candidates" / d2.MODEL_LOAO_CITIZEN / "2024-01.zarr") as c0,
        xr.open_zarr(tmp_path / "pert" / "candidates" / d2.MODEL_LOAO_CITIZEN / "2024-01.zarr") as c1,
    ):
        pre = b0.init_time.values <= t_cut
        post = b0.init_time.values > t_cut
        assert pre.any() and post.any()
        for var in UNITS:
            assert b0[var].values[pre].tobytes() == b1[var].values[pre].tobytes()
            assert c0[var].values[pre].tobytes() == c1[var].values[pre].tobytes()
        # Non-vacuous check: post-T_cut perturbation actually changes post-T_cut forecasts!
        assert b0["t2m"].values[post].tobytes() != b1["t2m"].values[post].tobytes()
        assert c0["t2m"].values[post].tobytes() != c1["t2m"].values[post].tobytes()

    # Default min_covered=50 must refuse scoring handoff on this 10-station build!
    with pytest.raises(builder.BuildRefusedError, match="covered_loao_count 10 < 50"):
        builder.build_track_b(
            tmp_path / "base",
            2024,
            issues=inits,
            stations=sids,
            stations_csv=stations_csv,
            verify_extract=True,
            min_free_bytes=0,
        )
    blocked_meta = json.loads(
        (tmp_path / "base" / "phase3-design2" / "2024" / "track_b" / "build.json").read_text()
    )
    assert blocked_meta["scoring_ready"] is False and blocked_meta["stop_reasons"]

    # Privacy check: Track B metadata & relay audit never leak citizen tokens or coordinate tile labels.
    coord_tile_re = re.compile(r"[+-]\d{2,3}_[+-]\d{3,4}")
    token_re = re.compile(r"CW\d{9}")
    meta_b = tmp_path / "base" / "phase3-design2" / "2024" / "track_b"
    for fname in ("build.json", "folds.csv", "covered_stations_loao.csv", "relay_audit.json"):
        txt = (meta_b / fname).read_text()
        assert not coord_tile_re.search(txt) and not token_re.search(txt)


# ------------------------------------------------------------------ post-freeze target-paired relay audit


def test_target_paired_relay_audit_deduplicates_distinct_official_times_and_handles_zero_denom():
    """Round 3 #1: full-res citizen stream paired directly against target s (non-nearest), one pair
    per distinct official measurement timestamp, and zero denominator -> 'unavailable'."""
    trial = p3.load_trial_module(START)
    # Two official stations: 0 (target s) and 1 (another airport).
    labels = np.arange(START, START + 4 * D, H)
    hours = np.arange(len(labels), dtype=np.float64)
    t2m_s = (280.0 + 6.0 * np.sin(2.0 * np.pi * hours / 24.0)).astype(np.float32)
    t2m_other = (290.0 + 3.0 * np.cos(2.0 * np.pi * hours / 24.0)).astype(np.float32)
    official_ds = xr.Dataset(
        coords={
            "valid_time": labels,
            "station_id": ["00000000001", "00000000002"],
            "latitude": ("station_id", np.float32([40.0, 40.2])),
            "longitude": ("station_id", np.float32([-100.0, -100.0])),
            "elevation_m": ("station_id", np.float32([500.0, 500.0])),
        }
    )
    for var, vals in (
        ("t2m", np.column_stack([t2m_s, t2m_other])),
        ("td2m", np.column_stack([t2m_s - 5.0, t2m_other - 5.0])),
        ("u10", np.ones((len(labels), 2), dtype=np.float32)),
        ("v10", np.zeros((len(labels), 2), dtype=np.float32)),
    ):
        official_ds[var] = (("valid_time", "station_id"), vals)
        official_ds[var + "_offset_min"] = (
            ("valid_time", "station_id"),
            np.full((len(labels), 2), -5, np.int16),
        )
    # Minimal forecasts tuple for official_series
    f_iss = np.array([START], dtype="datetime64[ns]")
    f_val = np.array([START + 3 * H], dtype="datetime64[ns]")
    f_means = {v: np.full((1, 2), 280.0) for v in (*trial.DIRECT, "wind_speed")}
    off_series = trial.official_series(official_ds, (f_iss, f_val, f_means))

    # Citizen 0 reports every 10 min for 3 days, copying target s (index 0) with 0.02 K noise.
    # Standard QC nearest-ref flag is False (imagine its nearest airport was index 1).
    cit_times_long = np.arange(START + 2 * H, START + 72 * H, 10 * M)
    obs_times = labels - 5 * M
    idx_s = np.clip(np.searchsorted(obs_times, cit_times_long, side="right") - 1, 0, len(labels) - 1)
    rng = np.random.default_rng(5)
    cit0_vals = t2m_s[idx_s].astype(float) + rng.normal(0.0, 0.01, len(cit_times_long))

    # Citizen 1 reports every 10 min for ONLY 10 hours (10 distinct official hours < 12 min_pairs!)
    # copying target s with 0.12 K noise: even though it has 60 ten-minute reports, distinct-pair
    # deduplication must keep count <= 10 so it does NOT falsely trigger the 48-pair close-copy branch!
    cit_times_short = np.arange(START + 2 * H, START + 12 * H, 10 * M)
    idx_short = np.clip(np.searchsorted(obs_times, cit_times_short, side="right") - 1, 0, len(labels) - 1)
    cit1_vals = t2m_s[idx_short].astype(float) + 0.12

    streams = {(0, "t2m"): (cit_times_long, cit0_vals), (1, "t2m"): (cit_times_short, cit1_vals)}
    contrib = [
        (0, 0, "t2m", 3, START + 70 * H, 0.75),  # flagged against target s!
        (1, 0, "t2m", 3, START + 12 * H, 0.25),  # NOT flagged (only 10 distinct official measurements)
    ]
    audit = d2.target_paired_relay_audit(
        contrib,
        streams,
        off_series,
        covered_stations_idx={0},
        trial=trial,
        standard_ever_copycat=np.array([False, False]),
    )
    assert audit["COVERED_LOAO"]["t2m"]["target_paired_relay_share"] == pytest.approx(0.75)
    assert audit["COVERED_LOAO"]["t2m"]["standard_qc_nearest_ref_ever_flagged_share"] == 0.0
    # Zero denominator for td2m -> explicitly "unavailable"
    assert audit["COVERED_LOAO"]["td2m"]["target_paired_relay_share"] == "unavailable"
    assert audit["COVERED_LOAO"]["td2m"]["standard_qc_nearest_ref_ever_flagged_share"] == "unavailable"


# ----------------------------------------------------------------------------- replay parity & frozen wrapper


def test_design1_2023_wrapper_isolated_replay_matches_reference_hashes(tmp_path):
    """§4 Stage 2a: run_phase3_2023.py replays Jan/Feb into isolated scratch and verifies against
    read-only build.json without touching frozen Design 1 code or live stores."""
    d1 = _load_d1_test()
    stations, truth, issues, snapshots = d1.world()
    CITIZEN = d1.CITIZEN
    (tmp_path / "truth").mkdir()
    truth.to_netcdf(tmp_path / "truth" / "isd_conus_2024.nc")
    months: dict[str, list] = {}
    for a in issues:
        months.setdefault(str(a.init_time.values[0])[:7], []).append(a)
    for month, parts in months.items():
        store = xr.concat(parts, dim="init_time", coords="minimal", compat="identical", join="exact")
        write(store, tmp_path / "candidates" / "gefs-station-corrected-v3" / f"{month}.zarr")
    rows = {k: [] for k in ("station", "issue", "var", "value", "weight", "earned", "age_ns")}
    for i, snap in enumerate(snapshots):
        for v, var in enumerate(arms.VARIABLES):
            value, weight = snap[var]
            for c in np.flatnonzero(weight > 0):
                rows["station"].append(c), rows["issue"].append(i), rows["var"].append(v)
                rows["value"].append(value[c]), rows["weight"].append(weight[c])
                rows["earned"].append(False), rows["age_ns"].append(60 * 10**9)
    cat = {
        "token": [f"CW{i:09d}" for i in range(len(CITIZEN))],
        "lat": CITIZEN[:, 0],
        "lon": CITIZEN[:, 1],
        "elevation_m": CITIZEN[:, 2],
        "stats": np.zeros((len(CITIZEN), 6), dtype=np.int64),
        **rows,
    }
    times = np.array([a.init_time.values[0] for a in issues])
    ids = truth.station_id.values.astype(str).tolist()
    qc_std = _load_script("run_phase3_qc")
    ext_path = tmp_path / "citizen" / "conus" / "issue_extract_2024.nc"
    qc_std.write_extract(
        ext_path,
        cat,
        times,
        ids,
        {
            "complete": 1,
            "year": 2024,
            "run_digest": "syn",
            "tiles_digest": "syn",
            "truth_sha256": p3.sha256_file(tmp_path / "truth" / "isd_conus_2024.nc"),
        },
    )
    wrapper = _load_script("run_phase3_2023")
    wrapper.build_design1_arms(
        tmp_path,
        2024,
        out_dir=tmp_path / "ref_live",
        issues=times,
        stations=ids,
        verify_extract=False,
    )
    ref_json = tmp_path / "ref_live" / "build.json"
    ref_bytes = ref_json.read_bytes()
    replay_jan_feb = wrapper.build_design1_arms(
        tmp_path,
        2024,
        out_dir=tmp_path / "scratch_replay",
        stop_after_month="2024-02",
        verify_against=ref_json,
        issues=times,
        stations=ids,
        verify_extract=False,
    )
    assert [m["period"] for m in replay_jan_feb["arm_b"]] == ["2024-01", "2024-02"]
    assert replay_jan_feb["frozen_code_digest"] == p3.code_digest()
    assert replay_jan_feb["parity"]["verified_months"] == ["2024-01", "2024-02"]

    # Empty replay and truncated one-month replay (missing 2024-02) must BOTH be refused!
    with pytest.raises(wrapper.BuildRefusedError, match="empty replay"):
        wrapper.verify_against_build_json({"arm_b": [], "arm_c": []}, ref_json)
    replay_jan = wrapper.build_design1_arms(
        tmp_path,
        2024,
        out_dir=tmp_path / "scratch_jan_only",
        stop_after_month="2024-01",
        issues=times,
        stations=ids,
        verify_extract=False,
    )
    with pytest.raises(wrapper.BuildRefusedError, match="replay periods incomplete"):
        wrapper.verify_against_build_json(replay_jan, ref_json)

    # Overlapping destination must be refused BEFORE overwriting reference build.json or live stores!
    with pytest.raises(wrapper.BuildRefusedError, match="must not overlap"):
        wrapper.build_design1_arms(
            tmp_path,
            2024,
            out_dir=ref_json.parent,
            stop_after_month="2024-02",
            verify_against=ref_json,
            issues=times,
            stations=ids,
            verify_extract=False,
        )
    assert ref_json.read_bytes() == ref_bytes
    with pytest.raises(wrapper.BuildRefusedError, match="isolated from <data-dir>/candidates"):
        wrapper.build_design1_arms(
            tmp_path,
            2024,
            out_dir=tmp_path / "candidates" / "bad_replay",
            stop_after_month="2024-02",
            issues=times,
            stations=ids,
            verify_extract=False,
        )

    # Tampered reference hash must refuse!
    bad = json.loads(ref_json.read_text())
    bad["arm_c"][0]["content_sha256"] = "0" * 64
    bad_json = tmp_path / "bad_build.json"
    bad_json.write_text(json.dumps(bad))
    with pytest.raises(wrapper.BuildRefusedError, match="hash mismatch"):
        wrapper.verify_against_build_json(replay_jan_feb, bad_json)

    # 2023 provenance verification must refuse explicitly rather than silently skipping!
    times_2023 = times - np.timedelta64(365, "D")
    qc_std.write_extract(
        tmp_path / "citizen" / "conus" / "issue_extract_2023.nc",
        cat,
        times_2023,
        ids,
        {
            "complete": 1,
            "year": 2023,
            "run_digest": "syn",
            "tiles_digest": "syn",
            "truth_sha256": p3.sha256_file(tmp_path / "truth" / "isd_conus_2024.nc"),
        },
    )
    with pytest.raises(wrapper.BuildRefusedError, match="provenance verification for 2023"):
        wrapper.build_design1_arms(
            tmp_path,
            2023,
            out_dir=tmp_path / "scratch_2023",
            issues=times_2023,
            stations=ids,
            verify_extract=True,
        )
    builder = _load_script("build_phase3_design2")
    with pytest.raises(builder.BuildRefusedError, match="provenance verification for 2023"):
        builder.build_track_a(
            tmp_path,
            2023,
            issues=times_2023,
            stations=ids,
            verify_extract=True,
            min_free_bytes=0,
        )


def test_design2_causality_year_isolation_and_isolated_scratch_replay(tmp_path):
    """§6 test 6: nothing after T enters forecasts up to T; year boundary isolated; Design 2
    builder isolated-scratch replay matches full-run content hashes without altering live stores."""
    d1 = _load_d1_test()
    CITIZEN = d1.CITIZEN
    cut = START + 30 * D
    stations, truth0, issues0, snaps0 = d1.world()
    _, truth1, issues1, snaps1 = d1.world(truth_shift_after=cut, citizen_shift_after=cut)
    p2 = d2.oriented_citizen_pairs(
        *CITIZEN.T, truth0.latitude.values, truth0.longitude.values, truth0.elevation_m.values
    )
    r0 = d2.TrackAReplay(2024, truth0, p2)
    r1 = d2.TrackAReplay(2024, truth1, p2)
    wind = (np.full(4, 6.0), np.full(4, -2.0))
    for i_t, (a0, s0, a1, s1) in enumerate(zip(issues0, snaps0, issues1, snaps1, strict=True)):
        w_in = None if i_t == 0 else wind
        b0, c0, dg0 = r0.issue(a0, s0, wind_prev=w_in)
        b1, c1, dg1 = r1.issue(a1, s1, wind_prev=w_in)
        if a0.init_time.values[0] <= cut:
            xr.testing.assert_identical(b0, b1)
            xr.testing.assert_identical(c0, c1)
            assert dg0 == dg1
    # Year isolation refuses truth starting before January 1.
    early = truth0.assign_coords(valid_time=truth0.valid_time.values - 2 * D)
    with pytest.raises(ValueError, match="before the start of the year"):
        d2.TrackAReplay(2024, early, p2)

    # Test build_phase3_design2.py Track A full run vs isolated-scratch --stop-after-month replay.
    (tmp_path / "truth").mkdir()
    truth0.to_netcdf(tmp_path / "truth" / "isd_conus_2024.nc")
    months: dict[str, list] = {}
    for a in issues0:
        months.setdefault(str(a.init_time.values[0])[:7], []).append(a)
        # Write matching 5-lead GEFS cycle for steering wind
        t = a.init_time.values[0]
        g = a.sel(lead_time=[np.timedelta64(h, "h") for h in GEFS_LEADS]).copy(deep=True)
        g.attrs["model_id"] = "noaa-gefs"
        stamp = np.datetime_as_string(t, unit="h").replace("-", "").replace("T", "")
        write(g, tmp_path / "baselines" / "gefs" / f"{stamp}.zarr")
    for month, parts in months.items():
        store = xr.concat(parts, dim="init_time", coords="minimal", compat="identical", join="exact")
        write(store, tmp_path / "candidates" / "gefs-station-corrected-v3" / f"{month}.zarr")
    rows = {k: [] for k in ("station", "issue", "var", "value", "weight", "earned", "age_ns")}
    for i, snap in enumerate(snaps0):
        for v, var in enumerate(arms.VARIABLES):
            value, weight = snap[var]
            for c in np.flatnonzero(weight > 0):
                rows["station"].append(c), rows["issue"].append(i), rows["var"].append(v)
                rows["value"].append(value[c]), rows["weight"].append(weight[c])
                rows["earned"].append(False), rows["age_ns"].append(60 * 10**9)
    cat = {
        "token": [f"CW{i:09d}" for i in range(len(CITIZEN))],
        "lat": CITIZEN[:, 0],
        "lon": CITIZEN[:, 1],
        "elevation_m": CITIZEN[:, 2],
        "stats": np.zeros((len(CITIZEN), 6), dtype=np.int64),
        **rows,
    }
    times = np.array([a.init_time.values[0] for a in issues0])
    ids = truth0.station_id.values.astype(str).tolist()
    qc_std = _load_script("run_phase3_qc")
    qc_std.write_extract(
        tmp_path / "citizen" / "conus" / "issue_extract_2024.nc",
        cat,
        times,
        ids,
        {
            "complete": 1,
            "year": 2024,
            "run_digest": "syn",
            "tiles_digest": "syn",
            "truth_sha256": p3.sha256_file(tmp_path / "truth" / "isd_conus_2024.nc"),
        },
    )
    builder = _load_script("build_phase3_design2")
    # Default min_covered=50 must block scoring handoff on this 4-station fixture (1 covered)!
    with pytest.raises(builder.BuildRefusedError, match="covered_count 1 < 50"):
        builder.build_track_a(
            tmp_path, 2024, issues=times, stations=ids, verify_extract=False, min_free_bytes=0
        )
    blocked = json.loads((tmp_path / "phase3-design2" / "2024" / "track_a" / "build.json").read_text())
    assert blocked["build_kind"] == "full_year" and blocked["scoring_ready"] is False
    assert blocked["stop_reasons"] == ["covered_count 1 < 50"]

    # With min_covered=1, full run succeeds and records scoring_ready=True; partial replay records False.
    full_a = builder.build_track_a(
        tmp_path, 2024, issues=times, stations=ids, verify_extract=False, min_covered=1, min_free_bytes=0
    )
    assert full_a["build_kind"] == "full_year" and full_a["scoring_ready"] is True
    assert full_a["covered_count"] == 1

    # Track A 2024 COVERED parity check against frozen Design 1 phase3/2024/build.json.
    d1_meta = tmp_path / "phase3" / "2024"
    d1_meta.mkdir(parents=True)
    (d1_meta / "build.json").write_text(json.dumps({"covered_sha256": "0" * 64}))
    with pytest.raises(builder.BuildRefusedError, match="COVERED sha256 differs"):
        builder.build_track_a(
            tmp_path, 2024, issues=times, stations=ids, verify_extract=False, min_covered=1, min_free_bytes=0
        )
    mismatch_meta = json.loads((tmp_path / "phase3-design2" / "2024" / "track_a" / "build.json").read_text())
    assert mismatch_meta["design1_covered_parity"] is False and mismatch_meta["scoring_ready"] is False
    (d1_meta / "build.json").write_text(json.dumps({"covered_sha256": full_a["covered_sha256"]}))
    matched = builder.build_track_a(
        tmp_path, 2024, issues=times, stations=ids, verify_extract=False, min_covered=1, min_free_bytes=0
    )
    assert matched["design1_covered_parity"] is True and matched["scoring_ready"] is True

    # Partial replay must require isolated out_root and mark scoring_ready=False.
    with pytest.raises(builder.BuildRefusedError, match="requires an isolated --out-dir"):
        builder.build_track_a(
            tmp_path,
            2024,
            stop_after_month="2024-01",
            issues=times,
            stations=ids,
            verify_extract=False,
            min_free_bytes=0,
        )
    replay_a = builder.build_track_a(
        tmp_path,
        2024,
        out_root=tmp_path / "scratch_d2",
        stop_after_month="2024-01",
        issues=times,
        stations=ids,
        verify_extract=False,
        min_free_bytes=0,
    )
    assert replay_a["build_kind"] == "partial_replay" and replay_a["scoring_ready"] is False
    assert [m["period"] for m in replay_a["arm_c2"]] == ["2024-01"]
    assert replay_a["arm_c2"][0]["content_sha256"] == full_a["arm_c2"][0]["content_sha256"]
