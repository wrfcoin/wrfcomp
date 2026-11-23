"""Scoring kernel tests. All data is synthetic and generated here with fixed seeds."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from wrfcomp.harness import metrics as M
from wrfcomp.harness.config import load_config
from wrfcomp.harness.schemas import GATE_NAMES, LeadMetrics, validate_thresholds

CFG = load_config()
FLOOR = CFG.gate.min_members_for_gate  # smallest ensemble that may pass the gate
MIN_SKILL = CFG.gate.min_crpss_vs_baseline  # skill must be strictly above this
N_MEM = max(20, FLOOR)  # default synthetic ensemble: always eligible for the gate
POINT_DIMS = ("init_time", "lead_time", "station_id")
FCST_DIMS = (*POINT_DIMS, "ens_member")


def station_ids(n: int) -> list[str]:
    return [f"7{i:05d}99999" for i in range(n)]


def coords(
    n_init: int, leads: tuple[int, ...], n_sta: int, start: str = "2024-02-01", freq: str = "6h"
) -> dict:
    return {
        "init_time": pd.date_range(start, periods=n_init, freq=freq).values.astype("datetime64[ns]"),
        "lead_time": np.array(leads, dtype="timedelta64[h]").astype("timedelta64[ns]"),
        "station_id": station_ids(n_sta),
    }


def as_fcst(values: np.ndarray, c: dict) -> xr.DataArray:
    return xr.DataArray(
        values.astype(np.float32), dims=FCST_DIMS, coords={**c, "ens_member": np.arange(values.shape[-1])}
    )


def as_obs(values: np.ndarray, c: dict) -> xr.DataArray:
    return xr.DataArray(values.astype(np.float32), dims=POINT_DIMS, coords=c)


def flat_threshold(n_sta: int, value: float) -> xr.DataArray:
    return xr.DataArray(
        np.full((n_sta, 12), value, dtype=np.float32),
        dims=("station_id", "month"),
        coords={"station_id": station_ids(n_sta), "month": np.arange(1, 13)},
    )


def score(
    fcst: xr.DataArray,
    obs: xr.DataArray,
    threshold: xr.DataArray,
    baseline: xr.DataArray | None = None,
    *,
    cfg: object = CFG,
) -> dict:
    """score_variable on the grid of the observations (the grid never comes from the forecast)."""
    return M.score_variable(
        fcst,
        obs,
        threshold,
        baseline,
        cfg=cfg,
        expected_init_times=obs["init_time"].values,
        expected_station_ids=obs["station_id"].values.tolist(),
    )


def brute_fair_crps(members: np.ndarray, y: float) -> float:
    m = len(members)
    term1 = sum(abs(float(x) - y) for x in members) / m
    term2 = sum(abs(float(a) - float(b)) for a in members for b in members) / (2 * m * (m - 1))
    return term1 - term2


def brute_ecdf_crps(members: np.ndarray, y: float) -> float:
    m = len(members)
    term1 = sum(abs(float(x) - y) for x in members) / m
    term2 = sum(abs(float(a) - float(b)) for a in members for b in members) / (2 * m * m)
    return term1 - term2


def brute_pointwise(
    fcst: xr.DataArray, obs: xr.DataArray, clip: np.ndarray | None = None, *, kernel: object = None
) -> np.ndarray:
    f = fcst.transpose(*FCST_DIMS).values.astype(np.float64)
    o = obs.transpose(*POINT_DIMS).values.astype(np.float64)
    out = np.full(o.shape, np.nan)
    for idx in np.ndindex(o.shape):
        if not np.isfinite(o[idx]):
            continue
        members, y = f[idx], o[idx]
        if clip is not None:
            members, y = np.maximum(members, clip[idx]), max(y, clip[idx])
        out[idx] = (kernel or brute_fair_crps)(members, y)
    return out


def synthetic(
    seed: int, *, n_init: int = 120, leads: tuple[int, ...] = (6, 24), n_sta: int = 8, n_mem: int = N_MEM
) -> dict:
    """Obs = 280 + 3*mu1 + 2*mu2 + 0.5*eps.

    The `good` ensemble knows mu1 and mu2 (spread 0.5); the `base` ensemble knows only mu1 and has the
    matching wider spread. Both are calibrated; `good` is sharper.
    """
    rng = np.random.default_rng(seed)
    c = coords(n_init, leads, n_sta)
    shape = (n_init, len(leads), n_sta)
    mu1, mu2, eps = (rng.normal(size=shape) for _ in range(3))
    known = 280.0 + 3.0 * mu1 + 2.0 * mu2
    obs = known + 0.5 * eps
    good = known[..., None] + 0.5 * rng.normal(size=(*shape, n_mem))
    base = (280.0 + 3.0 * mu1)[..., None] + math.sqrt(4.25) * rng.normal(size=(*shape, n_mem))
    return {
        "coords": c,
        "known": known,
        "obs": as_obs(obs, c),
        "good": as_fcst(good, c),
        "base": as_fcst(base, c),
        "threshold": flat_threshold(n_sta, 285.0),
        "rng": rng,
    }


# --- fair CRPS and tail CRPS against brute force -----------------------------------------------


def small_case(seed: int = 7):
    rng = np.random.default_rng(seed)
    c = coords(3, (6, 24), 4, start="2024-01-30", freq="12h")  # valid times cross into February
    shape = (3, 2, 4)
    fcst = as_fcst(rng.normal(280, 4, size=(*shape, 5)), c)
    obs = as_obs(rng.normal(280, 4, size=shape), c)
    return rng, c, fcst, obs


def test_fair_crps_matches_brute_force() -> None:
    _, _, fcst, obs = small_case()
    got = M.fair_crps(fcst, obs).transpose(*POINT_DIMS).values
    np.testing.assert_allclose(got, brute_pointwise(fcst, obs), rtol=1e-12, atol=1e-12)
    assert (got > 0).all()
    ecdf = M.ecdf_crps(fcst, obs).transpose(*POINT_DIMS).values
    brute_ecdf = brute_pointwise(fcst, obs, kernel=brute_ecdf_crps)
    np.testing.assert_allclose(ecdf, brute_ecdf, rtol=1e-12, atol=1e-12)
    assert (ecdf > got).all()  # the ecdf estimator is the fair one plus a positive spread term


def test_fair_crps_two_members_by_hand() -> None:
    c = coords(1, (6,), 1)
    fcst = as_fcst(np.array([1.0, 3.0]).reshape(1, 1, 1, 2), c)
    obs = as_obs(np.array([2.0]).reshape(1, 1, 1), c)
    # mean|x - y| = 1 ; fair spread term = (|1-3| + |3-1|) / (2 * 2 * 1) = 1
    assert float(M.fair_crps(fcst, obs).squeeze()) == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError, match="at least 2"):
        M.fair_crps(fcst.isel(ens_member=[0]), obs)


def test_tail_crps_matches_brute_force_with_station_month_thresholds() -> None:
    rng, c, fcst, obs = small_case()
    threshold = xr.DataArray(
        rng.normal(281, 2, size=(4, 12)).astype(np.float32),
        dims=("station_id", "month"),
        coords={"station_id": c["station_id"], "month": np.arange(1, 13)},
    )
    valid = c["init_time"][:, None] + c["lead_time"][None, :]
    months = pd.DatetimeIndex(valid.ravel()).month.values.reshape(valid.shape)
    assert set(months.ravel().tolist()) == {1, 2}  # the month really is taken from the valid time
    clip = np.empty((3, 2, 4))
    for i, j, k in np.ndindex(clip.shape):
        clip[i, j, k] = threshold.values.astype(np.float64)[k, months[i, j] - 1]

    np.testing.assert_array_equal(
        M.broadcast_threshold(threshold, fcst["init_time"], fcst["lead_time"]).values.astype(np.float64), clip
    )
    got = M.tail_crps(fcst, obs, threshold).transpose(*POINT_DIMS).values
    np.testing.assert_allclose(got, brute_pointwise(fcst, obs, clip), rtol=1e-12, atol=1e-12)
    assert (got >= -1e-12).all()
    got_ecdf = M.tail_crps(fcst, obs, threshold, method="ecdf").transpose(*POINT_DIMS).values
    np.testing.assert_allclose(
        got_ecdf, brute_pointwise(fcst, obs, clip, kernel=brute_ecdf_crps), rtol=1e-12, atol=1e-12
    )
    # a threshold far below everything gives back the plain fair CRPS
    np.testing.assert_allclose(
        M.tail_crps(fcst, obs, -1e6).transpose(*POINT_DIMS).values, brute_pointwise(fcst, obs), rtol=1e-9
    )


def test_tail_crps_is_zero_below_threshold() -> None:
    _, c, fcst, obs = small_case()
    above_everything = float(max(fcst.max(), obs.max())) + 1.0
    np.testing.assert_array_equal(M.tail_crps(fcst, obs, above_everything).values, 0.0)
    np.testing.assert_array_equal(M.tail_crps(fcst, obs, flat_threshold(4, above_everything)).values, 0.0)
    result = score(fcst, obs, flat_threshold(4, above_everything), cfg=CFG)
    assert result["all"]["twcrps_p95"] == 0.0
    assert result["all"]["twcrps_p95_ecdf"] == 0.0
    assert result["all"]["loss"] == pytest.approx(0.70 * result["all"]["crps"])


# --- thresholds --------------------------------------------------------------------------------


def train_truth(end: str, hours: int = 24 * 70) -> xr.DataArray:
    rng = np.random.default_rng(11)
    times = pd.date_range(end=end, periods=hours, freq="1h").values.astype("datetime64[ns]")
    values = rng.normal(280, 5, size=(hours, 3)).astype(np.float32)
    values[rng.random(values.shape) < 0.1] = np.nan
    return xr.DataArray(
        values, dims=("valid_time", "station_id"), coords={"valid_time": times, "station_id": station_ids(3)}
    )


def test_tail_thresholds_match_numpy_quantiles() -> None:
    truth = train_truth("2023-12-17T18:00")
    out = M.compute_tail_thresholds(truth, 0.95, cfg=CFG)
    assert out.dims == ("station_id", "month")
    assert out["month"].values.tolist() == list(range(1, 13))
    assert out.dtype == np.float32
    months = pd.DatetimeIndex(truth["valid_time"].values).month.values
    present = sorted(set(months.tolist()))
    assert present == [10, 11, 12]
    for s in range(3):
        for month in range(1, 13):
            got = out.values[s, month - 1]
            if month not in present:
                assert np.isnan(got)
                continue
            expected = np.nanquantile(truth.values[months == month, s].astype(np.float64), 0.95)
            assert got == np.float32(expected)
    ds = M.compute_tail_thresholds(truth.to_dataset(name="t2m"), CFG.gate.tail_quantile, cfg=CFG)
    validate_thresholds(ds, variables=["t2m"])
    np.testing.assert_array_equal(ds["t2m"].values, out.values)


def test_tail_thresholds_refuse_mixed_dates_and_nat() -> None:
    # Regression: a NaT made max() return NaT, the cutoff comparison was false, and a validation-year
    # observation went into the January threshold.
    def truth(times: list[str]) -> xr.DataArray:
        stamps = np.array(times, dtype="datetime64[ns]")
        values = np.random.default_rng(12).normal(280, 5, size=(len(times), 2)).astype(np.float32)
        coords_ = {"valid_time": stamps, "station_id": station_ids(2)}
        return xr.DataArray(values, dims=("valid_time", "station_id"), coords=coords_)

    train_only = ["2023-01-05T00", "2023-01-06T00", "2022-01-07T00"]
    M.compute_tail_thresholds(truth(train_only), 0.95, cfg=CFG)
    with pytest.raises(ValueError, match="after the train split end"):
        M.compute_tail_thresholds(truth([*train_only, "2024-01-05T00"]), 0.95, cfg=CFG)
    with pytest.raises(ValueError, match="after the train split end"):
        M.compute_tail_thresholds(truth(["2024-01-05T00", *train_only]), 0.95, cfg=CFG)  # unsorted
    with pytest.raises(ValueError, match="NaT"):
        M.compute_tail_thresholds(truth([*train_only, "2024-01-05T00", "NaT"]), 0.95, cfg=CFG)
    with pytest.raises(ValueError, match="NaT"):
        M.compute_tail_thresholds(truth([*train_only, "NaT"]), 0.95, cfg=CFG)
    with pytest.raises(ValueError, match="NaT"):
        as_dataset = truth(["NaT", "2024-01-05T00", *train_only]).to_dataset(name="t2m")
        M.compute_tail_thresholds(as_dataset, 0.95, cfg=CFG)


def test_tail_thresholds_refuse_post_train_data() -> None:
    M.compute_tail_thresholds(train_truth("2023-12-17T18:00", hours=48), 0.95, cfg=CFG)  # exactly the end
    with pytest.raises(ValueError, match="after the train split end"):
        M.compute_tail_thresholds(train_truth("2023-12-17T19:00", hours=48), 0.95, cfg=CFG)
    with pytest.raises(ValueError, match="after the train split end"):
        late = train_truth("2024-03-01T00:00", hours=48).to_dataset(name="t2m")
        M.compute_tail_thresholds(late, 0.95, cfg=CFG)
    with pytest.raises(ValueError, match="after the train split end"):
        M.compute_tail_thresholds(train_truth("2024-03-01T00:00", hours=48), 0.95)  # default config too


# --- spread-skill ------------------------------------------------------------------------------


def test_spread_skill_calibrated_near_one_and_overconfident_fails() -> None:
    rng = np.random.default_rng(3)
    n_mem = 50
    c = coords(200, (6, 24), 10)
    shape = (200, 2, 10)
    centre = 280 + 3 * rng.normal(size=shape)
    obs = as_obs(centre + 2.0 * rng.normal(size=shape), c)
    calibrated = as_fcst(centre[..., None] + 2.0 * rng.normal(size=(*shape, n_mem)), c)
    overconfident = as_fcst(centre[..., None] + 0.6 * rng.normal(size=(*shape, n_mem)), c)

    ratio = float(M.spread_skill_ratio(calibrated, obs))
    # 4000 points, tolerance +-0.03
    assert ratio == pytest.approx(1.0, abs=0.03)
    per_lead = M.spread_skill_ratio(calibrated, obs, preserve_dims=("lead_time",))
    assert per_lead.dims == ("lead_time",)
    assert np.all(np.abs(per_lead.values - 1.0) < 0.05)

    threshold = flat_threshold(10, 285.0)
    good = score(calibrated, obs, threshold, cfg=CFG)
    assert good["all"]["spread_skill_ratio"] == pytest.approx(ratio, rel=1e-12)
    assert M.evaluate_gates(good, CFG)["spread_skill"] is True

    bad = score(overconfident, obs, threshold, cfg=CFG)
    assert bad["all"]["spread_skill_ratio"] < CFG.gate.spread_skill_min
    assert bad["all"]["spread_skill_ratio"] == pytest.approx(0.3, abs=0.05)
    assert M.evaluate_gates(bad, CFG)["spread_skill"] is False


@pytest.mark.parametrize("n_mem", [2, 30])
def test_spread_skill_is_one_for_a_calibrated_ensemble_of_any_size(n_mem: int) -> None:
    # Obs and members are exchangeable draws around the same centre. 8000 points; tolerance +-0.04.
    rng = np.random.default_rng(40 + n_mem)
    c = coords(400, (6, 24), 10)
    shape = (400, 2, 10)
    centre = 280 + 3 * rng.normal(size=shape)
    obs = as_obs(centre + 2.0 * rng.normal(size=shape), c)
    fcst = as_fcst(centre[..., None] + 2.0 * rng.normal(size=(*shape, n_mem)), c)
    ratio = float(M.spread_skill_ratio(fcst, obs))
    assert ratio == pytest.approx(1.0, abs=0.04)

    # by hand: sqrt((M + 1) / M) * sqrt(mean unbiased variance) / rmse of the ensemble mean
    f, o = fcst.values.astype(np.float64), obs.values.astype(np.float64)
    spread = math.sqrt(f.var(axis=-1, ddof=1).mean())
    rmse = math.sqrt(((f.mean(axis=-1) - o) ** 2).mean())
    assert ratio == pytest.approx(math.sqrt((n_mem + 1) / n_mem) * spread / rmse, rel=1e-12)
    assert spread / rmse == pytest.approx(math.sqrt(n_mem / (n_mem + 1)), abs=0.04)  # uncorrected value

    result = score(fcst, obs, flat_threshold(10, 285.0), cfg=CFG)
    assert result["all"]["spread_skill_ratio"] == pytest.approx(ratio, rel=1e-12)
    assert M.evaluate_gates(result, CFG)["spread_skill"] is True


def test_composite_loss_uses_config_weights() -> None:
    assert M.composite_loss(2.0, 1.0, CFG) == pytest.approx(0.70 * 2.0 + 0.30 * 1.0)
    weights = CFG.gate.loss_weights
    assert M.composite_loss(1.0, 0.0, CFG) == weights.crps_fair
    assert M.composite_loss(0.0, 1.0, CFG) == weights.twcrps_p95


# --- score_variable and gates ------------------------------------------------------------------


def test_score_variable_matches_brute_force_and_schema() -> None:
    s = synthetic(21, n_init=12, n_sta=4, n_mem=6)
    obs = s["obs"].copy()
    obs.values[2, 0, 1] = np.nan
    obs.values[5, 1, :] = np.nan
    result = score(s["good"], obs, s["threshold"], s["base"], cfg=CFG)
    assert list(result) == ["all", "6", "24"]

    mask = np.isfinite(obs.values)
    clip = np.full(mask.shape, 285.0)
    crps_c, tw_c = brute_pointwise(s["good"], obs), brute_pointwise(s["good"], obs, clip)
    crps_b, tw_b = brute_pointwise(s["base"], obs), brute_pointwise(s["base"], obs, clip)
    for key, sel in (("all", slice(None)), ("6", slice(0, 1)), ("24", slice(1, 2))):
        m = mask[:, sel, :]
        entry = result[key]
        assert entry["n_pairs"] == int(m.sum())
        crps, tw = crps_c[:, sel, :][m].mean(), tw_c[:, sel, :][m].mean()
        base_crps, base_tw = crps_b[:, sel, :][m].mean(), tw_b[:, sel, :][m].mean()
        assert entry["crps"] == pytest.approx(crps, rel=1e-10)
        assert entry["twcrps_p95"] == pytest.approx(tw, rel=1e-10)
        assert entry["loss"] == pytest.approx(0.7 * crps + 0.3 * tw, rel=1e-10)
        assert entry["baseline_twcrps_p95"] == pytest.approx(base_tw, rel=1e-10)
        assert entry["n_members"] == 6
        ecdf = {
            name: brute_pointwise(s[who], obs, c_, kernel=brute_ecdf_crps)[:, sel, :][m].mean()
            for name, who, c_ in (
                ("crps", "good", None),
                ("tw", "good", clip),
                ("base_crps", "base", None),
                ("base_tw", "base", clip),
            )
        }
        assert entry["crps_ecdf"] == pytest.approx(ecdf["crps"], rel=1e-10)
        assert entry["twcrps_p95_ecdf"] == pytest.approx(ecdf["tw"], rel=1e-10)
        assert entry["loss_ecdf"] == pytest.approx(0.7 * ecdf["crps"] + 0.3 * ecdf["tw"], rel=1e-10)
        assert entry["baseline_twcrps_p95_ecdf"] == pytest.approx(ecdf["base_tw"], rel=1e-10)
        assert entry["crpss_vs_baseline_ecdf"] == pytest.approx(
            1 - (0.7 * ecdf["crps"] + 0.3 * ecdf["tw"]) / (0.7 * ecdf["base_crps"] + 0.3 * ecdf["base_tw"]),
            rel=1e-9,
        )
        assert entry["crpss_vs_baseline"] == pytest.approx(
            1 - (0.7 * crps + 0.3 * tw) / (0.7 * base_crps + 0.3 * base_tw), rel=1e-9
        )
        LeadMetrics.model_validate(entry)  # the dict is exactly what the Scorecard stores
    assert result["all"]["n_pairs"] == result["6"]["n_pairs"] + result["24"]["n_pairs"] == mask.sum()


def test_identical_baseline_has_zero_skill_and_fails() -> None:
    s = synthetic(22)
    result = score(s["good"], s["obs"], s["threshold"], s["good"].copy(), cfg=CFG)
    for entry in result.values():
        assert entry["crpss_vs_baseline"] == 0.0
        assert entry["crpss_vs_baseline_ecdf"] == 0.0
        assert entry["baseline_twcrps_p95"] == entry["twcrps_p95"]
        assert entry["dm_pvalue"] is None  # all differences are exactly zero: the test is undefined
    gates = M.evaluate_gates(result, CFG)
    assert gates["skill"] is False
    assert gates["dm"] is False
    assert not all(gates.values())


def test_clearly_better_candidate_passes_all_gates() -> None:
    s = synthetic(23)
    result = score(s["good"], s["obs"], s["threshold"], s["base"], cfg=CFG)
    gates = M.evaluate_gates(result, CFG)
    assert tuple(gates) == GATE_NAMES
    assert gates == dict.fromkeys(GATE_NAMES, True)
    assert result["all"]["crpss_vs_baseline"] > 0.5
    assert result["all"]["dm_pvalue"] < 1e-6
    # and the reverse comparison fails skill, tail and significance
    reverse = score(s["base"], s["obs"], s["threshold"], s["good"], cfg=CFG)
    reverse_gates = M.evaluate_gates(reverse, CFG)
    assert reverse["all"]["crpss_vs_baseline"] < 0
    assert reverse["all"]["dm_pvalue"] > 0.999
    assert (reverse_gates["skill"], reverse_gates["tail"], reverse_gates["dm"]) == (False, False, False)


def test_better_on_average_but_worse_in_tail_fails() -> None:
    s = synthetic(24)
    rng = s["rng"]
    shape = s["known"].shape
    n_mem = s["good"].sizes["ens_member"]
    # Sharp in ordinary weather, but falls back to a flat climatological guess whenever it gets warm.
    warm = s["known"] > 283.0
    members = np.where(
        warm[..., None],
        280.0 + 3.6 * rng.normal(size=(*shape, n_mem)),
        s["known"][..., None] + 0.5 * rng.normal(size=(*shape, n_mem)),
    )
    candidate = as_fcst(members, s["coords"])
    result = score(candidate, s["obs"], s["threshold"], s["base"], cfg=CFG)
    entry = result["all"]
    assert entry["crpss_vs_baseline"] > 0.1  # better on the headline loss
    assert entry["twcrps_p95"] > entry["baseline_twcrps_p95"]  # worse in the tail
    gates = M.evaluate_gates(result, CFG)
    assert gates["skill"] is True
    assert gates["tail"] is False
    assert not all(gates.values())


def test_no_baseline_means_gate_not_passed() -> None:
    s = synthetic(25, n_init=20)
    result = score(s["good"], s["obs"], s["threshold"], cfg=CFG)
    assert set(result["all"]) == {
        "crps", "twcrps_p95", "loss", "crps_ecdf", "twcrps_p95_ecdf", "loss_ecdf",
        "spread_skill_ratio", "n_pairs", "n_masked_no_threshold", "n_members",
    }  # fmt: skip
    gates = M.evaluate_gates(result, CFG)
    assert gates == {"skill": False, "tail": False, "spread_skill": True, "dm": False, "min_members": True}
    with pytest.raises(ValueError, match="'all'"):
        M.evaluate_gates({"6": result["6"]}, CFG)


def test_gate_boundaries() -> None:
    base = {"crps": 1.0, "twcrps_p95": 0.2, "loss": 0.76, "n_pairs": 10, "baseline_twcrps_p95": 0.2}

    def gates(**changes: float | None) -> dict[str, bool]:
        entry = {
            **base,
            "crpss_vs_baseline": MIN_SKILL + 0.1,
            "crpss_vs_baseline_ecdf": MIN_SKILL + 0.1,
            "spread_skill_ratio": 1.0,
            "dm_pvalue": 0.01,
            "n_members": FLOOR,
            **changes,
        }
        return M.evaluate_gates({"all": entry}, CFG)

    just_above = float(np.nextafter(MIN_SKILL, 1.0))
    assert all(gates().values())
    # skill: strictly greater than the configured minimum, under BOTH estimators
    assert gates(crpss_vs_baseline=MIN_SKILL)["skill"] is False
    assert gates(crpss_vs_baseline_ecdf=MIN_SKILL)["skill"] is False
    assert gates(crpss_vs_baseline=MIN_SKILL, crpss_vs_baseline_ecdf=MIN_SKILL)["skill"] is False
    assert gates(crpss_vs_baseline=just_above, crpss_vs_baseline_ecdf=just_above)["skill"] is True
    assert gates(crpss_vs_baseline=just_above, crpss_vs_baseline_ecdf=MIN_SKILL)["skill"] is False
    assert gates(crpss_vs_baseline=MIN_SKILL, crpss_vs_baseline_ecdf=just_above)["skill"] is False
    assert gates(crpss_vs_baseline=MIN_SKILL - 1e-9)["skill"] is False
    assert gates(crpss_vs_baseline_ecdf=-0.2, crpss_vs_baseline=0.9)["skill"] is False
    assert gates(crpss_vs_baseline_ecdf=None)["skill"] is False
    # members: at least the configured floor
    assert gates(n_members=FLOOR)["min_members"] is True
    assert gates(n_members=FLOOR - 1)["min_members"] is False
    assert gates(n_members=None)["min_members"] is False
    assert gates(n_members=FLOOR + 30)["min_members"] is True
    assert gates(twcrps_p95=0.2)["tail"] is True  # equal tail is "not worse"
    assert gates(twcrps_p95=0.2000001)["tail"] is False
    assert gates(spread_skill_ratio=0.85)["spread_skill"] is True
    assert gates(spread_skill_ratio=1.15)["spread_skill"] is True
    assert gates(spread_skill_ratio=0.8499)["spread_skill"] is False
    assert gates(spread_skill_ratio=1.1501)["spread_skill"] is False
    assert gates(dm_pvalue=0.05)["dm"] is False  # strictly below alpha
    assert gates(dm_pvalue=0.0499)["dm"] is True
    for key in ("crpss_vs_baseline", "spread_skill_ratio", "dm_pvalue", "twcrps_p95", "baseline_twcrps_p95"):
        assert not all(gates(**{key: None}).values())


def gate_config(**gate_values: float) -> object:
    """The repository config with some promotion-gate values replaced."""
    raw = CFG.model_dump(mode="json")
    raw["evaluation"]["promotion_gate"].update(gate_values)
    return type(CFG).model_validate(raw)


def test_skill_gate_boundary_at_exactly_five_percent() -> None:
    # Owner decision of 2026-11-23: skill must be ABOVE 5 percent, under both estimators, with at
    # least 20 members. Pinned here with explicit values, whatever configs/splits.yaml says later.
    cfg = gate_config(min_crpss_vs_baseline=0.05, min_members_for_gate=20)

    def gates(fair: float, ecdf: float, members: int = 20) -> dict[str, bool]:
        entry = {
            "crps": 1.0, "twcrps_p95": 0.2, "loss": 0.76, "n_pairs": 10, "baseline_twcrps_p95": 0.2,
            "spread_skill_ratio": 1.0, "dm_pvalue": 0.01, "n_members": members,
            "crpss_vs_baseline": fair, "crpss_vs_baseline_ecdf": ecdf,
        }  # fmt: skip
        return M.evaluate_gates({"all": entry}, cfg)

    above = float(np.nextafter(0.05, 1.0))
    below = float(np.nextafter(0.05, 0.0))
    assert gates(0.05, 0.05)["skill"] is False  # exactly 5 percent is not enough
    assert gates(above, above)["skill"] is True  # the next float64 above it is
    assert gates(below, 0.9)["skill"] is False
    assert gates(0.05, above)["skill"] is False  # fair at the boundary
    assert gates(above, 0.05)["skill"] is False  # ecdf at the boundary
    assert gates(0.9, 0.05)["skill"] is False
    assert gates(0.0501, 0.0501)["skill"] is True
    assert gates(0.04, 0.04)["skill"] is False  # passed the old rule (above 0), fails this one
    assert all(gates(above, above).values())
    # the member floor: 20 passes, 19 does not, and it does not depend on skill
    assert gates(0.9, 0.9, members=20)["min_members"] is True
    assert gates(0.9, 0.9, members=19)["min_members"] is False
    assert not all(gates(0.9, 0.9, members=19).values())
    # and the repository config carries exactly these values
    assert (CFG.gate.min_crpss_vs_baseline, CFG.gate.min_members_for_gate) == (0.05, 20)


def test_modest_gain_passes_or_fails_with_the_configured_minimum_skill() -> None:
    # A real scoring run: the same forecasts judged with a 0 and with a 5 percent minimum.
    rng = np.random.default_rng(41)
    c = coords(200, (6, 24), 6)
    shape = (200, 2, 6)
    centre = 280 + 3 * rng.normal(size=shape)
    obs = as_obs(centre + rng.normal(size=shape), c)
    thr = flat_threshold(6, 283.0)
    base = as_fcst(centre[..., None] + rng.normal(size=(*shape, N_MEM)), c)
    # A candidate that knows a little of what the baseline does not: a real but modest gain.
    partly_known = 0.85 * centre + 0.15 * obs.values
    marginal = as_fcst(partly_known[..., None] + 0.86 * rng.normal(size=(*shape, N_MEM)), c)
    result = score(marginal, obs, thr, base)
    fair, ecdf = result["all"]["crpss_vs_baseline"], result["all"]["crpss_vs_baseline_ecdf"]
    assert 0.0 < min(fair, ecdf) and max(fair, ecdf) < 0.30
    for minimum in (0.0, min(fair, ecdf), max(fair, ecdf), 0.5):
        expected = fair > minimum and ecdf > minimum
        assert M.evaluate_gates(result, gate_config(min_crpss_vs_baseline=minimum))["skill"] is expected
    assert M.evaluate_gates(result, gate_config(min_crpss_vs_baseline=min(fair, ecdf)))["skill"] is False
    assert M.evaluate_gates(result, gate_config(min_crpss_vs_baseline=0.0))["skill"] is True


# --- masking -----------------------------------------------------------------------------------


def test_nan_obs_masked_identically_for_candidate_and_baseline() -> None:
    s = synthetic(26, n_init=40)
    rng = np.random.default_rng(99)
    obs = s["obs"].copy()
    holes = rng.random(obs.shape) < 0.2
    obs.values[holes] = np.nan
    obs.values[7] = np.nan  # one init time with no observation at all
    n_valid = int(np.isfinite(obs.values).sum())

    result = score(s["good"], obs, s["threshold"], s["base"], cfg=CFG)
    assert result["all"]["n_pairs"] == n_valid

    # an init time with no observation at all leaves a hole in the DM series: undefined, gate fails
    assert result["all"]["dm_pvalue"] is None
    assert M.evaluate_gates(result, CFG)["dm"] is False

    # The baseline is scored on exactly the candidate's mask: scoring it alone gives the same numbers.
    base_alone = score(s["base"], obs, s["threshold"], cfg=CFG)
    for key in result:
        assert result[key]["n_pairs"] == base_alone[key]["n_pairs"]
        assert result[key]["baseline_twcrps_p95"] == base_alone[key]["twcrps_p95"]
        assert result[key]["crpss_vs_baseline"] == pytest.approx(
            1 - result[key]["loss"] / base_alone[key]["loss"], rel=1e-12
        )

    # What either forecast says at unobserved points cannot matter, even if it is NaN there.
    cand2, base2 = s["good"].copy(), s["base"].copy()
    cand2.values[~np.isfinite(obs.values)] = np.nan
    base2.values[~np.isfinite(obs.values)] = 1e6
    assert score(cand2, obs, s["threshold"], base2, cfg=CFG) == result

    # A hole in either forecast at an observed point is refused (no cherry-picking).
    i, j, k = np.argwhere(np.isfinite(obs.values))[0]
    broken = s["good"].copy()
    broken.values[i, j, k, 0] = np.nan
    with pytest.raises(ValueError, match="forecast has missing"):
        score(broken, obs, s["threshold"], s["base"], cfg=CFG)
    broken_base = s["base"].copy()
    broken_base.values[i, j, k, 3] = np.inf
    with pytest.raises(ValueError, match="baseline has missing"):
        score(s["good"], obs, s["threshold"], broken_base, cfg=CFG)


def test_missing_threshold_masks_the_point_for_every_model() -> None:
    # 30 days of 6-hourly runs from 1 February: valid times fall in February and March.
    s = synthetic(33, n_init=120, n_sta=6)
    obs = s["obs"].copy()
    obs.values[np.random.default_rng(34).random(obs.shape) < 0.1] = np.nan
    thr = s["threshold"].copy()
    thr.values[0, 2] = np.nan  # station 0 has no training data in March
    thr.values[3, :] = np.nan  # station 3 has no training data at all
    valid = s["coords"]["init_time"][:, None] + s["coords"]["lead_time"][None, :]
    march = (pd.DatetimeIndex(valid.ravel()).month.values == 3).reshape(valid.shape)
    assert march.any() and not march.all()
    no_threshold = np.zeros(obs.shape, dtype=bool)
    no_threshold[:, :, 0] = march
    no_threshold[:, :, 3] = True
    finite = np.isfinite(obs.values)
    expected_mask = finite & ~no_threshold

    mask = M.scoring_mask(obs, thr)
    assert mask.dims == POINT_DIMS
    np.testing.assert_array_equal(mask.values, expected_mask)
    np.testing.assert_array_equal(
        M.scoring_mask(obs.transpose("station_id", "init_time", "lead_time"), thr).values, expected_mask
    )
    np.testing.assert_array_equal(M.scoring_mask(obs, s["threshold"]).values, finite)

    result = score(s["good"], obs, thr, s["base"])
    for key, sel in (("all", slice(None)), ("6", slice(0, 1)), ("24", slice(1, 2))):
        assert result[key]["n_pairs"] == int(expected_mask[:, sel, :].sum())
        assert result[key]["n_masked_no_threshold"] == int((finite & no_threshold)[:, sel, :].sum())
        assert result[key]["n_pairs"] + result[key]["n_masked_no_threshold"] == int(finite[:, sel, :].sum())
    assert result["all"]["n_masked_no_threshold"] > 0

    # Same numbers as if truth itself were missing at those points; only the bookkeeping differs.
    cut = obs.where(~no_threshold)
    as_if_missing = score(s["good"], cut, s["threshold"], s["base"])
    for key in result:
        got = {k: v for k, v in result[key].items() if k != "n_masked_no_threshold"}
        want = {k: v for k, v in as_if_missing[key].items() if k != "n_masked_no_threshold"}
        assert got == want
        assert as_if_missing[key]["n_masked_no_threshold"] == 0

    # The mask does not depend on the forecast: the baseline alone is scored on the same points, and
    # what either forecast holds at a masked point (even NaN) changes nothing.
    base_alone = score(s["base"], obs, thr)
    for key in result:
        assert base_alone[key]["n_pairs"] == result[key]["n_pairs"]
        assert base_alone[key]["twcrps_p95"] == result[key]["baseline_twcrps_p95"]
    cand2, base2 = s["good"].copy(), s["base"].copy()
    cand2.values[no_threshold] = np.nan
    base2.values[no_threshold] = np.inf
    assert score(cand2, obs, thr, base2) == result

    # The NaN rule still holds on every scored point, for both.
    i, j, k = np.argwhere(expected_mask)[0]
    broken = s["good"].copy()
    broken.values[i, j, k, 0] = np.nan
    with pytest.raises(ValueError, match="forecast has missing"):
        score(broken, obs, thr, s["base"])
    broken_base = s["base"].copy()
    broken_base.values[i, j, k, 0] = np.nan
    with pytest.raises(ValueError, match="baseline has missing"):
        score(s["good"], obs, thr, broken_base)

    # No threshold anywhere: nothing is scored, nothing is defined, no gate passes.
    empty = score(s["good"], obs, thr.where(False), s["base"])
    assert empty["all"]["n_pairs"] == 0 and empty["all"]["crps"] is None
    assert empty["all"]["n_masked_no_threshold"] == int(finite.sum())
    gates = M.evaluate_gates(empty, CFG)
    assert (gates["skill"], gates["tail"], gates["spread_skill"], gates["dm"]) == (False,) * 4


@pytest.mark.parametrize("bad", [np.inf, -np.inf])
def test_infinite_threshold_is_refused_on_every_direct_call(bad: float) -> None:
    s = synthetic(35, n_init=12, n_sta=4, n_mem=6)
    good, obs, base = s["good"], s["obs"], s["base"]
    grid = {"cfg": CFG, "expected_init_times": obs["init_time"].values}
    grid["expected_station_ids"] = obs["station_id"].values.tolist()
    message = "infinite value"

    # Control: NaN is the legal marker. One station without thresholds is masked and counted.
    with_nan = s["threshold"].copy()
    with_nan.values[1, :] = np.nan
    control = M.score_variable(good, obs, with_nan, base, **grid)
    assert control["all"]["n_pairs"] == 3 * 12 * 2
    assert control["all"]["n_masked_no_threshold"] == 12 * 2
    assert int(M.scoring_mask(obs, with_nan).sum()) == 3 * 12 * 2
    assert np.isnan(M.tail_crps(good, obs, with_nan).sel(station_id=obs["station_id"].values[1])).all()

    # The same station with an infinite threshold must not be scored as "missing": it is refused.
    table = s["threshold"].copy()
    table.values[1, :] = bad
    one_cell = s["threshold"].copy()
    one_cell.values[3, 7] = bad  # August: no valid time of this case uses it, still refused
    for thresholds in (table, one_cell, table.chunk({"station_id": 1})):
        with pytest.raises(ValueError, match=message):
            M.scoring_mask(obs, thresholds)
        with pytest.raises(ValueError, match=message):
            M.score_variable(good, obs, thresholds, base, **grid)
        with pytest.raises(ValueError, match=message):
            M.score_variable(good, obs, thresholds, **grid)
        for method in ("fair", "ecdf"):
            with pytest.raises(ValueError, match=message):
                M.tail_crps(good, obs, thresholds, method=method)

    # tail_crps with a plain number, and with an array already on the forecast's points
    on_points = M.broadcast_threshold(s["threshold"], obs["init_time"], obs["lead_time"]).copy()
    on_points.values[0, 0, 0] = bad
    for method in ("fair", "ecdf"):
        with pytest.raises(ValueError, match=message):
            M.tail_crps(good, obs, bad, method=method)
        with pytest.raises(ValueError, match=message):
            M.tail_crps(good, obs, np.float32(bad), method=method)
        with pytest.raises(ValueError, match=message):
            M.tail_crps(good, obs, on_points, method=method)
        assert np.isfinite(M.tail_crps(good, obs, 285.0, method=method)).all()  # a number still works
        assert np.isnan(M.tail_crps(good, obs, np.nan, method=method)).all()  # NaN gives NaN, no error


WIDE_LONG_DOUBLE = np.finfo(np.longdouble).max > np.finfo(np.float64).max


@pytest.mark.skipif(not WIDE_LONG_DOUBLE, reason="long double is no wider than float64 on this platform")
@pytest.mark.parametrize("sign", [1, -1])
def test_long_double_threshold_that_overflows_float64_is_refused(sign: int) -> None:
    # 1e400 is a finite long double, but scoring works in float64, where it is infinity. The check
    # has to look at the values that are actually used.
    s = synthetic(36, n_init=12, n_sta=4, n_mem=6)
    good, obs, base = s["good"], s["obs"], s["base"]
    grid = {"cfg": CFG, "expected_init_times": obs["init_time"].values}
    grid["expected_station_ids"] = obs["station_id"].values.tolist()
    huge = sign * np.longdouble("1e400")
    assert np.isfinite(huge) and np.isinf(np.float64(huge))
    message = "infinite value\\(s\\) as float64"

    wide = s["threshold"].astype(np.longdouble)
    wide.values[1, :] = huge
    assert np.isfinite(wide.values).all()  # finite in its own type: the old check let it through
    on_points = M.broadcast_threshold(s["threshold"], obs["init_time"], obs["lead_time"])
    on_points = on_points.astype(np.longdouble)
    on_points.values[0, 0, 0] = huge
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # refused cleanly, not after an overflow warning
        for thresholds in (wide, wide.chunk({"station_id": 1})):
            with pytest.raises(ValueError, match=message):
                M.scoring_mask(obs, thresholds)
            with pytest.raises(ValueError, match=message):
                M.score_variable(good, obs, thresholds, base, **grid)
        for method in ("fair", "ecdf"):
            for thresholds in (wide, on_points, huge):
                with pytest.raises(ValueError, match=message):
                    M.tail_crps(good, obs, thresholds, method=method)

    # Controls. A long double table of ordinary values scores exactly like the float32 one, and NaN
    # in it still means "no training data".
    ordinary = s["threshold"].astype(np.longdouble)
    assert M.score_variable(good, obs, ordinary, base, **grid) == score(good, obs, s["threshold"], base)
    ordinary.values[1, :] = np.nan
    assert int(M.scoring_mask(obs, ordinary).sum()) == 3 * 12 * 2
    assert M.score_variable(good, obs, ordinary, base, **grid)["all"]["n_masked_no_threshold"] == 12 * 2


def test_lead_with_no_observations_is_undefined_not_zero() -> None:
    s = synthetic(27, n_init=30)
    obs = s["obs"].copy()
    obs.values[:, 0, :] = np.nan
    result = score(s["good"], obs, s["threshold"], s["base"], cfg=CFG)
    assert result["6"] == {
        "crps": None,
        "twcrps_p95": None,
        "loss": None,
        "crps_ecdf": None,
        "twcrps_p95_ecdf": None,
        "loss_ecdf": None,
        "spread_skill_ratio": None,
        "n_pairs": 0,
        "n_masked_no_threshold": 0,
        "n_members": N_MEM,
        "crpss_vs_baseline": None,
        "crpss_vs_baseline_ecdf": None,
        "baseline_twcrps_p95": None,
        "baseline_twcrps_p95_ecdf": None,
        "dm_pvalue": None,
    }
    assert result["all"]["n_pairs"] == result["24"]["n_pairs"] > 0
    assert result["all"]["crps"] == result["24"]["crps"]


def test_score_variable_refuses_bad_inputs() -> None:
    s = synthetic(28, n_init=10, n_sta=4, n_mem=4)
    good, obs, thr, base = s["good"], s["obs"], s["threshold"], s["base"]
    with pytest.raises(ValueError, match="baseline does not cover"):
        score(good, obs, thr, base.isel(lead_time=[0]), cfg=CFG)
    with pytest.raises(ValueError, match="does not cover every station"):
        score(good, obs, thr.isel(station_id=slice(1, None)), cfg=CFG)
    with pytest.raises(ValueError, match="must have dims"):
        score(good.isel(ens_member=0, drop=True), obs, thr, cfg=CFG)
    half_hour = good.assign_coords(lead_time=good["lead_time"].values + np.timedelta64(30, "m"))
    with pytest.raises(ValueError, match="whole number of hours"):
        score(half_hour, obs.assign_coords(lead_time=half_hour["lead_time"].values), thr, cfg=CFG)
    # a baseline with extra leads, stations and a different ensemble size is fine
    extra = synthetic(28, n_init=10, leads=(6, 24, 48), n_sta=5, n_mem=7)["base"]
    score(good, obs, thr, extra, cfg=CFG)


# --- evaluation grid ---------------------------------------------------------------------------


def test_forecast_cannot_drop_a_station_or_a_date() -> None:
    # 10 days at 00Z and 12Z; the grid comes from the protocol side, not from any forecast.
    grid_times = M.evaluation_init_times("2024-02-01T00", "2024-02-10T12", [0, 12])
    assert len(grid_times) == 20
    rng = np.random.default_rng(31)
    c = {
        "init_time": grid_times,
        "lead_time": np.array([6, 24], dtype="timedelta64[h]").astype("timedelta64[ns]"),
        "station_id": station_ids(5),
    }
    shape = (20, 2, 5)
    centre = 280 + 3 * rng.normal(size=shape)
    obs = as_obs(centre + rng.normal(size=shape), c)
    cand = as_fcst(centre[..., None] + rng.normal(size=(*shape, 6)), c)
    base = as_fcst(centre[..., None] + 2 * rng.normal(size=(*shape, 6)), c)
    thr = flat_threshold(5, 285.0)
    grid = {"expected_init_times": grid_times, "expected_station_ids": station_ids(5)}

    def run(fcst: xr.DataArray, truth: xr.DataArray, baseline: xr.DataArray | None = base) -> dict:
        return M.score_variable(fcst, truth, thr, baseline, cfg=CFG, **grid)

    full = run(cand, obs)
    assert full["all"]["n_pairs"] == 200

    no_station = cand.isel(station_id=[0, 1, 2, 4])
    no_date = cand.isel(init_time=[i for i in range(20) if i != 7])  # one 12Z run of an evaluated hour
    for dropped, dim in ((no_station, "station_id"), (no_date, "init_time")):
        with pytest.raises(ValueError, match=f"forecast does not cover the evaluation grid: 1 '{dim}'"):
            run(dropped, obs)
        # the reported hole: truth aligned to the shrunken candidate no longer hides the gap
        shrunk_obs = obs.sel({dim: dropped[dim]})
        with pytest.raises(ValueError, match=f"obs '{dim}' values are not exactly the evaluation grid"):
            run(dropped, shrunk_obs)
        with pytest.raises(ValueError, match=f"baseline does not cover the evaluation grid: 1 '{dim}'"):
            run(cand, obs, base.sel({dim: dropped[dim]}))

    # obs wider than the grid is also refused: the grid, not the data, decides what is scored
    wide = M.evaluation_init_times("2024-02-01T00", "2024-02-11T12", [0, 12])
    with pytest.raises(ValueError, match="obs 'init_time' values are not exactly"):
        run(cand, obs.reindex(init_time=wide))

    # documented exception: a forecast without a whole lead is scored on the leads it has
    one_lead = run(cand.isel(lead_time=[1]), obs)
    assert list(one_lead) == ["all", "24"]
    assert one_lead["24"] == full["24"]
    with pytest.raises(ValueError, match="obs does not cover every lead_time"):
        run(cand, obs.isel(lead_time=[0]))
    # a forecast with extra stations and dates is cut to the grid
    c_big = {**c, "init_time": wide, "station_id": station_ids(7)}
    big = as_fcst(np.full((22, 2, 7, 6), np.nan), c_big)
    big.loc[{"init_time": grid_times, "station_id": station_ids(5)}] = cand.values
    assert run(big, obs) == full

    duplicate = [*station_ids(5), station_ids(5)[0]]
    for bad in ({**grid, "expected_station_ids": []}, {**grid, "expected_station_ids": duplicate}):
        with pytest.raises(ValueError, match="evaluation grid"):
            M.score_variable(cand, obs, thr, base, cfg=CFG, **bad)
    nat_grid = np.array([*grid_times, "NaT"], dtype="datetime64[ns]")
    with pytest.raises(ValueError, match="evaluation grid"):
        M.score_variable(cand, obs, thr, base, cfg=CFG, **{**grid, "expected_init_times": nat_grid})


def test_evaluation_init_times() -> None:
    got = M.evaluation_init_times("2024-01-01T00", "2024-01-02T18", CFG.evaluation.init_hours_utc)
    expected = pd.date_range("2024-01-01", periods=8, freq="6h").values.astype("datetime64[ns]")
    np.testing.assert_array_equal(got, expected)
    assert got.dtype == np.dtype("datetime64[ns]")
    # start and end are inclusive and need not sit on an init hour
    got = M.evaluation_init_times("2024-01-01T01", "2024-01-03T11:59", [12, 0, 12])
    assert [str(t)[:13] for t in got] == ["2024-01-01T12", "2024-01-02T00", "2024-01-02T12", "2024-01-03T00"]
    validation = CFG.splits.validation
    hours = CFG.evaluation.init_hours_utc
    assert len(M.evaluation_init_times(validation.start, validation.end, hours)) == 352 * 4
    bad_args = (
        ("2024-01-02", "2024-01-01", [0]),
        ("2024-01-01", "2024-01-02", []),
        ("2024-01-01", "2024-01-02", [24]),
    )
    for args in bad_args:
        with pytest.raises(ValueError):
            M.evaluation_init_times(*args)


def test_placed_members_cannot_pass_the_gate() -> None:
    # Fair CRPS assumes members are random draws. Two members placed on purpose score exactly 0
    # whenever the observation lies between them, so "wide" members look perfect under fair CRPS.
    rng = np.random.default_rng(32)
    c = coords(200, (6, 24), 5)
    shape = (200, 2, 5)
    centre = 280 + 3 * rng.normal(size=shape)
    obs = as_obs(centre + rng.normal(size=shape), c)
    honest = as_fcst(centre[..., None] + rng.normal(size=(*shape, N_MEM)), c)  # calibrated, sampled
    thr = flat_threshold(5, 283.0)

    def placed(half_width: float) -> xr.DataArray:
        return as_fcst(np.stack([centre - half_width, centre + half_width], axis=-1), c)

    wide = placed(6.0)  # six error standard deviations either side: truth is always in between
    np.testing.assert_allclose(M.fair_crps(wide, obs).values, 0.0, atol=1e-4)
    result = score(wide, obs, thr, honest)
    entry = result["all"]
    assert entry["n_members"] == 2
    assert entry["crpss_vs_baseline"] > 0.99  # the hole: "perfect" skill under the fair estimator
    assert entry["crps_ecdf"] == pytest.approx(3.0, rel=1e-3)  # ecdf CRPS here is half the half-width
    assert entry["crpss_vs_baseline_ecdf"] < -3  # and far worse than the honest ensemble
    gates = M.evaluate_gates(result, CFG)
    assert gates["skill"] is False
    assert gates["min_members"] is False
    assert gates["spread_skill"] is False
    assert not all(gates.values())

    # Tuned to sit inside the spread-skill band: still beats the true CRPS under fair, still refused.
    tuned = score(placed(0.97 * 0.6745), obs, thr, honest)
    assert tuned["all"]["crps"] < 0.65 / math.sqrt(math.pi)  # true CRPS of the unit-variance error
    assert tuned["all"]["crpss_vs_baseline"] > 0.3
    assert tuned["all"]["crpss_vs_baseline_ecdf"] < 0
    tuned_gates = M.evaluate_gates(tuned, CFG)
    assert tuned_gates["spread_skill"] is True  # the calibration band alone does not catch it
    assert (tuned_gates["skill"], tuned_gates["min_members"]) == (False, False)

    # The member floor works on its own: an honest, better candidate with one member fewer than the
    # floor is not eligible; with exactly the floor it is.
    weak_centre = centre + rng.normal(size=shape)
    weak = as_fcst(weak_centre[..., None] + 2 * rng.normal(size=(*shape, N_MEM)), c)
    for n_mem, eligible in ((FLOOR - 1, False), (FLOOR, True)):
        gates = M.evaluate_gates(score(honest.isel(ens_member=slice(n_mem)), obs, thr, weak), CFG)
        assert gates["min_members"] is eligible
        assert (gates["skill"], gates["tail"], gates["dm"]) == (True, True, True)


# --- determinism -------------------------------------------------------------------------------


def test_result_independent_of_chunking_and_input_order() -> None:
    s = synthetic(29, n_init=30, n_sta=6, n_mem=8)
    obs = s["obs"].copy()
    obs.values[np.random.default_rng(5).random(obs.shape) < 0.15] = np.nan
    reference = score(s["good"], obs, s["threshold"], s["base"], cfg=CFG)
    assert score(s["good"], obs, s["threshold"], s["base"], cfg=CFG) == reference

    chunked = score(
        s["good"].chunk({"init_time": 7, "station_id": 4, "ens_member": 3}),
        obs.chunk({"init_time": 11}),
        s["threshold"],
        s["base"].chunk({"lead_time": 1, "ens_member": 5}),
        cfg=CFG,
    )
    assert chunked == reference

    rng = np.random.default_rng(6)
    order = {dim: rng.permutation(obs.sizes[dim]) for dim in POINT_DIMS}
    shuffled = score(
        s["good"].isel(order).isel(ens_member=rng.permutation(8)).transpose("ens_member", "station_id", ...),
        obs.isel(order).transpose("station_id", "lead_time", "init_time"),
        s["threshold"].isel(station_id=order["station_id"]).transpose("month", "station_id"),
        s["base"].isel(init_time=order["init_time"][::-1]),
        cfg=CFG,
    )
    assert shuffled == reference


# --- Diebold-Mariano ---------------------------------------------------------------------------


def series(values: np.ndarray, freq: str = "6h") -> xr.DataArray:
    times = pd.date_range("2024-01-01", periods=len(values), freq=freq).values.astype("datetime64[ns]")
    return xr.DataArray(np.asarray(values, dtype=np.float64), dims="init_time", coords={"init_time": times})


def ar1(rng: np.random.Generator, n: int, phi: float) -> np.ndarray:
    noise = rng.standard_normal(n + 100)
    out = np.zeros(n + 100)
    for i in range(1, n + 100):
        out[i] = phi * out[i - 1] + noise[i]
    return out[100:]


def test_dm_lag_grows_with_lead() -> None:
    assert [M.dm_lag(lead, 6) for lead in (1, 3, 6, 12, 24, 48)] == [1, 1, 1, 2, 4, 8]
    assert M.dm_lag(48, 24) == 2
    assert M.dm_lag(48, 12) == 4
    assert M.dm_lag(7, 6) == 2
    assert M.dm_lag(48, 6) > M.dm_lag(6, 6)
    with pytest.raises(ValueError):
        M.dm_lag(0, 6)


def test_dm_passes_the_lead_based_lag_to_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[int] = []
    real = M.diebold_mariano

    def spy(da: xr.DataArray, ts_dim: str, h_coord: str, **kwargs: object) -> xr.Dataset:
        seen.append(int(da[h_coord].values[0]))
        assert kwargs == {"method": "HG", "statistic_distribution": "t"}
        return real(da, ts_dim, h_coord, **kwargs)

    monkeypatch.setattr(M, "diebold_mariano", spy)
    values = ar1(np.random.default_rng(0), 200, 0.5)
    M.diebold_mariano_pvalue(series(values), 6)  # spacing inferred from the 6-hourly init times
    M.diebold_mariano_pvalue(series(values), 48)
    M.diebold_mariano_pvalue(series(values, freq="24h"), 48)
    M.diebold_mariano_pvalue(series(values, freq="12h"), 48, 12.0)
    assert seen == [1, 8, 2, 4]
    with pytest.raises(ValueError, match="init_spacing_hours"):
        M.diebold_mariano_pvalue(series(values), 48, 12.0)  # the init times are 6 h apart

    # score_variable: each lead gets its own lag, "all" gets the longest lead's
    seen.clear()
    s = synthetic(30, n_init=40, leads=(6, 24, 48))
    score(s["good"], s["obs"], s["threshold"], s["base"], cfg=CFG)
    assert seen == [8, 1, 4, 8]


def test_dm_rejection_rate_under_the_null_is_near_alpha() -> None:
    # Null: mean-zero AR(1) differences (phi = 0.5), 300 init times, 48 h lead at 6 h spacing.
    # 400 seeds. Nominal rate 0.05; binomial standard error 0.011; accepted band 0.02..0.08.
    alpha = CFG.gate.alpha_dm
    n_seeds = 400
    rejections = 0
    for seed in range(n_seeds):
        values = ar1(np.random.default_rng(seed), 300, 0.5)
        pvalue = M.diebold_mariano_pvalue(series(values), 48)
        assert pvalue is not None
        rejections += pvalue < alpha
    rate = rejections / n_seeds
    assert abs(rate - alpha) <= 0.03, rate


def test_dm_is_one_sided_and_detects_a_real_gain() -> None:
    noise = ar1(np.random.default_rng(1), 300, 0.5)
    better = M.diebold_mariano_pvalue(series(noise + 1.0), 24)
    worse = M.diebold_mariano_pvalue(series(-(noise + 1.0)), 24)
    assert better < 1e-6
    assert worse > 1 - 1e-6
    assert better + worse == pytest.approx(1.0, abs=1e-12)
    assert M.diebold_mariano_pvalue(series(np.full(50, 0.25)), 6) < 1e-12  # the same gain every time


def test_dm_nan_gaps_and_short_series() -> None:
    values = ar1(np.random.default_rng(2), 120, 0.4) + 0.2
    whole = M.diebold_mariano_pvalue(series(values), 24)
    assert whole is not None
    # a NaN inside the series, or a missing init time, would change the physical lag: undefined
    gappy = values.copy()
    gappy[40] = np.nan
    assert M.diebold_mariano_pvalue(series(gappy), 24) is None
    assert M.diebold_mariano_pvalue(series(values).isel(init_time=[*range(40), *range(41, 120)]), 24) is None
    uneven = series(values).isel(init_time=[*range(0, 60, 2), *range(60, 120)])
    assert M.diebold_mariano_pvalue(uneven, 24) is None
    # leading and trailing NaNs are only trimmed: the rest is still evenly spaced
    edges = values.copy()
    edges[[0, 1, 119]] = np.nan
    assert M.diebold_mariano_pvalue(series(edges), 24) == M.diebold_mariano_pvalue(series(values)[2:119], 24)
    # the same values one cycle apart or a week apart: same statistic only because both are regular
    # series; the lag follows the real spacing (24 h lead: h = 4 at 6 h, h = 1 at 168 h)
    assert M.dm_lag(24, 6) == 4 and M.dm_lag(24, 168) == 1
    # HG models the autocovariance over max((n - 1) // 2, h) lags: on a long series the lag does not
    # change the answer, on a short one it does.
    assert M.diebold_mariano_pvalue(series(values, freq="12h"), 24) == whole
    short = values[:12]
    six_hourly, two_daily = series(short), series(short, freq="48h")
    assert M.diebold_mariano_pvalue(six_hourly, 48) != M.diebold_mariano_pvalue(two_daily, 48)
    # order of the init_time coordinate does not matter
    assert M.diebold_mariano_pvalue(series(values).isel(init_time=slice(None, None, -1)), 24) == (
        M.diebold_mariano_pvalue(series(values), 24)
    )

    assert M.diebold_mariano_pvalue(series(np.zeros(50)), 6) is None  # all zero
    assert M.diebold_mariano_pvalue(series(values[:8]), 48) is None  # 8 values, lag 8
    assert M.diebold_mariano_pvalue(series(values[:9]), 48) is not None
    assert M.diebold_mariano_pvalue(series(values[:1]), 6) is None  # one init time: no spacing
    assert M.diebold_mariano_pvalue(series(np.full(20, np.nan)), 6) is None
    with pytest.raises(ValueError, match="infinite"):
        M.diebold_mariano_pvalue(series(np.array([1.0, np.inf, 2.0, 3.0])), 6)
    with pytest.raises(ValueError, match="dims"):
        M.diebold_mariano_pvalue(xr.DataArray(values, dims="time"), 6, 6.0)
