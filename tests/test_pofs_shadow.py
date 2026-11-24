"""Unit tests for the off-chain PoFS shadow payout ledger (src/wrfcomp/pofs_shadow.py)."""

from __future__ import annotations

import dataclasses
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest
from pofs_fixtures import recorded_epoch

from wrfcomp.pofs_shadow import (
    MinerEntry,
    ShadowLedgerParams,
    StationEntry,
    composite_forecast_loss,
    fair_ensemble_crps,
    load_scorecard,
    meta_ensemble_loss,
    miners_from_scorecards,
    settle_compute_pool,
    settle_data_pool,
    settle_epoch,
    settle_from_scorecards,
    weighted_mixture_crps,
)

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1] / "experiments"


def test_params_frozen_defaults_and_required_alpha_slash() -> None:
    with pytest.raises(TypeError):
        ShadowLedgerParams()  # type: ignore[call-arg]

    params = ShadowLedgerParams(alpha_slash=0.5)
    assert params.p_comp_share == 0.55
    assert params.p_data_share == 0.35
    assert params.p_val_share == 0.10
    assert params.w_delta == 0.90
    assert params.twcrps_weight == 0.30
    assert params.max_slash_cap == 0.10
    assert params.stake_exponent == 1.0
    assert params.k_min == 10.0
    assert params.h3_cell_cap == 1
    assert params.alpha_slash == 0.5

    with pytest.raises(dataclasses.FrozenInstanceError):
        params.w_delta = 0.5  # type: ignore[misc]


def test_fair_and_weighted_crps_match_brute_force() -> None:
    rng = np.random.default_rng(7)
    obs = rng.normal(0.0, 2.0, size=25)
    fc = obs[:, np.newaxis] + rng.normal(0.0, 1.0, size=(25, 6))

    fast = fair_ensemble_crps(fc, obs)
    brute = np.empty(25)
    for i in range(25):
        x = fc[i]
        m = len(x)
        t1 = np.mean(np.abs(x - obs[i]))
        t2 = np.sum(np.abs(x[:, None] - x[None, :])) / (2.0 * m * (m - 1))
        brute[i] = t1 - t2
    np.testing.assert_allclose(fast, brute, rtol=1e-12, atol=1e-12)

    fc2 = obs[:, np.newaxis] + rng.normal(0.5, 1.5, size=(25, 4))
    weights = np.array([3.0, 1.0])
    mix_fast = weighted_mixture_crps([fc, fc2], weights, obs)
    mix_brute = np.empty(25)
    w_all = np.concatenate([np.full(6, 0.75 / 6), np.full(4, 0.25 / 4)])
    for i in range(25):
        x = np.concatenate([fc[i], fc2[i]])
        t1 = np.sum(w_all * np.abs(x - obs[i]))
        t2 = 0.5 * np.sum(np.outer(w_all, w_all) * np.abs(x[:, None] - x[None, :]))
        mix_brute[i] = t1 - t2
    np.testing.assert_allclose(mix_fast, mix_brute, rtol=1e-12, atol=1e-12)


def test_unverified_scalar_skill_cannot_earn_at_any_delta_weight() -> None:
    """Without commitment and GEFS evidence neither delta weight authorizes payment."""
    rng = np.random.default_rng(42)
    n_obs = 800
    obs = rng.normal(0.0, 1.0, size=n_obs)

    # Construct a free public baseline forecast and a strictly better innovator forecast
    # with correlated error structure (innovator refines the public baseline).
    shared_err = rng.normal(0.0, 1.0, size=(n_obs, 16))
    public_fc = obs[:, np.newaxis] + 0.90 * shared_err
    innov_fc = obs[:, np.newaxis] + 0.70 * shared_err

    # Locked reference baseline (e.g. climatology/older baseline) with loss L_base so that:
    # CRPSS(public) = 0.10 and CRPSS(innovator) = 0.30 (exact 1:3 ratio).
    params_strict = ShadowLedgerParams(alpha_slash=0.5, w_delta=0.90)
    params_zero = ShadowLedgerParams(alpha_slash=0.5, w_delta=0.0)
    l_pub = composite_forecast_loss(public_fc, obs, params_strict)
    l_innov = composite_forecast_loss(innov_fc, obs, params_strict)
    # Solve (1 - l_pub/L_base) / (1 - l_innov/L_base) = 1/3 => L_base = (3*l_pub - l_innov) / 2
    l_base = (3.0 * l_pub - l_innov) / 2.0

    miners = [
        MinerEntry("innovator", stake=100.0, forecast=innov_fc),
        MinerEntry("public_copy", stake=100.0, forecast=public_fc),
    ]

    res_zero = settle_compute_pool(
        miners, pool_amount=1000.0, params=params_zero, baseline_loss=l_base, obs=obs
    )
    by_id_zero = {p.miner_id: p for p in res_zero.payouts}
    assert by_id_zero["public_copy"].reward == by_id_zero["innovator"].reward == 0.0

    res_strict = settle_compute_pool(
        miners, pool_amount=1000.0, params=params_strict, baseline_loss=l_base, obs=obs
    )
    by_id_strict = {p.miner_id: p for p in res_strict.payouts}
    assert by_id_strict["public_copy"].reward == pytest.approx(0.0, abs=1e-9)
    assert by_id_strict["innovator"].reward == 0.0

    # Exact copy of the locked baseline itself (L_copy == L_base => CRPSS == 0) earns 0 at w_delta=0.90.
    exact_miners = [
        MinerEntry("innovator", stake=100.0, forecast=innov_fc),
        MinerEntry("exact_baseline_copy", stake=100.0, forecast=public_fc),
    ]
    res_exact = settle_compute_pool(
        exact_miners, pool_amount=1000.0, params=params_strict, baseline_forecast=public_fc, obs=obs
    )
    by_id_exact = {p.miner_id: p for p in res_exact.payouts}
    assert by_id_exact["exact_baseline_copy"].crpss == pytest.approx(0.0, abs=1e-12)
    assert by_id_exact["exact_baseline_copy"].reward == 0.0


def test_ten_identical_clones_earn_no_more_than_one_alone() -> None:
    """Ten identical clones (whether splitting stake or duplicating stake) earn no more
    in total than a single copy of that model earns alone."""
    rng = np.random.default_rng(123)
    n_obs = 600
    obs = rng.normal(0.0, 2.0, size=n_obs)

    # Two independent skilled models beating baseline; clone model B ten times.
    baseline_fc = obs[:, np.newaxis] + rng.normal(0, 2.0, (n_obs, 1)) + rng.normal(0.0, 2.0, size=(n_obs, 31))
    model_a = obs[:, np.newaxis] + rng.normal(0, 1.1, (n_obs, 1)) + rng.normal(0.0, 1.1, size=(n_obs, 31))
    model_b = obs[:, np.newaxis] + rng.normal(0, 1.1, (n_obs, 1)) + rng.normal(0.0, 1.1, size=(n_obs, 31))

    params = ShadowLedgerParams(alpha_slash=0.5, w_delta=0.90)

    # Single instance of model B alongside model A.
    solo = recorded_epoch(
        [
            MinerEntry("miner_a", stake=100.0, forecast=model_a),
            MinerEntry("clone_0", stake=100.0, forecast=model_b),
        ],
        pool_amount=1000.0,
        params=params,
        baseline_forecast=baseline_fc,
        obs=obs,
    )
    solo_b_reward = {p.miner_id: p.reward for p in solo.payouts}["clone_0"]
    assert solo_b_reward > 0.0

    # Case 1: 10 clones splitting the same 100 stake (10 each) — Sybil stake splitting.
    split_miners = [MinerEntry("miner_a", stake=100.0, forecast=model_a)] + [
        MinerEntry(f"clone_{i}", stake=10.0, forecast=model_b) for i in range(10)
    ]
    res_split = recorded_epoch(
        split_miners, pool_amount=1000.0, params=params, baseline_forecast=baseline_fc, obs=obs
    )
    split_clones_total = sum(p.reward for p in res_split.payouts if p.miner_id.startswith("clone_"))
    assert split_clones_total <= solo_b_reward

    # Case 2: 10 full-stake clones under pure marginal contribution (w_delta=1.0).
    params_pure = ShadowLedgerParams(alpha_slash=0.5, w_delta=1.0)
    solo_pure = recorded_epoch(
        [
            MinerEntry("miner_a", stake=100.0, forecast=model_a),
            MinerEntry("clone_0", stake=100.0, forecast=model_b),
        ],
        pool_amount=1000.0,
        params=params_pure,
        baseline_forecast=baseline_fc,
        obs=obs,
    )
    solo_pure_b = {p.miner_id: p.reward for p in solo_pure.payouts}["clone_0"]
    dup_miners = [MinerEntry("miner_a", stake=100.0, forecast=model_a)] + [
        MinerEntry(f"clone_{i}", stake=100.0, forecast=model_b) for i in range(10)
    ]
    res_dup = recorded_epoch(
        dup_miners, pool_amount=1000.0, params=params_pure, baseline_forecast=baseline_fc, obs=obs
    )
    dup_clones_total = sum(p.reward for p in res_dup.payouts if p.miner_id.startswith("clone_"))
    assert dup_clones_total <= solo_pure_b


def test_worse_than_baseline_miner_is_slashed_and_cap_holds() -> None:
    rng = np.random.default_rng(9)
    n_obs = 200
    obs = rng.normal(0.0, 1.0, size=n_obs)
    baseline_fc = obs[:, np.newaxis] + rng.normal(0, 2.0, (n_obs, 1)) + rng.normal(0.0, 1.0, size=(n_obs, 31))
    good_fc = obs[:, np.newaxis] + rng.normal(0, 0.6, (n_obs, 1)) + rng.normal(0.0, 0.6, size=(n_obs, 31))
    mild_bad_fc = baseline_fc + 1.0
    terrible_fc = obs[:, np.newaxis] + rng.normal(10.0, 5.0, size=(n_obs, 31))

    params = ShadowLedgerParams(alpha_slash=0.25, max_slash_cap=0.10)

    # Mildly worse-than-baseline miner: slashed proportionally below the 10% cap.
    res_mild = recorded_epoch(
        [
            MinerEntry("good", stake=100.0, forecast=good_fc),
            MinerEntry("mild_bad", stake=100.0, forecast=mild_bad_fc),
        ],
        pool_amount=500.0,
        params=params,
        baseline_forecast=baseline_fc,
        obs=obs,
    )
    by_mild = {p.miner_id: p for p in res_mild.payouts}
    assert by_mild["good"].slash == 0.0
    assert by_mild["good"].reward == pytest.approx(500.0)
    assert by_mild["mild_bad"].reward == 0.0
    assert by_mild["mild_bad"].composite_score is not None and by_mild["mild_bad"].composite_score < 0.0
    expected_mild = min(0.10, 0.25 * abs(by_mild["mild_bad"].composite_score)) * 100.0
    assert by_mild["mild_bad"].slash == pytest.approx(expected_mild)
    assert 0.0 < by_mild["mild_bad"].slash < 10.0

    # Severely worse-than-baseline miner: hits the 10% max_slash_cap cleanly.
    res_cap = recorded_epoch(
        [
            MinerEntry("good", stake=100.0, forecast=good_fc),
            MinerEntry("terrible", stake=200.0, forecast=terrible_fc),
        ],
        pool_amount=500.0,
        params=params,
        baseline_forecast=baseline_fc,
        obs=obs,
    )
    by_cap = {p.miner_id: p for p in res_cap.payouts}
    assert by_cap["good"].reward == pytest.approx(500.0)
    assert by_cap["terrible"].reward == 0.0
    assert by_cap["terrible"].slash == pytest.approx(0.10 * 200.0)


def test_protocol_fault_gets_flat_slash_and_is_excluded_from_meta_ensemble() -> None:
    rng = np.random.default_rng(17)
    n_obs = 150
    obs = rng.normal(0.0, 1.0, size=n_obs)
    baseline_fc = obs[:, np.newaxis] + rng.normal(0, 2.0, (n_obs, 1)) + rng.normal(0.0, 1.5, size=(n_obs, 31))
    m1_fc = obs[:, np.newaxis] + rng.normal(0, 0.8, (n_obs, 1)) + rng.normal(0.0, 0.8, size=(n_obs, 31))
    m2_fc = obs[:, np.newaxis] + rng.normal(0, 0.9, (n_obs, 1)) + rng.normal(0.0, 0.9, size=(n_obs, 31))
    corrupt_fc = np.full((n_obs, 8), np.nan)  # invalid payload

    params = ShadowLedgerParams(alpha_slash=1.0, max_slash_cap=0.10)

    clean = recorded_epoch(
        [
            MinerEntry("m1", stake=100.0, forecast=m1_fc),
            MinerEntry("m2", stake=100.0, forecast=m2_fc),
        ],
        pool_amount=550.0,
        params=params,
        baseline_forecast=baseline_fc,
        obs=obs,
    )
    with_fault = recorded_epoch(
        [
            MinerEntry("m1", stake=100.0, forecast=m1_fc),
            MinerEntry("m2", stake=100.0, forecast=m2_fc),
            MinerEntry("faulty", stake=200.0, forecast=corrupt_fc, protocol_fault=True),
        ],
        pool_amount=550.0,
        params=params,
        baseline_forecast=baseline_fc,
        obs=obs,
    )

    assert with_fault.meta_ensemble_loss == clean.meta_ensemble_loss
    by_clean = {p.miner_id: p for p in clean.payouts}
    by_fault = {p.miner_id: p for p in with_fault.payouts}
    assert by_fault["m1"] == by_clean["m1"]
    assert by_fault["m2"] == by_clean["m2"]
    assert by_fault["faulty"].reward == 0.0
    assert by_fault["faulty"].slash == pytest.approx(params.max_slash_cap * 200.0)
    assert by_fault["faulty"].composite_score is None
    assert by_fault["faulty"].delta_crps is None


def test_eight_stations_in_one_cell_split_one_station_capacity() -> None:
    params = ShadowLedgerParams(alpha_slash=0.5, h3_cell_cap=1)
    crowded = [
        StationEntry(station_id=f"crowded_{i}", cell_id="872a1072bffffff", qod=1.0, pol=1.0) for i in range(8)
    ]
    solo = StationEntry(station_id="solo_0", cell_id="872a1072cffffff", qod=1.0, pol=1.0)

    res = settle_data_pool(crowded + [solo], pool_amount=350.0, params=params)
    by_id = {p.station_id: p for p in res.payouts}

    assert by_id["solo_0"].cell_cap_factor == 1.0
    assert by_id["solo_0"].reward == pytest.approx(175.0)
    crowded_total = sum(by_id[f"crowded_{i}"].reward for i in range(8))
    assert crowded_total == pytest.approx(by_id["solo_0"].reward)
    for i in range(8):
        assert by_id[f"crowded_{i}"].cell_cap_factor == pytest.approx(1.0 / 8.0)
        assert by_id[f"crowded_{i}"].reward == pytest.approx(175.0 / 8.0)


def test_pool_conservation_unpaid_remainder_and_determinism() -> None:
    params = ShadowLedgerParams(alpha_slash=0.5)
    # Pointwise-loss mode with positive scores + stations.
    miners = [
        MinerEntry("b_miner", stake=50.0, pointwise_loss=np.array([0.8, 0.7, 0.9])),
        MinerEntry("a_miner", stake=150.0, pointwise_loss=np.array([0.5, 0.6, 0.4])),
        MinerEntry("c_fault", stake=80.0, protocol_fault=True),
    ]
    stations = [
        StationEntry("st_2", cell_id="cell_a", qod=0.9, pol=0.95, hw=1.25, impact=1.2),
        StationEntry("st_1", cell_id="cell_b", qod=0.8, pol=1.0, hw=1.0, impact=1.0),
    ]

    e1 = settle_epoch(
        epoch_emission=1000.0, miners=miners, stations=stations, params=params, baseline_loss=1.0
    )
    e2 = settle_epoch(
        epoch_emission=1000.0,
        miners=list(reversed(miners)),
        stations=list(reversed(stations)),
        params=params,
        baseline_loss=1.0,
    )
    # Bitwise determinism regardless of input order.
    assert e1 == e2

    # Exact conservation across all three pools.
    assert e1.compute.total_paid + e1.compute.unpaid_remainder == pytest.approx(550.0, rel=1e-12)
    assert e1.data.total_paid + e1.data.unpaid_remainder == pytest.approx(350.0, rel=1e-12)
    assert e1.validator_pool == pytest.approx(100.0, rel=1e-12)
    assert (
        e1.compute.total_paid
        + e1.compute.unpaid_remainder
        + e1.data.total_paid
        + e1.data.unpaid_remainder
        + e1.validator_pool
    ) == pytest.approx(1000.0, rel=1e-12)

    # When nobody has positive score and all stations have zero QoD, unpaid remainders equal pools.
    no_winners = settle_epoch(
        epoch_emission=1000.0,
        miners=[MinerEntry("bad", stake=100.0, pointwise_loss=np.array([1.5, 1.6]))],
        stations=[StationEntry("spoof", cell_id="cell_x", qod=0.0, pol=1.0)],
        params=params,
        baseline_loss=1.0,
    )
    assert no_winners.compute.total_paid == 0.0
    assert no_winners.compute.unpaid_remainder == 550.0
    assert no_winners.data.total_paid == 0.0
    assert no_winners.data.unpaid_remainder == 350.0


def test_tail_threshold_weighting_and_single_miner_behavior() -> None:
    rng = np.random.default_rng(31)
    obs = rng.normal(0.0, 2.0, size=100)
    tau = np.quantile(obs, 0.95)
    base_fc = obs[:, np.newaxis] + rng.normal(0, 2.0, (100, 1)) + rng.normal(0.0, 1.5, size=(100, 31))
    good_fc = obs[:, np.newaxis] + rng.normal(0, 0.8, (100, 1)) + rng.normal(0.0, 0.8, size=(100, 31))
    params = ShadowLedgerParams(alpha_slash=0.5, twcrps_weight=0.30)

    res = recorded_epoch(
        [MinerEntry("solo", stake=100.0, forecast=good_fc)],
        pool_amount=550.0,
        params=params,
        baseline_forecast=base_fc,
        obs=obs,
        tail_threshold=tau,
    )
    assert res.payouts[0].reward == pytest.approx(550.0)
    assert res.payouts[0].crpss is not None and res.payouts[0].crpss > 0.0
    assert res.payouts[0].delta_crps is not None and res.payouts[0].delta_crps > 0.0


@pytest.mark.parametrize(
    "bad_threshold",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        np.nan,
        np.inf,
        -np.inf,
        10**400,
        -(10**400),
        Decimal("1e400"),
        np.longdouble("1e400"),
        np.array([1.0, np.nan, 2.0, 3.0]),
        np.array([1.0, np.inf, 2.0, 3.0]),
        np.array([1.0, -np.inf, 2.0, 3.0]),
        np.array([1.0, 10**400, 2.0, 3.0], dtype=object),
    ],
)
def test_non_finite_and_overflowing_tail_thresholds_are_refused(bad_threshold: object) -> None:
    obs = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    fc = np.column_stack([obs - 0.5, obs + 0.5])
    params = ShadowLedgerParams(alpha_slash=0.5)

    with pytest.raises(ValueError, match="tail_threshold"):
        composite_forecast_loss(fc, obs, params, tail_threshold=bad_threshold)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="tail_threshold"):
        meta_ensemble_loss([fc], np.array([100.0]), obs, params, tail_threshold=bad_threshold)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="tail_threshold"):
        settle_compute_pool(
            [MinerEntry("m1", stake=100.0, forecast=fc)],
            pool_amount=550.0,
            params=params,
            baseline_loss=1.0,
            obs=obs,
            tail_threshold=bad_threshold,  # type: ignore[arg-type]
        )


def test_gate1_scorecards_worked_example_january_and_july() -> None:
    params = ShadowLedgerParams(alpha_slash=0.5)

    jan_cards = {
        "climatology": load_scorecard(EXPERIMENTS_DIR / "climatology_2024-01.json"),
        "persistence": load_scorecard(EXPERIMENTS_DIR / "persistence_2024-01.json"),
        "noaa-gefs": load_scorecard(EXPERIMENTS_DIR / "gefs_2024-01.json"),
    }
    jan_miners, jan_base, jan_vars, jan_leads = miners_from_scorecards(
        jan_cards, baseline_key="climatology", stakes=100.0
    )
    assert jan_vars == ("msl", "t2m", "td2m", "u10", "v10")
    assert jan_leads == (3, 6, 12, 24, 48)
    assert jan_base == 1.0
    assert len(jan_miners) == 3

    for month in ("01", "07"):
        cards = {
            name: load_scorecard(EXPERIMENTS_DIR / f"{file}_2024-{month}.json")
            for name, file in (
                ("climatology", "climatology"),
                ("persistence", "persistence"),
                ("noaa-gefs", "gefs"),
            )
        }
        assert cards["noaa-gefs"].passed_gate is False
        for variables in (None, ["u10"]):
            res = settle_from_scorecards(
                cards,
                baseline_key="noaa-gefs",
                pool_amount=550.0,
                params=params,
                stakes=100.0,
                variables=variables,
            )
            for payout in res.payouts:
                assert payout.reward == 0.0
                assert payout.slash == 0.0
            gefs = next(p for p in res.payouts if p.miner_id == "noaa-gefs")
            assert gefs.crpss == 0.0
            if variables is None:
                persistence = next(p for p in res.payouts if p.miner_id == "persistence")
                assert persistence.crpss == pytest.approx({"01": -0.81864, "07": -0.21314}[month], abs=1e-5)
            assert res.unpaid_remainder == 550.0
            assert res.total_paid + res.unpaid_remainder == pytest.approx(550.0)


def test_miners_from_scorecards_refuses_mismatched_periods() -> None:
    jan_clim = load_scorecard(EXPERIMENTS_DIR / "climatology_2024-01.json")
    jul_gefs = load_scorecard(EXPERIMENTS_DIR / "gefs_2024-07.json")
    with pytest.raises(ValueError, match="does not share truth, thresholds, period"):
        miners_from_scorecards({"climatology": jan_clim, "noaa-gefs": jul_gefs}, baseline_key="climatology")


def test_commit_cutoff_digest_minimum_and_unrevealed_receipt():
    from wrfcomp.pofs_shadow import Commitment, EpochEvidence, forecast_digest

    rng = np.random.default_rng(34)
    n = 320
    t = np.datetime64("2024-01-01", "ns")
    times = t + np.arange(n) * np.timedelta64(6, "h")
    obs = rng.normal(size=n)
    fc = obs[:, None] + rng.normal(size=(n, 1)) + rng.normal(size=(n, 31))
    gefs = obs[:, None] + rng.normal(0, 2, (n, 1)) + rng.normal(0, 2, (n, 31))
    params = ShadowLedgerParams(alpha_slash=0.5)
    earlier = Commitment(forecast_digest(fc), t - np.timedelta64(1, "ns"))

    def settle(entry, receipt):
        e = EpochEvidence(t, forecast_digest(gefs), {"w": receipt} if receipt else {})
        return settle_compute_pool(
            [entry],
            pool_amount=550,
            params=params,
            obs=obs,
            baseline_forecast=gefs,
            evidence=e,
            issue_times=times,
            lead_hours=6,
            tail_threshold=1.65,
        ).payouts[0]

    at_floor = MinerEntry("w", params.k_min, forecast=fc)
    assert settle(at_floor, earlier).reward == 550
    assert settle(dataclasses.replace(at_floor, stake=np.nextafter(params.k_min, 0)), earlier).reward == 0
    assert settle(at_floor, None).status == "uncommitted"
    for delta in (0, 1):
        result = settle(at_floor, Commitment(earlier.digest, t + np.timedelta64(delta, "ns")))
        assert result.status == "late" and result.reward == 0
    for payload in (None, fc + 1, np.full_like(fc, np.nan)):
        result = settle(dataclasses.replace(at_floor, forecast=payload), earlier)
        assert result.reward == 0 and result.slash == params.k_min * params.max_slash_cap


def test_scorecard_gate_checks_gefs_and_every_variable_and_missing_wallet():
    from wrfcomp.pofs_shadow import Commitment, EpochEvidence

    # Synthetic evaluator evidence based on the shape/coverage of a real reference.
    # No synthetic metrics are used in the committed-reference replay.
    ref = load_scorecard(EXPERIMENTS_DIR / "gefs_2024-01.json")
    candidate = ref.model_dump()
    candidate["model_id"] = "fixture"
    candidate["forecast_sha256"] = "a" * 64
    candidate["baseline_id"] = "noaa-gefs"
    candidate["baseline_sha256"] = ref.forecast_sha256
    candidate["passed_gate"] = True
    for var, leads in candidate["metrics"].items():
        candidate["gates"][var] = dict.fromkeys(candidate["gates"][var], True)
        row = leads["all"]
        for name, val in dict(
            crpss_vs_baseline=0.25,
            crpss_vs_baseline_ecdf=0.24,
            spread_skill_ratio=1.0,
            dm_pvalue=0.01,
            twcrps_p95=0.1,
            baseline_twcrps_p95=0.2,
        ).items():
            row[name] = val
        for h in candidate["leads_scored"]:
            leads[str(h)]["loss"] *= 0.75
    t = np.datetime64(ref.period_start)
    receipt = Commitment(candidate["forecast_sha256"], t - np.timedelta64(1, "m"))
    evidence = EpochEvidence(t, ref.forecast_sha256, {"miner": receipt, "withheld": receipt})

    def settle():
        return {
            p.miner_id: p
            for p in settle_from_scorecards(
                {"gefs": ref, "miner": type(ref).model_validate(candidate)},
                baseline_key="gefs",
                pool_amount=550,
                params=ShadowLedgerParams(alpha_slash=0.5),
                stakes=100,
                evidence=evidence,
            ).payouts
        }

    assert settle()["miner"].reward == 550
    assert settle()["withheld"].slash == 10
    for stat, value in (
        ("dm_pvalue", 0.05),
        ("spread_skill_ratio", 1.16),
        ("spread_skill_ratio", 0.84),
        ("crpss_vs_baseline", 0.05),
    ):
        row = candidate["metrics"]["v10"]["all"]
        old = row[stat]
        row[stat] = value
        assert settle()["miner"].reward == 0
        row[stat] = old
    # Invalid evidence carries the exact honest digest and an earlier receipt.
    # It must neither claim a representative slot nor escape the missing-reveal penalty.
    for changes in ({"sealed_data_used": True}, {"truth_sha256": "0" * 64}, {"baseline_sha256": "0" * 64}):
        wrong = type(ref).model_validate({**candidate, **changes})
        proof = EpochEvidence(
            t,
            ref.forecast_sha256,
            {
                "miner": receipt,
                "early": Commitment(receipt.digest, receipt.received_at - np.timedelta64(1, "s")),
            },
        )
        rows = settle_from_scorecards(
            {"gefs": ref, "early": wrong, "miner": type(ref).model_validate(candidate)},
            baseline_key="gefs",
            pool_amount=550,
            params=ShadowLedgerParams(alpha_slash=0.5),
            evidence=proof,
        )
        by_id = {p.miner_id: p for p in rows.payouts}
        assert by_id["early"].status == "non_reveal_or_mismatch" and by_id["early"].slash == 10
        assert by_id["miner"].reward == 550

    candidate["baseline_sha256"] = "0" * 64
    assert settle()["miner"].reward == 0


def test_excluded_forecasts_do_not_change_the_valid_marginal():
    rng = np.random.default_rng(812)
    obs = rng.normal(size=500)
    gefs = obs[:, None] + rng.normal(0, 2, (500, 1)) + rng.normal(0, 2, (500, 31))
    first = MinerEntry(
        "innovator", 100, forecast=obs[:, None] + rng.normal(size=(500, 1)) + rng.normal(size=(500, 31))
    )
    kwargs = dict(
        pool_amount=550, params=ShadowLedgerParams(alpha_slash=0.5), baseline_forecast=gefs, obs=obs
    )
    solo = recorded_epoch([first], **kwargs)
    with_losers = recorded_epoch(
        [first, MinerEntry("relay", 1000, forecast=gefs), MinerEntry("bad", 1000, forecast=gefs + 5)],
        **kwargs,
    )
    assert solo.payouts[0].reward > 0
    assert with_losers.meta_ensemble_loss == solo.meta_ensemble_loss
    assert next(p for p in with_losers.payouts if p.miner_id == "innovator") == solo.payouts[0]


def test_explicit_fault_without_evidence_or_payload_gets_maximum_slash():
    params = ShadowLedgerParams(alpha_slash=0.5)
    payout = settle_compute_pool(
        [MinerEntry("fault", 100, protocol_fault=True)], pool_amount=550, params=params, baseline_loss=1
    ).payouts[0]
    assert payout.protocol_fault and payout.status == "protocol_fault"
    assert payout.reward == 0 and payout.slash == params.max_slash_cap * payout.stake


def test_raw_commitment_canonicalizes_signed_zero_without_mutating_reveal():
    from wrfcomp.pofs_shadow import forecast_digest

    left = np.array([[-0.0, 1.0], [2.0, 0.0]])
    right = left + 0.0
    assert forecast_digest(left) == forecast_digest(right)
    assert np.signbit(left[0, 0])


def test_rowwise_wasserstein_uses_member_probability_mass():
    from wrfcomp.pofs_shadow import _mean_wasserstein

    left = np.array([[0.0, 3.0], [4.0, 4.0]])
    right = np.array([[0.0, 0.0, 3.0], [4.0, 4.0, 4.0]])
    # First row moves 1/6 of its mass by 3; second row is identical.
    assert _mean_wasserstein(left, right) == pytest.approx(0.25)
    assert _mean_wasserstein(right, left) == pytest.approx(0.25)
