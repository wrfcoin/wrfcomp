"""Unit tests for the off-chain PoFS shadow payout ledger (src/wrfcomp/pofs_shadow.py)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from wrfcomp.pofs_shadow import (
    MinerEntry,
    ShadowLedgerParams,
    StationEntry,
    composite_forecast_loss,
    fair_ensemble_crps,
    settle_compute_pool,
    settle_data_pool,
    settle_epoch,
    weighted_mixture_crps,
)


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
    assert params.stake_exponent == 0.75
    assert params.protocol_fault_slash == 0.05
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


def test_baseline_copy_earns_near_zero_at_090_and_third_of_innovator_at_zero() -> None:
    """A public/baseline relay with 1/3 the innovator's CRPSS earns 1/3 at w_delta=0
    and ~0 at w_delta=0.90; an exact copy of the locked baseline earns 0 at both."""
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
    assert by_id_zero["public_copy"].reward == pytest.approx(
        by_id_zero["innovator"].reward / 3.0, rel=1e-9
    )

    res_strict = settle_compute_pool(
        miners, pool_amount=1000.0, params=params_strict, baseline_loss=l_base, obs=obs
    )
    by_id_strict = {p.miner_id: p for p in res_strict.payouts}
    assert by_id_strict["public_copy"].reward == pytest.approx(0.0, abs=1e-9)
    assert by_id_strict["innovator"].reward == pytest.approx(1000.0, rel=1e-9)

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
    baseline_fc = obs[:, np.newaxis] + rng.normal(0.0, 2.0, size=(n_obs, 12))
    model_a = obs[:, np.newaxis] + rng.normal(0.0, 1.1, size=(n_obs, 12))
    model_b = obs[:, np.newaxis] + rng.normal(0.0, 1.1, size=(n_obs, 12))

    params = ShadowLedgerParams(alpha_slash=0.5, w_delta=0.90)

    # Single instance of model B alongside model A.
    solo = settle_compute_pool(
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
    res_split = settle_compute_pool(
        split_miners, pool_amount=1000.0, params=params, baseline_forecast=baseline_fc, obs=obs
    )
    split_clones_total = sum(p.reward for p in res_split.payouts if p.miner_id.startswith("clone_"))
    assert split_clones_total <= solo_b_reward

    # Case 2: 10 full-stake clones under pure marginal contribution (w_delta=1.0).
    params_pure = ShadowLedgerParams(alpha_slash=0.5, w_delta=1.0)
    solo_pure = settle_compute_pool(
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
    res_dup = settle_compute_pool(
        dup_miners, pool_amount=1000.0, params=params_pure, baseline_forecast=baseline_fc, obs=obs
    )
    dup_clones_total = sum(p.reward for p in res_dup.payouts if p.miner_id.startswith("clone_"))
    assert dup_clones_total <= solo_pure_b


def test_worse_than_baseline_miner_is_slashed_and_cap_holds() -> None:
    rng = np.random.default_rng(9)
    n_obs = 200
    obs = rng.normal(0.0, 1.0, size=n_obs)
    baseline_fc = obs[:, np.newaxis] + rng.normal(0.0, 1.0, size=(n_obs, 8))
    good_fc = obs[:, np.newaxis] + rng.normal(0.0, 0.6, size=(n_obs, 8))
    mild_bad_fc = obs[:, np.newaxis] + rng.normal(0.0, 1.25, size=(n_obs, 8))
    terrible_fc = obs[:, np.newaxis] + rng.normal(10.0, 5.0, size=(n_obs, 8))

    params = ShadowLedgerParams(alpha_slash=0.25, max_slash_cap=0.10)

    # Mildly worse-than-baseline miner: slashed proportionally below the 10% cap.
    res_mild = settle_compute_pool(
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
    res_cap = settle_compute_pool(
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
    baseline_fc = obs[:, np.newaxis] + rng.normal(0.0, 1.5, size=(n_obs, 8))
    m1_fc = obs[:, np.newaxis] + rng.normal(0.0, 0.8, size=(n_obs, 8))
    m2_fc = obs[:, np.newaxis] + rng.normal(0.0, 0.9, size=(n_obs, 8))
    corrupt_fc = np.full((n_obs, 8), np.nan)  # invalid payload

    params = ShadowLedgerParams(alpha_slash=1.0, protocol_fault_slash=0.05)

    clean = settle_compute_pool(
        [
            MinerEntry("m1", stake=100.0, forecast=m1_fc),
            MinerEntry("m2", stake=100.0, forecast=m2_fc),
        ],
        pool_amount=550.0,
        params=params,
        baseline_forecast=baseline_fc,
        obs=obs,
    )
    with_fault = settle_compute_pool(
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
    assert by_fault["faulty"].slash == pytest.approx(0.05 * 200.0)
    assert by_fault["faulty"].composite_score is None
    assert by_fault["faulty"].delta_crps is None


def test_eight_stations_in_one_cell_split_one_station_capacity() -> None:
    params = ShadowLedgerParams(alpha_slash=0.5, h3_cell_cap=1)
    crowded = [
        StationEntry(station_id=f"crowded_{i}", cell_id="872a1072bffffff", qod=1.0, pol=1.0)
        for i in range(8)
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
    base_fc = obs[:, np.newaxis] + rng.normal(0.0, 1.5, size=(100, 8))
    good_fc = obs[:, np.newaxis] + rng.normal(0.0, 0.8, size=(100, 8))
    params = ShadowLedgerParams(alpha_slash=0.5, twcrps_weight=0.30)

    res = settle_compute_pool(
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
