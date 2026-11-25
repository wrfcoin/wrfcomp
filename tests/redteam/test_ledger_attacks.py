"""Red-team adversarial audit of ``src/wrfcomp/pofs_shadow.py`` (Lane R1, ledger).

Every test asserts that the intended PoFS anti-gaming defense HOLDS against a
synthetic, deterministically seeded attack under ``docs/TOKENOMICS.md`` defaults.
Fixtures include recorded commitments and a separate GEFS anchor.
"""

from __future__ import annotations

import numpy as np
import pytest
from pofs_fixtures import recorded_epoch

from wrfcomp.pofs_shadow import (
    MinerEntry,
    ShadowLedgerParams,
    _mean_wasserstein,
)

DEFAULT_PARAMS = ShadowLedgerParams(alpha_slash=0.50)
POOL_AMOUNT = 550.0  # 55% compute share of a 1,000-token epoch


def _spread_skill_ratio(forecast: np.ndarray, obs: np.ndarray) -> float:
    """Ensemble spread-skill ratio (sqrt(mean(var)) / RMSE of ensemble mean)."""
    ens_mean = np.mean(forecast, axis=1)
    rmse = float(np.sqrt(np.mean((ens_mean - obs) ** 2)))
    spread = float(np.sqrt(np.mean(np.var(forecast, axis=1, ddof=1))))
    return spread / rmse


def _make_synoptic_fixture(
    seed: int = 202407,
    n_obs: int = 300,
    n_members: int = 31,
) -> dict[str, np.ndarray]:
    """Generate deterministic synthetic truth, baselines, and miner ensembles."""
    rng = np.random.default_rng(seed)
    signal = rng.normal(0.0, 2.0, size=n_obs)
    obs = signal + rng.normal(0.0, 1.0, size=n_obs)
    tail_thr = np.full(n_obs, float(np.quantile(obs, 0.95)), dtype=np.float64)

    # Climatology baseline: zero synoptic signal, calibrated marginal spread.
    clim_std = float(np.sqrt(2.0**2 + 1.0**2))
    clim_fc = rng.normal(0.0, clim_std, size=(n_obs, n_members))

    # Free GEFS baseline: noisy synoptic mean (err std 1.2), calibrated spread.
    gefs_mean = signal + rng.normal(0.0, 1.2, size=n_obs)
    gefs_err_std = float(np.sqrt(1.0 + 1.2**2))
    gefs_fc = gefs_mean[:, None] + rng.normal(0.0, gefs_err_std, size=(n_obs, n_members))

    # Under-dispersed GEFS baseline (spread_skill_ratio ~ 0.25, mirroring Gate 1 GEFS).
    gefs_under_fc = gefs_mean[:, None] + rng.normal(0.0, 0.25 * gefs_err_std, size=(n_obs, n_members))

    # Two independent honest skilled innovators (mean err std 0.5, calibrated spread).
    innov_err_std = float(np.sqrt(1.0 + 0.5**2))
    innov1_mean = signal + rng.normal(0.0, 0.5, size=n_obs)
    innov1_fc = innov1_mean[:, None] + rng.normal(0.0, innov_err_std, size=(n_obs, n_members))

    innov2_mean = signal + rng.normal(0.0, 0.5, size=n_obs)
    innov2_fc = innov2_mean[:, None] + rng.normal(0.0, innov_err_std, size=(n_obs, n_members))

    # Weak/bad forecast strictly worse than GEFS (mean err std 2.2).
    bad_mean = signal + rng.normal(0.0, 2.2, size=n_obs)
    bad_fc = bad_mean[:, None] + rng.normal(0.0, float(np.sqrt(1.0 + 2.2**2)), size=(n_obs, n_members))

    return {
        "signal": signal,
        "obs": obs,
        "tail_thr": tail_thr,
        "clim_fc": clim_fc,
        "gefs_fc": gefs_fc,
        "gefs_under_fc": gefs_under_fc,
        "innov1_mean": innov1_mean,
        "innov1_fc": innov1_fc,
        "innov2_fc": innov2_fc,
        "bad_fc": bad_fc,
    }


def _by_id(settlement) -> dict[str, object]:
    return {p.miner_id: p for p in settlement.payouts}


# ---------------------------------------------------------------------------
# 1. Copycat of the free GEFS baseline
# ---------------------------------------------------------------------------


def test_gefs_baseline_copycat_blocked_against_skilled_innovator() -> None:
    """HELD: Exact GEFS copycat earns 0 and is slashed when GEFS is L_base and an innovator beats it."""
    fx = _make_synoptic_fixture()
    res = recorded_epoch(
        [
            MinerEntry("innovator", stake=100.0, forecast=fx["innov1_fc"]),
            MinerEntry("gefs_copycat", stake=100.0, forecast=fx["gefs_fc"].copy()),
        ],
        pool_amount=POOL_AMOUNT,
        params=DEFAULT_PARAMS,
        baseline_forecast=fx["gefs_fc"],
        obs=fx["obs"],
        tail_threshold=fx["tail_thr"],
    )
    by_id = _by_id(res)
    assert by_id["gefs_copycat"].crpss == pytest.approx(0.0, abs=1e-12)
    assert by_id["gefs_copycat"].reward == 0.0
    assert by_id["gefs_copycat"].slash > 0.0
    assert by_id["innovator"].reward == pytest.approx(POOL_AMOUNT, rel=1e-12)


def test_gefs_baseline_copycat_blocked_when_weaker_miner_present() -> None:
    """Defense should ensure a zero-skill copycat of the locked GEFS baseline never gets paid."""
    fx = _make_synoptic_fixture()
    res = recorded_epoch(
        [
            MinerEntry("bad_or_decoy", stake=100.0, forecast=fx["bad_fc"]),
            MinerEntry("gefs_copycat", stake=100.0, forecast=fx["gefs_fc"].copy()),
        ],
        pool_amount=POOL_AMOUNT,
        params=DEFAULT_PARAMS,
        baseline_forecast=fx["gefs_fc"],
        obs=fx["obs"],
        tail_threshold=fx["tail_thr"],
    )
    by_id = _by_id(res)
    assert by_id["gefs_copycat"].crpss == pytest.approx(0.0, abs=1e-12)
    assert by_id["gefs_copycat"].reward == 0.0


def test_gefs_copycat_pair_blocked_under_climatology_baseline() -> None:
    """Defense should ensure identical relays of free GEFS cannot farm P_comp under climatology."""
    fx = _make_synoptic_fixture()
    res = recorded_epoch(
        [
            MinerEntry("gefs_relay_a", stake=100.0, forecast=fx["gefs_fc"].copy()),
            MinerEntry("gefs_relay_b", stake=100.0, forecast=fx["gefs_fc"].copy()),
        ],
        pool_amount=POOL_AMOUNT,
        params=DEFAULT_PARAMS,
        baseline_forecast=fx["clim_fc"],
        gefs_forecast=fx["gefs_fc"],
        obs=fx["obs"],
        tail_threshold=fx["tail_thr"],
    )
    by_id = _by_id(res)
    assert by_id["gefs_relay_a"].reward == 0.0
    assert by_id["gefs_relay_b"].reward == 0.0


# ---------------------------------------------------------------------------
# 2. Copycat with tiny noise
# ---------------------------------------------------------------------------


def test_gefs_copycat_with_tiny_noise_blocked_against_skilled_innovator() -> None:
    """HELD: Adding tiny noise to calibrated GEFS still earns 0 and is slashed vs. an innovator."""
    fx = _make_synoptic_fixture()
    rng = np.random.default_rng(101)
    noisy_gefs = fx["gefs_fc"] + rng.normal(0.0, 1e-2, size=fx["gefs_fc"].shape)
    res = recorded_epoch(
        [
            MinerEntry("innovator", stake=100.0, forecast=fx["innov1_fc"]),
            MinerEntry("gefs_noisy_copy", stake=100.0, forecast=noisy_gefs),
        ],
        pool_amount=POOL_AMOUNT,
        params=DEFAULT_PARAMS,
        baseline_forecast=fx["gefs_fc"],
        obs=fx["obs"],
        tail_threshold=fx["tail_thr"],
    )
    by_id = _by_id(res)
    assert by_id["gefs_noisy_copy"].reward == 0.0
    assert by_id["gefs_noisy_copy"].slash > 0.0
    assert by_id["innovator"].reward == pytest.approx(POOL_AMOUNT, rel=1e-12)


def test_underdispersed_gefs_copycat_with_tiny_noise_blocked() -> None:
    """Defense should prevent trivial Gaussian jitter on under-dispersed GEFS from farming."""
    fx = _make_synoptic_fixture()
    rng = np.random.default_rng(102)
    # Small zero-mean jitter (sigma = 0.30 vs obs std ~ 2.2) added to under-dispersed GEFS.
    jittered_gefs = fx["gefs_under_fc"] + rng.normal(0.0, 0.30, size=fx["gefs_under_fc"].shape)
    res = recorded_epoch(
        [MinerEntry("gefs_jitter_copy", stake=100.0, forecast=jittered_gefs)],
        pool_amount=POOL_AMOUNT,
        params=DEFAULT_PARAMS,
        baseline_forecast=fx["gefs_under_fc"],
        obs=fx["obs"],
        tail_threshold=fx["tail_thr"],
    )
    by_id = _by_id(res)
    assert by_id["gefs_jitter_copy"].reward == 0.0


def test_miner_copycat_with_tiny_noise_blocked() -> None:
    """Defense should prevent copying a skilled miner plus tiny noise from stealing rewards."""
    fx = _make_synoptic_fixture()
    rng = np.random.default_rng(103)
    noisy_copy = fx["innov1_fc"] + rng.normal(0.0, 1e-3, size=fx["innov1_fc"].shape)

    res_two = recorded_epoch(
        [
            MinerEntry("honest_miner", stake=100.0, forecast=fx["innov1_fc"]),
            MinerEntry("noisy_copycat", stake=100.0, forecast=noisy_copy),
        ],
        pool_amount=POOL_AMOUNT,
        params=DEFAULT_PARAMS,
        baseline_forecast=fx["gefs_fc"],
        obs=fx["obs"],
        tail_threshold=fx["tail_thr"],
    )
    res_three = recorded_epoch(
        [
            MinerEntry("honest_miner", stake=100.0, forecast=fx["innov1_fc"]),
            MinerEntry("noisy_copycat", stake=100.0, forecast=noisy_copy),
            MinerEntry("honest_miner_2", stake=100.0, forecast=fx["innov2_fc"]),
        ],
        pool_amount=POOL_AMOUNT,
        params=DEFAULT_PARAMS,
        baseline_forecast=fx["gefs_fc"],
        obs=fx["obs"],
        tail_threshold=fx["tail_thr"],
    )
    assert _by_id(res_two)["noisy_copycat"].reward == 0.0
    assert _by_id(res_three)["noisy_copycat"].reward == 0.0
    assert _by_id(res_two)["honest_miner"].reward > 0.0
    assert _by_id(res_two)["noisy_copycat"].status == "clone"


# ---------------------------------------------------------------------------
# 3. Selective non-reveal (reveal only when good)
# ---------------------------------------------------------------------------


def test_selective_non_reveal_penalized_on_mild_loss_epoch() -> None:
    """HELD: Withholding costs strictly more than revealing a mild miss."""
    fx = _make_synoptic_fixture()
    # Compare the same committed batch with a reveal and with a missing reveal.
    revealed = recorded_epoch(
        [
            MinerEntry("innovator", stake=100.0, forecast=fx["innov1_fc"]),
            MinerEntry("gambler", stake=100.0, forecast=fx["gefs_fc"].copy()),
        ],
        pool_amount=POOL_AMOUNT,
        params=DEFAULT_PARAMS,
        baseline_forecast=fx["gefs_fc"],
        obs=fx["obs"],
        tail_threshold=fx["tail_thr"],
    )
    withheld = recorded_epoch(
        [
            MinerEntry("innovator", stake=100.0, forecast=fx["innov1_fc"]),
            MinerEntry("gambler", stake=100.0, protocol_fault=True),
        ],
        pool_amount=POOL_AMOUNT,
        params=DEFAULT_PARAMS,
        baseline_forecast=fx["gefs_fc"],
        obs=fx["obs"],
        tail_threshold=fx["tail_thr"],
    )
    rev_gambler = _by_id(revealed)["gambler"]
    fault_gambler = _by_id(withheld)["gambler"]
    assert 0.0 < rev_gambler.slash < fault_gambler.slash
    assert fault_gambler.slash == pytest.approx(DEFAULT_PARAMS.max_slash_cap * fault_gambler.stake, abs=1e-12)


def test_selective_non_reveal_cannot_beat_honest_reveal_across_epochs() -> None:
    """Defense should ensure withholding bad forecasts never yields higher net profit than revealing."""
    fx = _make_synoptic_fixture()

    # Epoch 1 (good epoch for gambler): gambler reveals its skilled forecast.
    ep1 = recorded_epoch(
        [
            MinerEntry("honest", stake=100.0, forecast=fx["innov1_fc"]),
            MinerEntry("gambler", stake=100.0, forecast=fx["innov2_fc"]),
        ],
        pool_amount=POOL_AMOUNT,
        params=DEFAULT_PARAMS,
        baseline_forecast=fx["gefs_fc"],
        obs=fx["obs"],
        tail_threshold=fx["tail_thr"],
    )
    # Epoch 2 (bad epoch for gambler): compare honest reveal vs selective non-reveal.
    ep2_honest = recorded_epoch(
        [
            MinerEntry("honest", stake=100.0, forecast=fx["innov1_fc"]),
            MinerEntry("gambler", stake=100.0, forecast=fx["bad_fc"]),
        ],
        pool_amount=POOL_AMOUNT,
        params=DEFAULT_PARAMS,
        baseline_forecast=fx["gefs_fc"],
        obs=fx["obs"],
        tail_threshold=fx["tail_thr"],
    )
    ep2_nonreveal = recorded_epoch(
        [
            MinerEntry("honest", stake=100.0, forecast=fx["innov1_fc"]),
            MinerEntry("gambler", stake=100.0, protocol_fault=True),
        ],
        pool_amount=POOL_AMOUNT,
        params=DEFAULT_PARAMS,
        baseline_forecast=fx["gefs_fc"],
        obs=fx["obs"],
        tail_threshold=fx["tail_thr"],
    )

    g1 = _by_id(ep1)["gambler"]
    g2_hon = _by_id(ep2_honest)["gambler"]
    g2_nr = _by_id(ep2_nonreveal)["gambler"]
    honest_net = (g1.reward - g1.slash) + (g2_hon.reward - g2_hon.slash)
    selective_net = (g1.reward - g1.slash) + (g2_nr.reward - g2_nr.slash)

    assert g2_nr.slash >= g2_hon.slash
    assert selective_net <= honest_net


# ---------------------------------------------------------------------------
# 4. Sybil split of one forecast into many miners
# ---------------------------------------------------------------------------


def test_small_sybil_split_does_not_increase_payout() -> None:
    """HELD (small m): Splitting stake across 2 or 5 identical wallets reduces total attacker payout."""
    fx = _make_synoptic_fixture(n_obs=250, n_members=31)
    single = recorded_epoch(
        [
            MinerEntry("honest", stake=100.0, forecast=fx["innov1_fc"]),
            MinerEntry("attacker", stake=100.0, forecast=fx["innov2_fc"]),
        ],
        pool_amount=POOL_AMOUNT,
        params=DEFAULT_PARAMS,
        baseline_forecast=fx["gefs_fc"],
        obs=fx["obs"],
        tail_threshold=fx["tail_thr"],
    )
    single_reward = _by_id(single)["attacker"].reward

    for m_sybils in (2, 5):
        miners = [MinerEntry("honest", stake=100.0, forecast=fx["innov1_fc"])] + [
            MinerEntry(f"sybil_{i:02d}", stake=100.0 / m_sybils, forecast=fx["innov2_fc"])
            for i in range(m_sybils)
        ]
        res = recorded_epoch(
            miners,
            pool_amount=POOL_AMOUNT,
            params=DEFAULT_PARAMS,
            baseline_forecast=fx["gefs_fc"],
            obs=fx["obs"],
            tail_threshold=fx["tail_thr"],
        )
        sybil_total = sum(p.reward for p in res.payouts if p.miner_id.startswith("sybil_"))
        assert sybil_total <= single_reward


def test_many_wallet_sybil_split_does_not_increase_payout() -> None:
    """Defense should ensure splitting one forecast across many wallets never increases payout."""
    fx = _make_synoptic_fixture(n_obs=250, n_members=31)
    single = recorded_epoch(
        [
            MinerEntry("honest", stake=100.0, forecast=fx["innov1_fc"]),
            MinerEntry("attacker", stake=100.0, forecast=fx["innov2_fc"]),
        ],
        pool_amount=POOL_AMOUNT,
        params=DEFAULT_PARAMS,
        baseline_forecast=fx["gefs_fc"],
        obs=fx["obs"],
        tail_threshold=fx["tail_thr"],
    )
    single_reward = _by_id(single)["attacker"].reward

    assert single_reward > 0
    for m_sybils in range(1, 101):
        miners = [MinerEntry("honest", stake=100.0, forecast=fx["innov1_fc"])] + [
            MinerEntry(f"sybil_{i:03d}", stake=100.0 / m_sybils, forecast=fx["innov2_fc"])
            for i in range(m_sybils)
        ]
        res = recorded_epoch(
            miners,
            pool_amount=POOL_AMOUNT,
            params=DEFAULT_PARAMS,
            baseline_forecast=fx["gefs_fc"],
            obs=fx["obs"],
            tail_threshold=fx["tail_thr"],
        )
        sybil_total = sum(p.reward for p in res.payouts if p.miner_id.startswith("sybil_"))
        assert sybil_total <= single_reward


# ---------------------------------------------------------------------------
# 5. Last-moment copying of another miner
# ---------------------------------------------------------------------------


def test_last_moment_exact_copy_of_honest_miner_blocked() -> None:
    """Defense should ensure a last-moment exact copier earns 0 and cannot harm the victim."""
    fx = _make_synoptic_fixture()

    baseline_two_honest = recorded_epoch(
        [
            MinerEntry("victim", stake=100.0, forecast=fx["innov1_fc"]),
            MinerEntry("honest_2", stake=100.0, forecast=fx["innov2_fc"]),
        ],
        pool_amount=POOL_AMOUNT,
        params=DEFAULT_PARAMS,
        baseline_forecast=fx["gefs_fc"],
        obs=fx["obs"],
        tail_threshold=fx["tail_thr"],
    )
    victim_unattacked = _by_id(baseline_two_honest)["victim"].reward

    # 2-miner attack: copier submits exact verbatim clone of victim.
    res_two = recorded_epoch(
        [
            MinerEntry("victim", stake=100.0, forecast=fx["innov1_fc"]),
            MinerEntry("last_moment_copier", stake=100.0, forecast=fx["innov1_fc"].copy()),
        ],
        pool_amount=POOL_AMOUNT,
        params=DEFAULT_PARAMS,
        baseline_forecast=fx["gefs_fc"],
        obs=fx["obs"],
        tail_threshold=fx["tail_thr"],
    )
    # 3-miner attack: copier joins with equal stake alongside victim and honest_2.
    res_three = recorded_epoch(
        [
            MinerEntry("victim", stake=100.0, forecast=fx["innov1_fc"]),
            MinerEntry("last_moment_copier", stake=100.0, forecast=fx["innov1_fc"].copy()),
            MinerEntry("honest_2", stake=100.0, forecast=fx["innov2_fc"]),
        ],
        pool_amount=POOL_AMOUNT,
        params=DEFAULT_PARAMS,
        baseline_forecast=fx["gefs_fc"],
        obs=fx["obs"],
        tail_threshold=fx["tail_thr"],
    )
    assert _by_id(res_two)["last_moment_copier"].reward == 0.0
    assert _by_id(res_three)["last_moment_copier"].reward == 0.0
    assert _by_id(res_three)["victim"].reward >= 0.9 * victim_unattacked
    assert victim_unattacked > 0.0
    assert _by_id(res_two)["last_moment_copier"].status == "clone"


# ---------------------------------------------------------------------------
# 6. Uncalibrated but sharp forecast
# ---------------------------------------------------------------------------


def test_uncalibrated_sharp_forecast_beaten_by_calibrated_equal_mean() -> None:
    """HELD (equal mean): Strictly proper CRPS + twCRPS rejects a sharp clone of a calibrated miner."""
    fx = _make_synoptic_fixture()
    rng = np.random.default_rng(601)
    sharp_same_mean = fx["innov1_mean"][:, None] + rng.normal(0.0, 0.05, size=fx["innov1_fc"].shape)
    assert 0.85 <= _spread_skill_ratio(fx["innov1_fc"], fx["obs"]) <= 1.15
    assert _spread_skill_ratio(sharp_same_mean, fx["obs"]) < 0.15

    res = recorded_epoch(
        [
            MinerEntry("calibrated", stake=100.0, forecast=fx["innov1_fc"]),
            MinerEntry("sharp_uncalibrated", stake=100.0, forecast=sharp_same_mean),
        ],
        pool_amount=POOL_AMOUNT,
        params=DEFAULT_PARAMS,
        baseline_forecast=fx["gefs_fc"],
        obs=fx["obs"],
        tail_threshold=fx["tail_thr"],
    )
    by_id = _by_id(res)
    assert by_id["sharp_uncalibrated"].reward == 0.0
    assert by_id["sharp_uncalibrated"].slash > 0.0
    assert by_id["calibrated"].reward == pytest.approx(POOL_AMOUNT, rel=1e-12)


def test_uncalibrated_sharp_forecast_gated_out_despite_better_mean() -> None:
    """Defense should prevent severely uncalibrated sharp ensembles from earning compute rewards."""
    fx = _make_synoptic_fixture()
    rng = np.random.default_rng(602)
    # Accurate mean (err std 0.2) but collapsed ensemble spread (0.05) -> spread_skill_ratio < 0.10.
    sharp_better_mean = (fx["signal"] + rng.normal(0.0, 0.2, size=fx["obs"].size))[:, None] + rng.normal(
        0.0, 0.05, size=fx["innov1_fc"].shape
    )
    assert 0.85 <= _spread_skill_ratio(fx["innov1_fc"], fx["obs"]) <= 1.15
    assert _spread_skill_ratio(sharp_better_mean, fx["obs"]) < 0.15

    res_pair = recorded_epoch(
        [
            MinerEntry("calibrated", stake=100.0, forecast=fx["innov1_fc"]),
            MinerEntry("sharp_uncalibrated", stake=100.0, forecast=sharp_better_mean),
        ],
        pool_amount=POOL_AMOUNT,
        params=DEFAULT_PARAMS,
        baseline_forecast=fx["gefs_fc"],
        obs=fx["obs"],
        tail_threshold=fx["tail_thr"],
    )
    res_solo = recorded_epoch(
        [MinerEntry("sharp_uncalibrated", stake=100.0, forecast=sharp_better_mean)],
        pool_amount=POOL_AMOUNT,
        params=DEFAULT_PARAMS,
        baseline_forecast=fx["gefs_fc"],
        obs=fx["obs"],
        tail_threshold=fx["tail_thr"],
    )
    assert _by_id(res_pair)["sharp_uncalibrated"].reward == 0.0
    assert _by_id(res_solo)["sharp_uncalibrated"].reward == 0.0


@pytest.mark.parametrize("variant", ["30_members", "sigma_0.003"])
@pytest.mark.parametrize("unit_scale", [1.0, 100.0])
def test_clone_reward_is_invariant_to_member_count_noise_and_field_units(variant, unit_scale):
    fx = _make_synoptic_fixture()
    original = fx["innov1_fc"]
    if variant == "30_members":
        copy = original[:, :-1]
    else:
        copy = original + np.random.default_rng(46).normal(0, 0.003, original.shape)
    miners = [
        MinerEntry("first", 100, forecast=unit_scale * original),
        MinerEntry("later", 100, forecast=unit_scale * copy),
    ]
    result = recorded_epoch(
        miners,
        pool_amount=POOL_AMOUNT,
        params=DEFAULT_PARAMS,
        obs=unit_scale * fx["obs"],
        tail_threshold=unit_scale * fx["tail_thr"],
        baseline_forecast=unit_scale * fx["gefs_fc"],
    )
    payouts = _by_id(result)
    assert payouts["later"].status == "clone" and payouts["later"].reward == 0
    assert payouts["first"].reward == pytest.approx(POOL_AMOUNT)


def test_minimum_stake_spoiler_does_not_increase_combined_reward():
    fx = _make_synoptic_fixture(n_obs=250)
    kwargs = dict(
        pool_amount=POOL_AMOUNT,
        params=DEFAULT_PARAMS,
        obs=fx["obs"],
        baseline_forecast=fx["gefs_fc"],
        tail_threshold=fx["tail_thr"],
    )
    original = MinerEntry("first", 100, forecast=fx["innov1_fc"])
    rival = fx["innov2_fc"]
    control = recorded_epoch([original, MinerEntry("rival", 100, forecast=rival)], **kwargs)
    floor = DEFAULT_PARAMS.k_min
    matched_control = recorded_epoch([original, MinerEntry("rival", 100 - floor, forecast=rival)], **kwargs)
    split = recorded_epoch(
        [
            original,
            MinerEntry("rival", 100 - floor, forecast=rival),
            MinerEntry("copy", floor, forecast=original.forecast[:, :-1]),
        ],
        **kwargs,
    )
    before, matched, after = _by_id(control), _by_id(matched_control), _by_id(split)
    assert before["rival"].reward > 0 and before["first"].reward > 0
    assert after["copy"].status == "clone" and after["copy"].reward == 0
    assert after["first"].delta_crps == matched["first"].delta_crps
    assert after["rival"].reward + after["copy"].reward <= before["rival"].reward


@pytest.mark.parametrize("stake", [0.0, np.nextafter(DEFAULT_PARAMS.k_min, 0)])
def test_below_minimum_commit_cannot_reserve_clone_slot(stake):
    fx = _make_synoptic_fixture()
    result = recorded_epoch(
        [
            MinerEntry("early", stake, forecast=fx["innov1_fc"]),
            MinerEntry("first_eligible", 100, forecast=fx["innov1_fc"]),
        ],
        pool_amount=POOL_AMOUNT,
        params=DEFAULT_PARAMS,
        obs=fx["obs"],
        tail_threshold=fx["tail_thr"],
        baseline_forecast=fx["gefs_fc"],
    )
    payouts = _by_id(result)
    assert payouts["early"].status == "below_minimum"
    assert payouts["first_eligible"].reward == pytest.approx(POOL_AMOUNT)


# ---------------------------------------------------------------------------
# 7. Stake-weighted Meta-Ensemble vs 50 perturbed minimum-stake Sybil wallets
# ---------------------------------------------------------------------------


def _walsh_hadamard(order: int) -> np.ndarray:
    mat = np.array([[1.0]], dtype=np.float64)
    while mat.shape[0] < order:
        mat = np.block([[mat, mat], [mat, -mat]])
    return mat


def test_fifty_min_stake_perturbed_sybils_do_not_distort_honest_delta_or_payout() -> None:
    """50 minimum-stake wallets perturbed just outside clone tolerance cannot distort honest miners."""
    n_obs = 256
    fx = _make_synoptic_fixture(n_obs=n_obs, n_members=31)
    gefs, obs, thr = fx["gefs_fc"], fx["obs"], fx["tail_thr"]

    rng = np.random.default_rng(999)
    innov3_mean = fx["signal"] + rng.normal(0.0, 0.6, size=n_obs)
    innov3_fc = innov3_mean[:, None] + rng.normal(0.0, float(np.sqrt(1.0 + 0.6**2)), size=(n_obs, 31))

    field_scale = max(float(np.sqrt(gefs.var(axis=1).mean())), float(np.std(obs)))
    clone_tol = DEFAULT_PARAMS.clone_relative_tolerance * field_scale
    hadamard = _walsh_hadamard(n_obs)
    # Row-wise +-1.05*clone_tol Walsh-Hadamard perturbations sit 5% outside tolerance pairwise
    # so all 50 minimum-stake wallets pass clone deduplication and enter the Meta-Ensemble.
    copies = [innov3_fc + (1.05 * clone_tol) * hadamard[i + 1, :, None] for i in range(50)]
    sorted_copies = [np.sort(c, axis=1) for c in copies]
    min_pair_w1 = min(
        _mean_wasserstein(sorted_copies[i], sorted_copies[j]) for i in range(50) for j in range(i + 1, 50)
    )
    assert min_pair_w1 > clone_tol

    total_attacker_stake = 50.0 * DEFAULT_PARAMS.k_min
    honest_1 = MinerEntry("honest_1", stake=2500.0, forecast=fx["innov1_fc"])
    honest_2 = MinerEntry("honest_2", stake=2500.0, forecast=fx["innov2_fc"])
    kwargs = dict(
        pool_amount=POOL_AMOUNT,
        params=DEFAULT_PARAMS,
        baseline_forecast=gefs,
        obs=obs,
        tail_threshold=thr,
    )

    single = recorded_epoch(
        [honest_1, honest_2, MinerEntry("attacker", stake=total_attacker_stake, forecast=innov3_fc)],
        **kwargs,
    )
    sybils = [MinerEntry(f"sybil_{i:02d}", stake=DEFAULT_PARAMS.k_min, forecast=copies[i]) for i in range(50)]
    split = recorded_epoch([honest_1, honest_2, *sybils], **kwargs)

    before, after = _by_id(single), _by_id(split)
    admitted = [p for p in split.payouts if p.miner_id.startswith("sybil_") and p.status == "eligible"]
    assert len(admitted) == 50

    # Stated small tolerances compared with one wallet holding the same total stake (500):
    # |Delta-CRPS change| <= 1e-3 and relative payout change <= 2%.
    for name in ("honest_1", "honest_2"):
        assert before[name].delta_crps is not None and after[name].delta_crps is not None
        assert abs(after[name].delta_crps - before[name].delta_crps) <= 1e-3
        assert abs(after[name].reward - before[name].reward) / before[name].reward <= 0.02

    attacker_split_total = sum(p.reward for p in split.payouts if p.miner_id.startswith("sybil_"))
    assert attacker_split_total <= before["attacker"].reward


# ---------------------------------------------------------------------------
# 8. Public promoted-candidate relays and consecutive-epoch nuisance slash
# ---------------------------------------------------------------------------


def test_public_promoted_candidate_first_relay_earns_zero_and_honest_still_earns() -> None:
    """First relay of a public promoted candidate (e.g. v3) is a clone earning zero; honest earns."""
    fx = _make_synoptic_fixture()
    v3_public_fc = fx["innov2_fc"]  # calibrated forecast passing all GEFS gates
    honest_fc = fx["innov1_fc"]
    kwargs = dict(
        pool_amount=POOL_AMOUNT,
        params=DEFAULT_PARAMS,
        baseline_forecast=fx["gefs_fc"],
        obs=fx["obs"],
        tail_threshold=fx["tail_thr"],
    )

    # Control: without v3 configured as a public reference, relaying v3 would earn the full pool.
    unlisted = recorded_epoch([MinerEntry("v3_relay", stake=100.0, forecast=v3_public_fc)], **kwargs)
    assert _by_id(unlisted)["v3_relay"].status == "eligible"
    assert _by_id(unlisted)["v3_relay"].reward == pytest.approx(POOL_AMOUNT)

    # With v3 listed as a public reference, even the FIRST relay earns zero.
    first_relay = recorded_epoch(
        [MinerEntry("v3_relay", stake=100.0, forecast=v3_public_fc)],
        public_references=[v3_public_fc],
        **kwargs,
    )
    p_first = _by_id(first_relay)["v3_relay"]
    assert p_first.status == "public_clone"
    assert p_first.reward == 0.0
    assert p_first.slash == 0.0
    assert first_relay.unpaid_remainder == pytest.approx(POOL_AMOUNT)
    assert first_relay.public_clone_streaks == {"v3_relay": 1}

    # Perturbed copy of v3 inside clone tolerance (30 members or small jitter) is also caught.
    jittered_v3 = v3_public_fc + np.random.default_rng(808).normal(0.0, 0.003, size=v3_public_fc.shape)
    near_relays = recorded_epoch(
        [
            MinerEntry("v3_drop_member", stake=100.0, forecast=v3_public_fc[:, :-1]),
            MinerEntry("v3_jitter", stake=100.0, forecast=jittered_v3),
            MinerEntry("honest", stake=100.0, forecast=honest_fc),
        ],
        public_references=[v3_public_fc],
        **kwargs,
    )
    honest_solo = recorded_epoch(
        [MinerEntry("honest", stake=100.0, forecast=honest_fc)],
        public_references=[v3_public_fc],
        **kwargs,
    )
    by_near = _by_id(near_relays)
    assert by_near["v3_drop_member"].status == "public_clone"
    assert by_near["v3_drop_member"].reward == 0.0
    assert by_near["v3_jitter"].status == "public_clone"
    assert by_near["v3_jitter"].reward == 0.0
    assert by_near["honest"].status == "eligible"
    assert by_near["honest"].reward == pytest.approx(POOL_AMOUNT)
    assert by_near["honest"] == _by_id(honest_solo)["honest"]


def test_consecutive_public_reference_relays_incur_nuisance_slash_and_reset_on_honesty() -> None:
    """Relaying a public reference in 2+ consecutive epochs triggers the small nuisance slash."""
    fx = _make_synoptic_fixture()
    v3_public_fc = fx["innov2_fc"]
    honest_fc = fx["innov1_fc"]
    kwargs = dict(
        pool_amount=POOL_AMOUNT,
        params=DEFAULT_PARAMS,
        baseline_forecast=fx["gefs_fc"],
        obs=fx["obs"],
        tail_threshold=fx["tail_thr"],
        public_references=[v3_public_fc],
    )

    # Epoch 1: first relay of v3 -> earns 0, no slash yet (streak 1).
    ep1 = recorded_epoch([MinerEntry("relayer", stake=100.0, forecast=v3_public_fc)], **kwargs)
    assert _by_id(ep1)["relayer"].status == "public_clone"
    assert _by_id(ep1)["relayer"].reward == 0.0
    assert _by_id(ep1)["relayer"].slash == 0.0
    assert ep1.public_clone_streaks == {"relayer": 1}

    # Epoch 2 (consecutive): jittered relay of v3 -> earns 0 AND incurs 2% nuisance slash (streak 2).
    jittered_v3 = v3_public_fc + np.random.default_rng(809).normal(0.0, 0.002, size=v3_public_fc.shape)
    ep2 = recorded_epoch(
        [MinerEntry("relayer", stake=100.0, forecast=jittered_v3)],
        prior_public_clone_streaks=ep1.public_clone_streaks,
        **kwargs,
    )
    expected_nuisance = 100.0 * DEFAULT_PARAMS.public_clone_slash
    assert expected_nuisance <= 0.5 * DEFAULT_PARAMS.max_slash_cap * 100.0
    assert _by_id(ep2)["relayer"].status == "public_clone"
    assert _by_id(ep2)["relayer"].reward == 0.0
    assert _by_id(ep2)["relayer"].slash == pytest.approx(expected_nuisance)
    assert ep2.public_clone_streaks == {"relayer": 2}

    # Epoch 3: wallet submits an honest independent forecast -> earns full pool and streak resets.
    ep3 = recorded_epoch(
        [MinerEntry("relayer", stake=100.0, forecast=honest_fc)],
        prior_public_clone_streaks=ep2.public_clone_streaks,
        **kwargs,
    )
    assert _by_id(ep3)["relayer"].status == "eligible"
    assert _by_id(ep3)["relayer"].reward == pytest.approx(POOL_AMOUNT)
    assert _by_id(ep3)["relayer"].slash == 0.0
    assert ep3.public_clone_streaks == {}

    # Epoch 4: single relay after honesty starts a fresh streak (no slash on first repeat).
    ep4 = recorded_epoch(
        [MinerEntry("relayer", stake=100.0, forecast=v3_public_fc)],
        prior_public_clone_streaks=ep3.public_clone_streaks,
        **kwargs,
    )
    assert _by_id(ep4)["relayer"].status == "public_clone"
    assert _by_id(ep4)["relayer"].slash == 0.0
    assert ep4.public_clone_streaks == {"relayer": 1}

    # Raw GEFS relay over two consecutive epochs (no innovator present) also triggers nuisance slash.
    gefs_ep1 = recorded_epoch([MinerEntry("gefs_spam", stake=100.0, forecast=fx["gefs_fc"])], **kwargs)
    assert _by_id(gefs_ep1)["gefs_spam"].reward == 0.0
    assert _by_id(gefs_ep1)["gefs_spam"].slash == 0.0
    assert gefs_ep1.public_clone_streaks == {"gefs_spam": 1}
    gefs_ep2 = recorded_epoch(
        [MinerEntry("gefs_spam", stake=100.0, forecast=fx["gefs_fc"])],
        prior_public_clone_streaks=gefs_ep1.public_clone_streaks,
        **kwargs,
    )
    assert _by_id(gefs_ep2)["gefs_spam"].reward == 0.0
    assert _by_id(gefs_ep2)["gefs_spam"].slash == pytest.approx(expected_nuisance)
    assert gefs_ep2.public_clone_streaks == {"gefs_spam": 2}


def test_negative_delta_splitter_cannot_dodge_slash_or_gain_reward_array_mode() -> None:
    """Array mode: splitting a negative-Delta miner into 10 or 50 wallets cannot raise pay or cut slash."""
    n_obs = 256
    fx = _make_synoptic_fixture(n_obs=n_obs, n_members=31)
    gefs, obs, thr = fx["gefs_fc"], fx["obs"], fx["tail_thr"]

    field_scale = max(float(np.sqrt(gefs.var(axis=1).mean())), float(np.std(obs)))
    clone_tol = DEFAULT_PARAMS.clone_relative_tolerance * field_scale
    hadamard = _walsh_hadamard(n_obs)

    # Shifted forecast: beats GEFS (passes gates) but is worse than the honest Meta-Ensemble (Delta < 0).
    att_fc = fx["innov1_fc"] + 0.7
    honest_1 = MinerEntry("honest_1", stake=2500.0, forecast=fx["innov1_fc"])
    honest_2 = MinerEntry("honest_2", stake=2500.0, forecast=fx["innov2_fc"])
    kwargs = dict(
        pool_amount=POOL_AMOUNT,
        params=DEFAULT_PARAMS,
        baseline_forecast=gefs,
        obs=obs,
        tail_threshold=thr,
    )

    total_stake = 500.0
    unsplit = recorded_epoch(
        [honest_1, honest_2, MinerEntry("attacker", stake=total_stake, forecast=att_fc)],
        **kwargs,
    )
    by_unsplit = _by_id(unsplit)
    assert by_unsplit["attacker"].status == "eligible"
    assert by_unsplit["attacker"].delta_crps is not None and by_unsplit["attacker"].delta_crps < 0.0
    unsplit_honest_total = by_unsplit["honest_1"].reward + by_unsplit["honest_2"].reward

    for n_wallets in (10, 50):
        copies = [att_fc + (1.05 * clone_tol) * hadamard[i + 1, :, None] for i in range(n_wallets)]
        sybils = [
            MinerEntry(f"sybil_{i:02d}", stake=total_stake / n_wallets, forecast=copies[i])
            for i in range(n_wallets)
        ]
        split = recorded_epoch([honest_1, honest_2, *sybils], **kwargs)
        by_split = _by_id(split)
        admitted = [p for p in split.payouts if p.miner_id.startswith("sybil_") and p.status == "eligible"]
        assert len(admitted) == n_wallets
        split_reward = sum(p.reward for p in admitted)
        split_slash = sum(p.slash for p in admitted)
        split_honest_total = by_split["honest_1"].reward + by_split["honest_2"].reward
        assert split_reward <= by_unsplit["attacker"].reward + 1e-12
        assert split_slash >= by_unsplit["attacker"].slash - 1e-12
        assert split_honest_total >= unsplit_honest_total - 1e-12


def test_negative_delta_splitter_cannot_dodge_slash_or_gain_reward_scorecard_mode() -> None:
    """Scorecard mode: splitting a negative-Delta whale into 10 or 50 wallets gives identical pay & slash."""
    from pathlib import Path

    from wrfcomp.pofs_shadow import Commitment, EpochEvidence, load_scorecard, settle_from_scorecards

    ref = load_scorecard(Path(__file__).resolve().parents[2] / "experiments" / "gefs_2024-Q1.json")

    def _make_card(model_id: str, digest: str, loss_scale: float):
        d = ref.model_dump()
        d["model_id"] = model_id
        d["forecast_sha256"] = digest
        d["baseline_id"] = "noaa-gefs"
        d["baseline_sha256"] = ref.forecast_sha256
        d["passed_gate"] = True
        for var, leads in d["metrics"].items():
            d["gates"][var] = dict.fromkeys(d["gates"][var], True)
            row = leads["all"]
            row["crpss_vs_baseline"] = 1.0 - loss_scale
            row["crpss_vs_baseline_ecdf"] = 1.0 - loss_scale
            row["spread_skill_ratio"] = 1.0
            row["dm_pvalue"] = 0.001
            row["twcrps_p95"] = 0.1
            row["baseline_twcrps_p95"] = 0.2
            for h in d["leads_scored"]:
                leads[str(h)]["loss"] *= loss_scale
        return type(ref).model_validate(d)

    t = np.datetime64(ref.period_start)
    honest = _make_card("honest", "a" * 64, 0.50)

    for whale_loss, whale_stake, n_wallets in ((0.80, 100.0, 10), (0.60, 100.0, 10), (0.95, 500.0, 50)):
        whale = _make_card("whale", "b" * 64, whale_loss)
        ev_single = EpochEvidence(
            t,
            ref.forecast_sha256,
            {
                "honest": Commitment(honest.forecast_sha256, t - np.timedelta64(10, "s")),
                "whale": Commitment(whale.forecast_sha256, t - np.timedelta64(9, "s")),
            },
        )
        res_single = settle_from_scorecards(
            {"gefs": ref, "honest": honest, "whale": whale},
            baseline_key="gefs",
            pool_amount=POOL_AMOUNT,
            params=DEFAULT_PARAMS,
            stakes={"gefs": 100.0, "honest": 500.0, "whale": whale_stake},
            evidence=ev_single,
        )

        split_cards = {"gefs": ref, "honest": honest}
        split_stakes = {"gefs": 100.0, "honest": 500.0}
        split_receipts = {"honest": Commitment(honest.forecast_sha256, t - np.timedelta64(10, "s"))}
        for idx in range(n_wallets):
            key = f"sybil_{idx:02d}"
            card = _make_card(key, f"{idx + 100:064x}", whale_loss)
            split_cards[key] = card
            split_stakes[key] = whale_stake / n_wallets
            split_receipts[key] = Commitment(card.forecast_sha256, t - np.timedelta64(8, "s"))

        res_split = settle_from_scorecards(
            split_cards,
            baseline_key="gefs",
            pool_amount=POOL_AMOUNT,
            params=DEFAULT_PARAMS,
            stakes=split_stakes,
            evidence=EpochEvidence(t, ref.forecast_sha256, split_receipts),
        )
        by_single, by_split = _by_id(res_single), _by_id(res_split)
        assert by_single["whale"].status == "eligible"
        assert by_single["whale"].delta_crps is not None and by_single["whale"].delta_crps < 0.0
        sybil_payouts = [p for p in res_split.payouts if p.miner_id.startswith("sybil_")]
        assert len(sybil_payouts) == n_wallets and all(p.status == "eligible" for p in sybil_payouts)
        assert sum(p.reward for p in sybil_payouts) == pytest.approx(
            by_single["whale"].reward, rel=1e-11, abs=1e-11
        )
        assert sum(p.slash for p in sybil_payouts) == pytest.approx(
            by_single["whale"].slash, rel=1e-11, abs=1e-11
        )
        assert by_split["honest"].reward == pytest.approx(by_single["honest"].reward, rel=1e-11, abs=1e-11)


def test_skipping_epochs_does_not_reset_public_clone_streak() -> None:
    """Relaying a public reference every other epoch (skipping in between) still triggers nuisance slash."""
    fx = _make_synoptic_fixture()
    v3_public_fc = fx["innov2_fc"]
    honest_fc = fx["innov1_fc"]
    kwargs = dict(
        pool_amount=POOL_AMOUNT,
        params=DEFAULT_PARAMS,
        baseline_forecast=fx["gefs_fc"],
        obs=fx["obs"],
        tail_threshold=fx["tail_thr"],
        public_references=[v3_public_fc],
    )
    expected_nuisance = 100.0 * DEFAULT_PARAMS.public_clone_slash

    # Epoch 1: relayer submits v3 -> streak 1, no slash yet.
    ep1 = recorded_epoch(
        [
            MinerEntry("honest", stake=100.0, forecast=honest_fc),
            MinerEntry("relayer", stake=100.0, forecast=v3_public_fc),
        ],
        **kwargs,
    )
    assert _by_id(ep1)["relayer"].status == "public_clone"
    assert _by_id(ep1)["relayer"].slash == 0.0
    assert ep1.public_clone_streaks == {"relayer": 1}

    # Epoch 2: relayer skips the epoch entirely -> streak MUST NOT reset.
    ep2 = recorded_epoch(
        [MinerEntry("honest", stake=100.0, forecast=honest_fc)],
        prior_public_clone_streaks=ep1.public_clone_streaks,
        **kwargs,
    )
    assert ep2.public_clone_streaks == {"relayer": 1}

    # Epoch 3: relayer submits v3 again (every-other-epoch pattern) -> streak 2, nuisance slash fires!
    ep3 = recorded_epoch(
        [
            MinerEntry("honest", stake=100.0, forecast=honest_fc),
            MinerEntry("relayer", stake=100.0, forecast=v3_public_fc),
        ],
        prior_public_clone_streaks=ep2.public_clone_streaks,
        **kwargs,
    )
    assert _by_id(ep3)["relayer"].status == "public_clone"
    assert _by_id(ep3)["relayer"].reward == 0.0
    assert _by_id(ep3)["relayer"].slash == pytest.approx(expected_nuisance)
    assert ep3.public_clone_streaks == {"relayer": 2}

    # Epoch 4: attempting a below-minimum dust submission does NOT reset the streak either.
    ep4 = recorded_epoch(
        [
            MinerEntry("honest", stake=100.0, forecast=honest_fc),
            MinerEntry("relayer", stake=1.0, forecast=honest_fc),
        ],
        prior_public_clone_streaks=ep3.public_clone_streaks,
        **kwargs,
    )
    assert _by_id(ep4)["relayer"].status == "below_minimum"
    assert ep4.public_clone_streaks == {"relayer": 2}

    # Epoch 5: submitting an actual valid non-clone forecast resets the streak.
    ep5 = recorded_epoch(
        [MinerEntry("relayer", stake=100.0, forecast=honest_fc)],
        prior_public_clone_streaks=ep4.public_clone_streaks,
        **kwargs,
    )
    assert _by_id(ep5)["relayer"].status == "eligible"
    assert _by_id(ep5)["relayer"].slash == 0.0
    assert ep5.public_clone_streaks == {}
