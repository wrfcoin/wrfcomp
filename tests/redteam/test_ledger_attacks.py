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
