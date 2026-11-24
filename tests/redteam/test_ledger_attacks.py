"""Red-team adversarial audit of ``src/wrfcomp/pofs_shadow.py`` (Lane R1, ledger).

Every test asserts that the intended PoFS anti-gaming defense HOLDS against a
synthetic, deterministically seeded attack under ``docs/TOKENOMICS.md`` defaults.
Attacks that currently break the ledger are marked ``@pytest.mark.xfail(strict=True)``
with the exact mathematical/structural reason; no production code is modified here.
"""

from __future__ import annotations

import numpy as np
import pytest

from wrfcomp.pofs_shadow import (
    MinerEntry,
    ShadowLedgerParams,
    settle_compute_pool,
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
    n_members: int = 12,
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
    res = settle_compute_pool(
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


@pytest.mark.xfail(
    strict=True,
    reason=(
        "BROKEN: settle_compute_pool has no CRPSS > 0 gate and includes worse-than-baseline "
        "miners in M_all, so a zero-skill exact copy of the locked GEFS baseline earns positive "
        "Delta_CRPS and 100% of P_comp simply by diluting a weaker or decoy miner."
    ),
)
def test_gefs_baseline_copycat_blocked_when_weaker_miner_present() -> None:
    """Defense should ensure a zero-skill copycat of the locked GEFS baseline never gets paid."""
    fx = _make_synoptic_fixture()
    res = settle_compute_pool(
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


@pytest.mark.xfail(
    strict=True,
    reason=(
        "BROKEN: Under a climatology baseline (as in Gate 1), free GEFS has CRPSS > 0; two "
        "identical GEFS relays have Delta_CRPS = 0 yet S_i = (1 - w_delta) * CRPSS > 0 and "
        "split 100% of P_comp because free public models are not anchored into M_all."
    ),
)
def test_gefs_copycat_pair_blocked_under_climatology_baseline() -> None:
    """Defense should ensure identical relays of free GEFS cannot farm P_comp under climatology."""
    fx = _make_synoptic_fixture()
    res = settle_compute_pool(
        [
            MinerEntry("gefs_relay_a", stake=100.0, forecast=fx["gefs_fc"].copy()),
            MinerEntry("gefs_relay_b", stake=100.0, forecast=fx["gefs_fc"].copy()),
        ],
        pool_amount=POOL_AMOUNT,
        params=DEFAULT_PARAMS,
        baseline_forecast=fx["clim_fc"],
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
    res = settle_compute_pool(
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


@pytest.mark.xfail(
    strict=True,
    reason=(
        "BROKEN: When the locked GEFS baseline is under-dispersed (like real Gate 1 GEFS), "
        "adding small i.i.d. zero-mean Gaussian jitter (sigma=0.30, <15% of obs std) to GEFS "
        "strictly reduces CRPS, giving CRPSS > 0 and Delta_CRPS > 0 and paying a zero-physics "
        "jitter copycat 100% of P_comp."
    ),
)
def test_underdispersed_gefs_copycat_with_tiny_noise_blocked() -> None:
    """Defense should prevent trivial Gaussian jitter on under-dispersed GEFS from farming."""
    fx = _make_synoptic_fixture()
    rng = np.random.default_rng(102)
    # Small zero-mean jitter (sigma = 0.30 vs obs std ~ 2.2) added to under-dispersed GEFS.
    jittered_gefs = fx["gefs_under_fc"] + rng.normal(0.0, 0.30, size=fx["gefs_under_fc"].shape)
    res = settle_compute_pool(
        [MinerEntry("gefs_jitter_copy", stake=100.0, forecast=jittered_gefs)],
        pool_amount=POOL_AMOUNT,
        params=DEFAULT_PARAMS,
        baseline_forecast=fx["gefs_under_fc"],
        obs=fx["obs"],
        tail_threshold=fx["tail_thr"],
    )
    by_id = _by_id(res)
    assert by_id["gefs_jitter_copy"].reward == 0.0


@pytest.mark.xfail(
    strict=True,
    reason=(
        "BROKEN: Copying a skilled miner and adding 1e-3 noise yields Delta_CRPS ~ 0 (or slightly "
        "negative in a 3-miner pool), but (1 - w_delta) * CRPSS keeps S_i > 0 so the noisy "
        "copycat steals ~50% of P_comp in a 2-miner pool and ~20% in a 3-miner pool."
    ),
)
def test_miner_copycat_with_tiny_noise_blocked() -> None:
    """Defense should prevent copying a skilled miner plus tiny noise from stealing rewards."""
    fx = _make_synoptic_fixture()
    rng = np.random.default_rng(103)
    noisy_copy = fx["innov1_fc"] + rng.normal(0.0, 1e-3, size=fx["innov1_fc"].shape)

    res_two = settle_compute_pool(
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
    res_three = settle_compute_pool(
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


# ---------------------------------------------------------------------------
# 3. Selective non-reveal (reveal only when good)
# ---------------------------------------------------------------------------


def test_selective_non_reveal_penalized_on_mild_loss_epoch() -> None:
    """HELD: On a mild miss (skill slash < 5%), non-reveal incurs a strictly larger 5% fault slash."""
    fx = _make_synoptic_fixture()
    # Mildly worse than GEFS (same as GEFS copycat vs innovator -> ~1.3% skill slash < 5%).
    revealed = settle_compute_pool(
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
    withheld = settle_compute_pool(
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
    assert fault_gambler.slash == pytest.approx(5.0, abs=1e-12)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "BROKEN: protocol_fault_slash (5%) is strictly cheaper than max_slash_cap (10%) and the "
        "ledger has no timelock commit-reveal enforcement. A miner facing a severe bad epoch "
        "halves its slash (5% vs 10%) by withholding/corrupting the reveal, so selective "
        "non-reveal strictly dominates honest full revealing in net profit across epochs."
    ),
)
def test_selective_non_reveal_cannot_beat_honest_reveal_across_epochs() -> None:
    """Defense should ensure withholding bad forecasts never yields higher net profit than revealing."""
    fx = _make_synoptic_fixture()

    # Epoch 1 (good epoch for gambler): gambler reveals its skilled forecast.
    ep1 = settle_compute_pool(
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
    ep2_honest = settle_compute_pool(
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
    ep2_nonreveal = settle_compute_pool(
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
    fx = _make_synoptic_fixture(n_obs=250, n_members=10)
    single = settle_compute_pool(
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
        res = settle_compute_pool(
            miners,
            pool_amount=POOL_AMOUNT,
            params=DEFAULT_PARAMS,
            baseline_forecast=fx["gefs_fc"],
            obs=fx["obs"],
            tail_threshold=fx["tail_thr"],
        )
        sybil_total = sum(p.reward for p in res.payouts if p.miner_id.startswith("sybil_"))
        assert sybil_total <= single_reward


@pytest.mark.xfail(
    strict=True,
    reason=(
        "BROKEN: With no K_min floor, splitting stake K across m identical wallets shrinks "
        "Delta_CRPS by O(1/m) but concave K_i^0.75 amplifies the constant (1 - w_delta) * CRPSS "
        "term by m^0.25. At m = 100 wallets, total Sybil payout strictly exceeds the "
        "single-wallet payout without adding any forecast skill."
    ),
)
def test_many_wallet_sybil_split_does_not_increase_payout() -> None:
    """Defense should ensure splitting one forecast across many wallets never increases payout."""
    fx = _make_synoptic_fixture(n_obs=250, n_members=10)
    single = settle_compute_pool(
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

    m_sybils = 100
    miners = [MinerEntry("honest", stake=100.0, forecast=fx["innov1_fc"])] + [
        MinerEntry(f"sybil_{i:03d}", stake=100.0 / m_sybils, forecast=fx["innov2_fc"])
        for i in range(m_sybils)
    ]
    res = settle_compute_pool(
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


@pytest.mark.xfail(
    strict=True,
    reason=(
        "BROKEN: pofs_shadow.py has no commit-reveal timelock, submission ordering, or clone "
        "deduplication. An exact last-moment copy of a skilled victim steals 50% of P_comp in a "
        "2-miner pool, steals ~20% while more than halving the victim's payout in a 3-miner pool, "
        "and earns positive reward even as a small-stake sniper."
    ),
)
def test_last_moment_exact_copy_of_honest_miner_blocked() -> None:
    """Defense should ensure a last-moment exact copier earns 0 and cannot harm the victim."""
    fx = _make_synoptic_fixture()

    baseline_two_honest = settle_compute_pool(
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
    res_two = settle_compute_pool(
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
    res_three = settle_compute_pool(
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

    res = settle_compute_pool(
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


@pytest.mark.xfail(
    strict=True,
    reason=(
        "BROKEN: settle_compute_pool has no calibration or spread-skill gate. A severely "
        "under-dispersed, sharp forecast (spread_skill_ratio < 0.15, failing the [0.85, 1.15] "
        "gate band by >5x) still collects large compute-pool payouts whenever its mean is "
        "accurate relative to the baseline or rivals (mirroring real Gate 1 GEFS)."
    ),
)
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

    res_pair = settle_compute_pool(
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
    res_solo = settle_compute_pool(
        [MinerEntry("sharp_uncalibrated", stake=100.0, forecast=sharp_better_mean)],
        pool_amount=POOL_AMOUNT,
        params=DEFAULT_PARAMS,
        baseline_forecast=fx["gefs_fc"],
        obs=fx["obs"],
        tail_threshold=fx["tail_thr"],
    )
    assert _by_id(res_pair)["sharp_uncalibrated"].reward == 0.0
    assert _by_id(res_solo)["sharp_uncalibrated"].reward == 0.0
