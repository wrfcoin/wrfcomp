# PoFS Shadow Payout Ledger (`src/wrfcomp/pofs_shadow.py`)

`src/wrfcomp/pofs_shadow.py` is a paper-only, deterministic NumPy implementation of the Proof-of-Forecast-Skill
(PoFS) reward and slashing formulas drafted in `docs/TOKENOMICS.md`. It performs no network I/O, touches no
wallet or chain, and modifies no numbers in `docs/TOKENOMICS.md`.

## What It Computes

- **Parameters (`ShadowLedgerParams`):** Frozen dataclass holding the `docs/TOKENOMICS.md` defaults (`55/35/10`
  emission split, `w_delta = 0.90`, `twcrps_weight = 0.30`, `max_slash_cap = 0.10`, `stake_exponent = 0.75`,
  `protocol_fault_slash = 0.05`, `h3_cell_cap = 1`), with `alpha_slash` required and without a default.
- **Compute Pool (`settle_compute_pool`, Section 4):** For each valid staked miner, computes composite loss
  `L_i = (1 - twcrps_weight) * CRPS_fair(i) + twcrps_weight * twCRPS_p95(i)`, individual skill
  `CRPSS_i = 1 - L_i / L_base`, leave-one-out marginal contribution
  `Delta_CRPS_i = (L(M_{-i}) - L(M_all)) / L_base`, composite score
  `S_i = (1 - w_delta) * CRPSS_i + w_delta * Delta_CRPS_i`, concave stake-weighted rewards for `S_i > 0`,
  capped slashes `min(max_slash_cap, alpha_slash * |S_i|) * K_i` for `S_i < 0`, and flat `0.05 * K_i`
  slashes (with exclusion from `M_all`) for protocol faults. Accepts either raw forecast arrays (forming
  the exact stake-weighted mixture meta-ensemble) or precomputed pointwise loss arrays. Non-finite tail
  thresholds (`NaN`, `±inf`, or values overflowing `float64`) are refused with `ValueError`.
- **Scorecard Loader (`load_scorecard`, `miners_from_scorecards`, `settle_from_scorecards`):** Validates
  `wrfcomp-eval` JSON scorecards via `wrfcomp.harness.schemas.Scorecard`, checks matching truth/threshold
  hashes and evaluation period, aligns onto common variables and common scored leads, and feeds the compute
  pool in pointwise-loss mode (using dimensionless `L_i(v) / L_base(v)` ratios across multiple variables so
  Pascals, Kelvins, and m/s carry equal weight).
- **Data Provider Pool (`settle_data_pool`, Section 5):** Computes per-station weight
  `W_s = QoD_s * PoL_s * CellCap_s * HW_s * Impact_s` with `CellCap_s = min(1, C_max / N_cell)` using
  caller-supplied cell IDs (no `h3` dependency), and distributes `P_data` pro-rata.
- **Epoch Conservation & Determinism (`settle_epoch`):** Sorts participants by ID prior to floating-point
  reductions so identical inputs produce bitwise-identical outputs regardless of input ordering, and reports
  explicit `unpaid_remainder` whenever no participant earns a positive weight so all pool totals conserve.

---

## Worked Example: Real Gate 1 Scorecards (January & July 2024)

Using the committed Gate 1 scorecards in `experiments/` (`climatology`, `persistence`, `noaa-gefs` for
`2024-01` and `2024-07`), we treat all three models as equal-stake compute miners (`K = 100` each) against
`climatology` as the locked baseline `L_base`, under `docs/TOKENOMICS.md` defaults (`P_comp = 550` of a
`1,000` epoch emission, `w_delta = 0.90`, `twcrps_weight = 0.30`, `max_slash_cap = 0.10`, `K^0.75`, and
illustrative `alpha_slash = 0.50`). Because GEFS is 3-hourly (`leads_scored = [3, 6, 12, 24, 48]`) while
climatology and persistence include `1h`, all three are aligned on their common leads `[3, 6, 12, 24, 48]`
using exact `n_pairs`-weighted composite loss per variable.

### Combined 5-Variable Settlement (Equal-Weight Baseline-Normalized Across `msl`, `t2m`, `td2m`, `u10`, `v10`)

| Month | Miner | Gate 1 `passed_gate` | Norm. Loss | `CRPSS_i` | `Delta_CRPS_i` | Composite `S_i` | Reward (of 550) | Slash (of 100, `alpha=0.5`) |
| :--- | :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **2024-01** | `noaa-gefs` | `False` | `0.5062` | `+0.4938` | `+0.1358` | `+0.1716` | **550.00** | `0.00` |
| **2024-01** | `persistence` | `False` | `0.8272` | `+0.1728` | `-0.0247` | `-0.0049` | `0.00` | **0.25** |
| **2024-01** | `climatology` | `False` | `1.0000` | `+0.0000` | `-0.1111` | `-0.1000` | `0.00` | **5.00** |
| **2024-07** | `noaa-gefs` | `False` | `0.8755` | `+0.1245` | `+0.0447` | `+0.0527` | **550.00** | `0.00` |
| **2024-07** | `climatology` | `False` | `1.0000` | `+0.0000` | `-0.0175` | `-0.0158` | `0.00` | **0.79** |
| **2024-07** | `persistence` | `False` | `1.0194` | `-0.0194` | `-0.0272` | `-0.0264` | `0.00` | **1.32** |

### Per-Variable Settlement (`P_comp = 550`, `K = 100`, `alpha_slash = 0.50`, Common Leads `[3, 6, 12, 24, 48]`)

| Month | Var | `climatology` (Reward / Slash) | `persistence` (Reward / Slash) | `noaa-gefs` (Reward / Slash) |
| :--- | :--- | ---: | ---: | ---: |
| **2024-01** | `msl` | `0.00 / 8.12` | `20.66 / 0.00` | `529.34 / 0.00` |
| **2024-01** | `t2m` | `0.00 / 6.24` | `65.38 / 0.00` | `484.62 / 0.00` |
| **2024-01** | `td2m` | `0.00 / 8.37` | `163.14 / 0.00` | `386.86 / 0.00` |
| **2024-01** | `u10` | `0.00 / 1.04` | `0.00 / 3.53` | `550.00 / 0.00` |
| **2024-01** | `v10` | `0.00 / 1.23` | `0.00 / 3.58` | `550.00 / 0.00` |
| **2024-07** | `msl` | `0.00 / 6.39` | `93.73 / 0.00` | `456.27 / 0.00` |
| **2024-07** | `t2m` | **93.05 / 0.00** | `0.00 / 1.77` | `456.95 / 0.00` |
| **2024-07** | `td2m` | `0.00 / 0.92` | **550.00 / 0.00** | `0.00 / 1.57` |
| **2024-07** | `u10` | **360.70 / 0.00** | `0.00 / 5.25` | **189.30 / 0.00** |
| **2024-07** | `v10` | **96.84 / 0.00** | `0.00 / 4.51` | `453.16 / 0.00` |

### What This Reveals About the `TOKENOMICS.md` Formulas

- **An uncalibrated model takes 100% of the pool:** `noaa-gefs` failed Gate 1 in both January and July because
  its ensembles are severely under-dispersed (`spread_skill_ratio` `0.27–0.39` in Jan, `0.34–0.48` in Jul, vs.
  the `[0.85, 1.15]` gate band), and in July it loses to climatology on `td2m` (`-2.4%`) and `u10` (`-4.1%`).
  Because Section 4 has no calibration or promotion-gate check, `noaa-gefs` still collects the entire `550.00`
  combined compute pool in both months.
- **Persistence is slashed in July—and even in January when paired with GEFS:** In July, persistence loses to
  climatology on `t2m` (`-7.4%`), `u10` (`-27.8%`), and `v10` (`-19.0%`) on leads `3–48h`, giving combined
  `CRPSS = -0.0194`, `S = -0.0264`, and a `1.32%` slash (even though it wins `td2m` outright at `550.00`).
  In January, persistence has strong positive skill (`CRPSS = +0.1728`), yet because `noaa-gefs` is much
  stronger (`+0.4938`), removing persistence improves the 3-miner pool (`Delta_CRPS = -0.0247`), so at
  `w_delta = 0.90` its composite score turns slightly negative (`S = -0.0049`) and it is slashed `0.25%`.
- **Negative-skill and baseline-copy miners get paid on real July variables (Open Problem #10 in practice):**
  On July `u10`, `persistence` is so weak (`CRPSS = -0.2780`) that both `climatology` (`CRPSS = 0.0`,
  `reward = 360.70`) and `noaa-gefs` (`CRPSS = -0.0414`, `reward = 189.30`) earn positive `Delta_CRPS` and
  split the entire pool simply by diluting `persistence` in `M_all` (the same happens to `climatology` on July
  `t2m` and `v10`).
- **Scorecard-only limitation:** Scorecards hold scalar losses rather than raw forecast ensembles, so
  `settle_from_scorecards` runs in linear pointwise-loss mode and cannot credit cross-model ensemble diversity
  (`CRPS` reduction from mixing independent errors); production settlement must evaluate `M_all` and `M_{-i}`
  on forecast arrays.

---

## Open Problems Found in `docs/TOKENOMICS.md`

1. **`alpha_slash` has no default or range.**
   Section 4.3 defines `Slash_i = min(max_slash_cap, alpha_slash * |S_i|) * K_i`, but `alpha_slash` is omitted
   from Section 8's parameter table; `ShadowLedgerParams` therefore makes it a required argument with no default.
   **Proposal for owner:** Add `alpha_slash` to Section 8's parameter table with an explicit default (e.g., `0.50`) and safe governance range (e.g., `0.25 – 1.00`).

2. **Single-miner leave-one-out `M_{-i}` is undefined.**
   When only one non-faulty staked miner participates, removing it leaves an empty meta-ensemble `M_{-i}` with no
   defined loss; the ledger falls back to `L(M_{-i}) = L_base` so a lone skilled miner still earns positive `Delta_CRPS`.
   **Proposal for owner:** Define `L(M_{-i}) = L_base` in Section 4.2 whenever removing miner `i` leaves no remaining valid staked miners.

3. **Zero positive scores / zero total weight creates division by zero.**
   Sections 4.3 and 5 divide by `sum_{j: S_j > 0} (K_j^0.75 * S_j)` and `sum_k W_k` without specifying where
   unclaimed emissions go when every participant scores `<= 0`; the ledger reports the full pool as `unpaid_remainder`.
   **Proposal for owner:** Specify in Sections 4.3 and 5 that when no participant has positive score or weight, payouts are `0` and the full `unpaid_remainder` is routed to treasury or burned.

4. **Stake splitting can still game the `(1 - w_delta) * CRPSS_i` term because `K_min` is unset.**
   Splitting stake `K` across `m` Sybil wallets shrinks `Delta_CRPS_i` by `O(1/m)`, but `K_i^0.75` amplifies the
   constant `0.10 * CRPSS_i` term by `m^0.25`; without a concrete numeric `K_min`, unbounded splitting eventually increases share.
   **Proposal for owner:** Set a numeric `K_min` in Section 8 and make payout stake weighting linear in `K_i` (removing the `m^0.25` split gain) or require `w_delta = 1.0` for positive payouts.

5. **Meta-ensemble aggregation and scoring rule for `L(M_all)` are underspecified.**
   Section 4.2 does not state whether `M_all` is a probability mixture of ensembles or a stake-weighted mean of
   forecasts, nor how to extend unbiased "fair" CRPS to unequal weights; the ledger uses exact weighted mixture CRPS.
   **Proposal for owner:** Specify in Section 4.2 that `M_all` is the stake-weighted probability mixture of valid miners' ensembles scored by exact weighted empirical CRPS.

6. **Raw `K_i` weights `M_all` while `K_i^0.75` weights payouts.**
   Weighting the meta-ensemble linearly by `K_i` lets a large whale dominate `M_all` and suppress smaller rivals'
   marginal `Delta_CRPS_i` even though payout scaling uses concave `K_i^0.75`.
   **Proposal for owner:** Weight `M_all` by the same `K_i^0.75` weights (or cap each miner's share in `M_all`) so meta-ensemble influence matches payout scaling.

7. **No guard for `L_base <= 0`.**
   Sections 4.1 and 4.2 divide by `L_base` without handling a zero baseline loss (possible on constant or
   near-deterministic truth slices); the ledger rejects `baseline_loss <= 0`.
   **Proposal for owner:** Specify in Section 4.1 that any variable-lead slice with `L_base <= 0` is excluded from skill normalization (or settles with `unpaid_remainder = P_comp` if the whole epoch has `L_base <= 0`).

8. **Zero-QoD / zero-PoL stations in `N_cell` enable cell-dilution griefing.**
   Section 5 dilutes cell capacity by `C_max / N_cell` over all registered stations in the cell without excluding
   spoofed (`QoD = 0`) or unverified (`PoL = 0`) sensors; the ledger follows the literal count of registered stations per cell.
   **Proposal for owner:** Define `N_cell` in Section 5 as the count of stations in the cell with `QoD_s > 0` and `PoL_s > 0` (or above minimum quality/location floors).

9. **Slashed stake accounting is separate from emission pools.**
   Section 4.3 states slashes burn stake, while Section 3 accounts only for minted epoch emissions; the ledger
   reports `total_slashed` separately so `total_paid + unpaid_remainder` always conserves `P_comp` and `P_data`.
   **Proposal for owner:** State explicitly in Sections 3 and 4.3 that slashed stake is burned out of supply (never recycled into `P_comp`/`P_data`/`P_val`) and audited on a separate ledger line.

10. **A worse-than-baseline miner can earn rewards by diluting an even worse miner in `M_all`.**
    Because `M_all` includes all non-faulty staked miners without gating on `CRPSS_i > 0`, a slightly negative-skill
    miner can earn a large positive `Delta_CRPS_i` (and positive `S_i`) simply by diluting a heavily staked terrible forecast.
    **Proposal for owner:** Gate compute payouts (and `M_all` membership) on `CRPSS_i > 0`—and optionally passing the harness calibration/promotion gates—so baseline copies, negative-skill models, and severely uncalibrated ensembles cannot earn rewards.
