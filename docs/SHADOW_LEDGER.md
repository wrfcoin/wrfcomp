# PoFS Shadow Payout Ledger (`src/wrfcomp/pofs_shadow.py`)

`src/wrfcomp/pofs_shadow.py` is a paper-only, deterministic NumPy implementation of the Proof-of-Forecast-Skill
(PoFS) reward and slashing formulas drafted in `docs/TOKENOMICS.md`. It computes paper rewards and slashes locally. No funds move.

## Compute settlement after the 2026-11-24 owner ruling

All amounts below are paper units. The recorder supplies an epoch start, an independently identified
NOAA GEFS forecast digest, the stake roster and a separate mapping from wallet to its first accepted
commitment (digest, UTC receipt time). The recorder is a trusted input to the simulation. Settlement
does not authenticate timestamps, hold keys or implement drand. A batch commitment covers the whole
forecast and must arrive strictly before the **first** initialization. A receipt at that instant is late.

The single-field array adapter hashes the domain tag, two little-endian uint64 dimensions and C-order
little-endian float64 member values using SHA-256. Both IEEE-754 signs of zero encode as positive zero. The scorecard adapter uses the locked evaluator's
forecast content hash. Supply evaluator-produced scorecards from the same epoch and the trusted GEFS
reference. Arbitrary self-reported metrics cannot establish forecast provenance. A standalone loss
vector or a scalar baseline provides no paired skill, calibration or commitment evidence.

Admission requires a matching reveal, timely receipt and minimum stake. For a full epoch each primary
variable must pass the configured GEFS comparison on matching leads and coverage: fair and ECDF skill
strictly above the minimum, tail loss at most GEFS tail loss, significant one-sided DM, minimum members,
and calibrated spread. A single-field array experiment supplies one observation per dated issue at a
single lead, the actual GEFS array and a tail threshold. It computes the paired losses, finite-member
spread ratio and locked DM test directly. A scalar loss cannot replace that GEFS input. A partial
scorecard variable selection can report diagnostics but cannot authorize payment.

A wallet may represent a forecast only after its timely reveal matches its commitment, its stake
meets the floor and it passes the GEFS gates. An invalid or sealed scorecard provides no reveal
digest. A committed wallet that submits one incurs the maximum slash. Among eligible wallets,
receipt time determines priority, with wallet ID breaking exact ties. Subsequent clones earn zero.

For raw ensembles, each member in a row has mass 1/M. The ledger takes the exact 1-Wasserstein
distance between the two empirical distributions at each observation, then averages over rows.
This integrates the absolute difference between their quantile functions even when the member counts
differ. The field scale is the larger of GEFS RMS within-row spread and the standard deviation of
observations (both population SD). At zero scale only zero distribution distance qualifies. Changing
units rescales the distance and cutoff together. Scorecards support exact forecast-content deduplication
only; they contain no arrays for a distance test.

Eligible representatives each contribute one equally weighted forecast to the meta-ensemble. The excluded miner's own diagnostic
downside may be computed by adding it to this fixed pool; its presence never changes an eligible
miner's marginal score. The scorecard approximation uses the mean of normalized losses across models;
it cannot measure physical ensemble diversity.

Eligible wallets with positive composite score share the compute pool in proportion to stake times
score at the default exponent. For a fixed forecast and total stake K, at most one of m copies can earn.
If its stake is k ≤ K and meets the floor, the set of distinct forecasts and their scores is unchanged.
Its share is w/(w+B), with competing weight B fixed and w proportional to k raised to the exponent.
This is nondecreasing in k. If the representative misses the floor its reward is zero. This establishes
the splitting bound at every m, including the tested range 1–100. It compares the same forecast and
competing submissions. New forecasting information is a different comparison.

A committed forecast withheld, corrupted or mismatched loses the maximum fraction of stake. An explicit
protocol fault also incurs that slash, even without a receipt or payload. An honest
negative score loses at most that fraction. Missing historical receipt evidence produces no reward and
does not establish a slashable commitment. Below-floor stakes, late matching reveals and duplicate
reveals earn zero. Reward returns to `unpaid_remainder` when no wallet earns. Slashes are accounted for
separately from emissions. The station and validator pool rules are unchanged.

### Residual risks

A distance cutoff can accept an independent but similar forecast as a clone and can miss a copy just
above the cutoff. Equal mixture weight per representative makes undetected copies consequential even
at minimum stake. Scorecard-only settlement cannot distinguish near-copies with different hashes:
a one-bit numerical change can change the digest. Keeping forecasts confidential until init time
through pre-init timelock encryption is the intended defense against live copying; the paper recorder
does not implement encryption.

The locked spread-skill gate pools squared errors and ensemble variances across observations for
each variable. A forecast can meet the global band while remaining overconfident during extreme events
if it concentrates spread in a different regime. Passing the aggregate band does not certify
calibration conditional on weather regime or threshold exceedance.

### Shadow defaults

The owner may change these research defaults. Ledger settings are centralized in
`ShadowLedgerParams`; protocol gates come directly from the locked configuration.

| Setting | Default | Reason |
| --- | ---: | --- |
| Minimum wallet stake | 10 paper units | One tenth of the illustrative 100-unit stake; dust wallets cannot participate. |
| Stake exponent | 1 | Linear stake weight gives each unit equal payout leverage. Exponents below 1 are rejected. |
| Raw clone distance tolerance | 0.05 times field scale | A single member carries about 3% of a 31-member empirical distribution. A 5% allowance covers small distribution changes from a missing member or numerical jitter. This is a shadow policy choice, not a universal similarity threshold. |
| Maximum slash | 10% | Same ceiling for the worst honest loss and a failed reveal. |
| Negative-score multiplier | Caller-supplied | The replay uses 0.50; this rate has no owner-set default. |
| Compute/data/validator shares | 55% / 35% / 10% | Existing shadow emission split. |
| Marginal score weight / tail loss weight | 0.90 / 0.30 | Existing uniqueness emphasis and proper-loss blend. Eligibility always uses the locked harness loss weights. |
| Default cell capacity | 1 | Existing rural station cap. |
| Protocol gate | Skill > 0.05, DM p < 0.05, spread-skill 0.85–1.15 inclusive, ≥20 members, no worse tail | Uses the shared evaluator gate and its configured parameters. |

### Gate 1 reference-only replay

All 48 committed reference cards (12 months and four quarters, three models per period) were replayed
using `settle_from_scorecards(cards, baseline_key="noaa-gefs", pool_amount=550,
params=ShadowLedgerParams(alpha_slash=0.5), stakes=100)`. Each period is a separate illustration;
quarters overlap the monthly replays. There are no recorder receipts in these historical artifacts.
Climatology and persistence do not have passing scorecards against GEFS in this inventory. GEFS is
the benchmark and has zero individual skill against itself.

| Every replay period | Reward | Slash |
| --- | ---: | ---: |
| NOAA GEFS | 0 | 0 |
| Climatology | 0 | 0 |
| Persistence | 0 | 0 |
| Unpaid compute pool | 550 | |

Diagnostic GEFS-normalized persistence skill was −0.81864 in January and −0.21314 in July. The
historical tables below use a different denominator and settlement policy.

## Historical pre-S3 comparison (superseded)

The following tables and observations describe the original experiment against climatology. They are
retained to show why the owner changed the rules; they are not outputs of current settlement.

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

## Historical design questions identified in `docs/TOKENOMICS.md`

The S3 rulings above resolve the payout eligibility, commit/reveal and stake-splitting questions.

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
