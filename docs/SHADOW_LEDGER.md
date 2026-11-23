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
  the exact stake-weighted mixture meta-ensemble) or precomputed pointwise loss arrays.
- **Data Provider Pool (`settle_data_pool`, Section 5):** Computes per-station weight
  `W_s = QoD_s * PoL_s * CellCap_s * HW_s * Impact_s` with `CellCap_s = min(1, C_max / N_cell)` using
  caller-supplied cell IDs (no `h3` dependency), and distributes `P_data` pro-rata.
- **Epoch Conservation & Determinism (`settle_epoch`):** Sorts participants by ID prior to floating-point
  reductions so identical inputs produce bitwise-identical outputs regardless of input ordering, and reports
  explicit `unpaid_remainder` whenever no participant earns a positive weight so all pool totals conserve.

## Open Problems Found in `docs/TOKENOMICS.md`

1. **`alpha_slash` has no default or range.**
   Section 4.3 defines `Slash_i = min(max_slash_cap, alpha_slash * |S_i|) * K_i`, but `alpha_slash` is omitted
   from Section 8's parameter table; `ShadowLedgerParams` therefore makes it a required argument with no default.

2. **Single-miner leave-one-out `M_{-i}` is undefined.**
   When only one non-faulty staked miner participates, removing it leaves an empty meta-ensemble `M_{-i}` with no
   defined loss; the ledger falls back to `L(M_{-i}) = L_base` so a lone skilled miner still earns positive `Delta_CRPS`.

3. **Zero positive scores / zero total weight creates division by zero.**
   Sections 4.3 and 5 divide by `sum_{j: S_j > 0} (K_j^0.75 * S_j)` and `sum_k W_k` without specifying where
   unclaimed emissions go when every participant scores `<= 0`; the ledger reports the full pool as `unpaid_remainder`.

4. **Stake splitting can still game the `(1 - w_delta) * CRPSS_i` term because `K_min` is unset.**
   Splitting stake `K` across `m` Sybil wallets shrinks `Delta_CRPS_i` by `O(1/m)`, but `K_i^0.75` amplifies the
   constant `0.10 * CRPSS_i` term by `m^0.25`; without a concrete numeric `K_min`, unbounded splitting eventually increases share.

5. **Meta-ensemble aggregation and scoring rule for `L(M_all)` are underspecified.**
   Section 4.2 does not state whether `M_all` is a probability mixture of ensembles or a stake-weighted mean of
   forecasts, nor how to extend unbiased "fair" CRPS to unequal weights; the ledger uses exact weighted mixture CRPS.

6. **Raw `K_i` weights `M_all` while `K_i^0.75` weights payouts.**
   Weighting the meta-ensemble linearly by `K_i` lets a large whale dominate `M_all` and suppress smaller rivals'
   marginal `Delta_CRPS_i` even though payout scaling uses concave `K_i^0.75`.

7. **No guard for `L_base <= 0`.**
   Sections 4.1 and 4.2 divide by `L_base` without handling a zero baseline loss (possible on constant or
   near-deterministic truth slices); the ledger rejects `baseline_loss <= 0`.

8. **Zero-QoD / zero-PoL stations in `N_cell` enable cell-dilution griefing.**
   Section 5 dilutes cell capacity by `C_max / N_cell` over all registered stations in the cell without excluding
   spoofed (`QoD = 0`) or unverified (`PoL = 0`) sensors; the ledger follows the literal count of registered stations per cell.

9. **Slashed stake accounting is separate from emission pools.**
   Section 4.3 states slashes burn stake, while Section 3 accounts only for minted epoch emissions; the ledger
   reports `total_slashed` separately so `total_paid + unpaid_remainder` always conserves `P_comp` and `P_data`.

10. **A worse-than-baseline miner can earn rewards by diluting an even worse miner in `M_all`.**
    Because `M_all` includes all non-faulty staked miners without gating on `CRPSS_i > 0`, a slightly negative-skill
    miner can earn a large positive `Delta_CRPS_i` (and positive `S_i`) simply by diluting a heavily staked terrible forecast.
