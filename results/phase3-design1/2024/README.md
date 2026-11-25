# Phase 3, design 1, 2024: scored once. Verdict: **citizen stations help** (small effect)

Design: `docs/PHASE3_DESIGN.md` (with Amendment 1). Builder commit `ec736eea`. Scored by the manager
with the locked harness 0.1.0 (`wrfcomp-eval`, paired `--forecast C --baseline B`), 2024 truth and the
period's thresholds. COVERED is the posted list (314 of 500 stations, sha256 `a75ab116...4955ee`),
scored through a copy of `configs/splits.yaml` that differs only in `stations_file`. 704 issues;
quarters 182 / 182 / 184 / 156.

Arms: A is v3, B is A plus the airport's own reading, C is B plus the QC-weighted citizen
neighbourhood. C against B decides.

**C against B, COVERED, skill in % (fair / ECDF), one-sided Diebold-Mariano p (C better):**

| cell | year | Q1 | Q2 | Q3 | Q4 |
|---|---|---|---|---|---|
| t2m 1 h | **+0.23 / +0.22, p=0.0016** | +0.24 | +0.08 | +0.10 | +0.51 |
| t2m 3 h | **+0.25 / +0.25, p<0.0001** | +0.35 | +0.15 | +0.08 | +0.43 |
| t2m 6 h | +0.07 / +0.07, p=0.053 | +0.17 | +0.00 | +0.01 | +0.09 |
| td2m 1 h | +0.16 / +0.16, p=0.028 | -0.01 | +0.09 | +0.44 | +0.14 |
| td2m 3 h | +0.12 / +0.11, p=0.015 | -0.03 | +0.07 | +0.41 | +0.03 |
| td2m 6 h | **+0.13 / +0.13, p=0.0024** | -0.00 | +0.18 | +0.35 | +0.02 |

On ALL (500 stations) the same direction, about two thirds of the size (t2m 3 h: +0.16 %, p=0.0009).

**Rule of section 2:**
1. Year p below 0.05/6 = 0.0083: t2m 1 h, t2m 3 h, td2m 6 h.
2. Skill above zero under both estimators for the year and in at least 3 of 4 quarters: t2m 1 h (4/4), t2m 3 h (4/4), td2m 6 h (3/4).
3. No harm: no cell on COVERED or ALL has a harm p below 0.05; year `twcrps_p95` of C is at or below B's for both variables on both sets (fair and ECDF); the builder's read-back confirmed byte identity with v3 outside the six cells.

Copycat-flagged share of the neighbourhood weight (t2m): 0.13 %.

**Size, in plain words:** citizen stations improve the forecast by about 0.1 to 0.5 % at these airports. That is real at this sample size, but small.

**B against A (official data, not a citizen result):** adding the airport's own current reading to v3 improves it by 2 to 17 % (t2m 3 h +10.3 %, td2m 3 h +16.3 % on COVERED), p<0.0001 in every cell and every quarter, no tail harm. Per the design, this is a v4 candidate for the model lane.

**Named for the 2023 confirmation (before any 2023 citizen data is fetched):** design frozen at `ec736eeac4d13324318cd8285ad0778c44d8942a`; cells **t2m 1 h, t2m 3 h, td2m 6 h**; confirmed if at least one meets the same three conditions on 2023 with p below 0.05/3 = 0.0167.

## Files

- `verdict.json`: the verdict record (`help`, the three help cells, no harm cells, alpha per cell).
- `<X>_vs_<Y>_<SET>_<PERIOD>.json`: the 30 scorecards, for X against Y in {C|B, C|A, B|A}, SET in
  {COVERED, ALL}, PERIOD in {Y (the year), Q1 to Q4}.
