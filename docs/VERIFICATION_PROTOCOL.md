# WRFComp Forecast Verification Protocol

**Target path:** `docs/VERIFICATION_PROTOCOL.md` in [`wrfcoin/wrfcomp`](https://github.com/wrfcoin)  
**Status:** Draft v0.1 (Immutable Grading Kernel Specification)

> `configs/splits.yaml` is the single source for split boundaries, initialisation hours, lead times,
> variables and gate thresholds. If this document and that file disagree, the file wins and this document
> is the bug.

---

## 1. Purpose & Kernel Immutability

This document defines the canonical, deterministic verification standard for `wrfcoin/wrfcomp`. Every forecasting model—whether produced by human researchers, an automated multi-provider LLM agent swarm, or future WRFCoin compute nodes—is graded exclusively through the `wrfcomp-eval` CLI.

**Immutability Policy:** Once Phase 1 passes unit tests and human sign-off, `src/wrfcomp/harness/` and `configs/splits.yaml` are **version-pinned and read-only**. No modeling, assimilation, or automated experiment PR may edit metric formulas, station lists, QC masks, or split boundaries. Any future protocol revision requires a major/minor version bump (`harness_version`) and full re-scoring of all baselines.

---

## 2. Temporal Splits & Anti-Leakage Rules

Weather fields are strongly autocorrelated across consecutive days. Random train/test splitting is strictly forbidden.

| Split | Initialization Window (`init_time`) | Embargo Buffer | Permitted Use |
| :--- | :--- | :--- | :--- |
| **Train** | `<= 2023-12-17 18:00 UTC` | 14 days before Val | Model training, QC calibration, and computing fixed 95th-percentile extreme thresholds |
| **Validation** | `2024-01-01 00:00 UTC` to `2024-12-17 18:00 UTC` | 14 days before Test | Architecture search, hyperparameter tuning, and bounded agentic experiment loops |
| **Sealed Test** | `>= 2025-01-01 00:00 UTC` | — | Final release verification only; **strictly off-limits** to automated research loops |

**Causal Input Cutoff:** For any forecast initialized at `init_time = T0`, models and assimilation pipelines may ingest **only** observations with measurement timestamp `obs_time <= T0`. Any look-ahead leakage disqualifies the submission.

---

## 3. Ground-Truth Sources & Spatiotemporal Matching

To prevent overfitting to legacy numerical weather prediction (NWP) reanalysis while still checking 3D atmospheric physics, `wrfcomp-eval` scores against two complementary truth tiers:

### Tier A — Primary Surface Truth (Direct Station Observations)
- **Sources:** [NOAA Global Hourly Integrated Surface Database (ISD)](https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database) and [ASOS/AWOS](https://www.ncei.noaa.gov/products/land-based-station/automated-surface-weather-observing-systems) stations listed in `configs/stations_conus.csv` (starting with 500 quality-vetted CONUS stations, expanding globally in v0.2).
- **Time Alignment:** Forecasts are matched to observations at `valid_time = init_time + lead_time` using nearest-neighbor matching within a strict **`±15 minute`** tolerance window.
- **Truth QC Filtering:** Only observations passing official NOAA quality flags are retained. Missing station readings are masked as `NaN` in the truth array and **never imputed**.
- **Scoring Mask:** A point `(init_time, lead_time, station_id)` of a variable is scored only if its truth value is finite **and** the tail threshold `tau_95(station, month)` of that variable is finite for the calendar month of the valid time. A threshold is missing exactly where the station has no training-split observation in that month, so the mask is fixed by the truth and the training data alone: no forecast can change it, and it is applied identically to the candidate and the baseline. The NaN rule of section 4 applies on the masked set. Every scorecard entry records `n_pairs` (points scored) and `n_masked_no_threshold` (points with valid truth that were left out for want of a threshold).
- **Grid-to-Station Interpolation (for gridded models):** Gridded forecasts evaluated at Tier A stations use deterministic bilinear interpolation in `(lat, lon)` with a standard environmental lapse-rate correction (`-6.5 K / km`) applied to `t2m` for the elevation difference between the model orography and the station's true altitude.

### Tier B — Secondary Upper-Air Truth (Gridded Reanalysis)
- **Source:** Google Research's cloud-optimized [ARCO-ERA5](https://github.com/google-research/arco-era5) (Zarr format) evaluated following [WeatherBench-X](https://github.com/google-research/weatherbenchX) conventions.
- **Spatial Weighting:** All global/regional gridded metrics apply cosine-latitude area weighting (`w(lat) = cos(lat) / mean(cos(lat))`) so polar grid cells do not dominate the score.

---

## 4. Canonical Variables, Lead Times & Submission Contract

### Core Evaluated Variables

| Variable | Standard Name | Units | Truth Tier | Notes |
| :--- | :--- | :--- | :---: | :--- |
| `t2m` | 2-meter air temperature | `K` | Tier A + B | Lapse-rate adjusted for gridded models |
| `u10`, `v10` | 10-meter wind components | `m s-1` | Tier A + B | Evaluated as components and derived speed `ws10` |
| `msl` | Mean sea-level pressure | `Pa` | Tier A + B | Primary synoptic dynamics check |
| `td2m` | 2-meter dewpoint temperature | `K` | Tier A | Must satisfy `td2m <= t2m` in truth |
| `z500`, `t850` | 500 hPa geopotential, 850 hPa temp | `m2 s-2`, `K` | Tier B | Upper-air physical consistency guardrail |

### Standard Lead Times & Schema Rules
- **Synoptic Initializations:** `00Z`, `06Z`, `12Z`, `18Z`. A scored run may cover a subset; the scorecard records which.
- **Lead Times (v1):** `1h, 3h, 6h, 12h, 24h, 48h`. Extended `72h, 120h, 168h` are optional and reported separately.
  A model that has no output at a lead (for example a 3-hourly ensemble at `1h`) is scored only at the leads it
  has, and the scorecard lists the leads that were scored. This is the one exception to the NaN rule below and
  it applies to whole leads only, never to single stations or dates.
- **Deferred:** precipitation (`tp6h`) and upper-air Tier B checks are not part of the v1 gate.
- **Required Dimensions:** `(init_time, lead_time, station_id, ens_member)` with `ens_member >= 2` (recommended `>= 32` for calibrated tails).
- **Zero-Cherry-Picking `NaN` Rule:** A candidate forecast MUST contain finite values for every `(init_time, lead_time, station_id)` where ground truth is valid. Any missing/NaN prediction at a valid truth point fails schema validation—preventing models or miners from skipping hard-to-predict stations or storm dates.

---

## 5. Metric Definitions (`scores` Reference Implementation)

All metrics are computed via the Australian Bureau of Meteorology's peer-reviewed [`scores`](https://github.com/nci/scores) package (`scores[fast]>=2.7.0`) to guarantee numerical reproducibility:

1. **Fair Ensemble CRPS (`crps`):** Computed via `scores.probability.crps_for_ensemble(fcst, obs, ensemble_member_dim="ens_member", method="fair")`. The `fair` estimator removes small-ensemble bias so a 16-member prototype and a 50-member operational ensemble compare on equal footing.
   - **Sampling assumption:** ensemble members must be exchangeable random draws from the forecast distribution. Fair CRPS is not a proper score for members placed deliberately: two fixed members score exactly `0` whenever the observation falls between them, and members fixed at inflated quantiles beat the true distribution while staying inside the calibration band (about 45 % with 2 members, 10 % with 10, 3 % with 32).
   - **ECDF cross-check:** every CRPS and tail CRPS is therefore also computed with `method="ecdf"` (`crps_ecdf`, `twcrps_p95_ecdf`, `loss_ecdf`, `crpss_vs_baseline_ecdf`), which has no such hole, and both are reported. The headline-skill gate must hold under both estimators, and an ensemble with fewer than `min_members_for_gate` members (20, set in `configs/splits.yaml`) is scored but cannot pass the gate.
   - **Known limitation (owner decision 2026-11-23: floor of 20 members and minimum skill of 5%, set in `configs/splits.yaml`):** the ECDF cross-check and the member floor do not fully close this. Members placed at plain Gaussian quantiles, with no extra information, still show skill against a *sampled* baseline of the same size and distribution under both estimators, while inside the calibration band, because a sampled ensemble pays sampling noise that a placed one does not. Measured in simulation (fair / ECDF skill): 8 members 12.3 % / 10.2 %, 10 members 9.7 % / 8.3 %, 20 members 4.9 % / 4.5 %, 32 members 3.1 % / 3.0 %, 50 members 2.0 % / 1.9 %. The tail gate and the Diebold-Mariano test use the fair estimator only.
2. **Extreme-Tail Threshold-Weighted CRPS (`twcrps_p95`):** Clips both `fcst` and `obs` below a station-and-month-specific 95th-percentile threshold `tau_95(station, month)` **precomputed strictly from the `<= 2023` training archive**, then applies fair CRPS. This focuses 30% of the headline loss on heatwaves, wind gusts, and heavy precipitation.
3. **Composite Proper Loss & Skill Score (`crpss_vs_baseline`):**
   ```text
   L(model) = 0.70 * crps + 0.30 * twcrps_p95
   crpss_vs_baseline = 1 - (L(candidate) / L(baseline))
   ```
   Every candidate is normalized against the locked operational baseline (e.g., Earth2Studio `AIFS` / `FCN3` or NOAA GFS/HRRR) evaluated on the exact same station-time mask.
4. **Ensemble Calibration (`spread_skill_ratio` & Rank Histogram):**
   - `spread_skill_ratio = sqrt((M + 1) / M) * sqrt(mean(ens_var)) / sqrt(mean((ens_mean - obs)^2))` (ideal ≈ `1.0`),
     where `M` is the number of ensemble members and `ens_var` is the unbiased (ddof = 1) variance. The
     `sqrt((M + 1) / M)` factor is the standard finite-ensemble correction, so a calibrated ensemble scores
     `1.0` whatever its size.
   - Diagnostic reliability verified via [`scores.probability.rank_histogram`](https://scores.readthedocs.io/en/stable/tutorials/Rank_Histogram.html) (which handles zero-precipitation ties cleanly).

---

## 6. Diebold-Mariano Statistical Promotion Gate

Because weather variance shifts dramatically from week to week, raw score improvements can easily be random luck. Every candidate PR must pass a four-part promotion gate evaluated on the 2024 validation split:

| Gate Criterion | Requirement | Why It Exists |
| :--- | :--- | :--- |
| **1. Headline Skill** | `crpss_vs_baseline` and `crpss_vs_baseline_ecdf` both above `min_crpss_vs_baseline` (0.05) across primary Tier A variables | Confirms candidate beats the locked operational baseline under both CRPS estimators |
| **2. Extreme-Tail Safety** | `twcrps_p95 <= baseline_twcrps_p95` (no tail degradation) | Prevents models that boost average scores by blurring storms |
| **3. Calibration Band** | `0.85 <= spread_skill_ratio <= 1.15` | Blocks overconfident (collapsed) or wildly over-dispersed ensembles |
| **4. Statistical Significance** | One-sided Diebold-Mariano `dm_pvalue < 0.05` (with horizon lag `h = lead_time / init_step`) | Accounts for autocorrelation across overlapping multi-day forecasts (`scores.statistical_tests.diebold_mariano`) |

**Eligibility:** a candidate needs at least `min_members_for_gate` ensemble members (20, in `configs/splits.yaml`)
to pass; smaller ensembles are scored and reported only. The scorecard records this as the `min_members` gate.

**Where the gates are judged:** each gate is evaluated per variable on its all-leads entry
(every scored lead pooled), not on each lead separately. Per-lead values are reported in the scorecard.
A Diebold-Mariano series with a missing init time inside it, or unevenly spaced init times, is undefined and
does not pass.

**Diebold-Mariano method:** the test uses the Hering-Genton ("HG") statistic with Student's t, because under
autocorrelated score differences it held the nominal 5 % false-pass rate in simulation (4.8–6.3 %) where the
Harvey-Leybourne-Newbold form at `h = 1` gave 15–31 %; with HG the lag `h` is a floor on the modelled
autocorrelation and does not change the result on a long series (details in `wrfcomp.harness.metrics`).

**Multiple-Testing Guard (Agentic Research Loops):** When an automated swarm tests `M` candidate architectures on the 2024 validation split within a single research sprint, apply a **Bonferroni-adjusted threshold (`alpha = 0.05 / M`)** or require the winning candidate to hold its gain across at least **3 of 4 calendar quarters** of 2024 before promoting it to a release candidate.

---

## 7. Deterministic CLI & Scorecard Provenance

Every official benchmark run is executed via `wrfcomp-eval` and committed as an immutable JSON artifact under `experiments/`:

```bash
uv run wrfcomp-eval \
  --forecast outputs/candidate_2024.zarr \
  --truth data/truth/isd_conus_2024.nc \
  --baseline outputs/aifs_baseline_2024.zarr \
  --model-id wrfcomp_healda_sda_v1 \
  --model-version $(git rev-parse --short HEAD) \
  --out experiments/wrfcomp_healda_sda_v1_2024.json
```

Each JSON scorecard records:
- `harness_version` (exact `wrfcomp` package + `scores` version)
- `truth_sha256` and `forecast_sha256` (content hashes guaranteeing reproducible inputs)
- `commit_sha256` (optional pre-window `drand` timelock commitment hash, bridging directly into `docs/TOKENOMICS.md`)
- Full per-variable, per-lead-time metric dictionaries and the boolean `passed_gate` verdict.
