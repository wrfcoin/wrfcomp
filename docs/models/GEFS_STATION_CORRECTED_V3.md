# Station-corrected GEFS v3 — pressure-spread pre-registration

**Registered decision.** Keep the v2 forecast centre and member ordering. Change only the spread of mean sea-level pressure (`msl`), using the realized errors of previously issued v2 pressure forecasts. This is candidate **M = 3** in the experiment family. This document precedes implementation. The diagnosis used existing 2024 v2 forecasts. No v3 forecasts, scores or parameter sweeps were produced, and no 2023 confirmation scores were consulted.

## What failed in 2024

Under promotion rule v2 and harness 0.1.0, v2 passed Q2 and Q3 against both references. Q1 failed pressure spread–skill. Q4 failed pressure spread–skill and the dewpoint tail gate against GEFS. The owner limited v3 to pressure. The dewpoint failure remains relevant to the quarterly verdict.

I aligned stored v2 pressure members to the official hourly truth by station and valid time. The six-lead comparison with climatology uses finite truth and a finite station/valid-month threshold. For each scored point, let `e` be the v2 member mean minus truth in Pa and `V = (32/31) × sample_variance(members, ddof=1)` in Pa². The reported spread is `sqrt(mean(V))`, and RMSE is `sqrt(mean(e²))`. Pool points before taking either square root. The reconstruction matches the official Q1 and Q4 counts and ratios (539,006 at 1.2397 and 459,931 at 0.7818). Against GEFS the five shared leads give 1.2459 and 0.7807, respectively.

| Issue period and v2 pressure branch | Valid points | RMS spread (Pa) | RMSE (Pa) | Spread–skill |
|---|---:|---:|---:|---:|
| January, fewer than 10 eligible pairs | 60,974 | 319.67 | 189.32 | 1.689 |
| January, fitted | 122,132 | 164.03 | 167.82 | 0.977 |
| February, all fitted | 172,301 | 144.05 | 117.77 | 1.223 |
| March, all fitted | 183,599 | 122.17 | 111.60 | 1.095 |
| Q2, all fitted | 539,285 | 103.04 | 100.55 | 1.025 |
| Q3, all fitted | 544,726 | 79.81 | 80.36 | 0.993 |
| October, all fitted | 183,535 | 76.04 | 92.84 | 0.819 |
| November, all fitted | 177,126 | 94.19 | 128.05 | 0.736 |
| December through the last official issue, all fitted | 99,270 | 118.16 | 145.09 | 0.814 |

**January and February have different causes.** I reconstructed the v2 branch per station, lead and issue from its same-hour 60-day history, including the per-variable measurement offset. The base pressure RMS spread on January warm-up points was 79.92 Pa. Multiplying it by four gives 319.67 Pa against a centre RMSE of 189.32 Pa. This branch contributed +4.05 billion Pa² to the sum of `V − e²`; the fitted branch contributed −0.15 billion Pa². Thus the warm-up explains more than the entire January surplus and about 73% of the Q1 surplus. February had zero warm-up points. Its analog residual RMS was 149.98 Pa for selected observations older than 30 days and 128.08 Pa for the more recent selections. The 144.05 Pa fitted spread exceeded the contemporaneous 117.77 Pa RMSE.

**Q4 had mature fits throughout.** Every scored point had at least 31 eligible analogs. I reconstructed the selected analogs and their regression residuals with v2's tie order and variance formula. The mean age of the selected observations was 30.95, 30.30 and 30.57 days in October, November and December. The table separates dispersion within the historical fit from the error actually encountered at forecast time.

| Issue month | Fitted residual RMS, age ≤30 d (Pa) | Fitted residual RMS, age >30 d (Pa) | RMS error of v2 forecasts originally issued for the selected analog cases (Pa) | Current v2 RMSE (Pa) |
|---|---:|---:|---:|---:|
| February | 128.08 | 149.98 | 156.49 | 117.77 |
| October | 72.50 | 70.14 | 78.41 | 92.84 |
| November | 99.03 | 76.80 | 96.61 | 128.05 |
| December | 125.70 | 95.16 | 121.79 | 145.09 |

Analog summaries count each selection, including reuse across target forecasts. V2 estimates spread from residuals around a fit to those historical cases. Even the as-issued errors on those cases were smaller than current error in each Q4 month. In November and December the older part of the fitted residual pool also pulled dispersion down. Q4 spread–skill was 0.76–0.81 at every lead, and the pooled signed centre error was +9.68 Pa against RMSE of 119.57 Pa. The evidence supports a lagging estimate of error magnitude across leads. It does not establish a particular weather system or instrument as the cause. Changing the pressure mean would also be outside this design.

**Evidence paths.** Under the external data directory, the inputs were `candidates/gefs-station-corrected-v2/2024-MM.zarr`, `gate1/2024-MM_h00-12/gefs.zarr`, `truth/isd_conus_2024.nc` restricted to 2024, and `thresholds/tail_p95.nc` solely for reproducing the diagnostic scoring mask. Official quarter cards and the annual decision are in `promotion/gefs-station-corrected-v2-official/`. The +1 h base follows v2's observation-time interpolation. Recent/older means measurement age relative to the target issue. [V2 construction](GEFS_STATION_CORRECTED_V2.md) · [Locked protocol](../VERIFICATION_PROTOCOL.md)

## Frozen v3 rule

1. **Inputs and identity.** Use model ID `gefs-station-corrected`, version `3`, the same 31 member IDs, station grid, 00Z/12Z issues, and leads 1, 3, 6, 12, 24 and 48 h as v2. Generate the v2 core once for the requested year Y, including its +1 h construction, analog fit, warm-up and paired dewpoint cap. Use its published-precision float32 output as the starting forecast. Copy `t2m`, `td2m`, `u10` and `v10` verbatim, with their dimensions, order, units and encoding. Carry the v2 pressure branch classification from its eligible-pair count (warm for 0–9, fitted for ≥10). The correction below applies to `msl` alone.

2. **Keep a causal record of issued forecasts.** For each past issue I, station s and lead L, retain the *v2* pressure mean, its corrected variance V as defined above, and its original branch. Compute both moments in float64 from the actual float32 members. At issue T use only I < T with the same UTC issue hour and the same lead. Its verifying hourly label I+L must be ≤ T. Require finite official pressure, finite offset in [−15,+15] minutes and actual measurement time U = I+L+offset in **(T−30 days, T]**, within Y. Use each eligible (I,s,L) once. The 30-day window halves v2's 60-day analog window: the older half retained larger residuals in February and smaller ones in November–December. The shorter window and the use of as-issued errors are fixed before testing v3. No threshold file, future truth, v3 error or scorecard enters the calibration.

3. **Estimate one amplitude gain per lead, issue hour and branch at T.** Pool eligible records over the scheduled stations. Lead-specific calibration preserves the distinction between +1 h and +48 h (Q1 ratios 1.04 and 1.42). Separating branches prevents the January ×4 warm-up error from influencing fitted forecasts. Let N be the number of paired forecast points in the pool, S the number of stations in the full forecast grid (500 on the 2024 grid), A the sum of squared as-issued v2 centre errors, and B the sum of V. The calibration target is a pooled spread–skill of one. For fitted records use prior gain `g₀ = 1`: v2's already-calibrated Q2 and Q3 spreads were close to target. For warm records use `g₀ = 1/2`: the January pressure-specific RMSE/base-spread ratio was 189.32/79.92 = 2.37. Rounded to the nearest whole base-spread multiplier this gives 2, or half the v2 multiplier of 4. This is a 2024-informed cold-start choice and must be confirmed independently.

   If N = 0 or B = 0, use g = g₀. Otherwise use the positive square root

   ```text
   g = sqrt((N × (A / B) + S × g₀²) / (N + S))
   ```

   A/B is the empirical variance multiplier for the historical issued ensemble. Shrink it towards the prior by the equivalent of one full network issue, so a handful of early station observations does not determine the gain. S is a deterministic weight, not a claim of S independent weather events. Pool squared errors including the mean error because the gate denominator is RMSE. Accumulate in float64 in ascending issue time and station-ID order. Apply no other gain limits, seasonal factors, station weights or fitted parameters.

4. **Form members.** For each current station/lead, take its matching gain and v2 pressure mean m. For member j publish `float32(m + g × (v2_msl[j] − m))`. All operations before the final cast use float64. This preserves the v2 centre up to float32 rounding and preserves member ordering; a zero-spread ensemble remains at its centre. Values of the other four variables never pass through this transformation.

5. **Year isolation.** Start fresh at 00Z January 1 of Y. Parameterize v2's existing 2024 date guards, schedule and archive paths by Y without changing the numerical rules. Every base forecast, historical issue, verification label and measurement used as input must be within Y; retain v2's rejection of forecasts whose maximum valid time reaches the following year. At +1 h also exclude lookback measurements preceding January 1. Build in chronological order across months or reconstruct exactly the same same-year state. No state crosses the year boundary. For the 2024 validation run use the official schedule through December 17 at 12Z. For the untouched 2023 confirmation use its authorized schedule and **2023 inputs only**, with the identical formula, priors and windows. The owner/manager controls that evaluation.

## Acceptance and handoff

After GO, require exact array dtype, dimensions, coordinates and C-order value bytes for all four non-pressure variables against v2 on the same inputs. Cover a cold start, the transition into the fitted branch, and a month boundary. For pressure, independently check the gain and the preserved centre, reject future/invalid-offset verification pairs, test both endpoints of the 30-day window, and require monthwise replay to equal uninterrupted replay. Include a year-start control that changes preceding-year data and leaves every output unchanged.

Only the manager runs the locked promotion check. A 2024 pass remains provisional until the unchanged candidate also passes the untouched 2023 confirmation. In particular, the known Q4 `td2m` tail failure against GEFS is outside scope and will remain for identical inputs. No implementation, v3 stores, or promotion run is part of Step 1.
