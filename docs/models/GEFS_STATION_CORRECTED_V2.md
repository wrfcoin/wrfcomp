# Station-corrected GEFS v2 — pre-registration

**Registration only.** This is the second candidate in the experiment family (M = 2).
The model below was specified after diagnosing v1 and before producing or scoring
any v2 forecast. The owner retains the locked harness, both reference comparisons
and the requirement to pass all gates in at least three of four calendar quarters.
A maintainer runs the official check. January–December 2024 supplied diagnostic
evidence and therefore cannot serve as an independent confirmation of this design.

## What failed in v1

The locked 0.1.0 results for the twelve 2024 periods rejected v1 every month.
The missing 1 h lead affected all twelve comparisons with climatology, including
February, April and May, when the numerical gates passed. Separately, u10 lost the
upper-tail gate to climatology in March, July, August, October and November; v10
lost in January, March, June, November and December. Both wind components beat
raw GEFS on tail loss in every month. July through December, dewpoint lost to
at least one reference each month.

**The immediate tail problem is excess probability or magnitude above p95 on
non-events.** For each valid station/issue/lead, clip members and observation from
below at the station and valid-month training p95. Compute the fair ensemble CRPS
there. Split its sum by observed exceedance, dividing both sums by the total
valid-pair count. The two columns below therefore add to the all-lead tail loss.
Exceedance frequency is the fraction of members above p95, averaged on the same
mask. Wind thresholds apply to the signed component. These comparisons use the
five v1 leads.

| Period, variable | Forecast | Event contribution | Non-event contribution | Forecast exceedance | Observed exceedance |
| --- | --- | ---: | ---: | ---: | ---: |
| July u10 (m/s) | v1 | 0.03324 | 0.01253 | 5.81% | 3.85% |
| | climatology | 0.04302 | 0.00170 | 4.16% | 3.85% |
| November v10 (m/s) | v1 | 0.03429 | 0.02209 | 6.75% | 4.01% |
| | climatology | 0.05340 | 0.00063 | 3.83% | 4.01% |
| October td2m (K) | v1 | 0.00894 | 0.01216 | 2.86% | 1.10% |
| | raw GEFS | 0.01094 | 0.00108 | 0.57% | 1.10% |

In October v1 reduced the fraction of observed dewpoint exceedances above the
warmest member from 74.7% to 18.1%. It gained 0.00200 K on events but paid an
extra 0.01108 K on non-events. A blanket increase in upper-tail spread would
target the wrong term. V1 shifts a raw mean with a single local bias and scales
its member anomalies with a single local factor, regardless of the forecast
level. The decomposition establishes where the loss arose; it does not identify
a unique cause such as residual skewness or the dewpoint cap.

Grouping by the **forecast** rather than the verifying observation also shows a
level-dependent mean error. Where the raw GEFS mean exceeded p95, v1 still
overpredicted July u10 by 0.93 m/s (5,142 pairs) and November v10 by 1.32 m/s
(7,784 pairs); on the remaining pairs its biases were −0.02 and −0.18 m/s.
For October dewpoint, v1 moved the bulk bias from −1.57 K to +0.01 K while the
bias on the 743 high-forecast cases increased from +1.20 to +1.42 K. This is
the evidence for conditioning the correction on the current forecast level.

**Temperature spread has a separate failure mode.** The following are pooled
v1 t2m ensemble-mean errors and RMS spreads, including the harness's 32/31
finite-member variance correction. “At cap” means v1 member SD divided by raw
member SD exceeded 5.999 (a diagnostic tolerance for float32 output).

| Period / subset | Valid pairs | Mean error (K) | RMSE (K) | RMS spread (K) | Spread-skill |
| --- | ---: | ---: | ---: | ---: | ---: |
| January 1–10 init | 49,656 | +0.17 | 2.39 | 0.58 | 0.245 |
| January 11–31 init | 103,993 | +0.01 | 2.23 | 2.01 | 0.901 |
| October all | 153,212 | +0.15 | 2.03 | 1.69 | 0.832 |
| October at cap | 14,638 | +0.10 | 3.07 | 2.13 | 0.694 |
| October below cap | 138,574 | +0.16 | 1.89 | 1.64 | 0.867 |
| December all | 82,994 | +0.09 | 2.25 | 1.89 | 0.839 |
| December at cap | 14,105 | +0.62 | 3.18 | 2.11 | 0.663 |
| December below cap | 68,889 | −0.02 | 2.01 | 1.84 | 0.916 |

The at-cap points accounted for 37% of October's and 64% of December's pooled
mean-squared-error minus variance deficit. Thus a multiplier limited to six
times a small raw spread leaves consequential high-error cases too tight; a
uniform monthly mean bias explains little of the overall RMSE. January also
has a cold start: v1 used its raw fallback at 259,570 of 775,000 variable-points.

Evidence can be reconstructed from the v1 official cards under
`promotion/gefs-station-corrected/YYYY-MM/promotion_*/`, the v1 candidate stores,
the matching Gate 1 GEFS and climatology stores, `truth/isd_conus_2024.nc` and
`thresholds/tail_p95.nc` under the external data directory. Thresholds are used
only for the diagnosis. [V1 model](GEFS_STATION_CORRECTED.md) · [Scoring rules](../VERIFICATION_PROTOCOL.md)

## One frozen forecast rule

Use model ID `gefs-station-corrected`, version `2`, 31 members with the GEFS member
IDs and leads 1, 3, 6, 12, 24 and 48 h. Apply the following rule separately to
each station, variable, lead and UTC init hour (00 or 12). All calculations use
float64 until the final float32 output. The inputs are existing 2024 forecasts
and QC-passing official hourly 2024 station values and their per-variable offsets.
Only complete finite raw ensembles are accepted. A missing source cycle is a build
error. No calendar-month reset occurs.

1. **Make a base ensemble at issue T.** At leads 3–48 h use the 31 stored GEFS
   station members, including the existing height correction. At 1 h inspect
   hourly labels T, T−1 h, T−2 h, T−3 h in that order for this station and
   variable. Use the first finite observation O with valid offset in [−15,+15]
   minutes and measurement time U in 2024 with U ≤ T. Let a = (T−U) in hours.
   For GEFS +3 h member G, the +1 h base member is
   `O + ((a + 1)/(a + 3)) * (G − O)`. These are the linear time-interpolation
   weights between the actual observation time and T+3 h. Without an eligible
   report, use G itself as the +1 h base member. Preserve member correspondence.
   Use these same rules at historical issue I with that issue's own cutoff,
   including the missing-report fallback. The +1 h verifying observation never
   supplies its own predictor.

2. **Collect causal pairs.** For a historical issue I < T at the same UTC hour,
   let x be its base ensemble mean at lead L and y the verifying observation at
   label I+L. Require I+L ≤ T, a finite y and valid offset, and measurement time
   in (T−60 days, T] and in 2024. Retain one pair per issue. Use the current
   base mean x₀ to order eligible pairs by ascending |x−x₀|, breaking ties by
   more recent issue first. Take the first 31, or all if fewer. The 60-day
   history doubles v1's 30-day window to provide roughly two same-hour cases
   per desired member; conditioning on a similar forecast level is intended to
   preserve changes in bias and tail shape across forecast regimes.

3. **Calibrate from observed errors.** With n ≥ 10 selected pairs compute the
   means x̄ and ȳ. Fit the local slope
   `b = clip(sum((x−x̄)(y−ȳ)) / sum((x−x̄)²), 0, 1)`; set b = 1 when the
   denominator is zero. Zero represents a local constant and one represents
   additive bias correction. The bounds prevent a noisy local fit from reversing
   or amplifying the raw forecast signal. Set the center to `ȳ + b(x₀−x̄)`.
   Save each signed residual `r = y−ȳ−b(x−x̄)`. These supply the distribution
   in physical units, including asymmetry and observed upper-tail misses. This
   decouples error spread from the current raw member SD and its sixfold ceiling.

   For ranks j = 0,…,30, sort residuals ascending, using earlier I first on ties,
   and take zero-based index `floor((j+0.5)*n/31)`. For n=31 this uses each
   actual residual exactly once; for 10–30 it gives each residual nearly equal
   weight. Center the 31 selected residuals by their arithmetic mean. Normalize
   their sample SD (divisor 30) to
   `sqrt(sum(r²)/(n−2) * (1+1/n))`, using the original n fitted residuals in
   the numerator. This accounts for two fitted mean parameters and uncertainty
   in a sample mean. If the selected SD is zero, use zero anomalies. Add the
   anomalies to the fitted center and assign them in ascending order of the
   current base members (lower integer member ID first on ties). This preserves
   raw ranks without imposing a Gaussian tail or promising an independent random
   ensemble. The locked evaluator retains both its fair and ECDF checks.

4. **Warm up from January 1 using 2024 only.** With 0–9 eligible pairs, retain
   the base mean and multiply each base member's deviation from it by **4**.
   The ten-pair floor is inherited from v1. Four is the rounded reciprocal of
   the January 1–10 t2m spread-skill (0.245); it is a declared design choice
   informed by v1, applied uniformly to all variables and leads. No separate
   rate is fitted to another variable, lead or month. At ten qualifying pairs
   switch directly to rule 3. No December 2023 cycles or training truth are
   required.

5. **Finalize.** After all five variables are formed, cap td2m at the paired
   t2m for every member. Publish one monthly store at
   `candidates/gefs-station-corrected-v2/YYYY-MM.zarr`, with exactly the
   reference initialization schedule, ending December 17 at 12Z. The 1 h
   forecast is included against climatology; the GEFS comparison uses the
   five shared leads. The pre-registration fixes one specification for all
   twelve months. No v2 scores, parameter sweeps or model comparisons were
   produced in this step.
