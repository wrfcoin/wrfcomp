# Citizen-station QC trial, 8–21 July 2024 UTC

The two-region sample contains **663,160 station-time reports** from **344 stations** and 4,961,551 individual variable observations. Median reporting cadence was ten minutes in both regions. Official station density prevented the spatial median test throughout the trial. No citizen weight could exceed the contract’s 0.8 comparison threshold because the new-station tier prior is 0.5.

## Gate outcomes

A report is counted once in a column if any of its variables raised that flag. Percentages use all station-time reports in that region. A temporal flag includes either a step spike or a flatline. Gate flags can overlap. The spatial column records actual outlier decisions. Insufficient-buddy coverage is listed separately.

| Region | Stations | Reports | Variable observations | Static | Temporal | Spatial | Background | Too few buddies |
|---|---|---|---|---|---|---|---|---|
| Amarillo | 20 | 41,869 | 301,453 | 1 (<0.01%) | 12,586 (30.06%) | 0 (0.00%) | 5,500 (13.14%) | 40,403 (96.50%) |
| Seattle | 324 | 621,291 | 4,660,098 | 1,059 (0.17%) | 281,246 (45.27%) | 0 (0.00%) | 39,447 (6.35%) | 617,381 (99.37%) |

Static and temporal checks ran on every variable observation. The next table counts flags per observation. Spatial and background decision counts give the applicable denominators. A dash indicates a variable without comparable official or GEFS data.

| Region | Variable | Observations | Static | Temporal | Spatial decisions | GEFS decisions | GEFS flags / decisions |
|---|---|---|---|---|---|---|---|
| Amarillo | precip | 40,329 | 0 (0.00%) | 40 (0.10%) | — | — | — |
| Amarillo | station_pressure | 23,178 | 0 (0.00%) | 208 (0.90%) | — | — | — |
| Amarillo | t2m | 40,402 | 0 (0.00%) | 4,786 (11.85%) | 0 | 40,402 | 2,091 (5.18%) |
| Amarillo | td2m | 40,402 | 0 (0.00%) | 1,454 (3.60%) | 0 | 40,402 | 1,052 (2.60%) |
| Amarillo | u10 | 38,928 | 0 (0.00%) | 7,883 (20.25%) | 0 | 38,928 | 1,184 (3.04%) |
| Amarillo | v10 | 38,928 | 0 (0.00%) | 6,169 (15.85%) | 0 | 38,928 | 2,211 (5.68%) |
| Amarillo | wind_gust | 38,884 | 1 (<0.01%) | 3,340 (8.59%) | — | — | — |
| Amarillo | wind_speed | 40,402 | 0 (0.00%) | 7,116 (17.61%) | 0 | 40,402 | 661 (1.64%) |
| Seattle | precip | 600,291 | 0 (0.00%) | 188 (0.03%) | — | — | — |
| Seattle | station_pressure | 474,032 | 0 (0.00%) | 17,707 (3.74%) | — | — | — |
| Seattle | t2m | 609,835 | 13 (<0.01%) | 72,162 (11.83%) | 0 | 609,835 | 28,882 (4.74%) |
| Seattle | td2m | 608,838 | 13 (<0.01%) | 17,189 (2.82%) | 0 | 608,838 | 13,101 (2.15%) |
| Seattle | u10 | 575,524 | 2 (<0.01%) | 183,369 (31.86%) | 0 | 575,524 | 429 (0.07%) |
| Seattle | v10 | 575,524 | 17 (<0.01%) | 179,640 (31.21%) | 0 | 575,524 | 187 (0.03%) |
| Seattle | wind_gust | 602,152 | 1,027 (0.17%) | 126,195 (20.96%) | — | — | — |
| Seattle | wind_speed | 613,902 | 0 (0.00%) | 219,153 (35.70%) | 0 | 613,902 | 2,220 (0.36%) |

## Final weights and failure-mode screens

Each weight is the final QC weight for one observation with an available GEFS background. Gate 4 compares the bias-corrected innovation with the Huber thresholds. A background flag can down-weight or reject a report. The distribution includes hard rejects at zero. Pressure, gust and precipitation lack trial weights.

| Region | Weighted observations | Zero | Min | P10 | P25 | Median | P75 | P90 | Max | Above 0.8 |
|---|---|---|---|---|---|---|---|---|---|---|
| Amarillo | 199,062 | 27,445 (13.79%) | 0.000 | 0.000 | 0.123 | 0.214 | 0.274 | 0.324 | 0.500 | 0 |
| Seattle | 2,983,623 | 677,531 (22.71%) | 0.000 | 0.000 | 0.037 | 0.235 | 0.337 | 0.377 | 0.500 | 0 |

Counts below are distinct stations that met a screen at least once. Categories overlap. The daytime temperature screen requires at least 12 accepted day and 12 night updates and a daytime minus nighttime innovation EMA above 1 K. Copycat means the shipped seven-day official-reference test fired. Flatline means six identical reports for a variable, with zero precipitation exempt.

| Region | Daytime bias | Any flatline | Wind-speed flatline | Copycat | Indoor-screen eligible | Indoor-like |
|---|---|---|---|---|---|---|
| Amarillo | 18 | 19 | 18 | 0 | 8 | 0 |
| Seattle | 225 | 322 | 318 | 0 | 114 | 0 |

The production indoor correlation screen requires a buddy median and had zero evaluable stations. The separate descriptive indoor screen uses temperature pairs with the nearest official station within 25 km. In a trailing 24 hours it requires 12 distinct official measurements spanning at least 18 hours, an official range of at least 5 K, and a paired citizen range of at most 1 K. These thresholds describe the screen only. Independent measurements would be needed to confirm a sensor fault.

## Error against official stations

The requested RMSE for citizen weights above 0.8 is **undefined in both regions** (zero qualifying rows and zero qualifying stations). For context, the table gives unfiltered citizen-to-official RMSE and the RMSE between that official station and its nearest distinct official neighbour. Each comparison uses the same citizen report clock. The official-to-official column additionally requires the second official measurement. Counts are pairs, and an official measurement can be reused by multiple reports. These are descriptive comparisons of different locations and sampling frequencies.

| Region | Variable | Unit | Citizen pairs | Citizen RMSE | Official pairs | Official RMSE |
|---|---|---|---|---|---|---|
| Amarillo | t2m | K | 10,568 | 3.67 | 9,629 | 2.60 |
| Amarillo | td2m | K | 10,568 | 3.60 | 9,629 | 2.23 |
| Amarillo | u10 | m/s | 9,831 | 2.87 | 8,724 | 2.92 |
| Amarillo | v10 | m/s | 9,831 | 3.74 | 8,724 | 3.36 |
| Amarillo | wind_speed | m/s | 10,199 | 3.76 | 9,064 | 2.25 |
| Seattle | t2m | K | 164,190 | 3.92 | 158,333 | 3.65 |
| Seattle | td2m | K | 163,983 | 4.03 | 158,142 | 1.89 |
| Seattle | u10 | m/s | 140,019 | 1.82 | 121,535 | 2.56 |
| Seattle | v10 | m/s | 140,019 | 2.74 | 121,535 | 2.76 |
| Seattle | wind_speed | m/s | 149,962 | 6.39 | 130,621 | 2.00 |

## Timing and interpretation

The runner reads only the two M1 NetCDF files with the citizen loader. Each independent station-variable history runs in measurement-time order. Static consistency uses other variables from the same report. The official comparison uses the nearest of the 500 ISD stations and its latest finite, quality-filtered measurement at or before the citizen report, at most 15 minutes old. Measurement time is the hourly label plus the variable’s own valid offset. Reference errors and copycat residuals use raw values. Derived official wind speed requires simultaneous components. The archive records measurement and forecast initialization times. Actual delivery delays are unknown.

The background is the ensemble mean at the nearest official station, using available GEFS leads (3, 6, 12, 24, 48 hours) from 00Z and 12Z runs issued strictly before the report. Among eligible forecasts the runner selects the closest valid time within three hours. Ties prefer the shorter lead, then the earlier valid time. It does not interpolate. Temperature and dewpoint backgrounds and buddy temperatures are adjusted to citizen elevation at 6.5 K/km. Wind speed uses the ensemble mean of member speeds. Trial inputs use 2 K for thermal variables and 2 m/s for wind as both background and total uncertainty. These assumptions affect absolute weights. Official storm corroboration uses same-sign official innovations relative to their own prior GEFS forecast with zero assumed official bias.

Citizen pressure is station pressure. Its static elevation check and temporal check ran. Comparing it with official or GEFS sea-level pressure would mix quantities, so that comparison was skipped. Gust and hourly precipitation also lack matching fields in the reference stores.

The 25 km spatial test requires three distinct recent official buddies. Positions were supplied at 0.1° and elevations at 10 m. Horizontal rounding can shift a distance by up to about 8 km, enough to change an individual buddy’s inclusion. Neither region had even one citizen station with three official sites within 33 km, before considering temporal availability. The spatial outlier and buddy-correlation rates among eligible points are therefore undefined. No exact position was inferred.

## Thresholds to revisit

- **Amarillo.** Six-report flatlining flagged 7,116 (17.61%) wind-speed observations. Of those, 6,437 (90.46%) occurred at zero wind. A calm or quantized sensor can repeat a legitimate value. Compare longer-duration and calm-aware wind rules against independent wind measurements before changing the gate.

- **Seattle.** Six-report flatlining flagged 219,083 (35.69%) wind-speed observations. Of those, 207,095 (94.53%) occurred at zero wind. A calm or quantized sensor can repeat a legitimate value. Compare longer-duration and calm-aware wind rules against independent wind measurements before changing the gate.

The 0.5 citizen prior makes the 0.8 error comparison unreachable even for a perfect history. Evaluate a separately specified eligibility metric or a calibrated tier transition in a later trial. The zero spatial coverage calls for denser official observations before judging the 25 km or three-buddy settings. The solar counts warrant checking persistence of the day/night differential across independent weather regimes. All production thresholds remained fixed.

Fourteen summer days in two regions cannot establish general false-positive rates, long-term reliability, fraud, sensor siting or absolute citizen accuracy. Local weather, elevation mismatch, instrument precision and GEFS timing can all contribute to an innovation. The trial has no collocated independent truth for the citizen sites.

Reproduce from the repository with `uv run python scripts/run_qc_trial.py --region Amarillo` and `uv run python scripts/run_qc_trial.py --region Seattle`. The commands emit aggregate JSON.

## V2 replay on the same July sample

The replay covered the same 20 Amarillo and 324 Seattle stations, 663,160 station-time reports and 4,961,551 variable observations. It used the same official observations, GEFS runs, uncertainty assumptions and time-matching rules as v1, adding two days of official thermal history for the background correction. All rates below measure changes in QC decisions. The trial has no independent labels for faulty sensors.

### Spatial coverage

| Region | Reports | V1 decisions | V2 reports with a decision | V2 temperature coverage | V2 reports with outliers / decided |
|---|---|---|---|---|---|
| Amarillo | 41,869 | 0 | 17,903 (42.76%) | 17,826 (44.12%) | 5,423 (30.29%) |
| Seattle | 621,291 | 0 | 582,768 (93.80%) | 569,480 (93.38%) | 71,645 (12.29%) |

A report has a spatial decision when any variable has three distinct eligible neighbours within 25 km, including at least one official or a citizen in another rounded cell. Different variables in one report can have different coverage. Each neighbour contributes its latest measurement at or before the target, at most 15 minutes old. A failed recent citizen report supersedes any older good report. Citizen buddies pass static and temporal screening and the official-reference copycat screen. Self-matches are excluded. Official observations have twice the weight of citizens in the median and MAD. The storm override also needs an agreeing independent buddy. Tests cover a same-cell cluster, a disagreeing outsider, a genuine cross-cell group, an official buddy and changes in report order.

Station pressure is compared only with citizen station pressure adjusted to the target elevation. Gust and precipitation also use citizen spatial data. The trial uses MAD scale-floor inputs of 200 Pa for pressure, 1 mm for precipitation and 2 K or 2 m/s for the other variables. Pressure, gust and precipitation still have no GEFS trial weights.

### Wind flatlines

Each percentage uses the full number of observations for that variable and region. Calm means absolute wind at or below 0.5 m/s. For other repeats, v2 requires a continuous 75-minute plateau and checks for a nearby official value consistent with the reporting bin. The original nonzero frozen-anemometer regression still passes.

| Region | Variable | Observations | V1 flatline | V2 flatline |
|---|---|---|---|---|
| Amarillo | wind_speed | 40,402 | 7,116 (17.61%) | 110 (0.27%) |
| Amarillo | wind_gust | 38,884 | 3,340 (8.59%) | 695 (1.79%) |
| Amarillo | u10 | 38,928 | 7,883 (20.25%) | 0 (0.00%) |
| Amarillo | v10 | 38,928 | 6,168 (15.84%) | 2 (0.01%) |
| Seattle | wind_speed | 613,902 | 219,083 (35.69%) | 2,839 (0.46%) |
| Seattle | wind_gust | 602,152 | 126,190 (20.96%) | 2,170 (0.36%) |
| Seattle | u10 | 575,524 | 183,331 (31.85%) | 0 (0.00%) |
| Seattle | v10 | 575,524 | 179,591 (31.20%) | 6 (0.00%) |

### Corrected daytime temperature bias

The screen still requires at least 12 accepted daytime and 12 nighttime reports and a day-minus-night EMA above 1 K at least once. V2 subtracts the nearest official site's expanding mean observation-minus-GEFS innovation for the same UTC measurement hour, using only earlier official measurement times within the 150 km radius. A report at the exact target timestamp cannot supply its own correction.

| Region | Stations | V1 screen | V2 screen | Temperature reports with past official correction |
|---|---|---|---|---|
| Amarillo | 20 | 18 | 18 | 40,402 (100.00%) |
| Seattle | 324 | 225 | 228 | 609,548 (99.95%) |

The corrected screen remained common in Amarillo and became slightly more common in Seattle. The replay gives no evidence that subtracting regional model bias alone identifies unshielded instruments. A single official site, UTC-hour sampling, local exposure and elevation can all affect the residual. Other QC decisions also changed between versions.

### Verified temperature tier and paired errors

Seven completed UTC days of independent temperature checks can raise the temperature prior from 0.5 to 1. Each day needs six distinct official measurements, at least 84 across the week, stable daily signed bias, and no disqualifying correlation or copycat evidence. The complete rule is in `docs/ASSIMILATION_SPEC.md`. Eligibility is determined from past history and can be revoked. End-of-trial counts use the last temperature report on July 21, with a final check for station-wide copycat evidence.

For stations qualified at the end, RMSE uses only report times when the credential was active and both official comparators had a measurement available. Both RMSE columns use exactly the same timestamps and count. They compare raw temperatures in K. V1 had no earned tier.

| Region | Ever qualified | Qualified at end / all stations | Matched reports | Citizen–official RMSE | Official–official RMSE | V2 max final weight |
|---|---|---|---|---|---|---|
| Amarillo | 2 | 2/20 | 391 | 1.08 K | 2.79 K | 0.686 |
| Seattle | 68 | 55/324 | 12,243 | 1.92 K | 4.07 K | 0.729 |

No individual final weight exceeded 0.8 in either region. A prior of 1 makes that weight possible, but the other factors in the weight still apply. The eligibility screen selects for agreement with the nearest official station. The official-to-official pair can have a different separation or terrain. Lower error in the selected citizen cohort therefore cannot establish higher instrument accuracy.

### Remaining limits and options

- A different rounded cell is a minimum independence check. It cannot establish independent ownership or exposure. Position rounding still allows roughly 8 km of distance uncertainty at the 25 km cutoff. The official-only three-buddy coverage remains zero in both regions.
- The calm exemption removes the original zero-wind false-alarm mechanism, but also removes flatline evidence for instruments genuinely stuck at zero. A future labeled trial could test requiring nearby sustained non-calm wind to flag that case. V1 found 6,437 zero-speed flatlines in Amarillo and 207,095 in Seattle.
- At roughly ten-minute cadence the 0.9 low-correlation decay compounds quickly. The overall median final weight moved from 0.214 to 0.226 in Amarillo and 0.235 to 0.054 in Seattle. Compare elapsed-time decay with report-count decay on independent periods before adjusting it.
- The same 14 summer days motivated and evaluated these changes. The absence of copycat hits or indoor-like hits in the independent temperature-range screen does not establish sensitivity. Verify the solar threshold, reporting-bin assumption and tier transitions on other seasons, regions and independently inspected sites before using them for rewards.

## Q4 replay after the red-team fixes

The two documented commands replayed the same 663,160 July reports. Each arrow compares the v2 aggregate with Q4. Flag columns count distinct station-time reports with at least one flagged variable. They overlap. The tier column counts stations qualified for temperature at trial end. The median uses the final weights for 199,062 eligible variable observations in Amarillo and 2,983,049 in Seattle.

| Region | Reports | Static | Temporal | Spatial | Background | Flatline | Copycat | End-tier stations | Median weight |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Amarillo | 41,869 | 1 → 1 | 5,796 → 5,905 | 5,423 → 7,643 | 3,421 → 3,424 | 4,989 → 5,147 | 0 → 0 | 2 → 1 | 0.2259 → 0.2002 |
| Seattle | 621,291 | 1,059 → 1,059 | 92,867 → 93,918 | 71,645 → 112,802 | 53,936 → 54,211 | 91,686 → 92,739 | 0 → 0 | 55 → 28 | 0.05378 → 0.0001416 |

Seattle’s median fell from 0.05378 to 0.0001416. Its spatial flag count rose by 41,157 reports. Sparse-buddy reference correlation, absolute error and official-anchored spatial consensus can reduce trust, but this combined replay does not isolate their contributions or measure false positives. Wind-speed flatline flags rose from 110 to 286 observations in Amarillo and from 2,839 to 3,931 in Seattle. No report triggered the copycat flag in either region.

**What changed.** The batch driver resolves rapid thermal steps using contemporaneous changes from previously usable, non-flatlined neighbours. One matching official or two citizens separated by at least 5 km from the target and one another can retain a step at Huber weight. Static failures and flatlines remain hard rejections. Geographic independence uses the supplied positions, with no attempt to refine rounded citizen locations. In the spatial check, two officials anchor the median and MAD; one official anchors them when citizen votes exceed its nominal weight. A strongly correlated near-copy now triggers at reference RMS error up to 0.04 background sigma. Wind plateaus can bridge report gaps up to 60 minutes, and calm repeats can flag when the reference differs by at least 5 m/s.

Temperature trust now tracks uncorrected day/night bias separately when official background history is absent. This evidence can reduce daytime weight but earns no verification credit. The verified tier requires at least 12 distinct comparisons in each solar regime and a weekly day-minus-night difference in signed reference error of at most 1 K. When the buddy correlation is undefined, thermal trust can use the official-reference correlation. The reliability factor also retains evidence of persistent absolute reference error or same-sign day and night bias beyond the existing one-sigma allowance, even when the EMA follows a drift.

**Replay coverage.** The prescribed trial uses the shared spatial, temporal and trust routines. It also has its own asynchronous buddy prefilter, continuity bookkeeping and storm-vote assembly. It filters spikes before nomination, supplies cell labels for independence and skips thermal background comparisons without an official correction. Consequently the July totals do not establish the batch driver's rapid-storm recovery, pairwise distance checks or rural solar behavior. Those paths were verified separately with the unchanged adversarial assertions and new storm controls. The 14-day sample has no independent labels for sensor faults or retained storm events.
