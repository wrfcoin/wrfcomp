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
