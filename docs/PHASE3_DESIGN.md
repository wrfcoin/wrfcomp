# Phase 3 design: do citizen stations improve the forecast?

**Status: pre-registration, step 1 (lane P3). Design and small measurements only.** No bulk download, no
model code, no forecast and no score was produced. Nothing here was chosen from 2024 or 2023 forecast
scores. Every number in sections 1 and 2 is fixed by this document before any citizen-corrected forecast
exists. This is design number 1 of the Phase 3 family (M = 1).

Owner ruling 2026-11-24: the premise of WRFCoin is that citizen weather stations make forecasts better;
design the test now. The control is `gefs-station-corrected` version 3 (v3), promoted on 2024.

## The short version

- **Three forecasts, not two.** A = v3. B = v3 plus the airport's *own* latest reading (official data
  only). C = B plus the QC-weighted citizen stations around the airport. **The citizen effect is C against
  B.** C against A is reported as well, as the ticket asks, but it is not the evidence for the premise.
- **Why the third forecast is needed.** At 3 h and longer v3 does not use the airport's current reading at
  all (it calibrates GEFS; see `docs/models/GEFS_STATION_CORRECTED_V2.md`, rule 1). A citizen station a few
  kilometres away is then a stand-in for the airport's own thermometer. "v3 plus citizens beats v3" would be
  true even if citizens added nothing the official network does not already have. B removes that credit.
- **Only temperature and dewpoint are changed, only at 1, 3 and 6 h.** Wind, pressure and the 12, 24 and
  48 h leads are copied from v3 byte for byte, so "no harm at 24 to 48 h" is an exact check, not a
  statistical one.
- **Truth stays the 500 official stations.** Citizen readings are inputs only. They are never truth and
  never scored.
- **Data:** a CONUS year of citizen reports is about 213 GiB to transfer (2024) or 187 GiB (2023) and
  about 6 GiB to keep. It fits on this disk today, one year at a time.

## 1. The treatment

### 1.1 What the three forecasts are

All three have 31 members, the v3 member ids, the v3 station grid, the 00Z and 12Z issues and the six
leads. B and C start from the frozen v3 store and move the members of `t2m` and `td2m` at leads 1, 3 and
6 h by one number per station, lead and issue (the same shift for all 31 members, so spread and member
order are unchanged). Everything else is copied verbatim.

| | Model id (proposed) | Uses |
| --- | --- | --- |
| A | `gefs-station-corrected` 3 | as promoted, unchanged |
| B | `gefs-station-corrected-own` 1 | A, plus official readings at or before the issue time |
| C | `gefs-station-corrected-citizen` 1 | B, plus citizen reports at or before the issue time |

For issue time T, official station s, variable v, lead L:

```text
B = A + phi   * own(s, T)          own     = the airport's reading at T minus what v3 expected for T
C = B + gamma * citizen(s, T)      citizen = how far the neighbourhood departs from the airport,
                                             compared with its usual difference at that hour
```

`citizen` is zero when the neighbourhood and the airport differ by their usual amount. A citizen station
can therefore not earn credit by repeating what the airport's own sensor says; it can only add what the
airport does not show yet (an approaching boundary, a local reading that is drifting away from its
surroundings).

### 1.2 The frozen rule

All arithmetic is float64 until the final float32 cast. "Label T" is the hourly truth label, as in v3.

1. **Airport reading.** `O(s,T)` is the official value at label T, finite, with measurement offset in
   [−15, 0] minutes (measured at or before T). No look-back to earlier hours. If it is missing, B = C = A at
   that point.
2. **Own innovation.** `own(s,T) = O(s,T) − mean of the 31 v3 members issued at T − 12 h for lead 12 h`.
   That is the latest earlier v3 forecast that verifies exactly at T. If it does not exist (first issue of
   the year), `own` is undefined and B = A.
3. **Citizen reports that count.** For citizen station c: its latest report of the variable with
   measurement time in [T − 15 min, T] and final QC weight `w > 0`. The weight is the one the shipped
   quality control (`wrfcomp.assimilation.qc`, `QCConfig` defaults, pinned by commit before the download)
   gives that report, run in time order, with state from reports at or before T only.
4. **Neighbours.** Station c is a neighbour of s when the distance from its stored position (rounded to
   0.1 degree) is at most **R = 50 km** and the stored elevations differ by at most **200 m**.
5. **Usual difference.** `usual(c,s)` is the median of `y_c − O_s` over the earlier issues at the same UTC
   hour in the **30 days** before T (T − 30 d to T − 1 d), each formed by rules 1 and 3. At least **10**
   values, otherwise c is not used at T.
6. **Departure.** `d_c = y_c(T) − usual(c,s) − O(s,T)`, clipped to **±5 K**.
7. **Neighbourhood value.** `citizen(s,T)` is the weighted lower median of `d_c` over the neighbours, with
   weights `w_c × exp(−(distance / 25 km)² / 2)`. It is defined only with at least **K = 3** neighbours in
   at least **2** different 0.1 degree cells. Otherwise C = B at that point.
8. **Gains, learned causally, per variable, lead (1, 3, 6 h) and issue hour, pooled over stations.** At T,
   take the earlier issues I at the same UTC hour whose verifying label I + L is at or before T, with a
   finite official value measured in (T − 30 days, T], inside the same calendar year. With `e_A` and `e_B`
   the errors (truth minus member mean) of A and of B *as issued* at I, N the number of records and
   S = 500:

   ```text
   phi   = clip( N  / (N  + S) * sum(own     * e_A) / sum(own^2),     0, 1 )
   gamma = clip( Nc / (Nc + S) * sum(citizen * e_B) / sum(citizen^2), 0, 1 )
   ```

   `Nc` counts the records where `citizen` was defined. A zero denominator gives a gain of 0. Sums run in
   ascending issue time and station id.
9. **Finish.** After both variables are shifted, cap `td2m` at the paired `t2m` for every member (v2
   rule 5). Start fresh at 00Z on 1 January; no state crosses a year boundary (as v3).

### 1.3 Where each number comes from

| Number | Value | Reason |
| --- | --- | --- |
| Variables | `t2m`, `td2m` | The QC has a calibrated trust path for temperature only; citizen wind is sheltered (a multiplicative error an offset cannot remove: 6.4 m/s RMSE against 2.0 for official pairs in `docs/QC_TRIAL_2024_07.md`); the feed has no sea-level pressure. |
| Leads | 1, 3, 6 h | A neighbourhood of 50 km is one to two hours of travel for a boundary moving at 10 m/s. Section 1.4 shows the neighbour signal exists at these leads against persistence. Beyond 6 h the model's own dynamics dominate. |
| Report window | 15 min | The harness tolerance for truth. Measured: 94 to 95 % of reporting stations have a temperature report in the last 15 minutes of an hour. |
| R, distance scale | 50 km, 25 km | 25 km is the QC's own buddy radius, and three times the position uncertainty from rounding (about 8 km). The cut is at twice that. Measured coverage: 251 of 500 airports have 3 or more reporting citizen stations within 25 km, 357 within 50 km. |
| Elevation limit | 200 m | 1.3 K at the standard lapse rate; inversions change the sign of the lapse rate, which a usual-difference cannot absorb. |
| Usual-difference window, floor | 30 days, 10 | The same window and floor v3 uses for its own calibration. |
| Clip | 5 K | The QC's own point where a temperature departure stops having full weight (2.5 sigma at 2 K). |
| K, cells | 3 stations, 2 cells | The QC's minimum for a spatial decision. A median of 3 ignores one bad station; two cells means one household or one cluster cannot form the value alone. |
| Median, not mean | | One broken or dishonest station cannot move it. This matters for a network that will pay stations. |
| Gain window, shrinkage | 30 days, S = 500 | Identical to v3's spread gain: recent as-issued errors, shrunk towards "no change" by one full network issue. |
| Clip of gains | [0, 1] | A negative gain would mean "move away from the observations". Zero is "do nothing". As v2's slope. |

**Why the gains are learned and not written down.** The right gain depends on how much of the neighbour
signal v3 already has, and that can only be measured from v3's errors. Measuring it on 2024 and fixing it
would be tuning on the scoring year. The rule above learns it the way v3 learns its spread: at each issue,
from errors of forecasts already issued and verified, starting from zero. Nothing later than T is used, and
a signal that is not there gives a gain near zero, which leaves the forecast as it was.

### 1.4 Evidence that a signal can exist (training truth only, no forecast)

Measured on `truth/isd_conus_train.nc` cut at **2022-12-17T18Z** (so that 2023 stays untouched), 00Z and
12Z, 2014 to 2022. Official neighbours stand in for citizen ones. Target: the airport's change over the
lead, beyond its usual change at that hour (30-day median). Predictors: the airport's own anomaly at T, and
the neighbour departure of rule 6.

| Median of 3 or more official neighbours within 100 km | 1 h | 3 h | 6 h | 12 h | 24 h | 48 h |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `t2m`: slope on the neighbour departure | 0.18 | 0.43 | 0.60 | 0.62 | 0.55 | 0.42 |
| `t2m`: extra error variance removed, after the airport's own anomaly | 4.2 % | 5.3 % | 4.7 % | 4.1 % | 2.7 % | 1.1 % |
| `td2m`: slope on the neighbour departure | 0.17 | 0.39 | 0.51 | 0.53 | 0.42 | 0.26 |
| `td2m`: extra error variance removed | 5.8 % | 8.8 % | 7.0 % | 4.1 % | 1.4 % | 0.4 % |

(About 229,000 station-issues at 37 airports; single neighbours at 50 to 100 km give slopes 0.09, 0.20,
0.26 at 1, 3, 6 h for `t2m`.)

What this does and does not show:

- Neighbours carry information the airport's own reading does not, against a forecast of "no change beyond
  the usual". The gain is a few percent of error variance (about half that in RMSE).
- v3 is better than that forecast, and GEFS already knows part of what neighbours 50 to 100 km away show.
  The gain against v3 will be smaller, possibly nothing. That is the experiment.
- The 500 official stations have **no pair closer than 50 km**. The training data cannot say how the signal
  behaves inside 50 km, which is exactly where the citizen stations are. Nearer should be more related;
  citizen sensors are noisier. The net is unknown.
- The same measurement without the airport's own anomaly gives a slope that rises to 0.45 at 48 h. That is
  only the airport's own anomaly fading, not neighbour information. It is the reason B exists.

### 1.5 Considered and not chosen

- **The ticket's example: citizen innovations against v3, applied to v3.** Not chosen as the test of the
  premise, for the reason in the short version. It is what "C against A" reports.
- **Fixed gains decaying with lead.** No allowed data can set them (section 1.3).
- **Upwind-only weighting.** Probably stronger, but it adds a wind forecast and a transport speed. One
  simple method first.
- **Citizens filling in when the airport reading is missing.** Real value, rare (about 1 % of points), and a
  second mechanism. Not tested here.

## 2. Fair scoring

- **Truth:** the official truth files of the period, as pinned. Citizen data never enters a truth file, a
  threshold file or a scored field.
- **Harness:** the locked `wrfcomp-eval` at 0.1.0, the same thresholds file as the period's references.
  The effect is the harness's own paired comparison: `--forecast C --baseline B` (and `C` against `A`,
  `B` against `A`). It reports `crpss_vs_baseline` (fair and ECDF), `twcrps_p95`, the spread-skill ratio
  and the one-sided Diebold-Mariano p-value per variable and lead.
- **References:** A, B and C also go through `check_promotion.py` against climatology and GEFS like any
  candidate. Reported; it does not decide the citizen question.

**Station sets (both always reported):**

1. **ALL:** the 500 stations of `configs/stations_conus.csv`.
2. **COVERED:** the official stations where `citizen(s,T)` for `t2m` (rule 7: K = 3 neighbours with positive
   QC weight, in 2 cells, within R = 50 km) is defined at **at least half of the year's issue times**. The
   list is written by the builder from the citizen QC record and the official readings. It reads no
   forecast and no score. Its sha256 is posted on the pull request before any forecast of B or C is scored.
   An upper bound from one measured hour: 364 airports have 3 raw reporters in 2 cells within 50 km. If the
   list has fewer than 50 stations, I stop and ask before scoring.

   The locked CLI scores exactly the stations of its config. COVERED therefore needs a config that is a copy
   of `configs/splits.yaml` with a different `stations_file`, kept outside `configs/`. The harness code is
   the same. The manager approves that file before use.

**How the effect is judged.** Six cells: (`t2m`, `td2m`) × (1, 3, 6 h). For each cell, on COVERED, C against
B, the whole year as one series (704 issues in 2024), and each quarter separately.

**"Citizen stations help"** is said only if all of these hold:

1. In at least one cell, the year's Diebold-Mariano p-value is below **0.05 / 6 = 0.0083**, and
2. in that cell the skill of C against B is above zero under both CRPS estimators, for the year and in
   **at least 3 of the 4 quarters**, and
3. no harm: in no cell, on COVERED or on ALL, is C worse than B with a harm p-value (1 − p) below 0.05; the
   year's `twcrps_p95` of C is not above B's for either variable on either set; and every value outside
   the six cells is byte-identical to v3.

**"Citizen stations harm"** if any cell fails the harm test of point 3. Otherwise **"no detectable
effect"**. The verdict is always given with the size (skill in percent and p-value for all six cells, both
sets, year and quarters) and with the same table for C against A and B against A. On ALL the effect is
expected to be smaller than on COVERED, because many airports have no usable neighbours.

B against A is a result about official data. If B is clearly better than v3, that is a candidate for the
model lane (it would be a v4), not a citizen result.

## 3. Leakage traps

| Trap | How it is closed |
| --- | --- |
| **A citizen station that copies the airport.** | (a) By construction: a copy has departure zero, so it pulls the neighbourhood value towards "no change" and cannot create skill. (b) The shipped copycat screens give such a station weight zero (`docs/ASSIMILATION_SPEC.md`; attacked in the red-team suite). (c) One station can never form the value: 3 stations, 2 cells, median. (d) Reported with the result: the share of the neighbourhood weight carried by stations ever flagged as copycat. |
| **Citizens standing in for the airport's own sensor.** | The comparison is C against B, and B already has the airport's reading. |
| **Citizen reports after T.** | Only measurement times at or before T are read, for the value, for the usual difference and for QC state. Test: changing or adding any citizen report after T leaves every forecast issued at T byte-identical. The archive holds measurement time, not arrival time; real delivery delay is not modelled (the same assumption v3 makes for official readings). |
| **QC trust learned from the scoring period.** | Trust is a running state built in time order from earlier reports only, starting fresh on 1 January. `QCConfig` is pinned by commit before the download. Known and stated: the QC thresholds were shaped on 14 days of July 2024 in two regions, inside the development year. 2023 is free of that. |
| **Gains learned from the scoring period.** | Same rule as v3: only errors of forecasts already issued and verified at or before T, same year. |
| **Numbers chosen from the confirmation year.** | Section 1.4 used truth up to 2022-12-17T18Z only. No 2023 or 2024 score was read for this design. |
| **Station selection that peeks.** | COVERED follows from a fixed rule that reads no forecast; its hash is posted before scoring. ALL is always reported next to it. Any other slice (region, season, airport size) is labelled exploratory and decides nothing. |
| **Trying again.** | One scoring of this design on 2024. A changed number is a new design (M = 2, and the 0.0083 becomes 0.0042). |
| **Sealed data.** | Nothing from 2025-01-01 on. The loader already refuses it in four places. |

## 4. Data

**Source:** NOAA MADIS public, provider `APRSWXNET` (CWOP), no account, as in `docs/CITIZEN_DATA.md`.

**Measured on 2026-11-24** (four archive hours, three CGI hours, eight directory listings; **140 MiB
transferred in total**, nothing kept on disk; station tokens for the size test used a throwaway key, the
real `.station_key` was not touched):

| | Hourly NetCDF archive | Bounding-box CGI, one CONUS request per hour |
| --- | ---: | ---: |
| Transfer, one hour | 21.5 to 28.3 MiB | 12.1 to 16.9 MiB |
| Time, one hour | 1.7 to 11.3 s, plus about 3.5 s to unpack and read | 17.8 to 21.3 s |
| CONUS citizen reports found (2024-04-15 11Z) | 57,067 from 9,957 stations | 56,951 from 9,937 stations |
| Other providers in the response | yes, 51 to 58 % of records (dropped in memory, never stored) | none |
| **Transfer, year 2024** | **about 213 GiB** (mean 24.9 MiB over 96 listed files) | about 116 GiB |
| **Transfer, year 2023** | **about 187 GiB** (mean 21.8 MiB) | about 101 GiB |
| **Time, one year** | 13 to 36 h one file at a time; 4 to 9 h at 4 at once | about 49 h one at a time; about 12 h at 4 at once |

| Stored after anonymising (keyed token, 0.1 degree, 10 m), one row per report | 2024 | 2023 |
| --- | ---: | ---: |
| Reports in the year (estimate) | about 546 million | about 478 million |
| `t2m` and `td2m` only (12.1 bytes a report, measured) | **about 6.2 GiB** | **about 5.4 GiB** |
| All seven variables (16.1 bytes a report, measured) | about 8.2 GiB | about 7.2 GiB |

Also measured: about 9,900 CONUS citizen stations report in an hour, 5.6 to 7.8 reports each; 71 % of
them are within 50 km of one of the 500 official stations and 91 % within 75 km.

**Recommendation: the NetCDF archive.** It is static files (fast, restartable hour by hour, no load on a
query service), it is complete (the CGI returned 0.2 % fewer reports), and the transfer is never written to
disk either way. Two cautions: the files contain restricted providers, which must be dropped in memory
before anything is stored; and CWOP has `altimeter` there, not station pressure (not needed for this
design).

**Disk.** Measured free space is **71.8 GB** (the ticket says about 56). Stop line 40 GB, so 31.8 GB of
room.

| Step | Adds | Room left above the stop line |
| --- | ---: | ---: |
| Today | | 31.8 GB |
| 2023 GEFS build, running (lane V1's figure) | 5.0 to 7.6 GB | about 24 GB |
| 2024 citizen year, `t2m` and `td2m` | 6.2 GiB (6.7 GB) | about 17.5 GB |
| 2024 stores of B and C | 2.0 GB | about 15.5 GB |
| 2023 citizen year, after the 2024 result | 5.4 GiB (5.8 GB) | about 9.7 GB |
| 2023 stores of B and C | 2.0 GB | about 7.7 GB |

So **2024 fits now, next to the running GEFS download, and 2023 fits afterwards.** Both years need about
16.5 GB. That breaks the sprint's 25 GB total cap (`00-common.md`; the data directory is at 11 GB), so the
manager has to raise the cap or free space. Once a year's forecasts are built and verified, its citizen
year can be reduced to the per-issue extract (under 0.1 GB) and fetched again if needed.

**Privacy (unchanged, absolute).** Raw responses are parsed in memory. Only the keyed token, the position
rounded to 0.1 degree and the elevation rounded to 10 m are stored. No identifier, call sign, name or
coordinate is printed, logged or committed. The token scheme and `.station_key` stay as they are. Logs
carry counts and exception class names only. Step 2 extends the existing loader to read the archive format
under the same rules, with the same tests.

**Not measured:** the running time of the QC on a full year (about 860 times the July trial). Step 2
measures one day first and reports before the bulk run.

## 5. Periods and the confirmation rule

- **2024: development.** The lane builds B and C for 2024 from the frozen v3 stores after the manager's
  GO. The manager runs the scoring once.
- **2023: untouched confirmation, as for v3.** Run only if 2024 says "help". Before any 2023 citizen data
  is fetched, the design is frozen at one commit and the cells that passed on 2024 are named on the pull
  request. The manager builds the 2023 stores from that commit (first rebuilding one 2024 month to match
  its content hash), with v3's 2023 schedule (698 issues, last 2023-12-15T12Z), the 2023 truth and the
  thresholds cut at 2022-12-17T18Z, and scores once.
- **Confirmed** means: on 2023, at least one of the named cells meets the same three conditions, with the
  p-value limit 0.05 divided by the number of named cells.
- **One shot per design.** No lane scores a 2023 store. If the design changes after its 2023 result is
  known, 2023 is used up for this family.

## 6. What a null result means

"No detectable effect" would mean: **this method, at these 500 airports, for temperature and dewpoint at
1 to 6 h, adds nothing measurable to a forecast that already uses the airport's own reading.** It would not
mean that citizen stations are useless. Three things it cannot separate:

1. **The place.** Airports are the best-observed points in the country. Citizen stations may matter most
   where there is no airport, and this truth cannot see that.
2. **The trust weights.** In the July trial the median final weight was 0.04 to 0.2 and few stations held
   the temperature credential. A strict QC may discard the signal with the noise.
3. **The method.** One neighbourhood value, no wind direction, two variables.

How it is reported: the full table of section 2 whatever the outcome, with sizes and p-values, the number
of COVERED stations, the learned gains over the year (a `gamma` that stays near zero is itself a finding),
and the smallest effect the test could have detected, computed from the spread of the paired differences.
The scorecards are committed. No second attempt on the same year under the same design number, and no
wording stronger than the verdict words of section 2.

## Where I think the ticket is incomplete

1. **The control.** "v3 plus citizens against v3 alone" overstates citizens at 3 h and longer, because v3
   ignores the airport's current reading there. This design judges the premise on C against B and reports C
   against A beside it. If the manager wants the ticket's comparison to decide, that is a one-line change
   to section 2, and the caveat should travel with the result.
2. **Free space** is 71.8 GB, not 56. The 25 GB total cap in `00-common.md` does not hold both years.
3. **COVERED cannot be scored with the locked config** (the CLI requires all 500 stations). It needs the
   derived config described in section 2.
4. **"Decaying with lead" gains** cannot be argued from allowed data; they are learned causally instead.

## Step 2, if GO (not started)

Archive loader with the privacy tests; one-day QC timing; the 2024 download (stop and report sizes after
the first day, as the common rules require); builders for B and C with the acceptance tests (byte identity
outside the six cells, the after-T control, month-wise replay equals uninterrupted replay, year-start
control); the COVERED list and its hash. No scoring by this lane.
