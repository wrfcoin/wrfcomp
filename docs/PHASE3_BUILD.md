# Phase 3, step 2: building the arms for 2024

The design is `docs/PHASE3_DESIGN.md`, registered at commit `12d0bff` (pull request 34), with **Amendment 1**
(arrival time, manager ruling 2026-11-24, recorded before any forecast of B or C existed). No number in it
is changed here. This page says how the registered rule is carried out, what was measured, and where the
results are. Nothing on this page is a score. Scoring is the manager's.

Rulings in force (manager and owner, 2026-11-24): 2024 only, no 2023 citizen data; NetCDF archive; data cap
45 GB, stop line 40 GB free; the COVERED config is approved by the manager before use; **C against B
decides**, C against A and B against A are reported; the readings below go to an outside review before the
build is relied on.

## The pipeline

| Step | Command | Reads | Writes (under the data directory) |
| --- | --- | --- | --- |
| 1 Fetch | `scripts/fetch_citizen_year.py` | hourly MADIS archive files, in memory | `citizen/conus/2024/<day>.nc` |
| 2 Partition | `scripts/run_phase3_qc.py partition` | step 1 | `citizen/conus/2024-tiles/` (working copy) |
| 3 QC | `scripts/run_phase3_qc.py run`, then `merge` | step 2, 2024 truth, 2024 GEFS station stores | `citizen/conus/issue_extract_2024.nc` |
| 4 Build | `scripts/build_phase3_arms.py` | step 3, 2024 truth, the twelve v3 stores | `candidates/gefs-station-corrected-own/`, `candidates/gefs-station-corrected-citizen/`, `phase3/2024/` |

Code: `wrfcomp.assimilation.madis_archive` (reader), `wrfcomp.assimilation.phase3_qc` (QC driver),
`wrfcomp.models.citizen_arms` (rule 1.2 of the design).

## How the registered rule is carried out

These are readings of the registered text, written down so a reviewer can disagree with them. None changes
a number. They were reviewed on pull request 36 (outside review, 2026-11-24, verdict CHANGES); readings 3,
4, 5 and 9 below are as corrected after that review, and each correction has a test built on the
reviewer's synthetic case.

1. **"The shipped quality control" (rule 3)** is the July trial runner's procedure
   (`scripts/run_qc_trial.py`) with the gates of `wrfcomp.assimilation.qc` and the state of
   `wrfcomp.assimilation.trust`, `QCConfig` defaults, for `t2m` and `td2m`. The year driver imports those
   functions; it defines no gate and no threshold of its own. **Checked on the real July sample: the driver
   returns the trial runner's weights exactly** (Amarillo 80,804 weights, Seattle 1,218,099 weights, zero
   differences). The same check runs in CI on synthetic data (`tests/test_phase3_qc.py`).
2. **Wind speed and direction are stored as well,** because the copycat screen looks at every channel and a
   hit on any channel withdraws the temperature credential. Wind is used for nothing else.
3. **"Its latest report in [T − 15 min, T] with weight above zero"** is carried out as written: the latest
   report of the window *among those with weight above zero*. A zero-weight report does not hide an
   earlier report that counts. (Corrected: the first version let a newer zero-weight report suppress it.)
4. **A station is a token.** If its stored position (0.1 degree, 10 m) changes, its QC history starts
   again at the new position with no trust. But it is still one station: of several rows of one token
   near an airport, only the row with the newest report counts towards the three-station minimum and
   enters the median. (Corrected: positions were counted as separate stations.)
5. **Year isolation is applied to measurement times.** Citizen: a report measured before 2024-01-01T00Z is
   dropped by the parser and again where the QC reads its input. Official: a reading whose label is in
   2024 but whose measurement time is not (the 00Z label of 1 January with a negative offset) is removed
   from the QC's official series and is not a reading for rule 1. QC state, usual differences and gains
   start empty. GEFS cycles from before 1 January are not read. (Corrected: labels were bounded, not
   measurement times.)
6. **"B as issued" (rule 8)** is the published float32 B store, after the dewpoint cap.
7. **Neighbour distance (rule 4)** is from the citizen station's stored position to the official station's
   position; heights are the stored citizen height and the official elevation.
8. **COVERED** counts every issue of the year (704), including the first days when nothing can be defined.
   The builder refuses to run unless the extract, the truth file and the v3 stores are all on exactly the
   704 registered issue times and the 500 stations of `configs/stations_conus.csv`.
9. **Amendment 1 (arrival time)** is decided in the archive reader, on the archive's own timestamps at
   full precision, **before** repeated records are merged and before any time is converted
   (`madis_archive.arrived_by_next_issue`). A record that had not arrived by the first 00Z or 12Z at or
   after its measurement, or that has no arrival time, never reaches a stored table. A late repeat
   therefore cannot displace a timely report; of several timely versions the one that arrived last is
   kept. Stored times are rounded up to the second, so a time just after an issue time stays after it. The
   same test is repeated on the stored values when tables are saved, loaded, partitioned and read by the
   QC. Official readings keep the v3 rule. (Corrected: the rule was applied after repeats had been merged,
   on times truncated to the second and a delay field that saturated.)

## Nothing incomplete or stale can become a forecast

- Every stored table names the arrival policy it was written under; a table without it is refused.
- Each daily file says which day it holds and which source hours the archive did not have. A missing
  source hour is recorded and carried into the extract. A missing day file is work that has not run:
  partition refuses.
- The tiles carry a digest of the daily files. The QC outputs carry a digest of the tiles, the truth file,
  the GEFS cycles and the QC code. Anything with another digest is refused, never reused.
- Merge needs an output for every tile. A timing run on some tiles has its own folder and cannot be merged.
- The builder checks the extract (complete, current policy, schedule, stations) and the v3 stores before it
  writes anything.

## Measured before the bulk run (2024-01-01)

| | One day | Year estimate (352 days) |
| --- | ---: | ---: |
| Transfer, 4 files at once | 550 MiB in 50 s | about 190 GiB, about 5 h |
| Stored, anonymised, four variables | 7.5 MiB (1,359,906 reports, 10,086 stations) | about 2.7 GiB |
| Tile working copy | 13 MB | about 4.5 GB |
| QC, 8 processes | 50 s wall, 397 CPU-seconds, 1,890,062 channel-reports | about 40 CPU-hours |

## Acceptance tests

`tests/test_madis_archive.py`: restricted providers and stations outside CONUS never reach a table; keyed
tokens; positions rounded in the parser and again on save; sealed times refused; no record content in an
error or a repr; the fetch script refuses 2023; a late repeat never displaces a timely report; an arrival a
quarter of a second after T is late; a long but timely delay is kept; nothing measured before the year is
kept; a table from before the amendment is refused.

`tests/test_phase3_qc.py`: driver equals trial runner; the forecast-choice table equals the trial's rule;
changing or removing every report after T leaves every weight at or before T, and the snapshot at T,
identical; the 15-minute window and the positive-weight rule; official readings measured before the year
are removed; partition, merge and a timing run refuse missing, partial and stale products.

`tests/test_model_citizen_arms.py`: only the six cells change and the rest is the v3 bytes; cold start
changes nothing; one shift for all members; dewpoint cap; gains are zero without history, shrunk and
clipped; fewer than three stations or a single cell never form a value; changing truth and citizen data
after T leaves every forecast issued up to T identical; truth from before 1 January is refused; issues must
be in order; the registered constants are asserted by value; an official reading measured before the year
is not a reading; one token at several positions counts once; the build refuses incomplete, stale or
off-schedule inputs before writing; stores published month by month equal the uninterrupted replay.

## Results

Filled in after the bulk run: store paths and content hashes, the COVERED count and hash, learned gains.
