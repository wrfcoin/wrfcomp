# Phase 3, step 2: building the arms for 2024

The design is `docs/PHASE3_DESIGN.md`, registered at commit `12d0bff` (pull request 34). **No number in it
is changed here.** This page says how the registered rule is carried out, what was measured, and where the
results are. Nothing on this page is a score. Scoring is the manager's.

Rulings in force (manager and owner, 2026-11-24): 2024 only, no 2023 citizen data; NetCDF archive; data cap
45 GB, stop line 40 GB free; the COVERED config is approved by the manager before use; **C against B
decides**, C against A and B against A are reported.

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
a number.

1. **"The shipped quality control" (rule 3)** is the July trial runner's procedure
   (`scripts/run_qc_trial.py`) with the gates of `wrfcomp.assimilation.qc` and the state of
   `wrfcomp.assimilation.trust`, `QCConfig` defaults, for `t2m` and `td2m`. The year driver imports those
   functions; it defines no gate and no threshold of its own. **Checked on the real July sample: the driver
   returns the trial runner's weights exactly** (Amarillo 80,804 weights, Seattle 1,218,099 weights, zero
   differences). The same check runs in CI on synthetic data (`tests/test_phase3_qc.py`).
2. **Wind speed and direction are stored as well,** because the copycat screen looks at every channel and a
   hit on any channel withdraws the temperature credential. Wind is used for nothing else.
3. **"Its latest report in [T − 15 min, T] with weight above zero"** is read as: take the latest report of
   the window; it counts if its weight is above zero. An older good report is not used in place of a newer
   failed one (the QC's own rule for neighbours).
4. **A station is a token at one stored position.** If its stored position (0.1 degree, 10 m) changes, it is
   a new station with no history and no trust.
5. **Year isolation:** QC state, usual differences and gains start on 2024-01-01T00Z. Official readings and
   GEFS cycles from before that are not read.
6. **"B as issued" (rule 8)** is the published float32 B store, after the dewpoint cap.
7. **Neighbour distance (rule 4)** is from the citizen station's stored position to the official station's
   position; heights are the stored citizen height and the official elevation.
8. **COVERED** counts every issue of the year (704), including the first days when nothing can be defined.

## One thing the archive has that the design did not know

The design says the archive holds measurement time only. The NetCDF archive also has `receivedTime`.
Measured on 2024-01-01: reports arrive a median of 5.8 minutes after they were measured, 95 % within
10.8 minutes, 99 % within 12.8. The registered rule uses measurement time (as v3 does for official
readings). Of the reports measured in the last 15 minutes of an hour, 61 % had arrived by the hour; **about
four in ten of the reports the rule reads had, in reality, not arrived yet.** The delay is stored with each report (`received_delay_s`). This build does not use it,
because that would change the registered rule. **It is a limit on what the result means for a live
system**, and a stricter rule (received at or before T) would be a new design.

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
error or a repr; the fetch script refuses 2023.

`tests/test_phase3_qc.py`: driver equals trial runner; the forecast-choice table equals the trial's rule;
changing or removing every report after T leaves every weight at or before T, and the snapshot at T,
identical; the 15-minute window.

`tests/test_model_citizen_arms.py`: only the six cells change and the rest is the v3 bytes; cold start
changes nothing; one shift for all members; dewpoint cap; gains are zero without history, shrunk and
clipped; fewer than three stations or a single cell never form a value; changing truth and citizen data
after T leaves every forecast issued up to T identical; truth from before 1 January is refused; issues must
be in order; the registered constants are asserted by value.

## Results

Filled in after the bulk run: store paths and content hashes, the COVERED count and hash, learned gains.
