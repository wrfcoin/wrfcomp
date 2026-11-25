# Official results

Each folder holds a candidate's official promotion records, exactly as the locked
`scripts/check_promotion.py promote` command wrote them: one folder per quarter with the two
scorecards (against climatology and against raw NOAA GEFS) and the quarter verdict, plus the
year's decision record. Nothing here is recomputed or edited.

`summary.json` is generated from those records by `scripts/summarize_results.py`; every value in
it names its source file. A test regenerates it and fails if it drifts from the records.

## gefs-station-corrected, version 3

NOAA GEFS corrected per station from recent verified errors (design:
`docs/models/GEFS_STATION_CORRECTED_V3.md`). Code frozen at commit `4eb752a`.

| Year | Role | Result |
| --- | --- | --- |
| 2024 | development (the design was shaped on this year) | PROMOTED, 3 of 4 quarters |
| 2023 | untouched confirmation, scored once | PROMOTED, 4 of 4 quarters |

For 2023, `schedules/` holds the forecast schedule taken from the pinned 2023 reference stores,
and `candidate_stores.json` the content hash of each monthly forecast store that was scored.

## gefs-station-corrected, version 4

Version 3 plus the airport's own current reading, for `t2m` and `td2m` at leads 1, 3 and 6 h. It is
Phase 3 arm B (design: `docs/PHASE3_DESIGN.md`, section 1.2), so its stores carry model id
`gefs-station-corrected-own`, version 1. Model code frozen at commit `ec736ee`. The B-only builder
`scripts/build_v4.py`, which needs no citizen data, is at commit `73ba89b`. Rebuilt with it, 2024
reproduced all 12 scored B stores byte for byte (same content hash).

| Year | Role | Result |
| --- | --- | --- |
| 2024 | development (the arm was registered and scored as Phase 3 arm B on this year) | PROMOTED, 4 of 4 quarters |
| 2023 | untouched confirmation, built and scored once | PROMOTED, 4 of 4 quarters |

For 2023, `build.json` is the build record of the scored stores: the truth file and the v3 input
stores with their hashes, and the content hash of each monthly v4 store. Paths are relative to the
data directory.

## Phase 3, design 1 (citizen stations), 2024

`phase3-design1/2024/` holds the 30 scorecards (arms C against B, C against A and B against A, on
the COVERED and ALL station sets, for the year and each quarter) and the verdict record. Its README
gives the deciding C against B table and the verdict.
