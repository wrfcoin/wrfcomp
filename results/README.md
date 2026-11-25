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
