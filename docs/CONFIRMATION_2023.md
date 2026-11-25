# 2023 confirmation period

**Status: design, code and reference data are done (sections 6 and 7).** The 2023 references are built.
Pinning them is a separate locked change that needs the owner's approval (#40). No candidate has been built
or scored on 2023.

Manager rulings, 2026-11-24: the Q4 end of 2023-12-15T12Z is accepted; the existing `promote` printing
`PROMOTED` for 2023 is read as confirmed, with no change to the locked script.

Owner ruling 2026-11-24: a candidate that passes on 2024 must also pass on an untouched 2023 period before
anyone relies on it. Nothing from 2025 onward is touched.

## 1. The period

All four calendar quarters of 2023, init hours 00Z and 12Z, the same leads, variables, stations and gates
as 2024. The 3-of-4 rule applies unchanged.

| | 2024 (as pinned) | 2023 confirmation (proposed) |
| --- | --- | --- |
| First init | 2024-01-01T00Z | 2023-01-01T00Z |
| Last init | 2024-12-17T12Z | **2023-12-15T12Z** |
| Init times | 704 | 698 (Q1 180, Q2 182, Q3 184, Q4 152) |
| References may use observations up to | 2023-12-17T18Z | **2022-12-17T18Z** |
| Gap between that cutoff and the first init | 14 days 6 h | 14 days 6 h |

- The cutoff 2022-12-17T18Z is the 2024 arrangement moved back exactly one year.
- Q4 stops at 2023-12-15T12Z, two days earlier than 2024-Q4, because the truth we hold ends at
  2023-12-17T18Z and the 48 h lead of a later init would verify after it. This costs 4 of 156 init times
  and needs no new truth download. The alternative (fetch two more days of 2023 truth to end on
  2023-12-17T12Z) reads data that sits inside the embargo before validation, so I do not recommend it.

## 2. References, and what has to change

**Rule: one fixed cutoff, 2022-12-17T18Z, for everything a reference learns from.**

- **Tail thresholds (p95):** per station and calendar month from truth 2014-01-01 to 2022-12-17T18Z
  (nine years; the 2024 thresholds use ten). New file `thresholds/tail_p95_to_2022-12-17.nc`.
- **Climatology:** same pools and member rule as today, archive cut at the same cutoff.
- **Persistence:** as today (latest observation at init), error quantiles from the archive up to the cutoff.
- **GEFS:** as today, raw station extraction, no training.

**A trap in today's baseline code.** For a request before the train end, `wrfcomp.baselines` takes its
cutoff as the earliest init of the request, not a fixed date. Built month by month, as `run_gate1.py`
does, February 2023 climatology would include January 2023 observations, with no embargo. That is causal,
but it is a stronger and different reference than the 2024 one. The confirmation must pass the cutoff
explicitly.

**Locked changes needed (one owner approval, one pull request):**

1. `experiments/REFERENCES.json`: four new periods, `2023-Q1` to `2023-Q4`. The file already names the truth
   and thresholds file per period, so each 2023 period points at its own truth and thresholds.
2. Twelve new reference scorecards (`climatology`, `gefs`, `persistence` for the four quarters), added to the
   lock scope (`scripts/harness_lock.py`, `HARNESS_LOCK.json`, the expectation in
   `tests/test_harness_lock.py`).
3. Recommended, same pull request: one paragraph in `docs/VERIFICATION_PROTOCOL.md` stating the
   confirmation rule and the cutoff.

**Not needed:** no change to `src/wrfcomp/harness/`, `configs/splits.yaml` or `scripts/check_promotion.py`,
and no version bump (lane L2 added the 2024 quarters the same way at 0.1.0). I read the locked scoring path
for anything that refuses a period before the validation split and found only the sealed-split checks. This
is from reading the code; it has not been run. Step 2 proves it on one month before anything else.

**The confirmation command is the existing one:**

```sh
uv run --locked python -P scripts/check_promotion.py promote --year 2023 \
  --forecast <the candidate's twelve 2023 monthly stores> --data-dir "$WRFCOMP_DATA_DIR" --out <dir>
```

It will print `PROMOTED` or `NOT PROMOTED` for 2023. Read that as "confirmed on 2023" or "not confirmed";
its `decision.json` records the year. A separate `confirm` mode that prints `CONFIRMED` and requires the 2024
result as well would be clearer, but it is a change to the locked script. I recommend the smaller option now.

Months of 2023 are deliberately **not** pinned: only the quarters decide, and unpinned months cannot be
scored through `check_promotion.py`.

**Unlocked changes (step 2, this lane unless marked):**

- `src/wrfcomp/baselines/`: an explicit `train_end` argument, default unchanged. Proof of no change to
  2024: rebuild one pinned 2024 month and match its pinned content hash.
- `scripts/build_thresholds.py`: a `--train-end` option that can only move the cutoff earlier, and that
  writes the real cutoff into the file's attributes.
- `scripts/run_gate1.py`: a confirmation mode (today it refuses any period that starts before validation
  and assumes the 2024 truth file).
- `scripts/build_candidate.py` (lane C1/C3's file, not mine): it is hard-wired to 2024 months, the 2024
  truth file and a 2024 start date. It needs a year argument.

## 3. Truth (measured)

`~/wrfcomp-data/truth/isd_conus_train.nc` covers 2014-01-01T01Z to 2023-12-17T18Z, every hour of 2023 up to
that end, with no missing hourly label.

- Stations: the same 500, in the same order as `configs/stations_conus.csv`; latitude, longitude and
  elevation equal the 2024 truth file.
- Same build: `parser_version` 3, the same passing quality codes and ±15 minute tolerance as the 2024 file.
- Finite share in 2023 against 2024: t2m 99.19 % / 98.94 %, td2m 99.09 % / 98.82 %, u10 and v10 95.69 % /
  95.72 %, msl 98.72 % / 98.59 %. No station is below 50 % in any variable in either year. Lowest quarter:
  wind in Q3, 94.09 %.
- Missing: 2023-12-17T19Z to 2023-12-31T17Z is in neither truth file. Section 1 avoids needing it.
- The scoring truth for 2023 would be a slice of this file (2022-12-31T18Z to 2023-12-17T18Z, about
  0.13 GB), the same shape as the 2024 file. No download.
- The station list was vetted on 2024 completeness, not 2023, so 2023 did not shape the list.

## 4. GEFS 2023 (measured on three cycles, the only download made)

| Cycle | HTTP bytes received | Seconds | Store on disk | Finite values per variable |
| --- | ---: | ---: | ---: | ---: |
| 2023-01-01 00Z | 181,548,420 | 69.8 | 1,288 KiB | 77,500 of 77,500 |
| 2023-07-01 12Z | 169,942,648 | 68.6 | 1,300 KiB | 77,500 of 77,500 |
| 2023-12-15 12Z | 183,381,366 | 69.4 | 1,268 KiB | 77,500 of 77,500 |

All three: 500 stations, 31 members, leads 3, 6, 12, 24, 48 h, no dewpoint above temperature. The 2023
archive has the same layout as 2024. The three stores are at their final place,
`~/wrfcomp-data/baselines/gefs/`, and are reused by the full build. Restart cache for the three: 5.6 MiB.

**Extrapolation to the 698 cycles (695 left):**

- Transfer: 698 × 178.3 MB ≈ **124 GB** received and discarded (GRIB is decoded in memory).
- Time: 698 × 69.3 s ≈ **13.4 h** one cycle at a time. The 2024 bulk runner logged 58 cycles in 1,103 s
  for December (about 19 s a cycle), which would be about **3.7 h**.
- Stored: GEFS stores ≈ **0.9 GB**, restart cache ≈ **1.3 GB** (can be deleted once a month is verified).

**Disk.** Estimated before the build: about 7.6 GB if built as 2024 was (monthly and quarterly stores and
the restart cache kept), about 5.0 GB lean (quarters kept, months removed after joining, restart cache
deleted as cycles are verified), in both cases including about 1 GB for one candidate's 2023 stores and
0.3 GB of scoring scratch. The lean layout was used; section 7 has the measured sizes.

## 5. How lane C3's candidate (v3) gets 2023 stores without seeing 2023 scores

1. C3 freezes v3 (specification and code at one commit) and the manager gets the 2024 verdict, as now.
2. v3's rule for 2023 is the 2024 rule moved back a year: cold start on 2023-01-01, 2023 data only, no
   December 2022 cycles. If v3 uses anything derived from the train split (climatology, thresholds), it must
   take the 2022-12-17T18Z cutoff for 2023. C3 states this in the pre-registration, before 2023 is built.
3. **The manager, not C3, builds the 2023 stores**, from the frozen commit, in the manager's own checkout.
   First rebuild one 2024 month there and match the content hash of the store that was scored, so the code
   that makes the 2023 stores is shown to be the code that was judged on 2024.
4. The manager records the twelve content hashes, then runs `promote --year 2023` **once**. The result
   directory and `decision.json` are the record.
5. No lane runs `wrfcomp-eval` or `check_promotion.py` on a 2023 candidate store, before or after. This is a
   rule for agents, not a control: the truth and thresholds are on the same disk and nothing technical stops
   it. The candidate needs 2023 observations as input (it is causal), so they cannot be hidden from the
   builder.
6. One shot per candidate version. If a design is changed after its 2023 result is known, 2023 is used up
   for that family and the next confirmation needs another year (2022 has GEFS and truth in the same form).

Raw-GEFS and climatology scorecards for 2023 (the pinned references) are not candidate scores, but they do
show how the references behave in 2023. Build and pin them only after v3 is frozen.

## 6. Code

All of it is unlocked code. The default behaviour of every changed function and command is unchanged.

- `wrfcomp.baselines`: `climatology.forecast` and `persistence.forecast` take `train_end`. It fixes the
  cutoff, can only be at or before the default one, and is written to the store's `train_end_utc`
  attribute. Without it nothing changes, attributes included.
- `scripts/build_thresholds.py --train-end <UTC>`: uses truth at or before that time only. It must be
  earlier than the train split end. The file's `train_end_utc` attribute is the real cutoff. Input after
  the train split end is still refused.
- `scripts/run_gate1.py --confirmation-train-end <UTC>`: scores a period inside the train split.
  - The period must start at least 14 days (the train split's embargo) after the cutoff and lie before
    validation. The last init is the last one whose 48 h lead verifies at or before the train split end.
  - `--truth` and `--thresholds` must be named, and the thresholds file must carry exactly this cutoff.
    The ordinary thresholds file is refused.
  - Train truth after the cutoff is dropped before loading. The scored truth may be the train truth
    file itself, so no separate 2023 truth file is needed.
  - A climatology or persistence store built with another cutoff, or with none, is not reused.

**No change to 2024 (measured).** With the changed code, rebuilt in memory from the real inputs and
compared by harness content hash with `experiments/REFERENCES.json`:

| Rebuilt | Content hash | Pinned |
| --- | --- | --- |
| Climatology 2024-01 (62 init times) | `ff1823d8f20d…9a636f53` | match |
| Persistence 2024-01 (62 init times) | `4b877417821d…5bbfa6db` | match |
| Tail thresholds (whole train split) | `7ee7e1e5428d…336587864c` | match |

Nothing was written to disk for this check.

Tests: `tests/test_baselines_models.py` (the rolling-cutoff trap and its fix, for both models),
`tests/test_gate1_scripts.py` (thresholds cut and attribute, a confirmation month end to end with a
check that train data after the cutoff cannot move the references, the refusals, and the 2023 schedule
with the real config: 180, 182, 184 and 152 init times). `scripts/slice_truth.py` (the scored truth slice) has its own test there.

## 7. The data build (done 2026-11-24)

All under the data directory (`$D`), built at commit `5cd7fe4` of this branch. The 2024 check of section 6
was repeated at that commit first and matched all three pinned hashes.

```sh
uv run python scripts/slice_truth.py --source "$D/truth/isd_conus_train.nc" \
  --start 2022-12-31T18 --end 2023-12-17T18 --out "$D/truth/isd_conus_2023.nc"
uv run python scripts/build_thresholds.py --train-truth "$D/truth/isd_conus_train.nc" \
  --train-end 2022-12-17T18 --out "$D/thresholds/tail_p95_to_2022-12-17.nc"
uv run python scripts/build_baselines.py gefs <each of the 698 init times>
uv run python scripts/run_gate1.py --data-dir "$D" --init-hours 0 12 --out <dir> \
  --confirmation-train-end 2022-12-17T18 --truth "$D/truth/isd_conus_2023.nc" \
  --thresholds "$D/thresholds/tail_p95_to_2022-12-17.nc" \
  --period 2023-01 ... --period 2023-12        # then the same with --period 2023-Q1 ... 2023-Q4
```

| Item | Result |
| --- | --- |
| Truth slice `truth/isd_conus_2023.nc` | 8,425 hours, 2022-12-31T18Z to 2023-12-17T18Z, 0.13 GB |
| Thresholds `thresholds/tail_p95_to_2022-12-17.nc` | 500 stations, no station-month without a threshold |
| GEFS 2023 | 698 cycles (695 downloaded here, 123.7 GB received and discarded, 5 h 43 min at four cycles at a time); 0.9 GB stored |
| GEFS verification | every cycle: 500 stations, 31 members, 5 leads, all values finite, no dewpoint above temperature; none failed. Each cycle's restart cache was deleted once the cycle was verified |
| Reference stores `gate1/2023-Qn_h00-12/` | climatology, persistence and GEFS for the four quarters, 2.0 GB. The monthly stores were removed after each quarter was joined |
| Init times | Q1 180, Q2 182, Q3 184, Q4 152 (ends 2023-12-15T12Z) |
| Cutoff recorded in the stores | `train_end_utc` = 2022-12-17T18Z on every climatology and persistence store |
| Reference scorecards | twelve, `<model>_2023-Qn.json`, in the locked-change pull request |

After the build every reference store, the truth slice and the thresholds file were hashed again from disk
and equal the hashes in the proposed `experiments/REFERENCES.json`.

The unchanged locked check was run once on 2023-Q4 with the persistence reference store as its input. It
scored against both references and answered `NOT PASSED`, with the expected reason that the input is a
reference forecast. That shows the locked script handles a 2023 period as it is. No candidate store for 2023
was built or scored.

Still open: `scripts/build_candidate.py` is hard-wired to 2024 and needs a year argument before a candidate's
2023 stores can be built (the candidate lane's file).

