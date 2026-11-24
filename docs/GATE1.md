# Gate 1: reproducing the reference scorecards

Gate 1 scores the three reference forecasts (climatology, persistence, NOAA GEFS) on the 2024
validation split with `wrfcomp-eval`, one scorecard per model and month. This page is the exact
command sequence. Everything after step 2 runs offline. The first run covered January and July; the
full year followed (results and reading: `experiments/README.md`, `experiments/SUMMARY_2024.md`).

All data lives outside the repository, in `$WRFCOMP_DATA_DIR` (default `~/wrfcomp-data`).

```sh
export WRFCOMP_DATA_DIR="${WRFCOMP_DATA_DIR:-$HOME/wrfcomp-data}"
uv sync --extra dev
```

## 1. What must be in the data directory

| Path under the data directory | What | Made by |
| --- | --- | --- |
| `truth/isd_conus_train.nc` | station truth for the training split, ending at or before the train split end (`2023-12-17T18:00Z`) | truth loader |
| `truth/isd_conus_2024.nc` | station truth for the scored year | truth loader |
| `baselines/gefs/<YYYYMMDDHH>.zarr` | one GEFS point store per cycle | `scripts/build_baselines.py gefs <init>` (network) |

Both truth files follow the data contract including addendum 1: every variable `X` has an
`X_offset_min` companion. The baselines need them.

Two things about the truth files:

- The training file must not reach past the train split end. A full-year 2023 file is refused; cut
  it first.
- The scored year's file starts before the first init time (the truth loader writes it from
  `2023-12-31T18:00Z`). Persistence may only use an observation taken at or before the init time,
  looking back at most 3 hours, so the first run of January needs those hours.

Other file names work too: pass `--train-truth` and `--truth` (and `--thresholds`) explicitly.

## 2. GEFS cycles (the only step that uses the network)

For every init time to be scored, once (see `docs/BASELINES.md` for size and time):

```sh
uv run python scripts/build_baselines.py gefs 2024-01-01T00
```

January and July 2024 at 00Z and 12Z are 124 cycles; the whole scored year is 704 (2024-01-01T00Z to
2024-12-17T12Z, see "Where a period ends" below). The runner never downloads: it lists the cycles that
are missing and scores the other models.

## 3. Tail thresholds

```sh
uv run python scripts/build_thresholds.py \
  --train-truth "$WRFCOMP_DATA_DIR/truth/isd_conus_train.nc" \
  --out "$WRFCOMP_DATA_DIR/thresholds/tail_p95.nc"
```

Per station and calendar month, the 0.95 quantile of the training truth for each primary variable.
The command refuses any input with a time after the train split end, and it prints how many
station-months have no threshold because there was no training data. Scoring refuses a point that
needs a missing threshold, so look at those counts.

Several stores may be given after `--train-truth` (for example one per year); they must not overlap.

## 4. The scorecards

```sh
uv run python scripts/run_gate1.py \
  --data-dir "$WRFCOMP_DATA_DIR" \
  --period 2024-01 --period 2024-07 \
  --init-hours 0 12 \
  --out experiments/
```

For each period this

1. builds the climatology and persistence forecasts for every scorable init time of the month at 00Z
   and 12Z
   (leads 1, 3, 6, 12, 24, 48 h, 30 members each) into
   `$WRFCOMP_DATA_DIR/gate1/<period>_h00-12/{climatology,persistence}.zarr`;
2. joins the GEFS cycle stores of those init times (62 for January) into `.../gefs.zarr` (leads 3, 6,
   12, 24, 48 h);
3. runs `wrfcomp-eval` three times: climatology alone, persistence against climatology, GEFS against
   climatology, each on the leads the model has;
4. writes the scorecards.

Result, six files:

```text
experiments/climatology_2024-01.json   experiments/climatology_2024-07.json
experiments/persistence_2024-01.json   experiments/persistence_2024-07.json
experiments/gefs_2024-01.json          experiments/gefs_2024-07.json
```

The file name uses the short model name. Inside the scorecard, `model_id` is `climatology`,
`persistence` or `noaa-gefs`, and `baseline_id` is `climatology`. The climatology scorecard has no
baseline, so its `passed_gate` is always `false`.

The summary at the end has one line per model and period: `written`, `unchanged` or `FAILED` with the
reason. Exit code 0 means every requested scorecard is in place, 1 means at least one failed, 2 means
the request or an input was refused.

For the whole year, give all twelve periods (`--period 2024-01` ... `--period 2024-12`): 36 scorecards.

### Where a period ends

A period is a calendar month, but its last init time is set by one rule, the same for all three
models:

> The last init of a period is the last one that is inside the validation split and whose longest
> scored lead stays before the sealed split.

Both limits are read from `configs/splits.yaml`; the longest scored lead is the longest of
`evaluation.lead_hours` (48 h), also for a model that lacks it. Init times after that are left out
whole (never scored at their shorter leads only), and the run prints a line saying how many.

With the present config this touches December only. The validation split ends on 2024-12-17T18Z, 14
days before the sealed split, so December is scored from 12-01T00Z to 12-17T12Z: 34 init times at
00Z and 12Z instead of 62. (The sealed-lead limit alone would allow 12-29T12Z, whose 48 h lead ends
on 12-31T12Z; the validation end is the stricter one.) The scorecards record the result in
`period_start`, `period_end` and `n_init_times`.

### Summary tables

```sh
uv run python scripts/run_gate1.py summary --scorecards experiments/ --year 2024
```

prints, from the scorecard files alone, one table per variable with the monthly skill of GEFS and of
persistence over climatology (fair and ecdf), the spread-skill ratios and the gates each fails, and a
list of the single leads at which GEFS scores below climatology. `experiments/SUMMARY_2024.md` holds
that output between two markers, and a test compares the two.

## 5. Rerunning

- The run is deterministic: the same inputs give the same scorecard bytes, on a rerun and on a
  rebuild from scratch.
- Forecast stores that already exist are reused. After the truth, the station list or the baseline
  code changes, delete `$WRFCOMP_DATA_DIR/gate1/` so they are rebuilt.
- A scorecard that already exists is never replaced silently. If the new result differs, the run
  says so and exits 1; pass `--overwrite` once you know why it changed.
- One model at a time: add `--models gefs` (or `climatology`, `persistence`). Climatology is still
  built when needed, because it is the baseline.

## 6. What the runner refuses

- A period that is not a calendar month, that starts before the validation split, or that has no
  scorable init time at all (for example `2025-01`). A month that only runs past the end is not
  refused; it is cut by the rule in "Where a period ends".
- An init hour that is not in `init_hours_utc` of `configs/splits.yaml`.
- Truth that contains any time in the sealed split (checked on the time coordinate, before values
  are read), or that does not span every valid time of the period.
- A data directory inside the repository or under `/tmp`.

It never opens a file dated 2025 or later: GEFS cycles are opened by name for the scored init times
only.

## 7. What is scored, and what still fails

- A point is scored when truth is valid there and the tail threshold of that station, variable and
  month exists. Points without a threshold (no training data for that station and month) are left
  out for every model alike, and each scorecard entry counts them as `n_masked_no_threshold`.
- Climatology and persistence fill their own gaps (see `docs/BASELINES.md`), so a station outage
  around an init time does not fail them.
- `wrfcomp-eval` still refuses a forecast that is not finite at a scored point (exit code 3, the
  first ten points are listed). That fails the scorecard of that model only. Climatology is the
  baseline, so a NaN in it at a scored point would fail the other two as well.
