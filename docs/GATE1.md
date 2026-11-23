# Gate 1: reproducing the reference scorecards

Gate 1 scores the three reference forecasts (climatology, persistence, NOAA GEFS) on two months of
the 2024 validation split with `wrfcomp-eval`. This page is the exact command sequence. Everything
after step 2 runs offline.

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

January and July 2024 at 00Z and 12Z are 124 cycles. The runner never downloads: it lists the cycles
that are missing and scores the other models.

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

1. builds the climatology and persistence forecasts for every init time of the month at 00Z and 12Z
   (leads 1, 3, 6, 12, 24, 48 h, 30 members each) into
   `$WRFCOMP_DATA_DIR/gate1/<period>_h00-12/{climatology,persistence}.zarr`;
2. joins the 62 GEFS cycle stores of the month into `.../gefs.zarr` (leads 3, 6, 12, 24, 48 h);
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
reason. Exit code 0 means all six are in place, 1 means at least one failed, 2 means the request or
an input was refused.

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

- A period that is not a whole month inside the validation split, or whose last valid time reaches
  2025. With the 48 h lead, `2024-12` is refused.
- An init hour that is not in `init_hours_utc` of `configs/splits.yaml`.
- Truth that contains any time in the sealed split (checked on the time coordinate, before values
  are read), or that does not span every valid time of the period.
- A data directory inside the repository or under `/tmp`.

It never opens a file dated 2025 or later: GEFS cycles are opened by name for the requested months
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
