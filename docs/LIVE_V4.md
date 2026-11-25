# Live v4 forecasts (testnet operation)

Candidate v4 (v3 plus the airport's own current reading, the frozen Phase 3 arm B) run every day on
current NOAA data. Owner ruling 2026-11-25. Code: `src/wrfcomp/live/`, `scripts/live_v4.py`.

Live data is for live operation only. It lives under `$WRFCOMP_DATA_DIR/live/`, is never written
under the evaluation folders, is never scored as a 2023 or 2024 result, and is never mixed into any
training, threshold or reference file. The sealed-split rule for evaluation is unchanged.

## What is forecast, and which leads count

- One store per issue time T at 00Z, built from the GEFS cycle that starts at T and the airport
  reading at T. That is the frozen model: its issue time is the GEFS cycle time.
- The GEFS cycle that starts at T is complete on NOAA's bucket about **4 h 05 min after T**
  (measured 2026-11-24, 00Z and 12Z). So a store for T can be committed on chain only after that,
  about T + 4 h 15 min, never before T.
- **A lead counts live only if its valid time is at least 15 minutes after the block time of the
  commit.** With a commit at about T + 4 h 15 min, leads 6, 12, 24 and 48 h count. Leads 1 h and 3 h
  are in the store and published, but they have already happened when the commit lands, so they are
  never scored live.
- v4 differs from v3 only at leads 1, 3 and 6 h (t2m, td2m). **Live, v4 differs from v3 only at the
  6 h lead in practice**; at 12, 24 and 48 h it is v3.
- The 12Z cycle is built every day too, and committing it is optional. v4 needs it: the airport-reading
  term at a 00Z issue is the reading at T minus the v3 12 h forecast of the issue 12 hours earlier.
  Without the 12Z issue, v4 at 00Z would be exactly v3.

## Inputs

**GEFS:** `wrfcomp.baselines.gefs.extract`, unchanged from the 2024 pipeline: the same NOAA bucket,
byte ranges taken from each file's index, the same 0.5-degree fields, bilinear point extraction,
height lapse and dewpoint. The live transfer also refuses any file that is not of cycle T and any
index line (of a field v4 uses) whose reference time, lead or member differs from its file.

**Airport readings:** METAR and SPECI from the NOAA Aviation Weather Center Data API
(`https://aviationweather.gov/api/data/metar`, JSON, free). Every fetch is stored unchanged; the
hourly truth is rebuilt from the stored fetches with the harness rule (nearest report within
15 minutes of the hour, earlier one on a tie, `X_offset_min` recorded). Station ids map to ICAO ids
through NOAA's `isd-history.csv` (`src/wrfcomp/live/stations_icao.csv`; all 500 are unique).

Differences from the 2024 truth (NOAA ISD) and how they are handled:

| | ISD (2024 truth) | Live METAR |
|---|---|---|
| Temperature, dewpoint | tenths of deg C | tenths from the remarks T-group (as ISD for ASOS); whole degrees from the body when there is no T-group |
| Pressure | `SLP`, tenths of hPa | remarks `SLPppp`, tenths of hPa; altimeter never used; no SLP gives NaN |
| Wind | tenths of m/s, converted from knots | knots to m/s rounded to 0.1 (5 kt = 2.6 m/s); calm = 0, 0; variable or missing direction = NaN; not in knots = NaN |
| Report types | FM-15, FM-16 | METAR, SPECI (the same reports) |
| Quality control | NCEI quality flags | none exist in real time; ISD's range limits applied; a correction replaces the report it corrects |

## Causality

- An issue at T is given only truth labels up to T, and inside them only readings measured at or
  before T (`metar.causal`). The code refuses otherwise (`metar.assert_causal`); the frozen model
  checks the same again.
- A store for T is not built until a METAR fetch made at least 15 minutes after T exists, so the
  reading at T is final.
- GEFS: only files of cycle T are read (above).

## Learned state (v2 regression, v3 pressure gain, v4 own-reading gain)

- The state is never stored. Each build replays every kept live cycle of the year in order with the
  frozen code (`gefs_station_v3.Replay`, then `citizen_arms.Replay` with no citizen data, exactly as
  `scripts/build_v4.py`). Each replayed issue t sees only the truth causal at t, so an error of an
  earlier issue enters the state only once its verifying reading has been measured. Anyone with the
  kept cycles (about 1 MB each) and the stored METAR fetches can recompute it.
- **Cold start.** The frozen models forget everything older than their windows (v2 60 days; v3 and
  v4 gains 30 days; same-hour issues only) and keep nothing across a calendar year. Nothing from the
  frozen 2023 or 2024 history therefore has any weight live. Live starts cold on the first live day,
  exactly as the frozen replay started every 1 January (and January 2024 was scored that way): v2
  uses four times the member spread until a station and lead has 10 verified pairs (about 10 to 12
  days), v3 uses its prior pressure gain, and v4's gain starts at 0 (v4 = v3 on day one).
- **30 to 31 December gap and 1 January reset.** The frozen code refuses an issue whose 48 h lead
  crosses 1 January, so there is no live forecast for 30 December 00Z and 12Z and 31 December 00Z
  and 12Z. On 1 January the state starts cold again.
- Missing days (a failed fetch) are recorded in `build.json` (`missing_cycles`); the model then has
  fewer past pairs, as it would with missing truth.

## The 2025 date checks

The frozen code refuses every time from 2025-01-01 on. `wrfcomp.live.dates.live_dates(year)` lifts
exactly these checks, for one live year, only while a live build runs, and restores them on exit:

1. `gefs_station_v2.year_bounds` (`END`), used by the v2, v3 and v4 replays;
2. `baselines.io.layout`, used by the GEFS extractor;
3. `harness.schemas.validate_forecast_dataset`, used when a store is checked or written.

No model file and nothing under `configs/` is changed. The truth parsers' cutoff stays: live readings
never go through them. A test replays a synthetic 2024 world through the live path and through the
frozen build path (identical content hashes), and the same world moved to 2026 under `live_dates`
(identical numbers).

## Output

`$WRFCOMP_DATA_DIR/live/<YYYY-MM-DD>T<HH>Z/`: `v4.zarr`, `v4.sha256` (its `content_sha256`) and
`build.json` (code commit, live start, missing cycles, hashes of every GEFS cycle and METAR fetch
used, the learned gains per issue). Kept inputs: `live/gefs/<YYYYMMDDHH>.zarr`, `live/metar/*.json`.

## Transfer

GEFS: about 184 MB per cycle (2024 measured 183.6 MB; 2026 index sizes agree), two cycles a day, so
about 370 MB a day, about 11 GB a month. METAR: under 1 MB a day (gzip). No download happens without
`--network`.

## Commands

```
uv sync --extra dev --extra grib
uv run python scripts/live_v4.py fetch-gefs --init 2026-12-02T00 --network
uv run python scripts/live_v4.py fetch-metar --hours 30 --network
uv run python scripts/live_v4.py build --init 2026-12-02T00
```
