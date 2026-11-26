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
| Dewpoint above temperature | dewpoint discarded per report, before hourly matching | the same, per report, before hourly matching |
| Report types | FM-15, FM-16 and SY-MT; FM-12 synoptic rows as a fallback for a station-year with none | METAR and SPECI only (FM-15, FM-16); this feed has no SY-MT or FM-12, so a station that sends no METAR has no live truth |
| Quality control | NCEI quality flags; no extra temperature or pressure limits | no flags exist in real time; live-only limits instead: temperature and dewpoint -80..60 deg C, sea-level pressure 850..1090 hPa, wind at most 90 m/s (ISD's limit) |
| Corrections | one record per report | two reports of one station at one exact time: the later fetch wins, among the fetches the issue is allowed to see (below) |

## Causality

- For an issue at T, every report measured after T, or before 1 January 00Z of T's year (exact
  times, to the second), is dropped **before** the hourly slots are chosen (`metar.truth_as_of`).
  A 31 December 23:59:45 report can therefore never become a 1 January 00Z reading at offset 0. A report after T can therefore never take the slot
  of an earlier one, nor be counted as measured at T because its offset rounds to 0 minutes. The
  result is checked again (`metar.assert_causal`), and the frozen model checks the same once more.
- A store for T is not built until a METAR fetch made at least 15 minutes after T exists, so the
  reading at T is final.
- GEFS: only files of cycle T are read (above).

## What each issue saw is frozen

- **When a fetch counts.** A METAR fetch file is named by, and records (`completed_at`), its
  completion time to the microsecond. That time is taken, and the file linked into place, under the
  fetch lock (`live/locks/metar.lock`). A build records its start and lists the eligible fetches
  once, under the same lock: name, completion time and sha256 of every fetch file in place that
  completed at or before the start. So a fetch is either complete and in place before a build starts
  (and may be used), or completed after it (and is never used by that build). Pinned files are
  re-hashed when read; a change is refused. A fetch file without a verified completion time (a
  legacy whole-second name, which recorded the request start, or a body that does not match its
  name) is refused.
- **Fetch window.** From that snapshot, an issue at T sees only fetches completed by its cutoff: the
  earlier of the build start and T + 24 h. The daily run builds the 00Z issue at about T + 4.5 h and
  the previous 12Z issue at about T + 16.5 h, both inside the window. If no fetch completed 15 minutes
  or more after T is inside the window (a delayed first build), the issue is a **retrospective
  backfill**: it sees the fetches up to the first one completed 15 minutes or more after T and
  nothing later, and its manifest says `"retrospective_backfill": true`.
- **Manifest.** `inputs.json` records the build start, the fetch cutoff, the pinned METAR fetches
  (name and sha256), the issue's GEFS cycle hash, and the ordered GEFS history it is replayed over
  (every earlier cycle with its hash, gaps listed as `missing_cycles`).
- **Publication is all or nothing.** A new issue is assembled in `live/.staging/` (manifest, store,
  hash, `build.json`) and moved into place in one rename after its store is built. Only issue folders
  in place are published. A failed build publishes nothing for the issue it failed on and leaves no
  pin; a crashed run can leave a folder under `.staging/`, which is never read.
- **Published issues are immutable.** A new issue extends the history of the latest published issue
  with the kept cycles after it, in order. A cycle that arrives after a later issue was published is
  ignored and listed as `ignored_late_cycles`, and **an issue older than an already published one is
  refused: a gap can never be filled in later.** A pinned cycle or fetch that is missing or changed is
  refused. Each build replays every published issue on the way and must reproduce its store exactly;
  otherwise it stops before anything new is published.
- **Writing.** One build at a time (`live/locks/build.lock`) and one writer per issue
  (`live/locks/<issue>.lock`); a second writer is refused, not queued. Each lock is a thread lock
  (threads of one process) plus an OS file lock (`flock` on Linux and macOS, `msvcrt.locking` on
  Windows; other processes), both released when the holder finishes or the process ends. A lock
  file or lock folder that is a link is refused, for builds and for the METAR writer alike. Every file is staged under an exclusive temporary name and linked into place
  without replacing anything.

## Learned state (v2 regression, v3 pressure gain, v4 own-reading gain)

- The state is never stored. Each build replays every kept live cycle of the year in order with the
  frozen code (`gefs_station_v3.Replay`, then `citizen_arms.Replay` with no citizen data, exactly as
  `scripts/build_v4.py`). Each replayed issue t sees only its pinned fetches and only readings
  measured at or before t, so an error of an earlier issue enters the state only once its verifying
  reading has been measured and fetched. Anyone with the kept cycles (about 1 MB each), the stored
  METAR fetches and the `inputs.json` files can recompute it.
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

The frozen code refuses every time from 2025-01-01 on. Live builds do not switch that off anywhere
shared. A `wrfcomp.live.dates.LiveYear(year)` policy object is passed explicitly down the live path.
It holds its own private copies of the frozen modules, loaded from the same source files (their
sha256 is checked and recorded in each `build.json`). Only in those copies:

1. `gefs_station_v2.year_bounds` accepts the live year and refuses every other year (a 2026 policy
   refuses 2025 and 2027);
2. `baselines.io.layout` (used by the GEFS extractor's copy) reads a config whose sealed-test start
   is 1 January after the live year;
3. `harness.schemas.validate_forecast_dataset` is called with that config passed as `cfg=`.

No attribute of the imported `wrfcomp` modules and nothing under `configs/` is changed. Ordinary
calls anywhere in the process, other threads included, keep refusing every 2025-or-later time while
a live build runs; a test checks all three guards from a second thread during a live replay. The
truth parsers' cutoff is not involved: live readings never go through them. A store is written only
if every init and valid time is inside the live year.

Equivalence: a test replays a synthetic 2024 world through the live replay and through the frozen
build path (identical content hashes), and the same world moved to 2026 through a `LiveYear(2026)`
policy (identical numbers).

## Where live data may go

- The data directory is `$WRFCOMP_DATA_DIR` (default `~/wrfcomp-data`). The script has no option to
  point live output elsewhere.
- Refused:
  - a data directory inside the repository or a temp directory;
  - a data directory that is, or is inside, an evaluation folder (`truth`, `candidates`, `baselines`,
    `phase3`, `gate1` and the others the pipeline uses);
  - a `live` folder that is a symlink or holds an evaluation folder;
  - any write whose FULL target path, resolved with `os.path.realpath` (every link on the way,
    the last component included), is not inside the real live folder. This is checked before every
    write: fetch files, locks, staging, issue folders, GEFS cycles and every entry of the GEFS point
    cache (before and after each fetch). Links are also refused by name with a clearer message.
- The check sits in the lowest-level writers themselves: `metar.publish_file`, `locks.file_lock`,
  and the two write paths of the frozen code a `LiveYear` policy carries (its private
  `io.write` and `gefs.extract`, wrapped). A `LiveYear` is bound to one live root; without one it can
  replay but not write. Everything else writes through these, so no library entry point can write
  around the check.
- A write-audit test runs a full synthetic live run (METAR fetch, GEFS fetch through the extractor
  and its cache, build) with every filesystem write primitive recorded (`open` in write modes,
  `os.open` with write flags, renames, links, folders, removals, `shutil`, temp files, Zarr and
  NetCDF writes, `Path.write_*`). Each write CALL is judged on its own target: outside a write
  scope, that exact target must have been approved by the check before the call (an approved folder
  does not cover its contents), and each approval is single-use, consumed by exactly one write; inside a `write_scope(root, target)` (used for a Zarr store, the
  GEFS point cache and a staged file linked into place, whose inner names are not known in advance),
  the call must stay under the scope's own checked target or its own temporary files. Three mutation
  tests prove the audit bites: with the GEFS cache guard disabled it fails on the cache writes, and
  with an extra unchecked metadata file next to each published file it fails on exactly those, and
  with each published file written a second time without a new check it fails on those second
  writes.
- **Live writes are refused on Windows**, at every entry point (`build_issue`, `fetch_cycle`,
  `metar.fetch`, the script) and again in every writer. Directory junctions and locked files make a full-path
  check unreliable there, and the daily live run is on Linux. Reading, tests and the evaluation code
  are unaffected; the live writer tests run on Linux and macOS, and a test checks the Windows
  refusal on every platform.

## Output

`$WRFCOMP_DATA_DIR/live/<YYYY-MM-DD>T<HH>Z/`: `v4.zarr`, `v4.sha256` (its `content_sha256`) and
`build.json` (code commit, frozen source hashes, live start, missing cycles, the learned gains per
issue), and `inputs.json` (the pinned METAR fetches and GEFS cycle hash). Every issue folder is
written once. Kept inputs: `live/gefs/<YYYYMMDDHH>.zarr`, `live/metar/*.json`.

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
