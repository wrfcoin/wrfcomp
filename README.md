# wrfcomp: Weather Research & Forecasting Computations

`wrfcomp` is the forecasting research side of [WRFCoin](https://github.com/wrfcoin): a scoring harness that
judges every forecast the same way, the baselines a new forecast has to beat, quality control for citizen
weather stations, and a paper-only payout ledger.

Status (2026-11-24): the harness is locked at version 0.1.0. One candidate, `gefs-station-corrected`
version 3 (NOAA GEFS corrected per station from recent verified errors), passes the promotion rule on 2024
(all gates against both climatology and raw GEFS in 3 of 4 quarters). That result is **provisional** until
the same, unchanged candidate passes an untouched 2023 confirmation period. Nothing here pays anyone or
touches a chain.

## Design rules

1. **Verification first.** Every forecast is graded by one harness (`wrfcomp.harness`) with proper scoring
   rules (fair CRPS, tail-weighted CRPS, spread-skill, Diebold-Mariano) against real station observations
   from [NOAA ISD](https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database).
   The rules are in [docs/VERIFICATION_PROTOCOL.md](docs/VERIFICATION_PROTOCOL.md).
2. **Strict time splits.** Train through 2023, validate on 2024, and 2025 onward is sealed. The exact
   boundaries are in [configs/splits.yaml](configs/splits.yaml), which is the single source for them.
3. **Real open data only.** No made-up observations and no hardcoded scores.
4. **Shadow economics only.** [docs/TOKENOMICS.md](docs/TOKENOMICS.md) is a research draft run as an
   off-chain ledger.

## Layout

```text
src/wrfcomp/
  harness/        scoring: schemas, truth loader, metrics, wrfcomp-eval CLI
  baselines/      climatology, persistence, NOAA GEFS ensemble
  assimilation/   station quality control and trust scores
  models/         candidate forecast models
configs/          splits.yaml, stations_conus.csv
experiments/      committed reference scorecards (2024 months and quarters), SUMMARY_2024.md
docs/             protocol, harness lock, baselines, candidate models (docs/models/), QC, tokenomics draft
```

## Quickstart

Needs [`uv`](https://docs.astral.sh/uv/). Python 3.13 is fetched automatically.

```bash
uv sync --extra dev
uv run ruff check
uv run pytest
```

To download and decode NOAA GEFS GRIB2 baselines (`wrfcomp.baselines.gefs`), install the optional `grib` extra:

```bash
uv sync --extra dev --extra grib
```

The `ai` extra (Earth2Studio, torch) needs an NVIDIA GPU and is not required for the harness or the tests.

## Platforms

- **Linux:** Full support for scoring, tests, and NOAA GEFS GRIB2 downloads (`uv sync --extra dev --extra grib`).
- **macOS:** Full support for scoring and tests (`uv sync --extra dev`). The prebuilt `eccodeslib` wheels for the `grib` extra require **macOS 15 (Sequoia)** or newer (`macosx_15_0` on Apple Silicon and Intel); earlier macOS releases can still run scoring and tests without `--extra grib`.
- **Windows:** Native Windows is supported for the scoring harness, tests, station QC, and shadow ledger (`uv sync --extra dev`, without `--extra grib`). Because `eccodeslib` does not publish native Windows wheels, use **WSL2** (Linux) for the NOAA GEFS GRIB2 download path.

## Harness lock

`src/wrfcomp/harness/`, `configs/`, the verification protocol, the promotion check and the Gate 1 reference
scorecards are **locked at harness version 0.1.0** (2026-11-23). A change there needs the project owner's
approval on the exact commit, a version bump and a re-score of every baseline. A candidate is promoted only if
it passes every gate against both climatology and NOAA GEFS in at least 3 of the 4 calendar quarters of the
year. What is enforced, and what is not, is in
[docs/HARNESS_LOCK.md](docs/HARNESS_LOCK.md). Everything else merges on green CI.

## How this was built

Most code and reviews here were written by AI coding agents (Claude, Codex and Gemini) working under one
maintainer. Changes to the harness were reviewed by an agent from a different provider than the author, and
locked changes need the project owner's approval on the exact commit. Pull request comments are posted
through the maintainer's GitHub account, so a comment's author is not always the person who wrote it.

## How to get the scoring data

Downloaded data lives outside the repo (default `~/wrfcomp-data/`, override with `WRFCOMP_DATA_DIR`).
Never commit data files.

To score a candidate locally against the locked harness (`scripts/check_promotion.py`), download the
per-year scoring data bundle from Zenodo (truth, p95 tail thresholds, and the pinned monthly and
quarterly climatology and GEFS reference stores — no citizen data and no sealed 2025+ data):

```bash
uv run python -P scripts/fetch_scoring_data.py \
  --url <ZENODO_RECORD_URL> \
  --expected-sha256 <ARCHIVE_SHA256> \
  --year 2024
```

`scripts/fetch_scoring_data.py` verifies every unpacked file against both the archive's
`MANIFEST.json` and the pinned content hashes in `experiments/REFERENCES.json` before installing
into `$WRFCOMP_DATA_DIR`. See [docs/SCORING_DATA.md](docs/SCORING_DATA.md) for bundle contents,
rebuilding instructions (`scripts/build_scoring_bundle.py`), data sources, and licensing.

## License

Apache-2.0.
