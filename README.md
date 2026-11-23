# wrfcomp: Weather Research & Forecasting Computations

`wrfcomp` is the forecasting research side of [WRFCoin](https://github.com/wrfcoin): a scoring harness that
judges every forecast the same way, the baselines a new forecast has to beat, quality control for citizen
weather stations, and a paper-only payout ledger.

Status: early. The harness is being built. Nothing here pays anyone or touches a chain.

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
experiments/      committed scorecard JSONs
docs/             protocol, tokenomics draft, design notes
```

## Quickstart

Needs [`uv`](https://docs.astral.sh/uv/). Python 3.13 is fetched automatically.

```bash
uv sync --extra dev
uv run ruff check
uv run pytest
```

The `ai` extra (Earth2Studio, torch) needs an NVIDIA GPU and is not required for the harness or the tests.

## Harness lock

`src/wrfcomp/harness/` and `configs/` are **locked at harness version 0.1.0** (2026-11-23). A change there
needs the project owner's approval, a version bump and a re-score of every baseline; CI refuses it otherwise.
See [docs/HARNESS_LOCK.md](docs/HARNESS_LOCK.md). Everything else merges on green CI.

## Data

Downloaded data lives outside the repo (default `~/wrfcomp-data/`, override with `WRFCOMP_DATA_DIR`).
Never commit data files.

## License

Apache-2.0.
