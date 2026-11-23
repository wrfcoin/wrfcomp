# Scorecards

Committed `wrfcomp-eval` scorecards. Each JSON is the byte-exact output of the harness; do not edit one.

## Gate 1 reference run: January and July 2024

Climatology, persistence and NOAA GEFS on the 2024 validation split, 00Z and 12Z runs, 500 stations.
Run on 2026-11-23 by following [docs/GATE1.md](../docs/GATE1.md). This is a measurement: no harness,
baseline or config code was changed for it.

**Result: no model passes the promotion gate** (20 members, skill above 5 %, set in
`configs/splits.yaml`). Climatology has no baseline, so it cannot pass by construction.

### Code

- Repository commit `0a5ada9d60d1dd097da3a52eefa411fd2769e03f` (main), `wrfcomp` 0.1.0, `scores` 2.7.0,
  protocol 0.1.0.

### Inputs (under `$WRFCOMP_DATA_DIR`, not in the repository)

| Input | File sha256 | Content hash in the scorecards |
| --- | --- | --- |
| `truth/isd_conus_2024.nc` (2023-12-31T18Z to 2024-12-31T23Z, 500 stations) | `5e6a5131a26b7181afc8bfdafef6d548ee0791b48f5506375d53d6f4f65d466f` | `truth_sha256` = `013466cf5348dd9d86ede15fcd847f436050bb7ba754c619f87043d0e1aede05` |
| `truth/isd_conus_train.nc` (2014-01-01T01Z to 2023-12-17T18Z, 500 stations) | `4659ca34a2fc7b1dcce2d56ea181048668e8213f05c91e5f3a5079e95e7639ac` | not scored; feeds thresholds and the two historical baselines |
| `thresholds/tail_p95.nc` (built in step 1 below; 0 of 6000 station-months missing for every variable) | `ad0aebe7d61d616e7466cbaa595a2e8ef180218067d4bf9d768b2abd07972816` | `thresholds_sha256` = `7ee7e1e5428d2736ab4f18ef9fe8f7b218c7038d4b377e54b18251336587864c` |
| `baselines/gefs/<YYYYMMDDHH>.zarr`, 124 cycles (62 per month) | sha256 of the sorted store names: `8c96c3682f3922fb6121332080e4193739022713c8a2117190c773918cf509bc` | see `forecast_sha256` of the GEFS scorecards |

The file hash of a NetCDF file depends on how it was written; the content hash in a scorecard does not.

| Scorecard | Model | `forecast_sha256` |
| --- | --- | --- |
| `climatology_2024-01.json` | `climatology` v3 | `ff1823d8f20d6a4af9a6824af28220218f800692d0bde5da36c83c629a636f53` |
| `persistence_2024-01.json` | `persistence` v3 | `4b877417821d04b38653c9e0540c9779f97cde094d1f12484b9be0b15bbfa6db` |
| `gefs_2024-01.json` | `noaa-gefs` v1 | `b95374540f6a12dcff2e0d3e9da1b68debdd031c54a307dac7590e3edd28c039` |
| `climatology_2024-07.json` | `climatology` v3 | `e45229867bfe9987132bbbce3274cd931c59d4768af60bfec907a100c32707af` |
| `persistence_2024-07.json` | `persistence` v3 | `0acef9dfcd1d58c686a3013214c7ba7ca83362c1c15b3b23a9fb8998689f9499` |
| `gefs_2024-07.json` | `noaa-gefs` v1 | `84f6122a33d72de00cddc62e312135a09a9fc18efb2b6fa1f4fea1ed81bc1d02` |

The persistence and GEFS scorecards of a month share one `baseline_sha256`, which is the
`forecast_sha256` of that month's climatology scorecard.

### Commands

```sh
export WRFCOMP_DATA_DIR="$HOME/wrfcomp-data"
uv sync --extra dev

# 1. tail thresholds from the training truth
uv run python scripts/build_thresholds.py \
  --train-truth "$WRFCOMP_DATA_DIR/truth/isd_conus_train.nc" \
  --out "$WRFCOMP_DATA_DIR/thresholds/tail_p95.nc"

# 2. forecasts and scorecards
uv run python scripts/run_gate1.py \
  --data-dir "$WRFCOMP_DATA_DIR" \
  --period 2024-01 --period 2024-07 \
  --init-hours 0 12 \
  --out experiments/
```

### Runtimes (one workstation, CPU only)

| Step | Wall time | Peak memory |
| --- | --- | --- |
| `build_thresholds.py` | 9 s | 0.7 GB |
| `run_gate1.py`, first run (builds both baselines for both months, joins GEFS, six evaluations) | 2 min 33 s | 5.8 GB |
| `run_gate1.py`, rerun (forecast stores reused, six evaluations) | 47 s | 0.9 GB |

The rerun reported all six scorecards as `unchanged` (identical bytes). Intermediate forecast stores:
`$WRFCOMP_DATA_DIR/gate1/`, 350 MB.

### Scorecard file sha256

```text
0396c101f3c79e11c7fea76ef622c0a40264b6d490344202f5f522a14d9d69af  climatology_2024-01.json
ab20799ec6f7289dc0b717faa16daf90cf596570a7e2af761cd9a209c93a8c52  climatology_2024-07.json
dc71d58f89160d3e1f6bb6d15b8d345f1364949d65094360aceded652ab18490  gefs_2024-01.json
21c9596181508588c651e1702e81dc049f309ff1f00f49fd6359dbd024c89a84  gefs_2024-07.json
78509b787394a552a525f802154e3012c7a9a3ed9dd0454362edea5974b62cd9  persistence_2024-01.json
21e94bb6e8970bcf70f19fbe4686d2a948b9675bc5ed6da1176ab8c0de013eb2  persistence_2024-07.json
```
