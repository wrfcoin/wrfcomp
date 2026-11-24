# Scorecards

Committed `wrfcomp-eval` scorecards. Each JSON is the byte-exact output of the harness; do not edit one.

## Quarterly reference scorecards: 2024-Q1 to 2024-Q4

The pinned references of the promotion rule (at least 3 of 4 quarters; `docs/HARNESS_LOCK.md`): climatology,
persistence and NOAA GEFS, each quarter scored as ONE period. 12 scorecards, `<model>_2024-Qn.json`, built on
2026-11-23 from the monthly reference stores already in the data directory. No download, no harness, baseline
or config change, nothing dated 2025.

- A quarter's forecast store is its three monthly stores joined along init time. Before joining, all 36
  monthly stores were checked against the content hashes pinned in `REFERENCES.json`: no mismatch.
- Init times: Q1 182, Q2 182, Q3 184, Q4 156 (Q4 ends at 2024-12-17T12Z, as December does). In every
  quarterly scorecard the number of scored points per variable and lead is the sum of its three months.
- Same truth and thresholds as the monthly run (`truth_sha256` `013466cf...`, `thresholds_sha256`
  `7ee7e1e5...`), harness 0.1.0, protocol 0.1.0. No model passes the gate in any quarter.

```sh
uv run python scripts/run_gate1.py \
  --data-dir "$WRFCOMP_DATA_DIR" \
  --period 2024-Q1 --period 2024-Q2 --period 2024-Q3 --period 2024-Q4 \
  --init-hours 0 12 \
  --out experiments/
```

Run after the twelve months. Wall time 5 min 38 s, peak memory 2.3 GB; the joined stores are
`$WRFCOMP_DATA_DIR/gate1/2024-Qn_h00-12/`, 2.0 GB.

```text
fae63b5edd0ed23934f6789a7249eedbd9763b627767fcb3923d1682f9b3ca0b  climatology_2024-Q1.json
8c60b06796f63e2cc374bb21c5a87aabb108e03ab5b13dd63aaa3eea7784c1c8  climatology_2024-Q2.json
f07873064364bf131e52b5671e1426ecb51a3db60119bf6e9a31ad4bad85067c  climatology_2024-Q3.json
678b309e70a324d477d223a627c715ff3ed768420fbbd0ee2dc3ff26390cc814  climatology_2024-Q4.json
e7b5fcc439c70220168229cd86ebfbf8d531e57b3787cc4423a75b5210822759  gefs_2024-Q1.json
0dd4cf6566a07ec628f3473ae31e27f750a6396385efcfbee470e8c9fb270145  gefs_2024-Q2.json
6f672b46a8071d99bd0edd26c1e85458dbc99b97a72c4ce1455e3972a50ff2a5  gefs_2024-Q3.json
0dba6f894be83b0488e5b3d45b8495072ea99ea5448d993e35cacfeddabbbe2b  gefs_2024-Q4.json
ff1286e4c7b7e5104fb801dd96707c0c1cec12dcf00418d2e55794be362e4b9c  persistence_2024-Q1.json
ab9f22ffce25d4eed859ed1765bcff6e4429924bdac45084a9f228748c3b9a2e  persistence_2024-Q2.json
293bd8d2bb9befc9864f87a67b0275976dc59cc389dd0713737edd84247f540f  persistence_2024-Q3.json
587c979e607602ded892fdc302fb9d8cd760142823d02fac848fd81b68476732  persistence_2024-Q4.json
```

## Gate 1 reference run: all of 2024

Climatology, persistence and NOAA GEFS for every month of the 2024 validation split, 00Z and 12Z runs,
500 stations: 36 scorecards, `<model>_2024-MM.json`. Run on 2026-11-23 by following
[docs/GATE1.md](../docs/GATE1.md). This is a measurement: no harness, baseline or config code was changed
for it. The reading is in [SUMMARY_2024.md](SUMMARY_2024.md).

**Result: no model passes the promotion gate in any month.**

The six January and July scorecards are the ones of the first run (next section), untouched. They were
rerun with the changed runner, on the existing forecast stores and rebuilt from scratch, and came out
byte for byte the same (sha256 below, identical to the list in the next section).

### Coverage

| Months | Init times | Last init |
| --- | --- | --- |
| January to November | every 00Z and 12Z of the month (58 to 62) | 12Z of the last day |
| December | 34 | 2024-12-17T12Z |

December stops at the last init inside the validation split, which `configs/splits.yaml` ends on
2024-12-17T18Z (14 days before the sealed 2025 split). The runner's rule: the last init of a period is the
last one that is inside the validation split and whose longest scored lead (48 h) stays before the sealed
split. The 28 later init times of December are left out for all three models alike. 704 init times in
all. The 24 GEFS cycles of 12-18 to 12-29 in the data directory were not opened. Nothing dated 2025 was
read: every scorecard has `sealed_data_used` false.

### Code

- Runner: commit `b91a04c51d86642169f4b07dd5fa16bb945c4a69` (this branch, on main `a0c14ff`). All 36
  scorecards were regenerated at that commit and reported `unchanged`.
- Harness, baselines and config: identical to commit `0a5ada9` of the first run
  (`git diff 0a5ada9 b91a04c -- src/wrfcomp/harness src/wrfcomp/baselines configs` is empty).
  `wrfcomp` 0.1.0, `scores` 2.7.0, protocol 0.1.0.

### Inputs (under `$WRFCOMP_DATA_DIR`, not in the repository)

Truth, training truth and thresholds are the same files as in the first run (same file sha256, same
content hashes in every scorecard: `truth_sha256` `013466cf...`, `thresholds_sha256` `7ee7e1e5...`;
see the table in the next section).

| Input | Hash |
| --- | --- |
| `baselines/gefs/<YYYYMMDDHH>.zarr`, the 704 cycles used (2024010100 to 2024121712) | sha256 of the sorted store names, one per line: `4034a1fb3d7a829b0b9d2cbc720f901b370bafd291a9f2f27e928f0c4c20413e` |

The content hash of each month's forecast is `forecast_sha256` in its scorecard. In every month the
persistence and GEFS scorecards share one `baseline_sha256`, the `forecast_sha256` of that month's
climatology scorecard.

### Commands

```sh
export WRFCOMP_DATA_DIR="$HOME/wrfcomp-data"
uv sync --extra dev

uv run python scripts/run_gate1.py \
  --data-dir "$WRFCOMP_DATA_DIR" \
  --period 2024-01 --period 2024-02 --period 2024-03 --period 2024-04 \
  --period 2024-05 --period 2024-06 --period 2024-07 --period 2024-08 \
  --period 2024-09 --period 2024-10 --period 2024-11 --period 2024-12 \
  --init-hours 0 12 \
  --out experiments/

# the tables of SUMMARY_2024.md
uv run python scripts/run_gate1.py summary --scorecards experiments/ --year 2024
```

The thresholds file is the one built for the first run (command in the next section).

### Runtimes (one workstation, CPU only)

| Step | Wall time | Peak memory |
| --- | --- | --- |
| January and July rebuilt from scratch (both baselines, GEFS joined, six evaluations) | 2 min 26 s | 5.8 GB |
| The nine months 02 to 06 and 08 to 11 built, January and July reused (33 evaluations) | 11 min 3 s | 6.1 GB |
| December built, eleven months reused (36 evaluations) | 4 min 56 s | 6.0 GB |
| Rerun of all twelve months, every forecast store reused (36 evaluations, all `unchanged`) | 4 min 23 s | 0.9 GB |

Other jobs were running on the workstation during the nine-month step, so its wall time is an upper
figure. Intermediate forecast stores: `$WRFCOMP_DATA_DIR/gate1/`, 2.0 GB.

### Scorecard file sha256

```text
0396c101f3c79e11c7fea76ef622c0a40264b6d490344202f5f522a14d9d69af  climatology_2024-01.json
d67858497ec7d5aba7f144073d8107b0d3b11b7e059136c6b82ba900c2e25274  climatology_2024-02.json
b000360631d127a248bce5ebe66b9d964c6959e123f971e996f5838e31d0a090  climatology_2024-03.json
23eced786dad2ef582b6f0fe87ca70c3aa8a3ccec63491bdc72afb440fcee1df  climatology_2024-04.json
faac05aba56567130e066e80215cf1c2614f9c05638c24be31a0762cd8503cec  climatology_2024-05.json
c2cec4441498d54c6a4373052408a18e05f80db427dae728967fd74704957bba  climatology_2024-06.json
ab20799ec6f7289dc0b717faa16daf90cf596570a7e2af761cd9a209c93a8c52  climatology_2024-07.json
c4f2c5651deb4105e6adb52d07e8814d435ebc3eda31684f37d36a2a1560be9f  climatology_2024-08.json
f9257153e23c347d7c5c69c8022ca76764f30650f19e275e2e49f3d1cedfa0b1  climatology_2024-09.json
adceac558ffc41c3cb6517a6c0eec22263588a5c06509feacfffe26f51200495  climatology_2024-10.json
ba08f52d03b81542b94a2aac65e9afe9d360c31c3aff062cc0512d4b745c4a96  climatology_2024-11.json
35e43384a5ff5e7723801d10e87c3d884d9f456e9e826886e92f4349b128c8d2  climatology_2024-12.json
dc71d58f89160d3e1f6bb6d15b8d345f1364949d65094360aceded652ab18490  gefs_2024-01.json
0c5d1ac75768cbaaf316878b07eda7bd01ee84a895e63501eaa02bb24477a9d1  gefs_2024-02.json
cb32bf7eca79cb66a3b17ea58250737d0ff896c62944faddc2fe1424165c3857  gefs_2024-03.json
ad3aa77152545b48af346f397fca302418f5ad0808a7fcd9aa13bb5d68520786  gefs_2024-04.json
c685a414bc4f31ec027dd3a4f6c9370e45faea1531df74527d7b48d1436c9cce  gefs_2024-05.json
4a163d10d5000d40d367f5a80ac2553c72eb7f74bf37c60d4a3816b4969e5f62  gefs_2024-06.json
21c9596181508588c651e1702e81dc049f309ff1f00f49fd6359dbd024c89a84  gefs_2024-07.json
6a72c6eb1cbbdb4f74300bf9ae66f6ab71739423ba3a68f9e1d7ffdcd7e03061  gefs_2024-08.json
f55956d1e242bda4efacb7ac586b19a11b6457a1b0a2b82ebdc3d770b3f8498f  gefs_2024-09.json
0ee737f6ae4379c27455b9f61eb975ad2c3df4067220a4fb710558e7f1c0c83d  gefs_2024-10.json
5d68494cff55c3a71a28a402724842e9392cdaea68ac1414b679fe4f6d4876cb  gefs_2024-11.json
41f8b378d18cfe0789a38518e231bd8c244238906e2918436422e4d3b795b928  gefs_2024-12.json
78509b787394a552a525f802154e3012c7a9a3ed9dd0454362edea5974b62cd9  persistence_2024-01.json
230c2426fa9f1616b7af2190416bd5256bff16589e5015e4980dc690a870a104  persistence_2024-02.json
6013b927408732eaffad77ed86103d23d8cd5bc38f959e9ee048204a4ddc1022  persistence_2024-03.json
fc23afefaf8f15814de3d3827bde794b3a87c1e15106c0af368d91fb731b54c8  persistence_2024-04.json
76c7665992cbce7b57b85c6f96a18c33c2bb5c4c83efb0292f95d3958d70caf5  persistence_2024-05.json
e607376d3b08fbc82cd57527ff76a38d2865f7462689da2e461696cde70dc503  persistence_2024-06.json
21e94bb6e8970bcf70f19fbe4686d2a948b9675bc5ed6da1176ab8c0de013eb2  persistence_2024-07.json
91acbc22df0e538d81f8324dbc2849c8cb32c7200f6ae7a586c9f2ef09952ee9  persistence_2024-08.json
231e5966befc17b1c71e3e48d16f50df5da054cd2619ac4786920209bfab922b  persistence_2024-09.json
e2dff43756c1ec14675d03ee3287148b0aab0694b0f5e78a28946e21dbb68419  persistence_2024-10.json
3e2ce439ab5a5757cd2386d804a04621237aef69eff9fea50e39f2f936a11e00  persistence_2024-11.json
dc5dd1c105fd0bddae01a0c5ee27bc14c3eae38bbec792450d1e92ade262c095  persistence_2024-12.json
```

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
