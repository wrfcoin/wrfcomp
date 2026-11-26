# Live v4: publication, check-reveal, ledger and the daily run

Deliverables 3–5 of lane I2. They come after the builder (`docs/LIVE_V4.md`) and the commit and
reveal (`docs/LIVE_COMMIT.md`). Testnet 888 only; nothing here pays anyone.

## 1. Where a store is published (zero cost)

Each revealed 00Z issue is published as one gzip tar, attached as an asset to a GitHub release on the
public wrfcomp repository:

- one release per month, tagged `live-v4-YYYY-MM`;
- one asset per issue, `wrfcomp-live-v4-<YYYY-MM-DD>T<HH>Z.tar.gz` (about 2 MB), plus its `.sha256`.

Release assets on a public repository cost nothing and need no new account. They keep a stable
download URL and stay out of the git history. The alternatives were weaker:

| Option | Why not |
|---|---|
| Commit the stores to the repository | Every clone grows by about 2 MB a day, forever. |
| Zenodo-style bundle | A manual deposit for every version; better for a yearly archive than a daily one. |

The asset is published only after the reveal is confirmed on chain, which is after init: the commit
lands at about T + 4 h 15 min.

The archive holds `<issue>/v4.zarr` and the exact revealed `data.json` and `manifest.json`. It also
holds `receipt.json` (heights, block hashes and times, counted leads) and `publication.json` (the
commitment's preimage parts: nonce, timestamp, station label, forecaster, chain). It holds no local
path, key path, CLI path or token. The archive is deterministic: members are sorted, and times and
owners are zero, so the same issue always packs to the same bytes.

```
uv run python -P scripts/live_publish.py pack --init 2026-12-02T00     # <live>/publish/<asset>
```

## 2. check-reveal

```
uv run python -P scripts/live_publish.py check-reveal --init 2026-12-02T00 --archive <file>
```

It takes a published archive, for example one downloaded back from the release, and:

1. **Extracts it safely** into a work folder under `<live>/verify/`, removed afterwards. Only
   regular files and folders, all under one `<issue>/` folder, with caps on the member count and
   size. A link, an absolute path or `..` is refused.
2. **Re-hashes the store** (the harness content hash). The hash must equal `forecast_sha256` in
   `data.json`, `output_hash` in `manifest.json` and `content_sha256` in the receipt. The store
   must be the v4 model at the receipt's init.
3. **Re-derives the commit id** from `data.json`, the nonce, the station label and the timestamp
   (the chain's formula), and the manifest digest. Both must equal the receipt.
4. **Re-reads the chain:**
   - both recorded blocks, with the same hash and time;
   - a ForecastCommit and a ForecastReveal sent by the forecaster in those blocks;
   - the reveal at commit + 1 to + 3;
   - at least 6 blocks after the reveal;
   - while the node still serves it, the commit status reporting our reveal at the recorded height.
5. **Recomputes the counted leads** from the block time read now.

The result is written to `<live>/verify/<issue>.json`. Re-running it is fine. A failure refuses
with the reason, and nothing is written.

**What the chain still shows later.** The commit status shows the reveal only for a limited time.
After that, check-reveal relies on the block reads: hash, time and our sender's transactions. The
receipt, the block hashes and the published archive are then the record (`commit_status_served:
false` in the result). The block read does not show a forecast's commit id. That would need a
read-only node API addition, as the I1 design notes.

## 3. The ledger: on-chain commits as `pofs_shadow` receipts

```
uv run python -P scripts/live_publish.py ledger --init 2026-12-02T00
```

`pofs_shadow` settles from `EpochEvidence`: a map of wallet to `Commitment(digest, received_at)`,
with "a receipt at or after `init_time` is late". `src/wrfcomp/live/ledger.py` builds that evidence
from the issue's `receipt.json`. Before use, the receipt is re-checked on the chain exactly as in
check-reveal (step 4), so a receipt file is never trusted on its own:

- **wallet** = the forecaster address; **digest** = the store's content hash (the hash the evaluator
  uses for a forecast); **received_at** = the commit block's time, from the chain, to the
  nanosecond.
- **Per lead.** The live rule (owner ruling) is that lead L of issue T counts only if T + L is at
  least 15 minutes after the commit block time. One `EpochEvidence` per lead, with
  `init_time = T + L − 15 min + 1 ns`, makes `pofs_shadow`'s existing test exactly that rule: the
  commit is late if and only if its block time is after T + L − 15 min. Nothing in `pofs_shadow` or
  `configs/` changes.
- **gefs_digest** = the content hash of the raw GEFS cycle the issue was built from, as pinned in its
  `inputs.json`.

Reads go to the local node only (`http://127.0.0.1:8556`): unauthenticated GETs at least 3 s apart,
so at most 20 a minute (the limit is 30). They never go to wrfcoin.com or any other host, never
through a proxy, and never follow a redirect. That is about 4 GETs per receipt.

## 4. The daily run

```
bash scripts/live_daily.sh 00            # after about 04:30 UTC: the 00Z issue
bash scripts/live_daily.sh 12            # after about 16:30 UTC: the 12Z cycle (build only)
bash scripts/live_daily.sh 00 2026-12-02 # a given day
```

The 00Z run:
1. fetch the airport readings (last 30 h);
2. fetch the GEFS cycle (skipped if kept);
3. build the v4 store (a published issue is re-verified, not rebuilt);
4. prepare the commit, once (`live_commit.py prepare`);
5. `submit.sh`, or `submit.sh resume` if a commit was already attempted (skipped once the receipt
   exists);
6. pack;
7. create the month's release if it is missing, and upload the asset if it is not there yet;
8. download the asset back and run check-reveal on it.

The 12Z run fetches and builds only. 12Z is not committed, but its cycle keeps the learned state
current. Every step can be re-run.

For 00Z the script needs, in the shell:
- `WRFCOMP_FORECASTER` (the throwaway forecaster address);
- `WRFCOIN_CLI` and `WRFCOIN_CLI_SHA256` (the CLI binary and its sha256);
- `WRFCOIN_OPERATOR_TOKEN`, for the CLI and the record step only;
- `gh`, logged in with permission to create releases on wrfcoin/wrfcomp.

The helper steps run without any bearer. The script unsets every proxy variable first.

**Timer.** Nothing installs one. If the owner approves a timer, two systemd user timers (or cron
lines) at 04:45 and 16:45 UTC running the two commands above are enough. The 00Z one needs the
variables above in its environment.

### Transfer per day

| What | Per day |
|---|---|
| GEFS byte ranges at the 500 stations, 00Z and 12Z | ~184 MB × 2 ≈ 368 MB |
| METAR/SPECI, last 30 h, twice | < 2 MB |
| Upload of the archive | ~2 MB |
| Download back for check-reveal | ~2 MB |
| Node reads | local, no internet transfer |
| **Total** | **about 372 MB/day** |

The ticket's target was under 300 MB/day; building both cycles was accepted at about 370 MB/day.
Building 00Z only would be about 188 MB/day, but the learned state would then miss every 12Z cycle.
The GEFS figure is the 2024 measurement (183.6 MB per cycle); the 2026 index sizes agree
(`docs/LIVE_V4.md`). It has not been measured on a live day yet.
