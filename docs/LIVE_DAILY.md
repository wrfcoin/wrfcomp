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

The asset is packed, and so published, only after two checks pass, which is after init (the commit
lands at about T + 4 h 15 min):
- the issue's files are verified as in check-reveal steps 2 and 3;
- the chain confirms the reveal now, as in step 4: the commit status served, both blocks unchanged,
  6 blocks after the reveal.

A `receipt.json` file alone never publishes anything.

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
2. **Verifies the files** (`publish.verify_files`):
   - the receipt is for testnet 888 and this issue, with the 6-block rule and the commit + 1 to + 3
     window, and `publication.json` agrees with it;
   - `data.json` is the compact revealed form and hashes to the receipt. It names the v4 model at
     this init and the receipt's content hash;
   - the manifest hashes to the receipt. It names the same model, the 48 h horizon, the content hash
     as `output_hash`, and this checkout's `HARNESS_LOCK.json` as `input_hash`. An older archive is
     therefore checked from the harness version it names;
   - the store is a real folder whose content hash is the committed one, the v4 model at this
     init.
3. **Re-derives the commit id** from `data.json`, the nonce, the station label and the timestamp
   (the chain's formula). It must equal the receipt.
4. **Re-reads the chain** (`publish.chain_check`), twice, with the tip read last in each pass:
   - `/network/info` must say chain 888, testnet;
   - the commit status must be **served** for that commit id and name our station label and our
     reveal, included, at the recorded height. A 404 or `not_found` is never a pass: the check is
     then **inconclusive** and refused;
   - both recorded blocks, with the same hash and time, carrying a ForecastCommit and a
     ForecastReveal sent by the forecaster;
   - the reveal at commit + 1 to + 3, with at least 6 blocks after it;
   - with the reader bearer (not used by the daily run), also the commit transaction's inclusion
     at the recorded height.
5. **Recomputes the counted and late leads** from the block time read now. Both must equal the
   receipt.

The result is written to `<live>/verify/<issue>.json`. Re-running it is fine. A failure refuses
with the reason, and nothing is written.

**What the chain still shows later.** The commit status shows the reveal only for a limited time.
After that, check-reveal refuses as inconclusive: the block read does not show a forecast's commit
id, so the commitment cannot be confirmed from blocks alone. The verify record written while it
was served, the receipt and the published archive remain. A read-only node API addition (the commit
id of forecast transactions in the block read, as the I1 design notes) would make later checks
possible.

## 3. The ledger: on-chain commits as `pofs_shadow` receipts

```
uv run python -P scripts/live_publish.py ledger --init 2026-12-02T00
```

`pofs_shadow` settles from `EpochEvidence(init_time, gefs_digest, commitments)`. Its contract:
`init_time` is the epoch's init (it must equal the reference scorecard's `period_start`), and a
commitment with `received_at >= init_time` is late. `src/wrfcomp/live/ledger.py` builds that evidence
for our issue only from a **verified** receipt. The issue's own files and store are checked as in
check-reveal steps 2 and 3, which binds the receipt's content hash to what `data.json` commits to and
to the local store. The chain is then checked as in step 4: served status, never a 404. A receipt
file is never trusted on its own.

- **wallet** = the forecaster address; **digest** = the store's content hash (the hash the evaluator
  puts in a scorecard's `forecast_sha256`); **init_time** = the issue's init T;
  **gefs_digest** = the content hash of the raw GEFS cycle the issue was built from, as pinned in its
  `inputs.json`.
- **Per lead.** The live rule (owner ruling) is that lead L counts only if T + L is at least 15
  minutes after the commit block time. `pofs_shadow` pays a scorecard only over all the leads it
  scores, with one receipt time. So `received_at` is the commit block time adjusted for the
  shortest scored lead: `block time + 15 min − min(leads) − 1 ns`. It is timely (`< T`) if and only
  if every scored lead counts.
- **Settling.** A live issue is settled with a scorecard over the leads it counts: 6, 12, 24 and 48 h
  for a commit near T + 4 h 15 min. A scorecard that also scores a lead the commit was too late for
  is late as a whole. The end-to-end test settles through `settle_from_scorecards`:
  - a verified receipt is paid for its counted leads;
  - the same receipt with lead 3 added is `late` and unpaid;
  - a forged digest is `non_reveal_or_mismatch`;
  - an unverified receipt gives no evidence at all.

  Nothing in `pofs_shadow` or `configs/` changes.

Reads go to the local node only (`http://127.0.0.1:8556`): unauthenticated GETs at least 3 s apart,
so at most 20 a minute (the limit is 30). They never go to wrfcoin.com or any other host, never
through a proxy, and never follow a redirect. That is about 10 GETs per receipt (two passes).

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
7. create the month's release if it is missing (from `gh release list`). Upload the asset and its
   `.sha256` together if neither is there. If only one of the two is there, stop with OPERATOR
   DECISION; nothing is replaced;
8. download both into a fresh folder. The asset's sha256 must equal its `.sha256` and the archive
   packed here (`verify-download`). Then run check-reveal on it.

Every `gh` call runs without the node bearers in its environment. Any `gh` failure stops the run;
it never falls through to an upload.

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
