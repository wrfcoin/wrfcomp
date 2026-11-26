# Committing a live v4 issue on testnet 888

Lane I2, deliverable 2. It follows the I1 design (stage 2), and needs no consensus or node change.
Code: `src/wrfcomp/live/chain.py`, `scripts/live_commit.py`.

Each 00Z live v4 issue is committed on chain, then revealed:

- **What is committed.** The commit holds the hash of a small data object: the store's
  `content_sha256`, the sha256 of its pinned `inputs.json`, the init, the model id and version.
- **What is revealed.** Within the chain's reveal window (3 blocks, about 4 minutes), the reveal
  discloses that object and a manifest whose `output_hash` is the store hash.
- **The store itself** is published off chain after its init and checked against the hash
  (deliverable 3).
- **Testnet 888 only.** Nothing here pays anyone: the fee and stake are zero, and the data object
  carries no weather field, no `temp_c_centi` or `pressure_pa`, and no schema marker. The station id
  is the label `wrfcomp-live-v4`, never a real or registered station.

The helper never signs or submits anything. It never opens the key file and never reads a token.
The operator runs one script.

## One-time setup (operator)

1. **CLI binary.** Build the `wrfcoin` CLI on a build machine that runs no live node, or take it from
   a release. Note its sha256.
2. **Throwaway forecaster key.** Make it with the CLI, made for this purpose only: never a producer,
   oracle or station key.

   ```
   wrfcoin --output json session-key keygen --out ~/wrfcomp-data/live/keys/forecaster.key
   ```

   The file is a 32-byte ed25519 seed in hex with mode 0600. Its `session_key_address` is the
   forecaster address.
3. **Bearer.** The CLI reveal sends no signature, so the node accepts it only from an Operator bearer
   (I1 design, section 3). `submit.sh` uses `WRFCOIN_OPERATOR_TOKEN` from the operator's shell and
   never prints it.

## Each day, for the 00Z issue

Prepare the commit files. This makes no network call and signs nothing:

```
uv run python -P scripts/live_commit.py prepare --init 2026-12-02T00 \
    --forecaster 0x... --signer-key ~/wrfcomp-data/live/keys/forecaster.key \
    --cli /path/to/wrfcoin --cli-sha256 <sha256>
```

This writes `~/wrfcomp-data/live/chain/2026-12-02T00Z/` once: `data.json`, `manifest.json`,
`commit.json` and `submit.sh`. The operator then runs:

```
bash ~/wrfcomp-data/live/chain/2026-12-02T00Z/submit.sh
```

`submit.sh` stops at the first failure. Its steps:

1. Check that the CLI binary's sha256 equals the one recorded at prepare.
2. Check that the node (`http://127.0.0.1:8556`, the local node; nothing else) is chain 888 `testnet`.
3. Validate the commit offline (`--validate-only`). The CLI's commit id must equal the helper's,
   and the body must name the same forecaster and station.
4. Commit.
5. Read the commit status every 20 s until it is `committed_pending_reveal`.
6. Reveal the data object and the manifest. No oracle receipt is attached.
7. Write `receipt.json`:
   - the commit and reveal transaction ids;
   - the commit height, block hash and block time;
   - the reveal status;
   - which leads count.

The script refuses to run again once a commit has been attempted in that folder. If a run stops
after step 4, check the commit status before doing anything else. A reveal later than 3 blocks
after the commit is refused by the chain.

## Which leads count

The GEFS cycle for T is complete about 4 h after T, so the commit lands after T. A lead counts live
only if its valid time is at least 15 minutes after the commit block's time (docs/LIVE_V4.md).
`receipt.json` lists `counted_leads_hours` and `late_leads_hours`. With a commit at about T + 4 h 15 min,
leads 6, 12, 24 and 48 count. Only 00Z issues are committed; 12Z needs `--allow-12z`.

## Hashing, as the chain does it

- **Commit id.** sha3-256 of `"{data}||{nonce}||{station_id}||{timestamp}||forecast"`. Here `data`
  is the data object as compact JSON with sorted keys: core4's serde_json has no `preserve_order`.
  The data object carries strings only, so number formatting cannot differ between Python and Rust.
- **Manifest.** The fields are written in the Rust declaration order:
  `schema_version, model_id, input_hash, output_hash, produced_at, region, horizon_hours,
  input_variables, output_variables`.
  - `input_hash` is the sha256 of `HARNESS_LOCK.json`, as in the I1 design.
  - `output_hash` is the store hash.
  - `produced_at` is the build start.
  - `horizon_hours` is 48.
  - The variables are `msl, t2m, td2m, u10, v10`.
  - The manifest digest is sha3-256 of its compact JSON.

## Reads

Only unauthenticated GETs to the local node, at least 3 s apart:

- `/network/info`
- `/transactions/commit/<id>/status`
- `/chain/block/<height>`

Never wrfcoin.com or any other host.

## Not verified

- No commit has been made. Nothing has run against a real CLI or the live node yet.
- The Python commit id matches the core4 formula as read from the source. It has not been compared
  against the Rust binary's output; step 3 of `submit.sh` makes that comparison on the first real run.
