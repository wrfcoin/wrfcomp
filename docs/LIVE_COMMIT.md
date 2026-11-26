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
   oracle or station key. Make the key folder first, owned by you with mode exactly 0700; prepare,
   and `submit.sh` again right before every signing step, refuse any other mode on the folder, and
   anything but 0600 on the key file:

   ```
   install -d -m 0700 ~/wrfcomp-data/live/keys
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

Prepare first checks the published issue against itself: the store's single init (exactly the
issue time, to the nanosecond), its published `v4.sha256`, and its own model attributes (`gefs-station-corrected-own`, version 1) must agree with the requested issue,
`build.json` and the pinned `inputs.json`, because the content hash covers the numbers and not the
attributes. It then writes `~/wrfcomp-data/live/chain/2026-12-02T00Z/` as a whole: `data.json`,
`manifest.json`, `commit.json` and `submit.sh` are staged together and moved into place in one rename,
so an interrupted prepare leaves nothing and can be run again. The operator then runs:

```
bash ~/wrfcomp-data/live/chain/2026-12-02T00Z/submit.sh
```

`submit.sh` stops at the first failure. One run at a time: a second run for the same issue is
refused while one is active. Its steps:

1. Check the CLI binary's sha256, the key folder and file, and that the node
   (`http://127.0.0.1:8556`, the local node; nothing else) is chain 888 `testnet`.
2. Validate the commit offline (`--validate-only`): the CLI's commit id must equal the helper's, and
   the body must name the same forecaster and station. Recheck the key, then claim the commit
   (`commit.claim`) and commit. The commit is never broadcast twice.
3. Wait until the node reports OUR commit (same id, station and type) as `committed_pending_reveal`,
   and save its heights. The reveal must land at a block from `committed_at + 1` to the earliest of
   `committed_at + 3`, `commit_expires_at` and `reveal_expires_at`, inclusive. That is the chain's
   own rule (commit_reveal.rs), with the 3-block window enforced here even if the node advertises
   later expiries, and the commit block itself excluded.
4. The reveal loop. Every attempt runs these steps in this order:
   1. **Freeze** read-only (0400) copies of the prepared `data.json` and `manifest.json` into a new
      folder `reveal_send/<n>/`. Rederive the commit id and the manifest digest from those exact
      bytes, and check that the manifest describes the committed store. A file changed since
      prepare (for example during a backoff) refuses the attempt before anything is sent.
   2. **Validate offline.** The CLI's offline reveal body, made from those frozen files, must
      equal the prepared reveal. The comparison is of the decoded objects: core4 prints the body,
      manifest included, with sorted keys, which is the same manifest. The digest is computed after
      putting the fields back in the Rust declaration order.
   3. **Re-read the node.** If our reveal is already included, nothing is sent. Otherwise:
      - the node must still report OUR commit pending at the saved height;
      - the commit block must be final;
      - then the tip, read **last**, must not be below the commit and must put the next block
        inside the window (tip + 1 ≤ commit + 3, and within the chain's expiries).
   4. **Send** with the same frozen files: claim the reveal (`reveal.claim`) and reveal the data
      object and manifest (no oracle receipt). `reveal.sent` is written only after the send
      returns successfully.

   A send that fails, whether a temporary refusal or a duplicate, does not end the run. The loop
   waits (5, 10, 20, then 30 s) and goes round again from step 1, so the chain decides:
   - our reveal is included: done;
   - the commit is still pending: send again (a duplicate is refused by the chain, harmlessly);
   - the next block is past the window: stop, with no receipt.
5. Wait until the chain reports OUR reveal included (`revealed_pending_attestation` or
   `revealed_attested`, the only two core4 statuses that mean included), and its block at least
   **6 deep**. Depth is tip − height + 1, so the tip block itself is 1 deep. Nothing is saved here.
6. Compute `receipt.json` from the chain as the node has it at that moment. No earlier checkpoint is
   used:
   - the commit block is located by **our commit's own transaction id** (the one the CLI returned
     for our commit id), through core4's `/transactions/<tx_id>/status`. That route needs the
     Reader role, so this step alone gets the bearer: `WRFCOIN_READER_TOKEN`, or else the
     operator bearer, exported inside a subshell (never on a command line), sent only to the
     local node and only on that route, and never printed or saved. The block must carry a
     `ForecastCommit` from our forecaster. Another issue's commit from the same key can never
     stand in for ours;
   - the reveal height is the one the chain reports for our commit id. If the reveal's own
     transaction id is known, its inclusion must be that same block; it is not known when a resend
     was refused as a duplicate. The block must carry our `ForecastReveal`;
   - the reveal must be at commit height + 1 to + 3;
   - everything is read twice, with the tip read last in each pass, after the block reads. The two
     passes must agree (heights and block hashes), and the depth comes from the final tip read. A
     difference (a reorg in progress), or a tip that rolled back below 6 deep, gives no receipt
     yet. Run `resume` later;
   - if the commit's transaction id was lost (interrupted around the commit send), its block cannot
     be located by id. There is then no counted receipt: OPERATOR DECISION.

   The receipt holds:
   - the commit and reveal transaction ids;
   - the confirmed commit and reveal heights, and the window's last block;
   - both block hashes and times;
   - the data sha256 and manifest digest recomputed from the files of every reveal attempt (all
     must equal the prepared ones);
   - which leads count, timed from the confirmed commit block to the nanosecond.

   A replacement chain that moved the commit, or the reveal, before confirmation is therefore
   followed: the receipt names the blocks the chain has, not the ones seen earlier. The status
   (`revealed_pending_attestation` or `revealed_attested`) is not part of the receipt.

   A reveal still pending when the window closes, outside the window, or reported for another
   commit gives no receipt and a non-zero exit. The chain itself refuses a reveal past the commit's
   `commit_expires_at` or `reveal_expires_at`.

Apart from `record`, the helper steps run without any bearer in their environment. Only the CLI
sees the operator bearer, and nothing prints it.

**If a run is interrupted**, run `bash submit.sh resume`. It continues from whatever stage was
saved and never broadcasts the commit twice:

- **prepared:** validate and commit;
- **committed:** wait for the pending commit. A commit that was claimed but is not on chain a few
  blocks later is never sent again: resume stops with OPERATOR DECISION;
- **included:** `status_pending.json` saved; the reveal loop above. If a reveal was claimed but
  `reveal.sent` is missing (killed around the send), the loop's first re-read asks the chain: our
  reveal included means nothing is sent; our commit still pending means it is sent again;
- **revealed:** wait for 6 confirmations and compute the receipt from the chain.

Running `resume` after the receipt recomputes it from the confirmed blocks. If it is identical, for
example after the status moved on to `revealed_attested`, nothing changes. If it differs (a reorg
deeper than 6 blocks), the run stops with OPERATOR DECISION and leaves `receipt.json` as it was.
Every saved checkpoint is rechecked before it is reused. If a run was killed, remove the `active`
folder only after checking that no run is still going.

## Which leads count

The GEFS cycle for T is complete about 4 h after T, so the commit lands after T. A lead counts live
only if its valid time is at least 15 minutes after the commit block's time (docs/LIVE_V4.md),
compared exactly in integer nanoseconds: a lead 1 ns past the cutoff is late.
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

- `/network/info` (chain id, network id, chain height)
- `/transactions/commit/<id>/status`
- `/chain/block/<height>`

Never wrfcoin.com or any other host.

## Not verified

- No commit has been made. Nothing has run against a real CLI or the live node yet.
- The Python commit id matches the core4 formula as read from the source. It has not been compared
  against the Rust binary's output; step 3 of `submit.sh` makes that comparison on the first real run.
