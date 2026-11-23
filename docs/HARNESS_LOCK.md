# Harness lock

**Locked on 2026-11-23 by the project owner, at harness version 0.1.0** (git tag `harness-v0.1.0`),
after the Gate 1 reference run (`experiments/`).

## What is locked

- `src/wrfcomp/harness/`: scoring rules, truth loader, alignment, content hash, the `wrfcomp-eval` command.
- `configs/`: split boundaries, station list, gate thresholds.
- The lock itself: `HARNESS_LOCK.json`, `scripts/harness_lock.py`, `tests/test_harness_lock.py`, `.github/`.

Everything else (baselines, assimilation, models, scripts, docs, experiments) is open as before.

## Why

Forecasts are judged by this code. If it can move while models are being improved, a better score can come
from an easier test instead of a better forecast. Scores from different harness versions must never be mixed.

## How it is enforced

1. `HARNESS_LOCK.json` records the sha256 of every locked file. CI and the test suite fail if any locked file
   differs, appears or disappears.
2. A pull request that touches a locked path fails CI unless it carries the label
   `owner-approved-harness-change`. Only the project owner applies that label. Agents never apply it,
   never remove the check, and never merge such a pull request without the owner's explicit yes.
3. `main` accepts changes only through pull requests with the `test` check green (repository rule).

## Changing the harness (owner-approved only)

1. Open a pull request with the change and say why in plain words.
2. Bump the version: `protocol_version` in `configs/splits.yaml` for rule changes; then run
   `python3 scripts/harness_lock.py update <new version>` and commit the new manifest.
3. A reviewer from a different model provider reviews it.
4. The owner applies the label. Merge. Tag `harness-v<new version>`.
5. Re-score every baseline under the new version and commit the new scorecards. Old scorecards stay, marked
   with their version; they are not comparable with new ones.

## Unlocking

The owner can lift the lock at any time by approving a pull request that removes it.
