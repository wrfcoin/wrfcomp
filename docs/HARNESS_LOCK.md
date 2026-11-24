# Harness lock

**Locked on 2026-11-23 by the project owner, at harness version 0.1.0**, after the Gate 1 reference run
(`experiments/`). Two owner decisions are behind it: the scoring harness is frozen at 0.1.0, and a candidate is
promoted only if it passes the gate against **both** climatology and NOAA GEFS.

This page says exactly what is enforced, by what, and what is not enforced.

## Why

Forecasts are judged by this code. If it can move while models are being improved, a better score can come
from an easier test instead of a better forecast. Scores from different harness versions must never be mixed.

## What is locked

| Locked | Hashed in `HARNESS_LOCK.json` | Needs the owner label to change |
| --- | --- | --- |
| `src/wrfcomp/harness/` (scoring rules, truth loader, alignment, content hash, `wrfcomp-eval`) | yes | yes |
| `configs/` (split boundaries, station list, gate thresholds) | yes | yes |
| `docs/VERIFICATION_PROTOCOL.md`, `scripts/check_promotion.py` | yes | yes |
| `experiments/REFERENCES.json` and the six Gate 1 scorecards (climatology, GEFS, persistence; 2024-01, 2024-07) | yes | yes |
| Versions of `scores`, `numpy`, `xarray`, `scipy`, `pandas` in `uv.lock` | yes (as versions) | yes, through the manifest |
| The lock itself: `HARNESS_LOCK.json`, `scripts/harness_lock.py`, `tests/test_harness_lock.py`, `tests/test_check_promotion.py`, `docs/HARNESS_LOCK.md`, everything under `.github/` | no | yes |

Everything else (baselines, assimilation, models, other scripts and docs, new scorecards, other dependencies)
is open as before.

## The three checks

### 1. `test` (`.github/workflows/ci.yml`): the tree is the locked tree

Runs `python3 scripts/harness_lock.py check`, then ruff and the test suite. The check reads the committed Git
tree (`git ls-tree -r HEAD`) and fails if

- a locked path was changed, added, removed or renamed (the manifest records every path with its Git blob id
  and mode; a rename shows as one path removed and one added);
- a locked entry is not a regular, non-executable file: a symlink, an executable or a submodule;
- there is an untracked, ignored or uncommitted file inside the locked scope (Python bytecode caches excepted);
- the manifest's harness version differs from `src/wrfcomp/__init__.py`, or its scope from the scope written
  in `scripts/harness_lock.py`;
- `uv.lock` resolves `scores`, `numpy`, `xarray`, `scipy` or `pandas` to a version other than the recorded one.

This check runs the pull request's own copy of the script. A pull request that edits the script, the manifest
or the workflow can therefore make it say anything. That is why the second check exists.

### 2. `harness-lock` (`.github/workflows/harness-lock.yml`): the owner approved this exact commit

Triggered by `pull_request_target`, so GitHub runs the workflow file of the **base** branch, not the pull
request's. It checks out nothing. It fetches its gate script (`.github/harness_lock_gate.py`) from the same
base commit through the API and reads the pull request through the API. Code from the pull request is never
executed by this workflow, so a pull request cannot change its own gate.

The gate lists every changed path (paginated, including the previous name of a rename). If none is locked, it
passes. If any is locked, it passes only when all of these hold:

1. the pull request carries the label `owner-approved-harness-change`;
2. the latest `labeled` event for that label was made by the owner's account (`tonykoop`), not by a bot or
   another account. A workflow inside the pull request that labels its own pull request does not count;
3. that event is newer than the pull request's latest head change in the timeline (latest commit date, force
   push, base change or reopen);
4. the run was not started by a new push. A push withdraws approval: a second job removes the label, and the
   gate fails that run outright. Every new commit needs a fresh label.

It fails closed: an API error, a file list GitHub cut off, or a head that moved during the run count as not
approved.

### 3. `scripts/check_promotion.py evaluate`: the candidate beat both references

```sh
uv run python scripts/check_promotion.py evaluate \
  --forecast <candidate.zarr> --data-dir "$WRFCOMP_DATA_DIR" --period 2024-01 --out <dir>
```

It refuses to run unless check 1 passes for the checkout it runs in. It takes the two reference forecasts of
the period from the places named in `experiments/REFERENCES.json` under the data directory, and refuses a
store whose content hash is not the pinned one. It then runs `wrfcomp-eval` twice itself, with the locked
config, and judges the two scorecards it has just produced:

- each validates as a harness scorecard, has all five primary variables, and its gates and `passed_gate` are
  what the harness computes from its own metrics;
- harness version, protocol version, truth, thresholds, period, number of init times, init hours and number
  of stations equal the pinned record; the baseline is the pinned reference forecast; no sealed data;
- scored leads are all v1 leads against climatology and 3 h and longer against GEFS, with the same number of
  scored and masked points per variable and lead as the reference scorecard;
- the candidate is not a reference, by model id or by forecast content;
- both gates are passed.

It prints `PROMOTED` only then, and only in this mode. Promotion is per period: run it for each pinned period
(`2024-01` and `2024-07`). No reference directory, reference file or config can be passed in.

`check_promotion.py verify A.json B.json --period P` applies the same rules to two scorecards that already
exist. It prints `CONSISTENT` or `INCONSISTENT`, never `PROMOTED`.

## What is NOT enforced

- **GitHub cannot tell the owner from an agent.** All agents in this project act through the owner's GitHub
  account. To GitHub, a label applied by an agent and a label applied by the owner are the same event from
  the same account. The `harness-lock` check proves that the owner's *account* approved the exact commit; it
  cannot prove that the owner did. The working rule is therefore a rule for agents, not a control: **only
  the owner applies the label `owner-approved-harness-change`; agents never apply it, never ask a tool to
  apply it, and never merge a pull request that carries it without the owner's explicit yes.** Separate
  accounts for agents would turn this rule into a control.
- **Nothing is enforced until the repository rules exist.** As of this change, `main` is not protected.
  Both `harness-lock` and `test` must be **required status checks** on `main`, with pull requests required,
  no direct pushes and no bypass. The manager sets that rule after this change merges and verifies it through
  the API. Until then the checks report but do not block. A pull request must also be up to date with `main`
  before merging for the `test` result to describe what lands.
- **The pull request that introduces or changes the gate is not judged by the new gate.** `harness-lock` runs
  the base branch's version. The first installation, and any later change to the gate, is reviewed by people.
- **Commit dates are set by whoever makes the commit.** Condition 3 above can be defeated with a backdated
  commit; condition 4 (the label is removed on every push) does not depend on dates and is the stronger of
  the two.
- **`verify` proves nothing about origin.** A hand-made pair of scorecards with consistent numbers passes it.
  Only `evaluate` computes the numbers.
- **`evaluate` on a private machine is as trustworthy as that machine.** It checks the harness code, config,
  references, truth and thresholds against pinned hashes, but it runs where the candidate's author runs it.
  A promotion that matters should be re-run by someone else from the same inputs; the scorecards are
  deterministic, so the bytes must match.
- **The truth and thresholds files are pinned by content hash only through the scorecards.** A wrong file is
  caught after scoring, not before.
- **The git tag `harness-v0.1.0` does not exist yet.** The owner or manager creates it on the merge commit.
- **A scorecard value exactly on a gate boundary** can be judged differently after rounding to 10 significant
  digits; the promotion check then refuses, it never promotes.

## Changing the harness (owner-approved only)

1. Open a pull request with the change and say why in plain words.
2. Bump the version (`__version__` in `src/wrfcomp/__init__.py`, and `protocol_version` in
   `configs/splits.yaml` for rule changes). Commit, then run
   `python3 scripts/harness_lock.py update <new version>` and commit the new manifest. `update` reads the
   committed tree, so the change must be committed first.
3. A reviewer from a different model provider reviews it.
4. The owner applies the label to the final commit. Any later push removes it. Merge. Tag
   `harness-v<new version>`.
5. Re-score every baseline under the new version, commit the new scorecards and a new
   `experiments/REFERENCES.json`. Old scorecards stay, marked with their version; they are not comparable
   with new ones.

## Unlocking

The owner can lift the lock at any time by approving a pull request that removes it.
