# Harness lock

**Locked on 2026-11-23 by the project owner, at harness version 0.1.0**, after the Gate 1 reference run
(`experiments/`). Two owner decisions are behind it: the scoring harness is frozen at 0.1.0, and a candidate is
promoted only if it passes the gate against **both** climatology and NOAA GEFS.

This page says exactly what is enforced, by what, and what is not enforced.

## Why

Forecasts are judged by this code. If it can move while models are being improved, a better score can come
from an easier test instead of a better forecast. Scores from different harness versions must never be mixed.

## What is locked

| Locked | Hashed in `HARNESS_LOCK.json` | Needs the owner's approval to change |
| --- | --- | --- |
| `src/wrfcomp/harness/` (scoring rules, truth loader, alignment, content hash, `wrfcomp-eval`) | yes | yes |
| `configs/` (split boundaries, station list, gate thresholds) | yes | yes |
| `src/wrfcomp/__init__.py` (the package initializer and its version) | yes | yes |
| `docs/VERIFICATION_PROTOCOL.md`, `scripts/check_promotion.py` | yes | yes |
| `experiments/REFERENCES.json` and the six Gate 1 scorecards (climatology, GEFS, persistence; 2024-01, 2024-07) | yes | yes |
| Versions of `scores`, `numpy`, `xarray`, `scipy`, `pandas` in `uv.lock` | yes (as versions) | yes, through the manifest |
| The lock itself: `HARNESS_LOCK.json`, `scripts/harness_lock.py`, `tests/test_harness_lock.py`, `tests/test_check_promotion.py`, `docs/HARNESS_LOCK.md`, everything under `.github/` | no | yes |

Everything else (baselines, assimilation, models, other scripts and docs, new scorecards, other dependencies)
is open as before.

## The three checks

**Every locked script is run with Python's `-P` option**, in CI, in the scoring processes and by hand:
`python3 -P scripts/harness_lock.py ...`, `python -P scripts/check_promotion.py ...`. Run by path without it,
Python puts the script's own directory first on its module path, ahead of the standard library and of every
installed package, and `scripts/` is not a locked directory: a file such as `scripts/tomllib.py` or a
directory `scripts/scores/` added by an ordinary pull request would be imported in place of the real module.
The three scripts (`harness_lock.py`, `check_promotion.py`, `.github/harness_lock_gate.py`) check this
themselves, before importing anything but `sys`, and refuse to run without `-P` (or `PYTHONSAFEPATH=1`).

### 1. `test` (`.github/workflows/ci.yml`): the tree is the locked tree

Runs `python3 -P scripts/harness_lock.py check`, then ruff and the test suite. The check reads the committed Git
tree (`git ls-tree -r HEAD`) and fails if

- a locked path was changed, added, removed or renamed (the manifest records every path with its Git blob id
  and mode; a rename shows as one path removed and one added);
- a locked entry is not a regular, non-executable file: a symlink, an executable or a submodule;
- there is an untracked, ignored or uncommitted file inside the locked scope. The single exception is an
  untracked or ignored `.pyc` file directly inside a `__pycache__` directory. Anything staged or tracked there,
  and any other kind of file there, is reported;
- the manifest's harness version differs from `src/wrfcomp/__init__.py`, or its scope from the scope written
  in `scripts/harness_lock.py`;
- `uv.lock` resolves `scores`, `numpy`, `xarray`, `scipy` or `pandas` to a version other than the recorded one.

This check runs the pull request's own copy of the script. A pull request that edits the script, the manifest
or the workflow can therefore make it say anything. That is why the second check exists.

### 2. `harness-lock` (`.github/workflows/harness-lock.yml`): the owner approved this exact commit

Triggered by `pull_request_target`, so GitHub runs the workflow file of the **base** branch, not the pull
request's. It checks out nothing. It fetches its gate script (`.github/harness_lock_gate.py`) from the same
base commit through the API and reads the pull request through the API. Code from the pull request is never
executed by this workflow, so a pull request cannot change its own gate. The workflow has no write permission
and no shared concurrency group.

The gate lists every changed path (paginated, including the previous name of a rename). If none is locked, it
passes. If any is locked, it passes only if the owner's account (`tonykoop`) has posted a comment on the pull
request whose **whole first line is exactly**

```text
APPROVE-HARNESS-CHANGE <full 40-character sha of the pull request's current head commit>
```

- The approval names the commit. After a new push the head is a different commit, so the old comment no longer
  matches and a new one is needed. No dates and no label are involved.
- The comment must come from the owner's account and be a normal user comment (not a bot's), and GitHub's own
  edit record for it must be empty (`lastEditedAt` is null and the creation was not itself an edit; read
  through the GraphQL API). Anything after the first line is free text.
- **Approval is observed when the check runs.** If the comment is edited or deleted after a green run, no new
  run starts and the earlier result stands. The gate does not revoke.
- The decision is about one head, on one base, under one policy, and all three are checked before either way
  of passing ("nothing locked" or "approved"):
  - the pull request is read before and after the files and the comments are paged through; if its head or
    base moved in between, the run fails;
  - the head must be the one the run was started for;
  - the pull request's base commit must be the base-branch commit the policy was fetched from. A re-run of an
    old run cannot apply an older policy to a newer base. In practice this means **a pull request whose base
    is behind the base branch fails `harness-lock` until it is updated onto the current base branch**, whether
    or not it touches a locked path.
- It prints the head and base it evaluated.
- A file list that GitHub cut off (more files than the API lists) is **treated as touching the locked set**:
  such a pull request needs the owner's approval of its head like any locked change. It does not fail outright.
- An API error is "not approved".

A comment does not start a new run of this check. After commenting, the owner re-runs the `harness-lock` job,
or adds or removes any label on the pull request, which starts a run. The label
`owner-approved-harness-change` may still be used as a marker for people; the gate does not look at it.

### 3. `scripts/check_promotion.py evaluate`: the candidate beat both references

```sh
uv run --locked python -P scripts/check_promotion.py evaluate \
  --forecast <candidate.zarr> --data-dir "$WRFCOMP_DATA_DIR" --period 2024-01 --out <dir>
```

Everything the verdict rests on comes from one tree, the checkout the script is in:

- that tree must pass check 1, and the harness modules the running Python has imported must be files of that
  tree (a harness installed from somewhere else is refused). The five scoring packages (`scores`, `numpy`,
  `xarray`, `scipy`, `pandas`) must resolve inside the Python environment, not inside the checkout. These
  packages are located before they are imported, so a foreign one is refused without running any of its code;
- the reference inventory and the config are read from that tree;
- each of the two evaluations runs in its own process, started with `uv run --locked` on that tree (so the
  dependency versions are the ones in its `uv.lock`) and with `-P`. That process is the locked script itself:
  it makes the same checks on *its own* imports (harness from the tree, scoring packages from the
  environment), refuses otherwise, and then calls the
  harness entry point `wrfcomp.harness.evaluate.app` directly. The installed `wrfcomp-eval` console command
  is not used, because what that name runs is set in `pyproject.toml`, which is not locked;
- the two reference forecasts of the period are taken from the places named in `experiments/REFERENCES.json`
  under the data directory, and a store whose content hash is not the pinned one is refused.

It then judges the two scorecards those processes produced:

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

The two scorecards and a `verdict.json` are written in a private directory created for the run and published
at the end as `<out>/promotion_<period>_<first 16 hex digits of the forecast hash>/`. The file names are
fixed by the program. The candidate's model id is recorded inside the files and is never part of a path.

`python -P scripts/check_promotion.py verify A.json B.json --period P` applies the same rules to two scorecards that already
exist. It prints `CONSISTENT` or `INCONSISTENT`, never `PROMOTED`.

## What is NOT enforced

- **GitHub cannot tell the owner from an agent.** All agents in this project act through the owner's GitHub
  account. To GitHub, a comment posted by an agent and a comment posted by the owner are the same event from
  the same account. The `harness-lock` check proves that the owner's *account* approved the exact commit; it
  cannot prove that the owner did. The working rule is therefore a rule for agents, not a control: **only
  the owner writes an `APPROVE-HARNESS-CHANGE` comment; agents never write one, never ask a tool to write
  one, not even as a test, and never merge a pull request that has one without the owner's explicit yes.**
  Separate accounts for agents would turn this rule into a control.
- **Nothing is enforced until the repository rules exist.** As of this change, `main` is not protected.
  Both `harness-lock` and `test` must be **required status checks** on `main`, with pull requests required,
  no direct pushes and no bypass. The manager sets that rule after this change merges and verifies it through
  the API. Until then the checks report but do not block. A pull request must also be up to date with `main`
  before merging for the `test` result to describe what lands.
- **The pull request that introduces or changes the gate is not judged by the new gate.** `harness-lock` runs
  the base branch's version. The first installation, and any later change to the gate, is reviewed by people.
- **A comment does not re-run the check, and neither does editing or deleting one.** Until someone re-runs
  it, the check shows the result from before the approval, which delays a merge. In the other direction, an
  approval withdrawn after a green run leaves that green result in place: approval is a fact at check time,
  not a standing condition.
- **`pyproject.toml` is not locked.** The promotion check does not go through the console command it
  defines, but anyone who runs `wrfcomp-eval` by name from a shell runs whatever `pyproject.toml` points
  it at.
- **`-P` protects against files next to a script, not against the environment.** `PYTHONPATH`, a `.pth` file
  or a `sitecustomize` module in the environment can still put other code on the path. The promotion check
  refuses a harness or scoring package that resolves to the wrong place, but it does not police every other
  module a scoring run imports.
- **No isolated, artifact-pinned runtime.** `evaluate` checks that the harness modules it imports are files
  of the locked tree and runs the evaluations with `uv run --locked`, which pins dependency *versions*. It
  does not verify the installed packages byte for byte, and it does not run in a clean container built from
  an approved release. A modified `numpy` with the right version number would not be noticed.
- **Inputs are hashed before they are read for scoring.** `wrfcomp-eval` hashes the forecast, baseline,
  truth and thresholds and then reads them again to score, and the two evaluations of a promotion open the
  inputs separately. The hashes in a scorecard therefore prove what was scored only if the inputs did not
  change during the run. Closing this needs a change to the locked harness and is not done here. A promotion
  that matters is re-run by a second party from read-only inputs; the scorecards are deterministic, so the
  bytes must match.
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
   `python3 -P scripts/harness_lock.py update <new version>` and commit the new manifest. `update` reads the
   committed tree, so the change must be committed first.
3. A reviewer from a different model provider reviews it.
4. The owner comments `APPROVE-HARNESS-CHANGE <head sha>` on the final commit and re-runs the
   `harness-lock` check. Any later push needs a new comment. Merge. Tag `harness-v<new version>`.
5. Re-score every baseline under the new version, commit the new scorecards and a new
   `experiments/REFERENCES.json`. Old scorecards stay, marked with their version; they are not comparable
   with new ones.

## Unlocking

The owner can lift the lock at any time by approving a pull request that removes it.
