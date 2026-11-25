# Harness lock

**Locked on 2026-11-23 by the project owner, at harness version 0.1.0**, after the Gate 1 reference run
(`experiments/`). Three owner decisions are behind it: the scoring harness is frozen at 0.1.0; a candidate
passes a period only if it passes the gate against **both** climatology and NOAA GEFS; and a candidate is
promoted only if it passes in **at least 3 of the 4 calendar quarters** of 2024, each quarter scored as one
period (promotion rule version 2).

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
| `experiments/REFERENCES.json`, the 36 monthly Gate 1 reference scorecards (climatology, GEFS, persistence; every month of 2024) and the 12 quarterly ones (the same three models, `2024-Q1` to `2024-Q4`), and the 12 quarterly reference scorecards of the 2023 confirmation year (`2023-Q1` to `2023-Q4`) | yes | yes |
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

### 3. `scripts/check_promotion.py`: both references, in 3 of 4 quarters

Two commands. `evaluate` scores one period and says PASSED or NOT PASSED. `promote` takes the candidate's
stores for the whole year, scores the four quarters itself and is the only command that says PROMOTED.

```sh
# one quarter; the candidate as one store, or as several stores (its months)
uv run --locked python -P scripts/check_promotion.py evaluate \
  --forecast <2024-01.zarr> <2024-02.zarr> <2024-03.zarr> \
  --data-dir "$WRFCOMP_DATA_DIR" --period 2024-Q1 --out <dir>

# the decision: the candidate's twelve monthly stores, scored here
uv run --locked python -P scripts/check_promotion.py promote --year 2024 \
  --forecast <2024-01.zarr> <2024-02.zarr> ... <2024-12.zarr> \
  --data-dir "$WRFCOMP_DATA_DIR" --out <dir>
```

#### `evaluate`: one period

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

It prints `PASSED for <period>` only then, otherwise `NOT PASSED for <period>` with the reasons. It never prints
`PROMOTED`. No reference directory, reference file or config can be passed in.

**Pinned periods.** The four quarters `2024-Q1` to `2024-Q4` are the ones promotion is decided on. A quarter
is ONE period: its reference forecasts are the monthly reference stores joined along init time
(`scripts/run_gate1.py --period 2024-Q1`, under `gate1/2024-Q1_h00-12/` in the data directory), and its
reference scorecards are one evaluation over the whole quarter, not an average of months. Q4 ends with the init
of 2024-12-17T12Z, as December does. The twelve months `2024-01` to `2024-12` stay pinned for diagnosis: **a
monthly result is diagnostic only** and counts for nothing in a promotion.

**The 2023 confirmation periods (owner ruling 2026-11-24).** `2023-Q1` to `2023-Q4` are pinned for the
confirmation run, `promote --year 2023`, which the manager runs once for a candidate that was promoted on 2024.
They lie inside the train split, so they have their own truth file (`truth/isd_conus_2023.nc`, a slice of the
train truth) and their own thresholds (`thresholds/tail_p95_to_2022-12-17.nc`), and their climatology and
persistence stores are built from truth at or before 2022-12-17T18Z only
(`scripts/run_gate1.py --confirmation-train-end`, see `docs/CONFIRMATION_2023.md`). Q4 ends with the init of
2023-12-15T12Z. Months of 2023 are not pinned, so they cannot be scored through this program. The check itself
is unchanged: it prints `PROMOTED` for 2023, which is read as confirmed and is not a promotion.

**A candidate given as several stores.** For a quarter, `--forecast` takes one store or several (for example
the three monthly stores). Several stores must be one model id and version, with the same variables,
coordinates, variable attributes (units), leads, stations and members, and their init times must not overlap
or interleave; anything else is refused. They are joined along init time in time order, whatever order they
are given in, written once into the run's private directory, scored from there and removed. Variable
attributes are carried into the joined store unchanged. The forecast hash in the scorecards is the harness content hash of the joined
forecast. That hash depends only on values, coordinates, dimensions and dtypes, not on chunking or file layout,
so the same monthly stores always give the same hash, and the same hash as the quarter stored whole.
`verdict.json` lists the content hash, first and last init time and number of init times of every input store.

**Every candidate store is checked before any of its values is read or hashed**, from coordinates and
attributes only: its init and valid times against the sealed split, its layout with the harness forecast
validator (dimensions, dtypes, units, model id and version), and its init times, which must be in time order.
The last rule is what makes the hash independent of how the forecast is handed in: the content hash depends
on the order of the init times, so a store holding the same forecasts in another order is refused rather than
hashed to something else.

The two scorecards and a `verdict.json` are written in a private directory created for the run and published
at the end as `<out>/promotion_<period>_<first 16 hex digits of the sha256 of verdict.json>/`. `verdict.json`
names the sha256 of both scorecards, so the directory name identifies the whole result: the same run always
lands in the same directory, and running it again is accepted and changes nothing. A quarter scored from one
store and from its three monthly stores gives the same scorecards byte for byte and two directories (the list
of input stores differs). If a directory of that name exists with other contents, it was changed after it was
written: the run is refused and nothing is overwritten. The file names are
fixed by the program. The candidate's model id is recorded inside the files and is never part of a path.
`verdict.json` holds the period, `PASSED` or `NOT PASSED` with the reasons, the candidate (model id, version,
forecast hash), the input stores, the sha256 of each scorecard file, the harness and protocol version, the
sha256 of `experiments/REFERENCES.json`, and `promotion_rule_version`.

#### `promote`: at least 3 of 4 quarters

`promote` takes a year, a data directory and the candidate's forecast stores for that year (its twelve monthly
stores). It does the scoring itself:

- every store is checked as above before anything is read; all stores must be one model id and version; each
  must lie inside one calendar quarter of the year;
- the stores are sorted into the four quarters by their init times, and `evaluate` is run for each quarter
  that has stores, exactly as described above: the candidate against both pinned references, every scoring
  in a separate process on the locked harness;
- a quarter with no store has not passed.

It prints one line per quarter with the reasons, then `PROMOTED` if at least 3 of the 4 quarters passed and
`NOT PROMOTED` otherwise. It refuses outright (exit code 2) when the tree is not the locked harness, when a
scoring process finds it is not running that harness, when the quarters of that year are not pinned, when no
store is given or one is given twice, when a store is outside the year or spans two quarters, and when the
stores are not one candidate.

**No file supplied by the caller is trusted for the verdict.** `promote` takes forecasts, not results. It reads
no scorecard, verdict or result directory that it did not compute in the same run, and the command line has
no argument for one. Whatever already lies in the output directory is not evidence: a result directory planted
there is ignored, or, if it sits under the very name this run would publish and holds something else, the run
is refused.

The four quarter results are published under `--out` as `evaluate` publishes them, next to
`promotion_<year>_<id>/decision.json`: the decision, the candidate, each quarter's verdict, reasons, result
directory and the sha256 of its `verdict.json`, the rule version, harness and protocol version and the sha256
of `REFERENCES.json`. These files are a record of the run. They are an input to nothing.

**Promotion rule version.** `promotion_rule_version` is 2: at least 3 of 4 quarters, each as one period.
Version 1 was the rule before this change: a single month decided. It is separate from `protocol_version` on
purpose. The rule change touches no metric, mask or threshold, so the scorecards and the pinned references are
exactly as before and `protocol_version` stays 0.1.0. A verdict written under rule version 1 has no
`promotion_rule_version` field.

#### `verify`

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
- **The repository rule on `main`.** Since 2026-11-24 a GitHub ruleset requires pull requests, makes
  `harness-lock` and `test` **required status checks**, requires the branch to be up to date with `main`,
  blocks force-pushes and deletion, and has no bypass list. It was verified with a test pull request that
  changed a locked file without approval and was refused. The rule is a repository setting, not code in
  this repository, so it is only as durable as the organisation's admin settings.
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
  Only `evaluate` and `promote` compute the numbers.
- **A printed `PROMOTED`, a `decision.json` or a result directory is not proof that a run happened.** They are
  text and files; anyone can write them. The proof is running `promote` again on the candidate's stores under
  the locked tree. The published files exist so that a rerun can be compared with them.
- **`promote` does not know that the twelve stores are one forecast system.** It requires one model id and
  version. A candidate whose author used different settings in different quarters under one version string
  would not be noticed; that is a matter for review of the candidate, not for this check.
- **`promote` does not check that the stores are months.** It needs every store to lie inside one quarter. A
  quarter given in part is scored and does not pass, because it is not the pinned quarter.
- **Which quarters failed is not weighed.** Any 3 of 4 promote. A candidate that fails the same season every
  year passes the rule.
- **`evaluate` on a private machine is as trustworthy as that machine.** It checks the harness code, config,
  references, truth and thresholds against pinned hashes, but it runs where the candidate's author runs it.
  A promotion that matters should be re-run by someone else from the same inputs; the scorecards are
  deterministic, so the bytes must match.
- **The truth and thresholds files are pinned by content hash only through the scorecards.** A wrong file is
  caught after scoring, not before.
- **Tags.** `harness-v0.1.0` marks the merge that locked the harness; `promotion-rule-v2` marks the merge that
  added the three-of-four-quarters rule (harness still 0.1.0). Tags can be moved by a repository admin.
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
