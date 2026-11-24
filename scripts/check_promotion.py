"""Promotion check: a candidate must pass the gate against BOTH reference baselines, in at least
3 of the 4 calendar quarters of the year.

Owner decisions 2026-11-23:
  * a candidate passes a period only if it passes the harness gate against climatology AND against
    NOAA GEFS, on the same forecast, truth, thresholds and period;
  * a candidate is PROMOTED only if it passes in at least 3 of the 4 calendar quarters, each quarter
    scored as ONE period. This is promotion rule version 2 (`PROMOTION_RULE_VERSION`); under rule
    version 1 a single month decided. A verdict for one month or one quarter is never a promotion.

    uv run --locked python -P scripts/check_promotion.py evaluate \\
        --forecast F.zarr --data-dir D --period 2024-Q1 --out DIR
    uv run --locked python -P scripts/check_promotion.py evaluate \\
        --forecast JAN.zarr FEB.zarr MAR.zarr --data-dir D --period 2024-Q1 --out DIR
    uv run --locked python -P scripts/check_promotion.py promote --year 2024 \\
        --model-id M --model-version V DIR/promotion_2024-Q1_... DIR/promotion_2024-Q2_... ...
    uv run --locked python -P scripts/check_promotion.py verify A.json B.json --period 2024-Q1

Always with `-P`: without it Python would look for modules next to this script first, and the
script refuses to run.

`evaluate` scores one pinned period (a quarter `YYYY-Qn`, or a month `YYYY-MM` for diagnosis) and
prints PASSED or NOT PASSED for it. Everything it relies on comes from one tree, the checkout this
script is in:
  1. that tree must pass the harness lock (`scripts/harness_lock.py check`), and the harness modules
     this process has imported must be files of that tree;
  2. the two reference forecasts of the period are taken from the data directory, at the places
     named in the tree's `experiments/REFERENCES.json`, and refused unless their content hash is the
     pinned one;
  3. the evaluation is run twice, with the tree's locked `configs/splits.yaml`: the candidate
     against climatology and against NOAA GEFS. Each run is a separate process started with
     `uv run --locked` on that tree. That process is this script again: it checks that the harness
     it has imported is the one in the tree and then calls the harness entry point
     (`wrfcomp.harness.evaluate.app`) directly, not the installed `wrfcomp-eval` command;
  4. the two scorecards are judged (rules below). They and `verdict.json` are written in a private
     directory made for the run and published as `<out>/promotion_<period>_<forecast hash>` at the
     end. File names are fixed by this program; the candidate's model id is data inside the files.

A quarter's candidate may be given as several stores, for example its three monthly stores. They
must be one model and version, with the same variables, leads, stations and members and no init
time in common. They are joined along init time in time order (whatever order they are given in),
written once into the run's private directory, scored from there and removed. The forecast hash in
the scorecards is the harness content hash of the joined forecast, which depends only on its
values, coordinates, dimensions and dtypes, so the same monthly stores always give the same hash.
`verdict.json` lists the content hash of each input store.

`promote` takes the result directories `evaluate` published for the four quarters of a year and one
candidate (model id and version). For each quarter it checks that the directory is this check's
output for that candidate under this tree: the two scorecards are judged again against the pinned
references, and `verdict.json` must be byte for byte what this program writes for them (same
candidate, rule version, harness and protocol version, the same `REFERENCES.json`, and the sha256
of each scorecard file). A quarter
with no result, or with a result that fails any of this, counts as NOT PASSED. It prints PROMOTED
only when at least 3 of the 4 quarters passed. It is the only mode that can print PROMOTED. It
checks consistency with the pinned references, NOT provenance: it reads files and cannot know who
wrote them.

`verify` applies the judging rules to two scorecards that already exist. It proves nothing about
where they came from, so it prints CONSISTENT or INCONSISTENT.

Rules, for each of the two scorecards, against the period's record in `experiments/REFERENCES.json`
(the only source of reference facts; no directory or file of references can be passed in):
  * it validates as a harness `Scorecard`, and has every primary variable, none missing;
  * its gates are exactly what the harness computes from its own metrics under the locked config,
    and `passed_gate` is exactly what those gates imply;
  * harness version, protocol version, truth hash, thresholds hash, period, number of init times,
    init hours and number of stations equal the record's; no sealed data;
  * `baseline_id` and `baseline_sha256` are the record's reference forecast;
  * the scored leads are the record's leads (all v1 leads against climatology, 3 h and longer
    against NOAA GEFS), and the number of scored and masked points per variable and lead equals the
    reference scorecard's;
  * the candidate is not a reference: neither its model id nor its forecast hash.
Both scorecards must describe the same candidate forecast, and both gates must be passed.

Exit codes: 0 PASSED (evaluate), PROMOTED (promote) or CONSISTENT (verify); 1 NOT PASSED,
NOT PROMOTED or INCONSISTENT; 2 refused (bad input, unpinned or missing references, harness not
locked or not from this tree, two results for one quarter).
Locked: see docs/HARNESS_LOCK.md, which also lists what this check does not establish.
"""

import sys

# Before any other import. Running a script by path puts its directory (scripts/, which is not a
# locked directory) first on sys.path, ahead of the standard library and of every installed
# package. A file planted there (scripts/tomllib.py, scripts/scores/) would then be imported in
# place of the real module. `python -P` leaves the script directory off the path; without it this
# script does not run. `sys` is built in and cannot be shadowed.
if __name__ == "__main__" and not getattr(sys.flags, "safe_path", False):
    sys.exit(
        f"REFUSED: run this script with Python's -P option (python -P {sys.argv[0]} ...), so that files "
        "next to it cannot shadow the standard library or the scoring packages. See docs/HARNESS_LOCK.md."
    )

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = "experiments/REFERENCES.json"
INVENTORY_VERSION = 1
REQUIRED_BASELINES = ("climatology", "noaa-gefs")
RECORD_KEYS = (
    "scorecard",
    "scorecard_sha256",
    "store",
    "model_id",
    "forecast_sha256",
    "truth_sha256",
    "thresholds_sha256",
    "period_start",
    "period_end",
    "n_init_times",
    "init_hours_scored",
    "n_stations",
    "leads_scored",
)
# Scorecard fields that must equal the reference record of the period.
CONTEXT_KEYS = (
    "truth_sha256",
    "thresholds_sha256",
    "period_start",
    "period_end",
    "n_init_times",
    "init_hours_scored",
    "n_stations",
)
SAME_CANDIDATE_KEYS = ("model_id", "model_version", "forecast_sha256")
EXIT_NOT_PROMOTED = 1  # also NOT PASSED and INCONSISTENT
EXIT_REFUSED = 2
EXIT_CHILD_REFUSED = 4  # the scoring process found that its harness is not the one under the root
CHILD_MODE = "score-one"  # internal: `check_promotion.py score-one <root> <wrfcomp-eval arguments>`
# Names this program gives the files of one run. Nothing from the candidate goes into a path.
STAGED_NAMES = {"climatology": "candidate_vs_climatology.json", "noaa-gefs": "candidate_vs_noaa-gefs.json"}
VERDICT_NAME = "verdict.json"
PERIOD_NAME = re.compile(r"[0-9]{4}-(?:[0-9]{2}|Q[1-4])")
QUARTER_NAME = re.compile(r"[0-9]{4}-Q[1-4]")
# Rule version 1: one period (a month) decided. Rule version 2: at least QUARTERS_REQUIRED of the four
# calendar quarters of a year, each scored as one period. Recorded in every verdict this program writes.
PROMOTION_RULE_VERSION = 2
QUARTERS_REQUIRED = 3
COMBINED_NAME = "candidate_combined.zarr"  # the joined candidate of one run; removed before publishing
JOIN_DIM = "init_time"
PROMOTE_NOTICE = (
    "promote checks that the quarter results are consistent with the pinned references and with each "
    "other. It reads files and proves nothing about who wrote them: re-run `evaluate` to be sure."
)
HEX_DIGEST = re.compile(r"[0-9a-f]{64}")
# Every module the scores come from. `evaluate` requires each to be loaded from the checked tree.
HARNESS_MODULES = (
    "wrfcomp",
    "wrfcomp.harness",
    "wrfcomp.harness.align",
    "wrfcomp.harness.config",
    "wrfcomp.harness.evaluate",
    "wrfcomp.harness.hashing",
    "wrfcomp.harness.metrics",
    "wrfcomp.harness.schemas",
    "wrfcomp.harness.truth",
)
# The packages the numbers come from (the same five scripts/harness_lock.py pins in uv.lock).
SCORING_PACKAGES = ("numpy", "pandas", "scipy", "scores", "xarray")
VERIFY_NOTICE = (
    "verify only compares two existing scorecards with the pinned references. It proves nothing about "
    "where the scorecards came from and is not a promotion. Use `evaluate` for a verdict."
)


class Refused(Exception):
    """The check cannot be made at all. Exit code 2."""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path, what: str) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise Refused(f"cannot read {what} {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise Refused(f"{what} {path} is not a JSON object")
    return data


# --------------------------------------------------------------------------- pinned references


def build_inventory(root: Path, periods: dict[str, dict]) -> dict:
    """The inventory for the given reference scorecards. Used once to write REFERENCES.json and by
    the tests to prove the committed file still describes the committed scorecards.

    `periods` maps a period label to {"truth": ..., "thresholds": ..., "references": {baseline id:
    {"scorecard": repo path, "store": data-dir path}}, "others": [repo paths of further scorecards]}.
    """
    inventory: dict = {"inventory_version": INVENTORY_VERSION, "periods": {}}
    for label, spec in sorted(periods.items()):
        entry: dict = {"truth": spec["truth"], "thresholds": spec["thresholds"], "references": {}}
        entry["other_reference_forecasts"] = {}
        for baseline, where in sorted(spec["references"].items()):
            card = _read_json(root / where["scorecard"], "reference scorecard")
            entry["references"][baseline] = {
                "scorecard": where["scorecard"],
                "scorecard_sha256": _sha256_file(root / where["scorecard"]),
                "store": where["store"],
                "model_id": card["model_id"],
                "forecast_sha256": card["forecast_sha256"],
                **{key: card[key] for key in CONTEXT_KEYS},
                "leads_scored": card["leads_scored"],
            }
            inventory.setdefault("harness_version", card["harness_version"])
            inventory.setdefault("protocol_version", card["protocol_version"])
        for path in sorted(spec.get("others", [])):
            card = _read_json(root / path, "reference scorecard")
            entry["other_reference_forecasts"][card["forecast_sha256"]] = path
        inventory["periods"][label] = entry
    return inventory


def load_inventory(root: Path) -> dict:
    """Read REFERENCES.json and check it against the reference scorecards it pins."""
    inventory = _read_json(root / INVENTORY, "reference inventory")
    if inventory.get("inventory_version") != INVENTORY_VERSION or not isinstance(
        inventory.get("periods"), dict
    ):
        raise Refused(f"{INVENTORY} is not a version {INVENTORY_VERSION} reference inventory")
    for label, entry in inventory["periods"].items():
        references = entry.get("references") if isinstance(entry, dict) else None
        if not isinstance(references, dict) or set(references) != set(REQUIRED_BASELINES):
            raise Refused(f"{INVENTORY}: period {label} must pin exactly {' and '.join(REQUIRED_BASELINES)}")
        for baseline, record in references.items():
            missing = [key for key in RECORD_KEYS if key not in record]
            if missing:
                raise Refused(f"{INVENTORY}: {label} {baseline} lacks {missing}")
            path = root / record["scorecard"]
            if not path.is_file() or _sha256_file(path) != record["scorecard_sha256"]:
                raise Refused(
                    f"reference scorecard {record['scorecard']} is missing or is not the pinned file"
                )
            card = _read_json(path, "reference scorecard")
            for key in ("model_id", "forecast_sha256", "leads_scored", *CONTEXT_KEYS):
                if card.get(key) != record[key]:
                    raise Refused(f"{INVENTORY}: {label} {baseline} disagrees with its scorecard on {key}")
            if record["model_id"] != baseline:
                raise Refused(
                    f"{INVENTORY}: {label} {baseline} pins a scorecard of model {record['model_id']}"
                )
            for key in ("harness_version", "protocol_version"):
                if card.get(key) != inventory.get(key):
                    raise Refused(f"{INVENTORY}: {label} {baseline} was scored with another {key}")
    return inventory


def period_record(inventory: dict, period: str) -> dict:
    if period not in inventory["periods"]:
        known = ", ".join(sorted(inventory["periods"]))
        raise Refused(f"period {period!r} has no pinned references (pinned: {known})")
    return inventory["periods"][period]


def reference_forecast_hashes(inventory: dict) -> dict[str, str]:
    """Every pinned reference forecast hash, of every period, with a label for messages."""
    out: dict[str, str] = {}
    for label, entry in inventory["periods"].items():
        for baseline, record in entry["references"].items():
            out[record["forecast_sha256"]] = f"{baseline} {label}"
        for digest, path in entry.get("other_reference_forecasts", {}).items():
            out[digest] = path
    return out


# --------------------------------------------------------------------------- judging


def judge(cards: list[dict], period: str, inventory: dict, root: Path) -> list[str]:
    """Reasons the two scorecards do NOT amount to a pass against both references. Empty = pass."""
    from wrfcomp.harness.config import load_config
    from wrfcomp.harness.metrics import evaluate_gates
    from wrfcomp.harness.schemas import Scorecard

    entry = period_record(inventory, period)
    cfg = load_config(root / "configs" / "splits.yaml")
    primary = list(cfg.evaluation.variables.primary)
    reasons: list[str] = []

    by_baseline: dict[str, dict] = {}
    for card in cards:
        baseline = card.get("baseline_id") if isinstance(card, dict) else None
        if baseline in by_baseline:
            reasons.append(f"two scorecards against the same baseline: {baseline}")
        by_baseline[baseline] = card
    for name in REQUIRED_BASELINES:
        if name not in by_baseline:
            reasons.append(f"no scorecard against the required baseline {name}")
    for name in by_baseline:
        if name not in REQUIRED_BASELINES:
            reasons.append(f"a scorecard against {name!r}, which is not a reference baseline")
    if reasons or len(cards) != len(REQUIRED_BASELINES):
        return sorted(set(reasons)) or ["exactly two scorecards are needed"]

    known_forecasts = reference_forecast_hashes(inventory)
    reference_models = {
        r["model_id"] for e in inventory["periods"].values() for r in e["references"].values()
    }
    for name in REQUIRED_BASELINES:
        card, record = by_baseline[name], entry["references"][name]
        where = f"against {name}"
        try:
            # the validated form, so optional fields are present with their defaults
            card = Scorecard.model_validate(card).model_dump(mode="json")
        except ValueError as exc:
            first = str(exc).splitlines()[0:3]
            reasons.append(f"the scorecard {where} is not a valid harness scorecard: {' '.join(first)}")
            continue
        by_baseline[name] = card

        # gates and verdict must follow from the card's own numbers under the locked config
        missing = [v for v in primary if v not in card["metrics"]]
        if missing or card["variables_missing"]:
            reasons.append(
                f"primary variables are missing {where}: {sorted({*missing, *card['variables_missing']})}"
            )
        implied = not missing and not card["variables_missing"]
        for variable, per_lead in card["metrics"].items():
            derived = evaluate_gates({"all": per_lead["all"]}, cfg)
            if derived != card["gates"][variable]:
                reasons.append(f"the gates recorded for {variable} {where} are not what its metrics give")
            implied = implied and all(derived.values())
        if card["passed_gate"] != implied:
            reasons.append(f"passed_gate {where} is {card['passed_gate']} but the metrics imply {implied}")
        if not implied:
            reasons.append(f"gate not passed {where}")

        # evaluation context: the locked harness, and the period's truth, thresholds and coverage
        for key in ("harness_version", "protocol_version"):
            if card[key] != inventory[key]:
                reasons.append(
                    f"{key} {where} is {card[key]}, the references were scored with {inventory[key]}"
                )
        for key in CONTEXT_KEYS:
            if card[key] != record[key]:
                reasons.append(f"{key} {where} differs from the pinned reference for {period}")
        if card["sealed_data_used"] is not False:
            reasons.append(f"sealed data was used {where}")
        if card["baseline_sha256"] != record["forecast_sha256"]:
            reasons.append(f"the baseline {where} is not the pinned reference forecast for {period}")
        if card["leads_scored"] != record["leads_scored"]:
            reasons.append(
                f"leads scored {where} are {card['leads_scored']}, expected {record['leads_scored']}"
            )
        else:
            reference_card = _read_json(root / record["scorecard"], "reference scorecard")
            for variable in primary:
                if variable not in card["metrics"]:
                    continue
                for lead in map(str, card["leads_scored"]):
                    ours, theirs = card["metrics"][variable][lead], reference_card["metrics"][variable][lead]
                    for key in ("n_pairs", "n_masked_no_threshold"):
                        if ours[key] != theirs[key]:
                            reasons.append(
                                f"{key} for {variable} at {lead} h {where} is {ours[key]}, "
                                f"the reference scored {theirs[key]}"
                            )

        # the candidate must not be a reference
        if card["model_id"] in reference_models:
            reasons.append(f"model id {card['model_id']!r} is a reference baseline, not a candidate")
        if card["forecast_sha256"] in known_forecasts:
            reasons.append(
                f"the candidate forecast is byte-for-byte a reference forecast "
                f"({known_forecasts[card['forecast_sha256']]})"
            )

    first = by_baseline[REQUIRED_BASELINES[0]]
    for key in SAME_CANDIDATE_KEYS:
        if any(by_baseline[name].get(key) != first.get(key) for name in REQUIRED_BASELINES):
            reasons.append(f"the two scorecards are not about the same candidate: {key} differs")
    return sorted(set(reasons))


# --------------------------------------------------------------------------- evaluate


@dataclass(frozen=True)
class Outcome:
    """Result of `evaluate`: why the candidate is not promoted (empty = promoted) and where the
    finished artifacts were published."""

    reasons: list[str]
    directory: Path


def _harness_lock_problems(root: Path) -> list[str]:
    """Problems reported by `root`'s scripts/harness_lock.py for the tree at `root`."""
    spec = importlib.util.spec_from_file_location("harness_lock", root / "scripts" / "harness_lock.py")
    if spec is None or spec.loader is None or not (root / "scripts" / "harness_lock.py").is_file():
        return [f"{root} has no scripts/harness_lock.py"]
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        return module.check(root)
    except module.LockError as exc:
        return [str(exc)]


def _spec_origin(name: str) -> Path | None:
    """Where `name` would be imported from, found WITHOUT importing it (top-level names only)."""
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return None
    if spec is None:
        return None
    if spec.origin and spec.origin not in ("built-in", "frozen", "namespace"):
        return Path(spec.origin).resolve()
    locations = list(spec.submodule_search_locations or [])  # a namespace package: a bare directory
    return Path(locations[0]).resolve() if locations else None


def _modules_outside_root(root: Path) -> list[str]:
    """Everything wrong with where this process takes its scoring code from. Empty = all in order.

    The tree check looks at files under `root`; the numbers come from whatever Python imports. So:
      * `wrfcomp` and every harness module must be files under `root/src/wrfcomp`;
      * the scoring packages (scores, numpy, xarray, scipy, pandas) must come from the Python
        environment (under `sys.prefix`), not from a directory of the checkout such as `scripts/`.
    Top-level packages are located with `find_spec` first, which runs none of their code: a foreign
    `wrfcomp` or `scores` is refused before it is ever imported.
    """
    package = (root / "src" / "wrfcomp").resolve()
    environment = Path(sys.prefix).resolve()
    problems = []
    origin = _spec_origin("wrfcomp")
    if origin is None or not origin.is_relative_to(package):
        return [f"wrfcomp would be loaded from {origin}, not from {package}"]
    for name in SCORING_PACKAGES:
        origin = _spec_origin(name)
        if origin is None or not origin.is_relative_to(environment):
            problems.append(f"{name} would be loaded from {origin}, not from the environment {environment}")
    if problems:
        return problems
    for name in HARNESS_MODULES:  # safe to import now: the package directory is the checked one
        module = importlib.import_module(name)
        origin = Path(getattr(module, "__file__", None) or "/nonexistent").resolve()
        if not origin.is_relative_to(package):
            problems.append(f"{name} is loaded from {origin}")
    return problems


def _open_forecast(path: Path, what: str):
    import xarray as xr

    if not path.is_dir():
        raise Refused(f"{what}: no such forecast store: {path}")
    try:
        try:
            return xr.open_zarr(path, chunks=None, consolidated=True)
        except (KeyError, ValueError, FileNotFoundError):
            return xr.open_zarr(path, chunks=None, consolidated=False)
    except Exception as exc:  # any reader failure is a refusal, not a traceback
        raise Refused(f"{what}: cannot open {path}: {exc}") from exc


def _eval_command(root: Path, args: list[str]) -> list[str]:
    """The scoring process for one evaluation.

    It is this locked script, run with `-P` by the interpreter of `root`'s locked environment, in
    its `score-one` mode: the child itself checks that the harness it imports is the one under
    `root` and that the scoring packages come from the environment, and then calls the harness
    entry point `wrfcomp.harness.evaluate.app` directly. The installed
    `wrfcomp-eval` console command is not used: what that name runs is decided in pyproject.toml,
    which is outside the lock.
    """
    uv = shutil.which("uv")
    if uv is None:
        raise Refused("uv is not installed; evaluations run through `uv run --locked`")
    script = root / "scripts" / "check_promotion.py"
    # -P: the child must not look for modules in scripts/ either (and refuses to start without it)
    python = ["python", "-P", str(script), CHILD_MODE, str(root), *args]
    return [uv, "run", "--locked", "--project", str(root), *python]


def _run_eval(root: Path, args: list[str]) -> int:
    environment = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}  # uv picks root's own
    return subprocess.run(_eval_command(root, args), cwd=root, env=environment, check=False).returncode


def score_one(root: Path, args: list[str]) -> int:
    """Body of the scoring process: verify where the harness comes from, then run its entry point.

    Returns the exit code of `wrfcomp.harness.evaluate.app` (0 scored, 2 bad input, 3 NaN or
    coverage rule), or EXIT_CHILD_REFUSED without scoring anything if this process was started
    without `-P`, if any harness module of THIS process is not a file under `root`, or if a scoring
    package would not come from the environment.
    """
    import click

    root = root.resolve()
    if not getattr(sys.flags, "safe_path", False):
        outside = ["the scoring process was started without Python's -P option"]
    else:
        try:
            outside = _modules_outside_root(root)
        except Exception as exc:  # a harness that cannot even be imported is not the locked one
            outside = [f"the harness cannot be imported: {type(exc).__name__}: {exc}"]
    if outside:
        print(f"REFUSED: the scoring process is not running the harness under {root}:", file=sys.stderr)
        for line in outside:
            print(f"  {line}", file=sys.stderr)
        return EXIT_CHILD_REFUSED
    app = importlib.import_module("wrfcomp.harness.evaluate").app
    try:
        code = app(args=args, standalone_mode=False)
    except click.exceptions.Exit as exc:
        return int(exc.exit_code)
    except click.ClickException as exc:
        print(f"wrfcomp-eval: {exc.format_message()}", file=sys.stderr)
        return int(exc.exit_code)
    return int(code or 0)


def _publish(staging: Path, out_dir: Path, name: str) -> Path:
    """Move the finished run directory into place. An identical earlier result is kept as it is."""
    final = out_dir / name
    if final.exists():
        ours = {p.name: p.read_bytes() for p in sorted(staging.iterdir())}
        theirs = {p.name: p.read_bytes() for p in sorted(final.iterdir())} if final.is_dir() else None
        if ours == theirs:
            shutil.rmtree(staging)
            return final
        final = out_dir / f"{name}-{staging.name.rsplit('-', 1)[-1]}"  # same candidate, different result
    staging.chmod(0o755)
    os.rename(staging, final)
    return final


def _references_sha256(root: Path) -> str:
    return _sha256_file(root / INVENTORY)


def _verdict_text(
    *,
    period: str,
    reasons: list[str],
    model_id: str,
    model_version: str,
    digest: str | None,
    scorecards: dict[str, str],
    forecast_inputs: list[dict],
    inventory: dict,
    root: Path,
) -> str:
    """The bytes of `verdict.json`. One function, so that `promote` can tell this program's output.

    `scorecards` maps each scorecard file of the run to the sha256 of its bytes.
    """
    verdict = {
        "period": period,
        "verdict": "NOT PASSED" if reasons else "PASSED",
        "reasons": reasons,
        "baselines": list(REQUIRED_BASELINES),
        "candidate": {"model_id": model_id, "model_version": model_version, "forecast_sha256": digest},
        "forecast_inputs": forecast_inputs,
        "scorecards": scorecards,
        "promotion_rule_version": PROMOTION_RULE_VERSION,
        "harness_version": inventory["harness_version"],
        "protocol_version": inventory["protocol_version"],
        "references_sha256": _references_sha256(root),
    }
    return json.dumps(verdict, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


def _describe_input(ds, digest: str) -> dict:
    import numpy as np

    times = np.sort(ds[JOIN_DIM].values.astype("datetime64[s]"))
    return {
        "content_sha256": digest,
        "init_first": str(times[0]),
        "init_last": str(times[-1]),
        "n_init_times": int(times.size),
    }


def _combine_forecasts(forecasts: list[Path], staging: Path) -> tuple[Path, list[dict], str, str]:
    """The candidate store to score, the description of every input, and the model id and version.

    One store is scored as it is. Several stores (a quarter given as its months) are joined along
    init time in time order and written once into `staging`; see the module docstring for the rule.
    """
    import numpy as np
    import xarray as xr

    from wrfcomp.harness.hashing import content_sha256

    parts = []
    try:
        for path in forecasts:
            ds = _open_forecast(path, "candidate")
            parts.append((path, ds))
            if JOIN_DIM not in ds.dims or ds.sizes[JOIN_DIM] == 0:
                raise Refused(f"candidate {path} has no init_time dimension or no init times")
        identity = {
            (str(ds.attrs.get("model_id", "")), str(ds.attrs.get("model_version", ""))) for _, ds in parts
        }
        if len(identity) != 1:
            raise Refused("the candidate stores are not one model and version")
        model_id, model_version = identity.pop()
        if not model_id or not model_version:
            raise Refused("the candidate forecast has no model_id or model_version attribute")
        try:
            inputs = [_describe_input(ds, content_sha256(ds)) for _, ds in parts]
        except Exception as exc:  # unreadable or unhashable content is a refusal, not a traceback
            raise Refused(f"cannot hash a candidate store: {type(exc).__name__}: {exc}") from exc
        order = sorted(range(len(parts)), key=lambda i: inputs[i]["init_first"])
        inputs = [inputs[i] for i in order]
        if len(parts) == 1:
            return forecasts[0].resolve(), inputs, model_id, model_version

        ordered = [parts[i] for i in order]
        first_path, first = ordered[0]
        all_times = np.concatenate([ds[JOIN_DIM].values for _, ds in ordered])
        if np.unique(all_times).size != all_times.size:
            raise Refused("the candidate stores have an init time in common")
        for path, ds in ordered[1:]:
            if set(ds.variables) != set(first.variables) or set(ds.data_vars) != set(first.data_vars):
                raise Refused(f"candidate {path} has other variables or coordinates than {first_path}")
            for name in first.variables:
                ours, theirs = ds[name], first[name]
                if ours.dims != theirs.dims or ours.dtype != theirs.dtype:
                    raise Refused(
                        f"candidate {path}: {name} has other dimensions or dtype than in {first_path}"
                    )
                if JOIN_DIM in theirs.dims:
                    continue  # joined along init time
                same = np.array_equal(ours.values, theirs.values, equal_nan=theirs.dtype.kind == "f")
                if not same:
                    raise Refused(f"candidate {path}: {name} differs from {first_path}")
        loaded = [ds.load() for _, ds in ordered]
        joined = xr.concat(loaded, dim=JOIN_DIM, data_vars="all", coords="minimal", compat="override")
        joined = joined.sortby(JOIN_DIM)
        joined.attrs = dict(first.attrs)
        for name in joined.variables:
            joined[name].encoding.clear()
        target = staging / COMBINED_NAME
        joined.to_zarr(target, mode="w-", consolidated=True, zarr_format=2)
        return target, inputs, model_id, model_version
    finally:
        for _, ds in parts:
            ds.close()


def evaluate(forecast: Path | list[Path], data_dir: Path, period: str, out_dir: Path, root: Path) -> Outcome:
    """Score the candidate against both pinned references and judge. One period: never a promotion.

    `forecast` is one store, or for a quarter several stores that are joined along init time
    (`_combine_forecasts`).

    Everything the verdict rests on comes from the one tree at `root`: the lock manifest and the
    tree it describes, the harness modules this process imports, the reference inventory, the
    config, and the harness entry point called by the scoring processes, which check their own
    imports against `root` (see `_eval_command`). If any of them is somewhere else, the call is
    refused.

    The two scorecards and `verdict.json` are written into a private directory made for this run,
    under names this program chooses, and published as `<out_dir>/promotion_<period>_<hash>` when
    the run is complete. The candidate's model id is recorded inside the files; it is never part of
    a path.
    """
    from wrfcomp.harness.hashing import content_sha256

    root = root.resolve()
    problems = _harness_lock_problems(root)
    if problems:
        raise Refused(f"{root} is not the locked harness: " + "; ".join(problems[:5]))
    outside = _modules_outside_root(root)
    if outside:
        raise Refused(
            f"the harness in use is not the one under {root}: " + "; ".join(outside[:3]) + ". "
            "Run this script with `uv run --locked` from that checkout."
        )
    inventory = load_inventory(root)
    entry = period_record(inventory, period)
    if not PERIOD_NAME.fullmatch(period):
        raise Refused(f"period {period!r} is not a name this program writes files for")
    data_dir = data_dir.resolve()
    truth, thresholds = data_dir / entry["truth"], data_dir / entry["thresholds"]
    for path, what in ((truth, "truth"), (thresholds, "thresholds")):
        if not path.exists():
            raise Refused(f"{what} not found under the data directory: {path}")

    forecasts = [forecast] if isinstance(forecast, Path) else list(forecast)
    if not forecasts:
        raise Refused("no candidate forecast was given")
    if len(forecasts) > 1 and not QUARTER_NAME.fullmatch(period):
        raise Refused("several candidate stores can be given for a quarter only; a month takes one store")
    if len({path.resolve() for path in forecasts}) != len(forecasts):
        raise Refused("a candidate store is listed twice")
    for path in forecasts:  # refuse a missing or unreadable candidate before anything is created
        with _open_forecast(path, "candidate") as ds:
            if not str(ds.attrs.get("model_id", "")) or not str(ds.attrs.get("model_version", "")):
                raise Refused("the candidate forecast has no model_id or model_version attribute")

    stores: dict[str, Path] = {}
    for name in REQUIRED_BASELINES:
        record = entry["references"][name]
        store = data_dir / record["store"]
        with _open_forecast(store, f"reference {name}") as ds:
            digest = content_sha256(ds)
        if digest != record["forecast_sha256"]:
            raise Refused(
                f"reference {name} for {period} at {store} is not the pinned forecast "
                f"(content hash {digest[:12]}..., pinned {record['forecast_sha256'][:12]}...). "
                "Rebuild it with scripts/run_gate1.py from the original inputs."
            )
        stores[name] = store

    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".promotion-run-", dir=out_dir))  # private: mode 0700
    try:
        scored, forecast_inputs, model_id, model_version = _combine_forecasts(forecasts, staging)
        cards: list[dict] = []
        reasons: list[str] = []
        for name in REQUIRED_BASELINES:
            target = staging / STAGED_NAMES[name]
            args = [
                "--forecast",
                str(scored),
                "--truth",
                str(truth),
                "--thresholds",
                str(thresholds),
            ]
            args += ["--baseline", str(stores[name]), "--baseline-id", name, "--out", str(target)]
            args += ["--model-id", model_id, "--model-version", model_version]
            args += ["--config", str(root / "configs" / "splits.yaml")]
            code = _run_eval(root, args)
            if code == EXIT_CHILD_REFUSED:
                raise Refused(f"the scoring process did not run the harness under {root}; nothing was scored")
            if code != 0 or not target.is_file():
                target.unlink(missing_ok=True)
                reasons.append(f"wrfcomp-eval refused the candidate against {name} (exit code {code})")
                continue
            cards.append(_read_json(target, "scorecard"))
        if not reasons:
            reasons = judge(cards, period, inventory, root)

        if (staging / COMBINED_NAME).exists():
            shutil.rmtree(staging / COMBINED_NAME)  # scored; only its hash and its inputs' hashes are kept
        digests = {str(card.get("forecast_sha256")) for card in cards}
        digest = digests.pop() if len(digests) == 1 else ""
        text = _verdict_text(
            period=period,
            reasons=reasons,
            model_id=model_id,
            model_version=model_version,
            digest=digest or None,
            scorecards={p.name: _sha256_file(p) for p in sorted(staging.iterdir())},
            forecast_inputs=forecast_inputs,
            inventory=inventory,
            root=root,
        )
        (staging / VERDICT_NAME).write_text(text, encoding="ascii")
        token = digest[:16] if HEX_DIGEST.fullmatch(digest) else "unscored-" + staging.name.rsplit("-", 1)[-1]
        return Outcome(reasons=reasons, directory=_publish(staging, out_dir, f"promotion_{period}_{token}"))
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


# --------------------------------------------------------------------------- promote: 3 of 4 quarters


@dataclass(frozen=True)
class Quarter:
    """One quarter in a promotion decision: passed or not, and why not."""

    period: str
    passed: bool
    reasons: list[str]
    directory: Path | None


def _inspect_quarter_result(
    directory: Path, period: str, model_id: str, model_version: str, inventory: dict, root: Path
) -> list[str]:
    """Reasons this directory is NOT a passing result of this check for this candidate and period.

    The scorecards are judged again against the pinned references, and `verdict.json` must be byte
    for byte what `evaluate` writes for them under this tree. Empty = the quarter passed.
    """
    verdict_path = directory / VERDICT_NAME
    try:
        recorded_text = verdict_path.read_text(encoding="ascii")
        recorded = json.loads(recorded_text)
    except (OSError, ValueError):
        return [f"{VERDICT_NAME} cannot be read"]
    if not isinstance(recorded, dict):
        return [f"{VERDICT_NAME} is not a JSON object"]

    problems: list[str] = []
    candidate = recorded.get("candidate") if isinstance(recorded.get("candidate"), dict) else {}
    if (candidate.get("model_id"), candidate.get("model_version")) != (model_id, model_version):
        problems.append(
            f"the result is for another candidate ({candidate.get('model_id')!r} version "
            f"{candidate.get('model_version')!r})"
        )
    if recorded.get("promotion_rule_version") != PROMOTION_RULE_VERSION:
        problems.append(
            "the result was written under promotion rule version "
            f"{recorded.get('promotion_rule_version')!r}, not {PROMOTION_RULE_VERSION}"
        )
    for key in ("harness_version", "protocol_version"):
        if recorded.get(key) != inventory[key]:
            problems.append(f"the result was scored with another {key}: {recorded.get(key)!r}")
    if recorded.get("references_sha256") != _references_sha256(root):
        problems.append(f"the result was judged against another {INVENTORY}")

    present = sorted(p.name for p in directory.iterdir())
    names = recorded.get("scorecards")
    expected_names = sorted(STAGED_NAMES.values())
    listed = sorted(names) if isinstance(names, dict) else None
    if present != sorted([*expected_names, VERDICT_NAME]) or listed != expected_names:
        problems.append(
            "the directory does not hold exactly the two scorecards and the verdict of a scored run"
        )
        return problems + [
            f"recorded: {line}" for line in recorded.get("reasons", []) if isinstance(line, str)
        ]

    cards = [_read_json(directory / name, "scorecard") for name in expected_names]
    reasons = judge(cards, period, inventory, root)
    digests = {str(card.get("forecast_sha256")) for card in cards}
    digest = digests.pop() if len(digests) == 1 else ""
    inputs = recorded.get("forecast_inputs")
    expected_text = _verdict_text(
        period=period,
        reasons=reasons,
        model_id=str(cards[0].get("model_id")),
        model_version=str(cards[0].get("model_version")),
        digest=digest or None,
        scorecards={name: _sha256_file(directory / name) for name in expected_names},
        forecast_inputs=inputs if isinstance(inputs, list) else [],
        inventory=inventory,
        root=root,
    )
    if recorded_text != expected_text:
        problems.append(f"{VERDICT_NAME} is not what this check writes for these two scorecards")
    token = digest[:16] if HEX_DIGEST.fullmatch(digest) else None
    if directory.name != f"promotion_{period}_{token}":
        problems.append("the directory name is not the one this check gives this result")
    return sorted(set(problems)) + reasons


def promote(
    directories: list[Path], year: int, model_id: str, model_version: str, root: Path
) -> list[Quarter]:
    """The four quarters of `year` for one candidate, each passed or not. The caller counts them.

    A quarter passes only with a result directory that `_inspect_quarter_result` accepts. A quarter
    with no directory, or with one that is refused as evidence, has not passed. Refused outright
    (exit code 2): a tree that is not the locked harness, quarters that are not pinned, a directory
    that names no quarter of the year, and two directories for the same quarter.
    """
    root = root.resolve()
    problems = _harness_lock_problems(root)
    if problems:
        raise Refused(f"{root} is not the locked harness: " + "; ".join(problems[:5]))
    outside = _modules_outside_root(root)
    if outside:
        raise Refused(
            f"the harness in use is not the one under {root}: " + "; ".join(outside[:3]) + ". "
            "Run this script with `uv run --locked` from that checkout."
        )
    if not model_id or not model_version:
        raise Refused("the candidate's model id and model version are both needed")
    inventory = load_inventory(root)
    quarters = [f"{year:04d}-Q{n}" for n in range(1, 5)]
    unpinned = [q for q in quarters if q not in inventory["periods"]]
    if unpinned:
        raise Refused(f"no pinned references for {', '.join(unpinned)}")

    given: dict[str, Path] = {}
    for directory in directories:
        directory = directory.resolve()
        if not directory.is_dir():
            raise Refused(f"not a result directory: {directory}")
        try:
            period = json.loads((directory / VERDICT_NAME).read_text(encoding="ascii")).get("period")
        except (OSError, ValueError, AttributeError):
            raise Refused(f"{directory} has no readable {VERDICT_NAME}") from None
        if period not in quarters:
            raise Refused(f"{directory} is a result for {period!r}, not for a quarter of {year}")
        if period in given:
            raise Refused(f"two results were given for {period}: {given[period]} and {directory}")
        given[period] = directory

    results = []
    for period in quarters:
        directory = given.get(period)
        if directory is None:
            results.append(Quarter(period, False, ["no result was given for this quarter"], None))
            continue
        reasons = _inspect_quarter_result(directory, period, model_id, model_version, inventory, root)
        results.append(Quarter(period, not reasons, reasons, directory))
    return results


# --------------------------------------------------------------------------- command line


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Promotion check against both pinned reference baselines.",
        epilog="See docs/HARNESS_LOCK.md.",
    )
    modes = parser.add_subparsers(dest="mode", required=True)
    run = modes.add_parser(
        "evaluate", help="score the candidate against both references and give the verdict"
    )
    run.add_argument(
        "--forecast",
        type=Path,
        required=True,
        nargs="+",
        help="candidate forecast (Zarr); for a quarter, several stores (its months) are joined",
    )
    run.add_argument("--data-dir", type=Path, required=True, help="data directory with truth and references")
    run.add_argument("--period", required=True, help="a period pinned in experiments/REFERENCES.json")
    run.add_argument("--out", type=Path, required=True, help="directory for the two scorecards")
    decide = modes.add_parser(
        "promote", help=f"PROMOTED if at least {QUARTERS_REQUIRED} of the 4 quarter results of a year passed"
    )
    decide.add_argument(
        "results", type=Path, nargs="*", metavar="RESULT_DIR", help="directories from evaluate"
    )
    decide.add_argument("--year", type=int, required=True)
    decide.add_argument("--model-id", required=True, help="the candidate the results must be about")
    decide.add_argument("--model-version", required=True)
    look = modes.add_parser("verify", help="compare two existing scorecards with the pinned references")
    look.add_argument("scorecards", type=Path, nargs=2, metavar="SCORECARD")
    look.add_argument("--period", required=True, help="a period pinned in experiments/REFERENCES.json")
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == [CHILD_MODE]:  # internal mode, started by `evaluate` for each of its two evaluations
        if len(argv) < 2:
            parser.error(f"{CHILD_MODE} needs the root it scores for")
        return score_one(Path(argv[1]), argv[2:])
    args = parser.parse_args(argv)

    try:
        if args.mode == "promote":
            print(PROMOTE_NOTICE)
            quarters = promote(args.results, args.year, args.model_id, args.model_version, root=ROOT)
            return _report_promotion(quarters, args.year, args.model_id, args.model_version)
        if args.mode == "evaluate":
            forecast = args.forecast[0] if len(args.forecast) == 1 else args.forecast
            outcome = evaluate(forecast, args.data_dir, args.period, args.out, root=ROOT)
            reasons = outcome.reasons
            print(f"Scorecards and verdict: {outcome.directory}")
            verdict = ("PASSED", "NOT PASSED")
        else:
            print(VERIFY_NOTICE)
            inventory = load_inventory(ROOT)
            cards = [_read_json(path, "scorecard") for path in args.scorecards]
            reasons = judge(cards, args.period, inventory, ROOT)
            verdict = ("CONSISTENT", "INCONSISTENT")
    except Refused as exc:
        print(f"REFUSED: {exc}")
        return EXIT_REFUSED

    if reasons:
        print(f"{verdict[1]} for {args.period}" if args.mode == "evaluate" else verdict[1])
        for line in reasons:
            print(f"- {line}")
        return EXIT_NOT_PROMOTED
    baselines = " and ".join(REQUIRED_BASELINES)
    if args.mode == "evaluate":
        print(f"PASSED for {args.period}: the candidate passes the gate against {baselines}.")
        if QUARTER_NAME.fullmatch(args.period):
            print(
                f"One quarter is not a promotion: at least {QUARTERS_REQUIRED} of the 4 quarters must pass "
                "(`promote`)."
            )
        else:
            print("A monthly result is diagnostic only: promotion is decided on quarters (`promote`).")
    else:
        print(
            f"CONSISTENT for {args.period}: both scorecards pass and match the pinned references "
            f"({baselines})."
        )
        print("This is not a promotion.")
    return 0


def _report_promotion(quarters: list[Quarter], year: int, model_id: str, model_version: str) -> int:
    passed = [q.period for q in quarters if q.passed]
    for quarter in quarters:
        print(f"{quarter.period}: {'PASSED' if quarter.passed else 'NOT PASSED'}")
        for line in quarter.reasons:
            print(f"  - {line}")
    count = f"{len(passed)} of {len(quarters)} quarters of {year} passed (rule: at least {QUARTERS_REQUIRED})"
    who = f"{model_id} version {model_version}"
    if len(passed) >= QUARTERS_REQUIRED:
        print(f"PROMOTED: {who}. {count}, each against {' and '.join(REQUIRED_BASELINES)}.")
        return 0
    print(f"NOT PROMOTED: {who}. {count}.")
    return EXIT_NOT_PROMOTED


if __name__ == "__main__":
    sys.exit(main())
