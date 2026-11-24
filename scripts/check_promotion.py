"""Promotion check: a candidate must pass the gate against BOTH reference baselines.

Owner decision 2026-11-23: a candidate is promoted only if it passes the harness gate against
climatology AND against NOAA GEFS, on the same forecast, truth, thresholds and period.

    uv run --locked python scripts/check_promotion.py evaluate \\
        --forecast F.zarr --data-dir D --period 2024-01 --out DIR
    uv run --locked python scripts/check_promotion.py verify A.json B.json --period 2024-01

`evaluate` is the only mode that can print PROMOTED. Everything it relies on comes from one tree,
the checkout this script is in:
  1. that tree must pass the harness lock (`scripts/harness_lock.py check`), and the harness modules
     this process has imported must be files of that tree;
  2. the two reference forecasts of the period are taken from the data directory, at the places
     named in the tree's `experiments/REFERENCES.json`, and refused unless their content hash is the
     pinned one;
  3. `wrfcomp-eval` is run twice through `uv run --locked` on that tree, with its locked
     `configs/splits.yaml`: the candidate against climatology and against NOAA GEFS;
  4. the two scorecards are judged (rules below). They and `verdict.json` are written in a private
     directory made for the run and published as `<out>/promotion_<period>_<forecast hash>` at the
     end. File names are fixed by this program; the candidate's model id is data inside the files.

`verify` applies the same rules to two scorecards that already exist. It proves nothing about where
they came from, so it prints CONSISTENT or INCONSISTENT and never PROMOTED.

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

Exit codes: 0 PROMOTED (evaluate) or CONSISTENT (verify); 1 NOT PROMOTED or INCONSISTENT;
2 refused (bad input, unpinned or missing references, harness not locked or not from this tree).
Locked: see docs/HARNESS_LOCK.md, which also lists what this check does not establish.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
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
EXIT_NOT_PROMOTED = 1
EXIT_REFUSED = 2
# Names this program gives the files of one run. Nothing from the candidate goes into a path.
STAGED_NAMES = {"climatology": "candidate_vs_climatology.json", "noaa-gefs": "candidate_vs_noaa-gefs.json"}
VERDICT_NAME = "verdict.json"
PERIOD_NAME = re.compile(r"[0-9]{4}-[0-9]{2}")
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


def _modules_outside_root(root: Path) -> list[str]:
    """Harness modules that this process did NOT load from `root`'s source tree.

    The tree check looks at files under `root`; the numbers come from whatever Python imports. Both
    must be the same files, so every harness module has to resolve inside `root/src/wrfcomp`.
    """
    package = (root / "src" / "wrfcomp").resolve()
    outside = []
    for name in HARNESS_MODULES:
        module = importlib.import_module(name)
        origin = Path(getattr(module, "__file__", None) or "/nonexistent").resolve()
        if not origin.is_relative_to(package):
            outside.append(f"{name} is loaded from {origin}")
    return outside


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
    """`wrfcomp-eval` as installed from `root`, with the dependency versions of `root`'s uv.lock."""
    uv = shutil.which("uv")
    if uv is None:
        raise Refused("uv is not installed; evaluations run through `uv run --locked`")
    return [uv, "run", "--locked", "--project", str(root), "wrfcomp-eval", *args]


def _run_eval(root: Path, args: list[str]) -> int:
    environment = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}  # uv picks root's own
    return subprocess.run(_eval_command(root, args), cwd=root, env=environment, check=False).returncode


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


def evaluate(forecast: Path, data_dir: Path, period: str, out_dir: Path, root: Path) -> Outcome:
    """Score the candidate against both pinned references and judge.

    Everything the verdict rests on comes from the one tree at `root`: the lock manifest and the
    tree it describes, the harness modules this process imports, the reference inventory, the
    config, and the `wrfcomp-eval` that `uv run --locked --project root` starts. If any of them is
    somewhere else, the call is refused.

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

    with _open_forecast(forecast, "candidate") as ds:
        model_id, model_version = str(ds.attrs.get("model_id", "")), str(ds.attrs.get("model_version", ""))
    if not model_id or not model_version:
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
        cards: list[dict] = []
        reasons: list[str] = []
        for name in REQUIRED_BASELINES:
            target = staging / STAGED_NAMES[name]
            args = [
                "--forecast",
                str(forecast.resolve()),
                "--truth",
                str(truth),
                "--thresholds",
                str(thresholds),
            ]
            args += ["--baseline", str(stores[name]), "--baseline-id", name, "--out", str(target)]
            args += ["--model-id", model_id, "--model-version", model_version]
            args += ["--config", str(root / "configs" / "splits.yaml")]
            code = _run_eval(root, args)
            if code != 0 or not target.is_file():
                target.unlink(missing_ok=True)
                reasons.append(f"wrfcomp-eval refused the candidate against {name} (exit code {code})")
                continue
            cards.append(_read_json(target, "scorecard"))
        if not reasons:
            reasons = judge(cards, period, inventory, root)

        digests = {str(card.get("forecast_sha256")) for card in cards}
        digest = digests.pop() if len(digests) == 1 else ""
        verdict = {
            "period": period,
            "verdict": "NOT PROMOTED" if reasons else "PROMOTED",
            "reasons": reasons,
            "baselines": list(REQUIRED_BASELINES),
            "candidate": {
                "model_id": model_id,
                "model_version": model_version,
                "forecast_sha256": digest or None,
            },
            "scorecards": sorted(p.name for p in staging.iterdir()),
        }
        text = json.dumps(verdict, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
        (staging / VERDICT_NAME).write_text(text, encoding="ascii")
        token = digest[:16] if HEX_DIGEST.fullmatch(digest) else "unscored-" + staging.name.rsplit("-", 1)[-1]
        return Outcome(reasons=reasons, directory=_publish(staging, out_dir, f"promotion_{period}_{token}"))
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


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
    run.add_argument("--forecast", type=Path, required=True, help="candidate forecast (Zarr)")
    run.add_argument("--data-dir", type=Path, required=True, help="data directory with truth and references")
    run.add_argument("--period", required=True, help="a period pinned in experiments/REFERENCES.json")
    run.add_argument("--out", type=Path, required=True, help="directory for the two scorecards")
    look = modes.add_parser("verify", help="compare two existing scorecards with the pinned references")
    look.add_argument("scorecards", type=Path, nargs=2, metavar="SCORECARD")
    look.add_argument("--period", required=True, help="a period pinned in experiments/REFERENCES.json")
    args = parser.parse_args(argv)

    try:
        if args.mode == "evaluate":
            outcome = evaluate(args.forecast, args.data_dir, args.period, args.out, root=ROOT)
            reasons = outcome.reasons
            print(f"Scorecards and verdict: {outcome.directory}")
            verdict = ("PROMOTED", "NOT PROMOTED")
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
        print(verdict[1])
        for line in reasons:
            print(f"- {line}")
        return EXIT_NOT_PROMOTED
    baselines = " and ".join(REQUIRED_BASELINES)
    if args.mode == "evaluate":
        print(f"PROMOTED for {args.period}: the candidate passes the gate against {baselines}.")
    else:
        print(
            f"CONSISTENT for {args.period}: both scorecards pass and match the pinned references "
            f"({baselines})."
        )
        print("This is not a promotion.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
