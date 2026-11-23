"""Promotion check: a candidate must pass the gate against BOTH reference baselines.

Owner decision 2026-11-23: a candidate is promoted only if it passes the harness gate against
climatology AND against NOAA GEFS, on the same forecast, truth, thresholds and period. This script
reads the two `wrfcomp-eval` scorecards and the committed reference scorecards and prints a verdict.
It computes no score itself. Standard library only. Locked: see docs/HARNESS_LOCK.md.

Exit code 0 = promoted, 1 = not promoted, 2 = bad input.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REQUIRED_BASELINES = ("climatology", "noaa-gefs")
MUST_MATCH = (
    "model_id",
    "model_version",
    "forecast_sha256",
    "truth_sha256",
    "thresholds_sha256",
    "protocol_version",
    "harness_version",
    "period_start",
    "period_end",
    "init_hours_scored",
    "n_stations",
)


def _load(path: Path) -> dict:
    try:
        card = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise SystemExit2(f"cannot read scorecard {path}: {exc}") from exc
    if not isinstance(card, dict) or "passed_gate" not in card:
        raise SystemExit2(f"{path} is not a wrfcomp-eval scorecard")
    return card


class SystemExit2(Exception):
    pass


def reference_forecast_hashes(reference_dir: Path) -> dict[str, set[str]]:
    """Map each required baseline id to the forecast hashes of its committed reference scorecards."""
    out: dict[str, set[str]] = {name: set() for name in REQUIRED_BASELINES}
    for path in sorted(reference_dir.glob("*.json")):
        try:
            card = json.loads(path.read_text())
        except ValueError:
            continue
        if isinstance(card, dict) and card.get("model_id") in out and card.get("forecast_sha256"):
            out[card["model_id"]].add(card["forecast_sha256"])
    return out


def check(cards: list[dict], references: dict[str, set[str]]) -> list[str]:
    """Return the reasons the candidate is NOT promoted. An empty list means promoted."""
    reasons: list[str] = []
    by_baseline: dict[str, dict] = {}
    for card in cards:
        baseline = card.get("baseline_id")
        if baseline in by_baseline:
            reasons.append(f"two scorecards against the same baseline: {baseline}")
        by_baseline[baseline] = card
    for name in REQUIRED_BASELINES:
        if name not in by_baseline:
            reasons.append(f"no scorecard against required baseline: {name}")
    for name in by_baseline:
        if name not in REQUIRED_BASELINES:
            reasons.append(f"scorecard against a baseline that is not a reference: {name}")
    if reasons:
        return reasons

    first = by_baseline[REQUIRED_BASELINES[0]]
    for name in REQUIRED_BASELINES:
        card = by_baseline[name]
        if card.get("passed_gate") is not True:
            reasons.append(f"gate not passed against {name}")
        if card.get("sealed_data_used") is not False:
            reasons.append(f"sealed data used (or not recorded) in the scorecard against {name}")
        if card.get("variables_missing"):
            reasons.append(f"primary variables missing in the scorecard against {name}")
        if card.get("model_id") in REQUIRED_BASELINES:
            reasons.append("a reference baseline cannot be its own candidate")
        if card.get("baseline_sha256") not in references.get(name, set()):
            reasons.append(f"baseline {name} is not a committed reference forecast (baseline_sha256 unknown)")
        for key in MUST_MATCH:
            if card.get(key) != first.get(key) or card.get(key) is None:
                reasons.append(f"scorecards disagree on, or lack, {key}")
    return sorted(set(reasons))


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print("usage: check_promotion.py <scorecard A> <scorecard B> <reference scorecards dir>")
        return 2
    try:
        cards = [_load(Path(argv[1])), _load(Path(argv[2]))]
    except SystemExit2 as exc:
        print(exc)
        return 2
    reference_dir = Path(argv[3])
    if not reference_dir.is_dir():
        print(f"{reference_dir} is not a directory")
        return 2
    reasons = check(cards, reference_forecast_hashes(reference_dir))
    if reasons:
        print("NOT PROMOTED")
        for line in reasons:
            print(f"- {line}")
        return 1
    print(f"PROMOTED: {cards[0]['model_id']} {cards[0]['model_version']} passes the gate against "
          + " and ".join(REQUIRED_BASELINES))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
