"""Summarise a candidate's official promotion records into one small JSON file.

Reads only the committed records under results/<candidate>/<year>/promotion_*/ (the locked
check's own output) and writes results/<candidate>/summary.json. With no argument, every
candidate folder under results/ that holds promotion records is summarised. Numbers are copied from the
scorecards, never recomputed, so anyone can trace each value to its source file.
"""

import argparse
import json
from pathlib import Path

VARIABLES = ("t2m", "td2m", "u10", "v10", "msl")
RESULTS = Path(__file__).resolve().parents[1] / "results"


def candidate_dirs(results: Path = RESULTS) -> list[Path]:
    """Every folder under ``results`` with at least one year's promotion decision."""
    return sorted(p for p in results.iterdir() if p.is_dir() and any(p.glob("*/promotion_*/decision.json")))


def summarize(candidate_dir: Path) -> dict:
    years = {}
    for year_dir in sorted(p for p in candidate_dir.iterdir() if p.is_dir() and p.name.isdigit()):
        decision_file = next(year_dir.glob(f"promotion_{year_dir.name}_*/decision.json"))
        decision = json.loads(decision_file.read_text())
        quarters = {}
        for qdir in sorted(year_dir.glob(f"promotion_{year_dir.name}-Q*")):
            quarter = qdir.name.split("_")[1]
            verdict = json.loads((qdir / "verdict.json").read_text())
            refs = {}
            for card in sorted(qdir.glob("candidate_vs_*.json")):
                data = json.loads(card.read_text())
                metrics = {v: data["metrics"][v]["all"] for v in VARIABLES}
                refs[data["baseline_id"]] = {
                    "all_gates_passed": all(all(g.values()) for g in data["gates"].values()),
                    "crpss_vs_baseline": {v: m["crpss_vs_baseline"] for v, m in metrics.items()},
                    "spread_skill_ratio": {v: m["spread_skill_ratio"] for v, m in metrics.items()},
                    "n_init_times": data["n_init_times"],
                    "n_stations": data["n_stations"],
                    "source": card.relative_to(candidate_dir.parent.parent).as_posix(),
                }
            quarters[quarter] = {"verdict": verdict["verdict"], "references": refs}
        years[year_dir.name] = {
            "decision": decision["decision"],
            "quarters_passed": decision["quarters_passed"],
            "quarters_required": decision["quarters_required"],
            "promotion_rule_version": decision["promotion_rule_version"],
            "harness_version": decision["harness_version"],
            "source": decision_file.relative_to(candidate_dir.parent.parent).as_posix(),
            "quarters": quarters,
        }
    return {"candidate": candidate_dir.name, "years": years}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_dir", type=Path, nargs="*", help="default: every candidate in results/")
    args = parser.parse_args()
    for candidate_dir in args.candidate_dir or candidate_dirs():
        summary = summarize(candidate_dir)
        out = candidate_dir / "summary.json"
        out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(out)


if __name__ == "__main__":
    main()
