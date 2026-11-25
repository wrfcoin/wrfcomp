import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("summarize_results", ROOT / "scripts" / "summarize_results.py")
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def test_committed_summaries_match_their_records() -> None:
    for summary in (ROOT / "results").glob("*/summary.json"):
        assert json.loads(summary.read_text()) == tool.summarize(summary.parent)


def test_v3_results_are_as_reported() -> None:
    years = json.loads((ROOT / "results" / "gefs-station-corrected-v3" / "summary.json").read_text())["years"]
    assert (years["2024"]["decision"], years["2024"]["quarters_passed"]) == ("PROMOTED", 3)
    assert (years["2023"]["decision"], years["2023"]["quarters_passed"]) == ("PROMOTED", 4)
