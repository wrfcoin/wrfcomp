import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("summarize_results", ROOT / "scripts" / "summarize_results.py")
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def test_committed_summaries_match_their_records() -> None:
    candidates = tool.candidate_dirs()
    names = [c.name for c in candidates]
    assert names == ["gefs-station-corrected-v3", "gefs-station-corrected-v4"]
    assert sorted(s.parent.name for s in (ROOT / "results").glob("*/summary.json")) == names
    for candidate in candidates:
        assert json.loads((candidate / "summary.json").read_text()) == tool.summarize(candidate)


def test_v3_results_are_as_reported() -> None:
    years = json.loads((ROOT / "results" / "gefs-station-corrected-v3" / "summary.json").read_text())["years"]
    assert (years["2024"]["decision"], years["2024"]["quarters_passed"]) == ("PROMOTED", 3)
    assert (years["2023"]["decision"], years["2023"]["quarters_passed"]) == ("PROMOTED", 4)


def test_v4_results_are_as_reported() -> None:
    years = json.loads((ROOT / "results" / "gefs-station-corrected-v4" / "summary.json").read_text())["years"]
    assert (years["2024"]["decision"], years["2024"]["quarters_passed"]) == ("PROMOTED", 4)
    assert (years["2023"]["decision"], years["2023"]["quarters_passed"]) == ("PROMOTED", 4)
    for year in years.values():
        assert all(q["verdict"] == "PASSED" for q in year["quarters"].values())
