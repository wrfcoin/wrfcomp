import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _tool():
    spec = importlib.util.spec_from_file_location("check_promotion", ROOT / "scripts" / "check_promotion.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REFS = {"climatology": {"clim-hash"}, "noaa-gefs": {"gefs-hash"}}


def _card(baseline: str, baseline_hash: str) -> dict:
    return {
        "model_id": "candidate",
        "model_version": "7",
        "forecast_sha256": "f" * 64,
        "truth_sha256": "t" * 64,
        "thresholds_sha256": "h" * 64,
        "protocol_version": "0.1.0",
        "harness_version": {"wrfcomp": "0.1.0", "scores": "2.7.0"},
        "period_start": "2024-01-01T00:00:00",
        "period_end": "2024-01-31T12:00:00",
        "init_hours_scored": [0, 12],
        "n_stations": 500,
        "baseline_id": baseline,
        "baseline_sha256": baseline_hash,
        "passed_gate": True,
        "sealed_data_used": False,
        "variables_missing": [],
    }


def _pair() -> list[dict]:
    return [_card("climatology", "clim-hash"), _card("noaa-gefs", "gefs-hash")]


def test_passing_both_references_is_promoted() -> None:
    assert _tool().check(_pair(), REFS) == []


@pytest.mark.parametrize("which", [0, 1])
def test_failing_either_reference_is_not_promoted(which: int) -> None:
    cards = _pair()
    cards[which]["passed_gate"] = False
    reasons = _tool().check(cards, REFS)
    assert any("gate not passed" in r for r in reasons)


def test_one_scorecard_twice_is_not_enough() -> None:
    cards = [_card("noaa-gefs", "gefs-hash"), _card("noaa-gefs", "gefs-hash")]
    reasons = _tool().check(cards, REFS)
    assert any("climatology" in r for r in reasons)


def test_a_stand_in_baseline_is_refused() -> None:
    cards = _pair()
    cards[0]["baseline_sha256"] = "weaker-climatology"
    assert any("not a committed reference" in r for r in _tool().check(cards, REFS))


@pytest.mark.parametrize("key", ["forecast_sha256", "truth_sha256", "thresholds_sha256", "period_end",
                                 "model_version", "protocol_version", "init_hours_scored", "n_stations"])
def test_scorecards_must_describe_the_same_run(key: str) -> None:
    cards = _pair()
    cards[1] = copy.deepcopy(cards[1])
    cards[1][key] = "different"
    assert any(key in r for r in _tool().check(cards, REFS))


def test_sealed_data_or_missing_variables_block_promotion() -> None:
    cards = _pair()
    cards[0]["sealed_data_used"] = True
    cards[1]["variables_missing"] = ["msl"]
    reasons = _tool().check(cards, REFS)
    assert any("sealed" in r for r in reasons) and any("missing" in r for r in reasons)


def test_a_reference_cannot_promote_itself() -> None:
    cards = _pair()
    for c in cards:
        c["model_id"] = "noaa-gefs"
    assert any("own candidate" in r for r in _tool().check(cards, REFS))


def test_committed_references_are_found_and_no_reference_passes() -> None:
    tool = _tool()
    refs = tool.reference_forecast_hashes(ROOT / "experiments")
    assert refs["climatology"] and refs["noaa-gefs"]
    gefs = json.loads((ROOT / "experiments" / "gefs_2024-01.json").read_text())
    pers = json.loads((ROOT / "experiments" / "persistence_2024-01.json").read_text())
    assert tool.check([gefs, pers], refs) != []
