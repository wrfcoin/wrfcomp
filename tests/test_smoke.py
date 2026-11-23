from datetime import datetime
from pathlib import Path

import yaml

import wrfcomp

SPLITS = Path(__file__).resolve().parents[1] / "configs" / "splits.yaml"


def _ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def test_package_imports() -> None:
    assert wrfcomp.__version__


def test_splits_do_not_overlap() -> None:
    cfg = yaml.safe_load(SPLITS.read_text())
    s = cfg["splits"]
    assert _ts(s["train"]["end_utc"]) < _ts(s["validation"]["start_utc"])
    assert _ts(s["validation"]["end_utc"]) < _ts(s["sealed_test"]["start_utc"])
    gap = _ts(s["validation"]["start_utc"]) - _ts(s["train"]["end_utc"])
    assert gap.days >= s["train"]["embargo_days_after"]


def test_gate_weights_sum_to_one() -> None:
    cfg = yaml.safe_load(SPLITS.read_text())
    weights = cfg["evaluation"]["promotion_gate"]["loss_weights"]
    assert abs(sum(weights.values()) - 1.0) < 1e-12
