"""The promotion check: evaluation against both pinned references, and every refusal.

The world is synthetic and seeded (tests/conftest.py): a "climatology" and a "noaa-gefs" reference
forecast, their reference scorecards, an inventory that pins them, and a candidate that beats both.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from conftest import make_synthetic, open_zarr, station_ids, write_config, write_netcdf, write_zarr
from typer.testing import CliRunner

from wrfcomp.harness.evaluate import app as eval_app
from wrfcomp.harness.schemas import Scorecard

ROOT = Path(__file__).resolve().parents[1]
PERIOD = "2024-03"
N_STATIONS = 20


def _load():
    spec = importlib.util.spec_from_file_location("check_promotion", ROOT / "scripts" / "check_promotion.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_promotion"] = module
    spec.loader.exec_module(module)
    return module


tool = _load()


def _eval(
    root: Path,
    forecast: Path,
    out: Path,
    model_id: str,
    baseline: tuple[Path, str] | None,
    truth: str = "truth/truth.nc",
    thresholds: str = "thresholds/tail.nc",
) -> None:
    args = ["--forecast", str(forecast), "--truth", str(root / "data" / truth)]
    args += ["--thresholds", str(root / "data" / thresholds), "--out", str(out)]
    args += ["--model-id", model_id, "--model-version", "1", "--config", str(root / "configs/splits.yaml")]
    if baseline:
        args += ["--baseline", str(baseline[0]), "--baseline-id", baseline[1]]
    result = CliRunner().invoke(eval_app, args)
    assert result.exit_code == 0, f"{result.output}\n{result.exception!r}"


def build_world(root: Path) -> Path:
    """Repository-shaped directory: configs, experiments with pinned references, and a data dir."""
    syn = make_synthetic()
    rng = np.random.default_rng(7)
    data = root / "data"
    for sub in ("truth", "thresholds", "refs", "candidates"):
        (data / sub).mkdir(parents=True)
    write_config(root, list(station_ids(N_STATIONS)))
    write_netcdf(syn.truth, data / "truth/truth.nc")
    write_netcdf(syn.thresholds, data / "thresholds/tail.nc")

    # "noaa-gefs": the fixture's weaker forecast, without the shortest lead (as GEFS has no 1 h lead).
    gefs = syn.baseline.isel(lead_time=[1, 2]).copy(deep=True)
    gefs.attrs["model_id"] = "noaa-gefs"
    # "climatology": weaker still. A wide, unsharp ensemble around the gefs-like centre.
    climatology = syn.baseline.copy(deep=True)
    for name in climatology.data_vars:
        spread = float(syn.truth[name].std()) * 0.5
        noise = rng.normal(0, spread, climatology[name].shape).astype("float32")
        climatology[name] = (
            climatology[name].dims,
            climatology[name].values + noise,
            climatology[name].attrs,
        )
    climatology.attrs["model_id"] = "climatology"
    weak = climatology.copy(deep=True)
    weak.attrs["model_id"] = "weak-candidate"
    for name in weak.data_vars:
        weak[name] = (weak[name].dims, weak[name].values + np.float32(0.25), weak[name].attrs)

    write_zarr(climatology, data / "refs/climatology.zarr")
    write_zarr(gefs, data / "refs/gefs.zarr")
    write_zarr(syn.candidate, data / "candidates/good.zarr")
    write_zarr(weak, data / "candidates/weak.zarr")

    experiments = root / "experiments"
    experiments.mkdir()
    _eval(
        root, data / "refs/climatology.zarr", experiments / f"climatology_{PERIOD}.json", "climatology", None
    )
    reference = (data / "refs/climatology.zarr", "climatology")
    _eval(root, data / "refs/gefs.zarr", experiments / f"gefs_{PERIOD}.json", "noaa-gefs", reference)
    periods = {
        PERIOD: {
            "truth": "truth/truth.nc",
            "thresholds": "thresholds/tail.nc",
            "references": {
                "climatology": {
                    "scorecard": f"experiments/climatology_{PERIOD}.json",
                    "store": "refs/climatology.zarr",
                },
                "noaa-gefs": {"scorecard": f"experiments/gefs_{PERIOD}.json", "store": "refs/gefs.zarr"},
            },
        }
    }
    inventory = tool.build_inventory(root, periods)
    (root / tool.INVENTORY).write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n")
    return root


def in_process_eval(root: Path, args: list[str]) -> int:
    """Stands in for `uv run --locked wrfcomp-eval`: the same command, run in this process."""
    return CliRunner().invoke(eval_app, args).exit_code


def trust(patch: pytest.MonkeyPatch) -> None:
    """Pretend the synthetic world is a locked checkout whose harness is the one imported here.

    The real checks are tested on their own below; the synthetic world is not a Git repository and
    holds no source tree, so they would (rightly) refuse it.
    """
    patch.setattr(tool, "_harness_lock_problems", lambda root: [])
    patch.setattr(tool, "_modules_outside_root", lambda root: [])
    patch.setattr(tool, "_run_eval", in_process_eval)


@pytest.fixture(scope="module")
def base_world(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Built once, never modified. `cards` are the two scorecards of the good candidate."""
    root = build_world(tmp_path_factory.mktemp("promotion") / "repo")
    with pytest.MonkeyPatch.context() as patch:
        trust(patch)
        out = root.parent / "good-cards"
        outcome = tool.evaluate(root / "data/candidates/good.zarr", root / "data", PERIOD, out, root)
    cards = [json.loads((outcome.directory / name).read_text()) for name in tool.STAGED_NAMES.values()]
    return {
        "root": root,
        "reasons": outcome.reasons,
        "cards": cards,
        "out": out,
        "directory": outcome.directory,
        "inventory": tool.load_inventory(root),
    }


@pytest.fixture
def locked(monkeypatch: pytest.MonkeyPatch) -> None:
    trust(monkeypatch)


def judge(base_world: dict, cards: list[dict]) -> list[str]:
    return tool.judge(cards, PERIOD, base_world["inventory"], base_world["root"])


def good_cards(base_world: dict) -> tuple[dict, dict]:
    """Fresh copies: (against climatology, against noaa-gefs)."""
    cards = {card["baseline_id"]: copy.deepcopy(card) for card in base_world["cards"]}
    return cards["climatology"], cards["noaa-gefs"]


# --------------------------------------------------------------------------- the passing case


def test_evaluate_runs_both_evaluations_and_promotes_a_better_candidate(base_world: dict) -> None:
    assert base_world["reasons"] == []
    # one published directory, named by the program from the period and the hash of its verdict file
    directory = base_world["directory"]
    digest = base_world["cards"][0]["forecast_sha256"]
    assert [p.name for p in base_world["out"].iterdir()] == [directory.name]
    assert directory.name == tool._result_name(PERIOD, (directory / "verdict.json").read_text())
    assert sorted(p.name for p in directory.iterdir()) == [
        "candidate_vs_climatology.json",
        "candidate_vs_noaa-gefs.json",
        "verdict.json",
    ]
    verdict = json.loads((directory / "verdict.json").read_text())
    assert verdict["scorecards"] == {
        name: tool._sha256_file(directory / name) for name in sorted(tool.STAGED_NAMES.values())
    }
    # one period passed: PASSED, never PROMOTED (promotion is decided on quarters, by `promote`)
    assert verdict["verdict"] == "PASSED" and verdict["reasons"] == [] and verdict["period"] == PERIOD
    assert verdict["candidate"] == {"model_id": "synth", "model_version": "1", "forecast_sha256": digest}
    assert verdict["promotion_rule_version"] == tool.PROMOTION_RULE_VERSION == 2
    assert verdict["harness_version"] == base_world["inventory"]["harness_version"]
    assert verdict["protocol_version"] == base_world["inventory"]["protocol_version"]
    assert verdict["references_sha256"] == tool._sha256_file(base_world["root"] / tool.INVENTORY)
    assert verdict["forecast_inputs"] == [
        {
            "content_sha256": digest,
            "init_first": "2024-03-01T00:00:00",
            "init_last": "2024-03-15T12:00:00",
            "n_init_times": 30,
        }
    ]
    assert "PROMOTED" not in (directory / "verdict.json").read_text()
    against_climatology, against_gefs = good_cards(base_world)
    for card in (against_climatology, against_gefs):
        Scorecard.model_validate(card)
        assert card["passed_gate"] is True and card["model_id"] == "synth"
    record = base_world["inventory"]["periods"][PERIOD]["references"]
    assert against_climatology["leads_scored"] == record["climatology"]["leads_scored"] == [6, 12, 24]
    assert (
        against_gefs["leads_scored"] == record["noaa-gefs"]["leads_scored"] == [12, 24]
    )  # the leads GEFS has
    assert against_climatology["baseline_sha256"] == record["climatology"]["forecast_sha256"]
    assert against_gefs["baseline_sha256"] == record["noaa-gefs"]["forecast_sha256"]
    assert judge(base_world, [against_gefs, against_climatology]) == []  # order does not matter


def test_command_line_verdicts(base_world: dict, locked: None, monkeypatch, capsys, tmp_path: Path) -> None:
    monkeypatch.setattr(tool, "ROOT", base_world["root"])
    data = base_world["root"] / "data"
    args = ["--data-dir", str(data), "--period", PERIOD, "--out", str(tmp_path / "out")]
    assert tool.main(["evaluate", "--forecast", str(data / "candidates/good.zarr"), *args]) == 0
    out = capsys.readouterr().out
    assert f"PASSED for {PERIOD}" in out and "NOT PASSED" not in out
    assert "A monthly result is diagnostic only" in out and "PROMOTED" not in out

    assert tool.main(["evaluate", "--forecast", str(data / "candidates/weak.zarr"), *args]) == 1
    out = capsys.readouterr().out
    assert f"NOT PASSED for {PERIOD}" in out and "gate not passed against" in out and "PROMOTED" not in out

    # verify: the same good scorecards are only ever CONSISTENT
    paths = [str(base_world["directory"] / name) for name in tool.STAGED_NAMES.values()]
    assert tool.main(["verify", *paths, "--period", PERIOD]) == 0
    out = capsys.readouterr().out
    assert "CONSISTENT" in out and "proves nothing about where the scorecards came from" in out
    assert "This is not a promotion." in out
    assert "PROMOTED" not in out and "PASSED" not in out
    bad = tmp_path / "edited.json"
    card = json.loads(Path(paths[0]).read_text())
    card["truth_sha256"] = "0" * 64
    bad.write_text(json.dumps(card))
    assert tool.main(["verify", str(bad), paths[1], "--period", PERIOD]) == 1
    out = capsys.readouterr().out
    assert "INCONSISTENT" in out and "PROMOTED" not in out


# --------------------------------------------------------------------------- refusals: the cards


def test_hand_edited_verdicts_are_refused(base_world: dict) -> None:
    # 1. passed_gate flipped by hand: the scorecard model itself rejects the contradiction
    card, other = good_cards(base_world)
    card["passed_gate"] = False
    assert any("not a valid harness scorecard" in r for r in judge(base_world, [card, other]))
    card, other = good_cards(base_world)
    card["gates"]["t2m"]["dm"] = False  # a failing gate, verdict left at "passed"
    assert any("not a valid harness scorecard" in r for r in judge(base_world, [card, other]))

    # 2. numbers made worse, gates and verdict left alone: valid JSON, valid model, false claim
    card, other = good_cards(base_world)
    card["metrics"]["t2m"]["all"]["crpss_vs_baseline"] = 0.01
    reasons = judge(base_world, [card, other])
    assert "the gates recorded for t2m against climatology are not what its metrics give" in reasons
    assert "passed_gate against climatology is True but the metrics imply False" in reasons
    assert "gate not passed against climatology" in reasons

    # 3. a failing gate and verdict edited to pass together, numbers untouched
    card, other = good_cards(base_world)
    card["metrics"]["msl"]["all"]["n_members"] = 4  # too few members for the gate
    reasons = judge(base_world, [card, other])
    assert "the gates recorded for msl against climatology are not what its metrics give" in reasons

    # 4. the other card is judged with the same care
    card, other = good_cards(base_world)
    other["metrics"]["u10"]["all"]["dm_pvalue"] = 0.9
    assert "gate not passed against noaa-gefs" in judge(base_world, [card, other])


def test_card_without_metrics_or_gates_is_refused(base_world: dict) -> None:
    for missing in ("metrics", "gates", "n_pairs", "leads_scored", "harness_version", "passed_gate"):
        card, other = good_cards(base_world)
        del card[missing]
        reasons = judge(base_world, [card, other])
        assert any("not a valid harness scorecard" in r for r in reasons), missing
    # the kind of record the first version of this check accepted: a verdict and a few hashes
    card, other = good_cards(base_world)
    bare = {
        k: card[k] for k in ("model_id", "model_version", "forecast_sha256", "truth_sha256", "passed_gate")
    }
    bare.update(baseline_id="climatology", baseline_sha256=card["baseline_sha256"])
    assert any("not a valid harness scorecard" in r for r in judge(base_world, [bare, other]))
    assert judge(base_world, ["not a card", other])


def test_wrong_set_of_scorecards_is_refused(base_world: dict) -> None:
    against_climatology, against_gefs = good_cards(base_world)
    assert "two scorecards against the same baseline: climatology" in judge(
        base_world, [against_climatology, copy.deepcopy(against_climatology)]
    )
    assert "no scorecard against the required baseline noaa-gefs" in judge(base_world, [against_climatology])
    third = copy.deepcopy(against_gefs)
    third["baseline_id"] = "persistence"
    reasons = judge(base_world, [against_climatology, third])
    assert "a scorecard against 'persistence', which is not a reference baseline" in reasons
    assert judge(base_world, [against_climatology, against_gefs, third])
    assert judge(base_world, [])


CONTEXT_EDITS = {
    "truth_sha256": "0" * 64,
    "thresholds_sha256": "1" * 64,
    "period_start": "2024-03-02T00:00:00",
    "period_end": "2024-03-14T12:00:00",
    "n_init_times": 29,
    "init_hours_scored": [0],
    "n_stations": 19,
}


@pytest.mark.parametrize("key", sorted(CONTEXT_EDITS))
@pytest.mark.parametrize("which", [0, 1])
def test_evaluation_context_must_equal_the_pinned_reference(base_world: dict, key: str, which: int) -> None:
    cards = list(good_cards(base_world))
    cards[which][key] = CONTEXT_EDITS[key]
    name = ("climatology", "noaa-gefs")[which]
    assert f"{key} against {name} differs from the pinned reference for {PERIOD}" in judge(base_world, cards)


def test_harness_version_sealed_data_and_missing_variables_are_refused(base_world: dict) -> None:
    card, other = good_cards(base_world)
    card["harness_version"] = {"wrfcomp": "0.2.0", "scores": card["harness_version"]["scores"]}
    assert any(r.startswith("harness_version against climatology") for r in judge(base_world, [card, other]))
    card, other = good_cards(base_world)
    other["protocol_version"] = "0.2.0"
    assert any(r.startswith("protocol_version against noaa-gefs") for r in judge(base_world, [card, other]))
    card, other = good_cards(base_world)
    card["sealed_data_used"] = True
    assert "sealed data was used against climatology" in judge(base_world, [card, other])

    card, other = good_cards(base_world)
    for key in ("metrics", "gates", "n_pairs"):
        del card[key]["msl"]
    card.update(variables_missing=["msl"], passed_gate=False)
    Scorecard.model_validate(card)  # an honest card of a forecast without msl
    reasons = judge(base_world, [card, other])
    assert "primary variables are missing against climatology: ['msl']" in reasons
    assert "gate not passed against climatology" in reasons
    # the same variable dropped silently, verdict left at "passed"
    card, other = good_cards(base_world)
    for key in ("metrics", "gates", "n_pairs"):
        del card[key]["msl"]
    reasons = judge(base_world, [card, other])
    assert "primary variables are missing against climatology: ['msl']" in reasons
    assert "passed_gate against climatology is True but the metrics imply False" in reasons


def test_leads_and_coverage_must_match_each_reference(base_world: dict) -> None:
    # against climatology every lead is expected: dropping the hardest one is refused
    card, other = good_cards(base_world)
    card["leads_scored"] = [6, 12]
    for per_lead in card["metrics"].values():
        del per_lead["24"]
    Scorecard.model_validate(card)
    assert "leads scored against climatology are [6, 12], expected [6, 12, 24]" in judge(
        base_world, [card, other]
    )
    # against GEFS only the leads GEFS has: a card claiming more is refused too
    card, other = good_cards(base_world)
    other["leads_scored"] = [6, 12, 24]
    for per_lead in other["metrics"].values():
        per_lead["6"] = copy.deepcopy(per_lead["12"])
    assert "leads scored against noaa-gefs are [6, 12, 24], expected [12, 24]" in judge(
        base_world, [card, other]
    )
    # fewer points scored than the reference scored at the same lead
    card, other = good_cards(base_world)
    card["metrics"]["t2m"]["12"]["n_pairs"] -= 5
    reasons = judge(base_world, [card, other])
    assert any(r.startswith("n_pairs for t2m at 12 h against climatology") for r in reasons)
    card, other = good_cards(base_world)
    other["metrics"]["v10"]["24"]["n_masked_no_threshold"] += 3
    reasons = judge(base_world, [card, other])
    assert any(r.startswith("n_masked_no_threshold for v10 at 24 h against noaa-gefs") for r in reasons)


def test_the_two_cards_must_describe_one_candidate(base_world: dict) -> None:
    for key, value in (("forecast_sha256", "2" * 64), ("model_version", "2"), ("model_id", "other")):
        card, other = good_cards(base_world)
        other[key] = value
        assert f"the two scorecards are not about the same candidate: {key} differs" in judge(
            base_world, [card, other]
        )


# --------------------------------------------------------------------------- refusals: references


def test_stand_in_baseline_is_refused(base_world: dict) -> None:
    card, other = good_cards(base_world)
    card["baseline_sha256"] = "3" * 64  # scored against something that only calls itself climatology
    reasons = judge(base_world, [card, other])
    assert f"the baseline against climatology is not the pinned reference forecast for {PERIOD}" in reasons
    # the two real references swapped
    card, other = good_cards(base_world)
    card["baseline_sha256"], other["baseline_sha256"] = other["baseline_sha256"], card["baseline_sha256"]
    assert len([r for r in judge(base_world, [card, other]) if "not the pinned reference forecast" in r]) == 2


def test_no_reference_directory_can_be_passed_in(base_world: dict, capsys) -> None:
    paths = [str(base_world["directory"] / name) for name in tool.STAGED_NAMES.values()]
    stand_in = str(base_world["root"] / "experiments")
    for argv in (
        ["verify", *paths, stand_in, "--period", PERIOD],  # the old third argument
        ["verify", *paths, "--period", PERIOD, "--references", stand_in],
        [
            "evaluate",
            "--forecast",
            "f.zarr",
            "--data-dir",
            "d",
            "--period",
            PERIOD,
            "--out",
            "o",
            "--references",
            stand_in,
        ],
        [
            "evaluate",
            "--forecast",
            "f.zarr",
            "--data-dir",
            "d",
            "--period",
            PERIOD,
            "--out",
            "o",
            "--config",
            "c.yaml",
        ],
        [*paths, stand_in],  # the old call form
    ):
        with pytest.raises(SystemExit) as exit_info:
            tool.main(argv)
        assert exit_info.value.code == 2
    capsys.readouterr()


def test_stand_in_reference_store_is_refused(base_world: dict, locked: None, tmp_path: Path) -> None:
    root = tmp_path / "repo"
    shutil.copytree(base_world["root"], root)
    # a weakened "climatology" put where the real one belongs: easier to beat
    store = root / "data/refs/climatology.zarr"
    weakened = open_zarr(store).load()
    weakened["t2m"] = (weakened["t2m"].dims, weakened["t2m"].values + np.float32(5.0), weakened["t2m"].attrs)
    shutil.rmtree(store)
    write_zarr(weakened, store)
    with pytest.raises(tool.Refused, match="reference climatology .* is not the pinned forecast"):
        tool.evaluate(root / "data/candidates/good.zarr", root / "data", PERIOD, tmp_path / "out", root)
    assert not (tmp_path / "out").exists()
    shutil.rmtree(store)
    with pytest.raises(tool.Refused, match="reference climatology: no such forecast store"):
        tool.evaluate(root / "data/candidates/good.zarr", root / "data", PERIOD, tmp_path / "out", root)


@pytest.mark.parametrize("reference", ["climatology", "gefs"])
def test_reference_store_with_a_sealed_valid_time_is_refused_before_it_is_hashed(
    base_world: dict, locked: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reference: str
) -> None:
    """A store in the data directory under a reference's pinned name, with a valid time in the
    sealed split: refused on its coordinates. Its content hash is never taken, so no value is read."""
    import wrfcomp.harness.hashing as hashing

    root = tmp_path / "repo"
    shutil.copytree(base_world["root"], root)
    sealed_start = tool._load_locked_config(root).splits.sealed_test.start
    store = root / f"data/refs/{reference}.zarr"
    misplaced = open_zarr(store).load()
    # init times stay in March 2024; the longest lead now ends on the first sealed instant
    leads = misplaced["lead_time"].values.copy()
    leads[-1] = sealed_start - misplaced["init_time"].values.max()
    misplaced = misplaced.assign_coords(lead_time=leads)
    assert misplaced["init_time"].values.max() < sealed_start
    shutil.rmtree(store)
    write_zarr(misplaced, store)

    real = hashing.content_sha256

    def sentinel(ds):
        latest = ds["init_time"].values.max() + ds["lead_time"].values.max()
        assert latest < sealed_start, "a store with sealed valid times was hashed"
        return real(ds)

    monkeypatch.setattr(hashing, "content_sha256", sentinel)
    name = {"climatology": "climatology", "gefs": "noaa-gefs"}[reference]
    with pytest.raises(tool.Refused, match=f"reference {name} at .* refused before it is read.*sealed split"):
        tool.evaluate(root / "data/candidates/good.zarr", root / "data", PERIOD, tmp_path / "out", root)
    assert not (tmp_path / "out").exists()


def test_tampered_inventory_or_reference_scorecard_is_refused(base_world: dict, tmp_path: Path) -> None:
    def world() -> Path:
        root = tmp_path / f"repo{len(list(tmp_path.iterdir()))}"
        shutil.copytree(base_world["root"], root, ignore=shutil.ignore_patterns("data"))
        return root

    def edit_inventory(root: Path, change) -> None:
        inventory = json.loads((root / tool.INVENTORY).read_text())
        change(inventory["periods"][PERIOD]["references"])
        (root / tool.INVENTORY).write_text(json.dumps(inventory))

    root = world()
    card_path = root / f"experiments/gefs_{PERIOD}.json"
    card_path.write_text(card_path.read_text().replace('"n_stations": 20', '"n_stations": 21'))
    with pytest.raises(tool.Refused, match="is missing or is not the pinned file"):
        tool.load_inventory(root)

    root = world()
    edit_inventory(root, lambda refs: refs["climatology"].update(forecast_sha256="4" * 64))
    with pytest.raises(tool.Refused, match="disagrees with its scorecard on forecast_sha256"):
        tool.load_inventory(root)

    root = world()
    edit_inventory(root, lambda refs: refs.pop("noaa-gefs"))
    with pytest.raises(tool.Refused, match="must pin exactly climatology and noaa-gefs"):
        tool.load_inventory(root)

    root = world()
    edit_inventory(root, lambda refs: refs["noaa-gefs"].pop("leads_scored"))
    with pytest.raises(tool.Refused, match="lacks"):
        tool.load_inventory(root)

    root = world()
    (root / tool.INVENTORY).unlink()
    with pytest.raises(tool.Refused, match="cannot read reference inventory"):
        tool.load_inventory(root)
    with pytest.raises(tool.Refused, match="has no pinned references"):
        tool.period_record(base_world["inventory"], "2024-04")


# --------------------------------------------------------------------------- refusals: the candidate


def test_candidate_that_is_a_copy_of_a_reference_is_refused(
    base_world: dict, locked: None, tmp_path: Path
) -> None:
    root = base_world["root"]
    # the GEFS reference submitted as it is
    reasons = tool.evaluate(root / "data/refs/gefs.zarr", root / "data", PERIOD, tmp_path / "a", root).reasons
    assert any("is a reference baseline, not a candidate" in r for r in reasons)
    assert any("byte-for-byte a reference forecast (noaa-gefs" in r for r in reasons)

    # the same values under a new name: the content hash ignores attributes, so it is still caught
    renamed = open_zarr(root / "data/refs/climatology.zarr").load()
    renamed.attrs.update(model_id="brand-new-model", model_version="9")
    write_zarr(renamed, tmp_path / "renamed.zarr")
    reasons = tool.evaluate(tmp_path / "renamed.zarr", root / "data", PERIOD, tmp_path / "b", root).reasons
    assert any("byte-for-byte a reference forecast (climatology" in r for r in reasons)
    assert not any("is a reference baseline, not a candidate" in r for r in reasons)

    # and at the level of the cards: a good card whose forecast hash is a pinned one
    card, other = good_cards(base_world)
    pinned = base_world["inventory"]["periods"][PERIOD]["references"]["noaa-gefs"]["forecast_sha256"]
    card["forecast_sha256"] = other["forecast_sha256"] = pinned
    assert any("byte-for-byte a reference forecast" in r for r in judge(base_world, [card, other]))
    card, other = good_cards(base_world)
    card["model_id"] = other["model_id"] = "climatology"
    assert "model id 'climatology' is a reference baseline, not a candidate" in judge(
        base_world, [card, other]
    )


def test_weaker_candidate_is_not_promoted_and_refused_inputs_stop_early(
    base_world: dict, locked: None, tmp_path: Path
) -> None:
    root = base_world["root"]
    outcome = tool.evaluate(
        root / "data/candidates/weak.zarr", root / "data", PERIOD, tmp_path / "weak", root
    )
    assert "gate not passed against climatology" in outcome.reasons
    assert "gate not passed against noaa-gefs" in outcome.reasons
    verdict = json.loads((outcome.directory / "verdict.json").read_text())
    assert verdict["verdict"] == "NOT PASSED" and verdict["reasons"] == outcome.reasons
    assert len(list(outcome.directory.glob("candidate_vs_*.json"))) == 2  # both evaluations ran and were kept

    good = root / "data/candidates/good.zarr"
    with pytest.raises(tool.Refused, match="has no pinned references"):
        tool.evaluate(good, root / "data", "2024-04", tmp_path / "x", root)
    with pytest.raises(tool.Refused, match="candidate: no such forecast store"):
        tool.evaluate(root / "data/candidates/none.zarr", root / "data", PERIOD, tmp_path / "x", root)
    with pytest.raises(tool.Refused, match="truth not found under the data directory"):
        tool.evaluate(good, tmp_path, PERIOD, tmp_path / "x", root)
    assert not (tmp_path / "x").exists()  # a refusal leaves nothing behind
    # a candidate wrfcomp-eval itself refuses (a NaN at a scored point) is not promoted
    broken = open_zarr(good).load()
    broken["t2m"].values[3, 1, 2, 0] = np.nan
    write_zarr(broken, tmp_path / "broken.zarr")
    outcome = tool.evaluate(tmp_path / "broken.zarr", root / "data", PERIOD, tmp_path / "y", root)
    assert outcome.reasons == [
        "wrfcomp-eval refused the candidate against climatology (exit code 3)",
        "wrfcomp-eval refused the candidate against noaa-gefs (exit code 3)",
    ]
    assert outcome.directory.name == tool._result_name(
        PERIOD, (outcome.directory / "verdict.json").read_text()
    )
    assert (
        json.loads((outcome.directory / "verdict.json").read_text())["candidate"]["forecast_sha256"] is None
    )
    assert [p.name for p in outcome.directory.iterdir()] == ["verdict.json"]


# --------------------------------------------------------------------------- one tree, private output


def test_candidate_metadata_never_becomes_a_path(base_world: dict, locked: None, tmp_path: Path) -> None:
    # Model ids written to climb out of the output directory or to collide with the program's own
    # file names. They are recorded as data; none changes where anything is written.
    root = base_world["root"]
    out = tmp_path / "published"
    hostile_ids = ("../../escaped", "/tmp/wrfcomp-escaped", "candidate_vs_climatology.json", "a/b\\c")
    for index, model_id in enumerate(hostile_ids):
        hostile = open_zarr(root / "data/candidates/good.zarr").load()
        hostile.attrs["model_id"] = model_id
        hostile["t2m"].values[0, 0, 0, 0] += np.float32(0.001 * (index + 1))  # a distinct forecast each time
        write_zarr(hostile, tmp_path / f"hostile{index}.zarr")
        outcome = tool.evaluate(tmp_path / f"hostile{index}.zarr", root / "data", PERIOD, out, root)
        assert outcome.reasons == [], model_id
        assert outcome.directory.parent == out.resolve()
        assert sorted(p.name for p in outcome.directory.iterdir()) == [
            "candidate_vs_climatology.json",
            "candidate_vs_noaa-gefs.json",
            "verdict.json",
        ]
        verdict = json.loads((outcome.directory / "verdict.json").read_text())
        assert verdict["candidate"]["model_id"] == model_id  # kept, as data
        card = json.loads((outcome.directory / "candidate_vs_climatology.json").read_text())
        assert card["model_id"] == model_id
    assert not (tmp_path / "escaped.json").exists() and not Path("/tmp/wrfcomp-escaped").exists()
    everything = (p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*.json"))
    written = sorted(path for path in everything if ".zarr/" not in path)  # not the stores' own metadata
    assert len(written) == 12
    assert all(path.startswith("published/promotion_") for path in written), written
    assert len({p.name for p in out.iterdir()}) == 4 and not list(out.glob(".promotion-run-*"))


def test_runs_are_staged_privately_and_published_whole(
    base_world: dict, locked: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root, out = base_world["root"], tmp_path / "published"
    good = root / "data/candidates/good.zarr"
    seen: list[tuple[Path, int, list[str]]] = []

    def watching_eval(eval_root: Path, args: list[str]) -> int:
        target = Path(args[args.index("--out") + 1])
        seen.append((target, target.parent.stat().st_mode & 0o777, sorted(p.name for p in out.iterdir())))
        return in_process_eval(eval_root, args)

    monkeypatch.setattr(tool, "_run_eval", watching_eval)
    first = tool.evaluate(good, root / "data", PERIOD, out, root)
    assert [target.name for target, _, _ in seen] == list(tool.STAGED_NAMES.values())
    for target, mode, published in seen:
        assert target.parent.name.startswith(".promotion-run-") and target.parent.parent == out.resolve()
        if os.name != "nt":
            assert mode == 0o700  # private while the run is in progress
        assert published == [target.parent.name]  # nothing is published before the run is complete
    assert [p.name for p in out.iterdir()] == [first.directory.name]  # staging is gone, one result

    # the same candidate again: each run has its own staging directory, the identical result is kept
    before = {p.name: p.read_bytes() for p in first.directory.iterdir()}
    again = tool.evaluate(good, root / "data", PERIOD, out, root)
    assert again.directory == first.directory
    assert {p.name: p.read_bytes() for p in again.directory.iterdir()} == before
    assert len({target.parent for target, _, _ in seen}) == 2
    assert [p.name for p in out.iterdir()] == [first.directory.name]

    # an evaluator that crashes half way leaves no staging directory and publishes nothing new
    def crashing_eval(eval_root: Path, args: list[str]) -> int:
        raise RuntimeError("evaluator crashed")

    monkeypatch.setattr(tool, "_run_eval", crashing_eval)
    with pytest.raises(RuntimeError, match="evaluator crashed"):
        tool.evaluate(good, root / "data", PERIOD, out, root)
    assert [p.name for p in out.iterdir()] == [first.directory.name]
    if os.name == "nt":
        pytest.skip(
            "atomic staging and publication verified; POSIX 0o700 staging-directory privacy "
            "cannot be asserted on Windows because tempfile.mkdtemp ignores mode bits"
        )


def test_every_input_must_come_from_the_one_checked_tree(
    base_world: dict, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = base_world["root"]
    good = root / "data/candidates/good.zarr"
    monkeypatch.setattr(tool, "_run_eval", in_process_eval)

    # 1. The synthetic world is not a locked checkout: with the real lock check it is refused.
    with pytest.raises(tool.Refused, match="is not the locked harness"):
        tool.evaluate(good, root / "data", PERIOD, tmp_path / "a", root)
    monkeypatch.setattr(tool, "_harness_lock_problems", lambda r: ["changed: src/wrfcomp/harness/metrics.py"])
    with pytest.raises(tool.Refused, match="not the locked harness: changed: src/wrfcomp/harness/metrics.py"):
        tool.evaluate(good, root / "data", PERIOD, tmp_path / "a", root)

    # 2. A tree that passes its own lock, but is not where this process's harness was imported from:
    #    the inventory and config would come from one place and the scoring code from another.
    monkeypatch.setattr(tool, "_harness_lock_problems", lambda r: [])
    with pytest.raises(tool.Refused, match="the harness in use is not the one under") as refusal:
        tool.evaluate(good, root / "data", PERIOD, tmp_path / "a", root)
    assert "wrfcomp would be loaded from" in str(refusal.value)
    assert not (tmp_path / "a").exists()


def test_origin_and_lock_checks_look_at_the_given_root(base_world: dict, tmp_path: Path) -> None:
    # this checkout's harness modules are files of this checkout
    assert tool._modules_outside_root(ROOT) == []
    outside = tool._modules_outside_root(base_world["root"])
    assert len(outside) == 1 and outside[0].startswith("wrfcomp would be loaded from")
    expected = {"wrfcomp", "wrfcomp.harness.metrics", "wrfcomp.harness.evaluate", "wrfcomp.harness.schemas"}
    assert expected <= set(tool.HARNESS_MODULES)
    # the lock check runs that root's own script on that root
    assert tool._harness_lock_problems(ROOT) == []
    assert tool._harness_lock_problems(tmp_path) == [f"{tmp_path} has no scripts/harness_lock.py"]


def test_evaluations_run_through_uv_locked_on_the_checked_tree(tmp_path: Path) -> None:
    command = tool._eval_command(ROOT, ["--help"])
    if os.name == "nt":
        assert Path(command[0]).name.lower() == "uv.exe"
    else:
        assert Path(command[0]).name == "uv"
    script = str(ROOT / "scripts" / "check_promotion.py")
    assert command[1:] == [
        "run",
        "--locked",
        "--project",
        str(ROOT),
        "python",
        "-P",  # never look for modules next to the script
        script,
        "score-one",
        str(ROOT),
        "--help",
    ]
    assert "wrfcomp-eval" not in command  # the console command's target is set outside the lock
    # a real run of that command: a synthetic forecast scored by the locked environment of this tree
    world = build_world(tmp_path / "repo")

    def args_for(out: Path) -> list[str]:
        args = ["--forecast", str(world / "data/candidates/good.zarr")]
        args += [
            "--truth",
            str(world / "data/truth/truth.nc"),
            "--thresholds",
            str(world / "data/thresholds/tail.nc"),
        ]
        args += [
            "--baseline",
            str(world / "data/refs/gefs.zarr"),
            "--baseline-id",
            "noaa-gefs",
            "--out",
            str(out),
        ]
        args += [
            "--model-id",
            "synth",
            "--model-version",
            "1",
            "--config",
            str(world / "configs/splits.yaml"),
        ]
        return args

    assert tool._run_eval(ROOT, args_for(tmp_path / "locked.json")) == 0
    assert in_process_eval(ROOT, args_for(tmp_path / "in-process.json")) == 0
    # the in-process stand-in used by the other tests writes the same bytes as the real command
    assert (tmp_path / "locked.json").read_bytes() == (tmp_path / "in-process.json").read_bytes()
    assert tool._run_eval(ROOT, ["--forecast", "nope.zarr"]) != 0


def test_scoring_process_refuses_a_harness_that_is_not_under_the_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capfd: pytest.CaptureFixture
) -> None:
    # Regression: the parent used to check its own imports and then start the installed
    # `wrfcomp-eval` command. Here the child process finds another `wrfcomp` first on its path: a
    # stand-in whose evaluator would write a card. The child must refuse without running any of it.
    shadow = tmp_path / "shadow"
    (shadow / "wrfcomp" / "harness").mkdir(parents=True)
    imported, ran = tmp_path / "stand-in-imported", tmp_path / "stand-in-ran"
    for name in ("align", "config", "hashing", "metrics", "schemas", "truth"):
        (shadow / "wrfcomp" / "harness" / f"{name}.py").write_text("")
    (shadow / "wrfcomp" / "__init__.py").write_text(
        f'from pathlib import Path\n\nPath({str(imported)!r}).write_text("imported")\n__version__ = "0.1.0"\n'
    )
    (shadow / "wrfcomp" / "harness" / "__init__.py").write_text("")
    (shadow / "wrfcomp" / "harness" / "evaluate.py").write_text(
        f"from pathlib import Path\n\n\ndef app(args=None, standalone_mode=True):\n"
        f"    Path({str(ran)!r}).write_text('ran')\n    return 0\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(shadow))
    assert tool._run_eval(ROOT, ["--forecast", "x.zarr"]) == tool.EXIT_CHILD_REFUSED
    assert not ran.exists()
    assert not imported.exists()  # located with find_spec, refused, never imported
    err = capfd.readouterr().err
    assert "the scoring process is not running the harness under" in err
    assert f"wrfcomp would be loaded from {shadow}" in err
    monkeypatch.delenv("PYTHONPATH")

    # the child checks against the root it is given, and it insists on -P for itself
    assert tool.score_one(tmp_path, ["--help"]) == tool.EXIT_CHILD_REFUSED
    assert tool.main(["score-one", str(tmp_path), "--help"]) == tool.EXIT_CHILD_REFUSED
    err = capfd.readouterr().err
    if sys.flags.safe_path:  # pytest itself was started with -P: the origin check answers instead
        assert "wrfcomp would be loaded from" in err
    else:
        assert "started without Python's -P option" in err


SHADOW_SCORES = """from pathlib import Path

Path({marker!r}).write_text("the planted scores package was imported")
"""


def checkout_with_planted_scores(tmp_path: Path) -> tuple[Path, Path]:
    """A copy of the source tree and the promotion script, plus `scripts/scores/` planted next to it.

    This is the pull request the review described: nothing locked is touched, but a package named
    like the scoring library sits in the script's own directory.
    """
    root = tmp_path / "checkout"
    shutil.copytree(ROOT / "src", root / "src", ignore=shutil.ignore_patterns("__pycache__"))
    (root / "scripts").mkdir()
    shutil.copy(ROOT / "scripts" / "check_promotion.py", root / "scripts" / "check_promotion.py")
    marker = tmp_path / "planted-scores-imported"
    planted = root / "scripts" / "scores"
    (planted / "probability").mkdir(parents=True)
    (planted / "stats" / "statistical_tests").mkdir(parents=True)
    for package in (
        planted,
        planted / "probability",
        planted / "stats",
        planted / "stats" / "statistical_tests",
    ):
        (package / "__init__.py").write_text(SHADOW_SCORES.format(marker=str(marker)))
    return root, marker


def run_child(root: Path, *flags: str, env: dict | None = None) -> subprocess.CompletedProcess:
    """The scoring child, started the way `_eval_command` starts it but with this interpreter."""
    script = str(root / "scripts" / "check_promotion.py")
    inherited = {k: v for k, v in os.environ.items() if k != "PYTHONSAFEPATH"}
    environment = {**inherited, "PYTHONPATH": str(root / "src"), **(env or {})}
    command = [sys.executable, *flags, script, "score-one", str(root), "--help"]
    return subprocess.run(command, capture_output=True, text=True, env=environment, check=False)


def test_scores_planted_next_to_the_script_is_never_imported(tmp_path: Path) -> None:
    root, marker = checkout_with_planted_scores(tmp_path)

    # Without -P Python would put scripts/ first on its path: the script refuses before any import.
    refused = run_child(root)
    assert refused.returncode != 0 and refused.returncode != tool.EXIT_CHILD_REFUSED
    assert "REFUSED: run this script with Python's -P option" in refused.stderr
    assert not marker.exists()

    # With -P the planted package is not on the path: the child scores with the real library.
    safe = run_child(root, "-P")
    assert safe.returncode == 0, safe.stderr
    assert "Score a forecast against truth" in safe.stdout  # the harness entry point ran
    assert not marker.exists()
    # the interpreter's own switch for the same thing is accepted too
    assert run_child(root, env={"PYTHONSAFEPATH": "1"}).returncode == 0
    assert not marker.exists()

    # Belt and braces: even on a safe path, a scoring package found inside the checkout is refused.
    # (-P keeps the script's directory off the path; PYTHONPATH can still put it back.)
    forced = run_child(root, "-P", env={"PYTHONPATH": f"{root / 'scripts'}{os.pathsep}{root / 'src'}"})
    assert forced.returncode == tool.EXIT_CHILD_REFUSED
    assert f"scores would be loaded from {root / 'scripts' / 'scores'}" in forced.stderr
    assert not marker.exists()  # located with find_spec, refused, never imported


def test_scoring_packages_must_come_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    assert tool._modules_outside_root(ROOT) == []
    assert set(tool.SCORING_PACKAGES) == {"scores", "numpy", "xarray", "scipy", "pandas"}
    real = tool._spec_origin
    for name in tool.SCORING_PACKAGES:
        planted = ROOT / "scripts" / name / "__init__.py"
        monkeypatch.setattr(
            tool, "_spec_origin", lambda n, name=name, planted=planted: planted if n == name else real(n)
        )
        problems = tool._modules_outside_root(ROOT)
        assert problems == [
            f"{name} would be loaded from {planted}, not from the environment {Path(sys.prefix).resolve()}"
        ]
    monkeypatch.setattr(tool, "_spec_origin", lambda n: None if n == "scores" else real(n))
    assert tool._modules_outside_root(ROOT)[0].startswith("scores would be loaded from None")
    monkeypatch.setattr(tool, "_spec_origin", real)
    assert tool._spec_origin("sys") is None and tool._spec_origin("no_such_package_anywhere") is None
    assert tool._spec_origin("wrfcomp") == (ROOT / "src" / "wrfcomp" / "__init__.py").resolve()


# --------------------------------------------------------------------------- the real repository


MONTHS = [f"2024-{month:02d}" for month in range(1, 13)]
QUARTERS = [f"2024-Q{quarter}" for quarter in range(1, 5)]


def test_committed_inventory_pins_the_committed_reference_scorecards() -> None:
    inventory = tool.load_inventory(ROOT)
    # every month of the 2024 validation split (diagnostic) and its four quarters (promotion)
    assert sorted(inventory["periods"]) == sorted(MONTHS + QUARTERS)
    assert inventory["harness_version"] == {"scores": "2.7.0", "wrfcomp": "0.1.0"}
    periods = {
        label: {
            "truth": "truth/isd_conus_2024.nc",
            "thresholds": "thresholds/tail_p95.nc",
            "references": {
                "climatology": {
                    "scorecard": f"experiments/climatology_{label}.json",
                    "store": f"gate1/{label}_h00-12/climatology.zarr",
                },
                "noaa-gefs": {
                    "scorecard": f"experiments/gefs_{label}.json",
                    "store": f"gate1/{label}_h00-12/gefs.zarr",
                },
            },
            "others": [f"experiments/persistence_{label}.json"],
        }
        for label in MONTHS + QUARTERS
    }
    assert tool.build_inventory(ROOT, periods) == json.loads((ROOT / tool.INVENTORY).read_text())
    truth = {r["truth_sha256"] for e in inventory["periods"].values() for r in e["references"].values()}
    thresholds = {
        r["thresholds_sha256"] for e in inventory["periods"].values() for r in e["references"].values()
    }
    assert len(truth) == 1 and len(thresholds) == 1  # one truth file and one thresholds file for the year
    init_times = {}
    for label, entry in inventory["periods"].items():
        references = entry["references"]
        assert references["climatology"]["leads_scored"] == [1, 3, 6, 12, 24, 48]  # all v1 leads
        assert references["noaa-gefs"]["leads_scored"] == [3, 6, 12, 24, 48]  # 3 h and longer
        for record in references.values():
            assert (record["n_stations"], record["init_hours_scored"]) == (500, [0, 12])
            first_month = label[-2:] if label in MONTHS else f"{3 * int(label[-1]) - 2:02d}"
            assert record["period_start"] == f"2024-{first_month}-01T00:00:00"
            assert record["n_init_times"] == references["climatology"]["n_init_times"]
        init_times[label] = references["climatology"]["n_init_times"]
        assert len(entry["other_reference_forecasts"]) == 1, label
    # 00Z and 12Z of every day; December stops at the last init inside the validation split
    days = {"2024-02": 29, "2024-04": 30, "2024-06": 30, "2024-09": 30, "2024-11": 30, "2024-12": 17}
    assert {m: init_times[m] for m in MONTHS} == {label: 2 * days.get(label, 31) for label in MONTHS}
    assert inventory["periods"]["2024-12"]["references"]["noaa-gefs"]["period_end"] == "2024-12-17T12:00:00"
    # a quarter is its three months as one period; Q4 ends where December does
    for number, label in enumerate(QUARTERS, start=1):
        months = MONTHS[3 * number - 3 : 3 * number]
        assert init_times[label] == sum(init_times[m] for m in months)
        for baseline, record in inventory["periods"][label]["references"].items():
            last = inventory["periods"][months[-1]]["references"][baseline]
            assert record["period_end"] == last["period_end"]
            assert (
                record["store"]
                == f"gate1/{label}_h00-12/{'gefs' if baseline == 'noaa-gefs' else baseline}.zarr"
            )
            for key in ("truth_sha256", "thresholds_sha256", "leads_scored", "n_stations"):
                assert record[key] == last[key]
            quarter_card = json.loads((ROOT / record["scorecard"]).read_text())
            for variable, per_lead in quarter_card["metrics"].items():
                for lead, entry in per_lead.items():
                    monthly = [
                        json.loads(
                            (ROOT / inventory["periods"][m]["references"][baseline]["scorecard"]).read_text()
                        )
                        for m in months
                    ]
                    assert entry["n_pairs"] == sum(c["metrics"][variable][lead]["n_pairs"] for c in monthly)
    assert inventory["periods"]["2024-Q4"]["references"]["noaa-gefs"]["period_end"] == "2024-12-17T12:00:00"
    assert len(tool.reference_forecast_hashes(inventory)) == 48  # no forecast is shared between periods
    # the inventory pins exactly the reference scorecards the manifest locks
    pinned = {r["scorecard"] for e in inventory["periods"].values() for r in e["references"].values()}
    pinned |= {
        path for e in inventory["periods"].values() for path in e["other_reference_forecasts"].values()
    }
    lock = json.loads((ROOT / "HARNESS_LOCK.json").read_text())["entries"]
    assert pinned == {path for path in lock if path.startswith("experiments/") and path != tool.INVENTORY}


def test_real_reference_scorecards_are_never_a_promotion(capsys: pytest.CaptureFixture) -> None:
    # GEFS and persistence were both scored against climatology in January: not a candidate pair.
    cards = [str(ROOT / "experiments" / f"{name}_2024-01.json") for name in ("gefs", "persistence")]
    assert tool.main(["verify", *cards, "--period", "2024-01"]) == 1
    out = capsys.readouterr().out
    assert "INCONSISTENT" in out and "two scorecards against the same baseline: climatology" in out
    assert "PROMOTED" not in out
    # January scorecards do not pass for another pinned month
    assert tool.main(["verify", *cards, "--period", "2024-02"]) == 1
    out = capsys.readouterr().out
    assert "INCONSISTENT" in out and "PROMOTED" not in out
    # a period that is not pinned
    assert tool.main(["verify", *cards, "--period", "2023-12"]) == 2
    assert "REFUSED" in capsys.readouterr().out


# =============================================================================== quarters, 3 of 4

QUARTER_STARTS = {"2024-Q1": "2024-01-01T00:00", "2024-Q2": "2024-04-01T00:00"}
QUARTER_STARTS |= {"2024-Q3": "2024-07-01T00:00", "2024-Q4": "2024-10-01T00:00"}
PARTS = ("a", "b", "c")  # a quarter's candidate in three stores, standing in for its three months


def build_quarter_world(root: Path) -> Path:
    """Like `build_world`, with the four quarters of 2024 pinned, each with its own truth and thresholds.

    Per quarter: a good candidate split into three stores of ten init times, the same candidate under
    another model id, and a weak candidate in one store.
    """
    data = root / "data"
    for sub in ("truth", "thresholds", "refs", "candidates"):
        (data / sub).mkdir(parents=True)
    write_config(root, list(station_ids(N_STATIONS)))
    (root / "experiments").mkdir()
    periods = {}
    for number, (label, start) in enumerate(QUARTER_STARTS.items()):
        syn = make_synthetic(seed=500 + number, start=start)
        rng = np.random.default_rng(900 + number)
        write_netcdf(syn.truth, data / f"truth/{label}.nc")
        write_netcdf(syn.thresholds, data / f"thresholds/{label}.nc")
        gefs = syn.baseline.isel(lead_time=[1, 2]).copy(deep=True)
        gefs.attrs["model_id"] = "noaa-gefs"
        climatology = syn.baseline.copy(deep=True)
        for name in climatology.data_vars:
            noise = rng.normal(0, float(syn.truth[name].std()) * 0.5, climatology[name].shape).astype(
                "float32"
            )
            climatology[name] = (
                climatology[name].dims,
                climatology[name].values + noise,
                climatology[name].attrs,
            )
        climatology.attrs["model_id"] = "climatology"
        weak = climatology.copy(deep=True)
        weak.attrs["model_id"] = "synth"  # the same candidate, having a bad quarter
        for name in weak.data_vars:
            weak[name] = (weak[name].dims, weak[name].values + np.float32(0.25), weak[name].attrs)
        other = syn.candidate.copy(deep=True)
        other.attrs["model_id"] = "someone-else"

        write_zarr(climatology, data / f"refs/climatology_{label}.zarr")
        write_zarr(gefs, data / f"refs/gefs_{label}.zarr")
        write_zarr(weak, data / f"candidates/weak_{label}.zarr")
        write_zarr(other, data / f"candidates/other_{label}.zarr")
        write_zarr(syn.candidate, data / f"candidates/whole_{label}.zarr", {"init_time": 7, "station_id": 3})
        for index, part in enumerate(PARTS):
            piece = syn.candidate.isel(init_time=slice(10 * index, 10 * index + 10))
            write_zarr(piece, data / f"candidates/good_{label}_{part}.zarr")

        files = {"truth": f"truth/{label}.nc", "thresholds": f"thresholds/{label}.nc"}
        clim_card, gefs_card = f"experiments/climatology_{label}.json", f"experiments/gefs_{label}.json"
        _eval(root, data / f"refs/climatology_{label}.zarr", root / clim_card, "climatology", None, **files)
        reference = (data / f"refs/climatology_{label}.zarr", "climatology")
        _eval(root, data / f"refs/gefs_{label}.zarr", root / gefs_card, "noaa-gefs", reference, **files)
        periods[label] = {
            **files,
            "references": {
                "climatology": {"scorecard": clim_card, "store": f"refs/climatology_{label}.zarr"},
                "noaa-gefs": {"scorecard": gefs_card, "store": f"refs/gefs_{label}.zarr"},
            },
        }
    inventory = tool.build_inventory(root, periods)
    (root / tool.INVENTORY).write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n")
    return root


def good_parts(root: Path, label: str, order: str = "abc") -> list[Path]:
    return [root / f"data/candidates/good_{label}_{part}.zarr" for part in order]


LABELS = list(QUARTER_STARTS)


def year_stores(root: Path, kinds: str) -> list[Path]:
    """The candidate's stores for the year. `kinds` has one letter per quarter: g good (three
    stores), w weak (one store), - none."""
    stores: list[Path] = []
    for label, kind in zip(LABELS, kinds, strict=True):
        if kind == "g":
            stores += good_parts(root, label)
        elif kind == "w":
            stores.append(root / f"data/candidates/weak_{label}.zarr")
    return stores


@pytest.fixture(scope="module")
def quarter_world(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Built once, never modified. `good` is the published Q2 result of the good candidate, given as
    three stores that are deliberately not in time order."""
    root = build_quarter_world(tmp_path_factory.mktemp("quarters") / "repo")
    data, out = root / "data", root.parent / "results"
    with pytest.MonkeyPatch.context() as patch:
        trust(patch)
        outcome = tool.evaluate(good_parts(root, "2024-Q2", "cab"), data, "2024-Q2", out, root)
        assert outcome.reasons == []
    return {"root": root, "data": data, "good": outcome.directory, "cfg": tool._load_locked_config(root)}


def decide(world: dict, kinds: str | list[Path], out: Path):
    stores = year_stores(world["root"], kinds) if isinstance(kinds, str) else kinds
    return tool.promote(stores, world["data"], 2024, out, world["root"])


def passed(decision) -> list[str]:
    return [q.period for q in decision.quarters if q.passed]


def shifted(source: Path, dest: Path, first: str) -> Path:
    """A copy of a store with its init times moved to start at `first` (same 12 h step)."""
    ds = open_zarr(source).load()
    times = np.datetime64(first, "ns") + np.arange(ds.sizes["init_time"]) * np.timedelta64(12, "h")
    return write_zarr(ds.assign_coords(init_time=times), dest)


def test_quarter_given_as_monthly_stores_is_joined_and_hashed_reproducibly(
    quarter_world: dict, locked: None, tmp_path: Path
) -> None:
    from wrfcomp.harness.hashing import content_sha256

    root, data, label = quarter_world["root"], quarter_world["data"], "2024-Q2"
    published = quarter_world["good"]  # given as c, a, b
    assert sorted(p.name for p in published.iterdir()) == [
        "candidate_vs_climatology.json",
        "candidate_vs_noaa-gefs.json",
        "verdict.json",
    ]  # the joined store is not kept
    verdict = json.loads((published / "verdict.json").read_text())
    whole = open_zarr(data / f"candidates/whole_{label}.zarr")
    digest = content_sha256(whole)
    # the hash of the joined forecast is the hash of the same forecast stored whole, with other chunks
    assert verdict["candidate"] == {"model_id": "synth", "model_version": "1", "forecast_sha256": digest}
    assert published.name == tool._result_name(label, (published / "verdict.json").read_text())
    assert verdict["verdict"] == "PASSED" and verdict["promotion_rule_version"] == 2
    inputs = verdict["forecast_inputs"]
    assert [i["n_init_times"] for i in inputs] == [10, 10, 10]
    assert [i["init_first"] for i in inputs] == sorted(i["init_first"] for i in inputs)  # time order
    assert [i["content_sha256"] for i in inputs] == [
        content_sha256(open_zarr(path)) for path in good_parts(root, label, "abc")
    ]
    assert (
        inputs[0]["init_first"] == "2024-04-01T00:00:00" and inputs[-1]["init_last"] == "2024-04-15T12:00:00"
    )

    # the order the stores are given in changes nothing: the same hash, the same bytes, the same name
    again = tool.evaluate(good_parts(root, label, "bca"), data, label, tmp_path / "again", root)
    assert again.directory.name == published.name
    for name in (*tool.STAGED_NAMES.values(), "verdict.json"):
        assert (again.directory / name).read_bytes() == (published / name).read_bytes()
    # nothing but the published directory is left in the output directory
    assert [p.name for p in (tmp_path / "again").iterdir()] == [published.name]


def test_candidate_stores_that_do_not_form_one_forecast_are_refused(
    quarter_world: dict, locked: None, tmp_path: Path
) -> None:
    root, data, label = quarter_world["root"], quarter_world["data"], "2024-Q1"
    a, b, c = good_parts(root, label)
    out = tmp_path / "out"

    def refused(forecast: list[Path], match: str, period: str = label) -> None:
        with pytest.raises(tool.Refused, match=match):
            tool.evaluate(forecast, data, period, out, root)

    refused([a, b, a], "listed twice")
    assert not out.exists()  # refused before anything is created
    overlap = open_zarr(b).load()
    write_zarr(overlap.assign_coords(init_time=open_zarr(a)["init_time"].values), tmp_path / "overlap.zarr")
    refused([a, tmp_path / "overlap.zarr", c], "an init time in common")
    # b moved half a step: no init time in common with a, but in between a's
    shifted(b, tmp_path / "between.zarr", "2024-01-01T06:00")
    refused([a, tmp_path / "between.zarr"], "interleave in time")
    version = open_zarr(b).load()
    version.attrs["model_version"] = "2"
    write_zarr(version, tmp_path / "version.zarr")
    refused([a, tmp_path / "version.zarr", c], "not one model and version")
    refused([a, data / f"candidates/other_{label}.zarr"], "not one model and version")
    fewer = open_zarr(b).load().isel(station_id=slice(0, N_STATIONS - 1))
    write_zarr(fewer, tmp_path / "fewer.zarr")
    refused([a, tmp_path / "fewer.zarr", c], "fewer.zarr: .* differs from")
    dropped = open_zarr(b).load().drop_vars("u10")
    write_zarr(dropped, tmp_path / "dropped.zarr")
    refused([a, tmp_path / "dropped.zarr", c], "other variables or coordinates")
    refused([a, tmp_path / "nowhere.zarr"], "no such forecast store")
    assert not list(out.glob("promotion_*"))

    # two of the three stores: scored, but it is not the pinned quarter, so it does not pass
    outcome = tool.evaluate([a, b], data, label, tmp_path / "short", root)
    assert any("n_init_times against climatology differs" in line for line in outcome.reasons)
    assert not list((tmp_path / "short").glob(".promotion-run-*"))


def test_several_stores_are_for_quarters_only(base_world: dict, locked: None, tmp_path: Path) -> None:
    good = base_world["root"] / "data/candidates/good.zarr"
    weak = base_world["root"] / "data/candidates/weak.zarr"
    with pytest.raises(tool.Refused, match="for a quarter only"):
        tool.evaluate([good, weak], base_world["root"] / "data", PERIOD, tmp_path / "x", base_world["root"])
    with pytest.raises(tool.Refused, match="no candidate forecast"):
        tool.evaluate([], base_world["root"] / "data", PERIOD, tmp_path / "x", base_world["root"])
    assert not (tmp_path / "x").exists()


# ------------------------------------------------------------------ review items: inputs and identity


def test_candidate_is_checked_against_the_sealed_split_before_it_is_hashed(
    quarter_world: dict, locked: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A store that reaches the sealed split, or is not a forecast, is refused on its coordinates
    and attributes: no content hash is taken, so none of its values is read."""
    import wrfcomp.harness.hashing as hashing

    root, data = quarter_world["root"], quarter_world["data"]
    a, b, c = good_parts(root, "2024-Q4")
    sealed_start = quarter_world["cfg"].splits.sealed_test.start
    late = shifted(c, tmp_path / "late.zarr", "2024-12-28T00:00")  # ten init times: the last is in 2025
    assert open_zarr(late)["init_time"].values.max() >= sealed_start
    # init times all in 2024, but the longest lead carries the valid times past the start of 2025
    edge = shifted(c, tmp_path / "edge.zarr", "2024-12-26T12:00")
    assert open_zarr(edge)["init_time"].values.max() < sealed_start
    wrong_units = open_zarr(b).load()
    wrong_units["t2m"].attrs["units"] = "degF"
    write_zarr(wrong_units, tmp_path / "degf.zarr")

    def never(ds):
        raise AssertionError("a candidate store was hashed before it was checked")

    monkeypatch.setattr(hashing, "content_sha256", never)
    out = tmp_path / "out"
    for stores, match in (
        ([a, b, late], "sealed split"),
        ([edge], "sealed split"),
        ([a, tmp_path / "degf.zarr", c], "has units 'degF'"),
    ):
        with pytest.raises(tool.Refused, match=match):
            tool.evaluate(stores, data, "2024-Q4", out, root)
        with pytest.raises(tool.Refused, match=match):
            tool.promote(stores, data, 2024, out, root)
    assert not out.exists()


def test_units_survive_the_join_and_must_agree_between_stores(
    quarter_world: dict, locked: None, tmp_path: Path
) -> None:
    root, data, cfg = quarter_world["root"], quarter_world["data"], quarter_world["cfg"]
    a, b, c = good_parts(root, "2024-Q1")
    (tmp_path / "staging").mkdir()
    joined_path, _, _, _ = tool._combine_forecasts([c, a, b], tmp_path / "staging", cfg)
    with open_zarr(joined_path) as joined, open_zarr(a) as first:
        assert joined.sizes["init_time"] == 30 and set(joined.data_vars) == set(first.data_vars)
        for name in first.data_vars:
            assert joined[name].attrs == first[name].attrs and joined[name].attrs["units"]
        assert joined.attrs == first.attrs

    # a later store that declares another unit: an accepted spelling of the same unit still differs
    # from the first store, and no unit at all would otherwise vanish in the join
    kelvin = open_zarr(b).load()
    kelvin["t2m"].attrs["units"] = "kelvin"
    write_zarr(kelvin, tmp_path / "kelvin.zarr")
    bare = open_zarr(c).load()
    del bare["msl"].attrs["units"]
    write_zarr(bare, tmp_path / "bare.zarr")
    for stores, variable in (
        ([a, tmp_path / "kelvin.zarr", c], "t2m"),
        ([a, b, tmp_path / "bare.zarr"], "msl"),
    ):
        with pytest.raises(tool.Refused, match=f"{variable} has other attributes \\(units\\)"):
            tool.evaluate(stores, data, "2024-Q1", tmp_path / "out", root)
    assert not list((tmp_path / "out").glob("promotion_*"))


def test_a_repeated_evaluation_is_the_same_result_whatever_form_the_input_took(
    quarter_world: dict, locked: None, tmp_path: Path
) -> None:
    """One output directory, the same quarter scored from one store and from three, twice each:
    every run is accepted, and each form has one directory with a name the program derives."""
    root, data, label = quarter_world["root"], quarter_world["data"], "2024-Q2"
    out = tmp_path / "out"
    whole = data / f"candidates/whole_{label}.zarr"
    runs = [
        tool.evaluate(whole, data, label, out, root),
        tool.evaluate(good_parts(root, label), data, label, out, root),
        tool.evaluate(good_parts(root, label, "cba"), data, label, out, root),
        tool.evaluate(whole, data, label, out, root),
    ]
    assert all(run.reasons == [] for run in runs)
    one, three = runs[0].directory, runs[1].directory
    assert runs[3].directory == one and runs[2].directory == three and one != three
    assert sorted(p.name for p in out.iterdir()) == sorted([one.name, three.name])  # nothing suffixed
    for directory in (one, three):
        assert directory.name == tool._result_name(label, (directory / "verdict.json").read_text())
    # the same forecast: the same scorecards byte for byte; only the list of input stores differs
    for name in tool.STAGED_NAMES.values():
        assert (one / name).read_bytes() == (three / name).read_bytes()
    verdicts = [json.loads((d / "verdict.json").read_text()) for d in (one, three)]
    assert [len(v.pop("forecast_inputs")) for v in verdicts] == [1, 3]
    assert verdicts[0] == verdicts[1] and verdicts[0]["verdict"] == "PASSED"

    # the whole year twice into one directory: the same decision, the same directories
    first = decide(quarter_world, "gggw", tmp_path / "year")
    listing = sorted(p.name for p in (tmp_path / "year").iterdir())
    second = decide(quarter_world, "gggw", tmp_path / "year")
    assert passed(first) == passed(second) == LABELS[:3]
    assert first.directory == second.directory
    assert sorted(p.name for p in (tmp_path / "year").iterdir()) == listing and len(listing) == 5

    # a published result that was changed afterwards is neither overwritten nor taken over
    (three / "verdict.json").write_text((three / "verdict.json").read_text() + " ")
    with pytest.raises(tool.Refused, match="is not the result this run produced"):
        tool.evaluate(good_parts(root, label), data, label, out, root)


def test_the_hash_does_not_depend_on_the_order_of_the_input(
    quarter_world: dict, locked: None, tmp_path: Path
) -> None:
    """Whole or split, a forecast has one content hash: init times must be in time order in every
    store, so a store with the same forecasts in another order is refused, not hashed differently."""
    from wrfcomp.harness.hashing import content_sha256

    root, data, label = quarter_world["root"], quarter_world["data"], "2024-Q2"
    whole = open_zarr(data / f"candidates/whole_{label}.zarr").load()
    backwards = whole.isel(init_time=slice(None, None, -1))
    assert content_sha256(backwards) != content_sha256(whole)  # why the order has to be fixed
    write_zarr(backwards, tmp_path / "backwards.zarr")
    a, b, c = good_parts(root, label)
    part = open_zarr(b).load()
    write_zarr(part.isel(init_time=slice(None, None, -1)), tmp_path / "part.zarr")
    for stores in ([tmp_path / "backwards.zarr"], [a, tmp_path / "part.zarr", c]):
        with pytest.raises(tool.Refused, match="init times are not in time order"):
            tool.evaluate(stores, data, label, tmp_path / "out", root)
        with pytest.raises(tool.Refused, match="init times are not in time order"):
            tool.promote(stores, data, 2024, tmp_path / "out", root)
    assert not (tmp_path / "out").exists()

    # in time order, every way of giving the quarter is the one forecast with the one hash
    digest = content_sha256(whole)
    forms = [[data / f"candidates/whole_{label}.zarr"], [a, b, c], [c, b, a], [b, c, a]]
    for number, stores in enumerate(forms):
        outcome = tool.evaluate(stores, data, label, tmp_path / f"form{number}", root)
        verdict = json.loads((outcome.directory / "verdict.json").read_text())
        assert verdict["candidate"]["forecast_sha256"] == digest and outcome.reasons == []


# ------------------------------------------------------------------------- promote: 3 of 4 quarters


def test_three_of_four_quarters_promote_and_two_do_not(
    quarter_world: dict, locked: None, tmp_path: Path
) -> None:
    all_four = decide(quarter_world, "gggg", tmp_path / "four")
    assert passed(all_four) == LABELS and all(q.reasons == [] for q in all_four.quarters)
    assert all_four.promoted and (all_four.model_id, all_four.model_version) == ("synth", "1")

    # exactly three pass: promoted, and the fourth says why it did not pass
    three = decide(quarter_world, "gggw", tmp_path / "three")
    assert passed(three) == LABELS[:3] and three.promoted
    assert "gate not passed against climatology" in three.quarters[3].reasons
    assert "gate not passed against noaa-gefs" in three.quarters[3].reasons
    assert tool._report_promotion(three) == 0

    # two pass, two do not: not promoted
    two = decide(quarter_world, "gwgw", tmp_path / "two")
    assert passed(two) == [LABELS[0], LABELS[2]] and not two.promoted
    assert tool._report_promotion(two) == 1
    assert tool.QUARTERS_REQUIRED == 3

    # the order the stores are given in does not matter: the same record, byte for byte
    stores = year_stores(quarter_world["root"], "gggw")
    shuffled = decide(quarter_world, stores[::-1], tmp_path / "shuffled")
    assert shuffled.directory.name == three.directory.name
    assert (shuffled.directory / "decision.json").read_bytes() == (
        three.directory / "decision.json"
    ).read_bytes()

    # the record of the run: what was decided, from which quarter results, under which rule
    record = json.loads((three.directory / "decision.json").read_text())
    assert record["decision"] == "PROMOTED" and record["quarters_passed"] == 3
    assert record["promotion_rule_version"] == 2 and record["quarters_required"] == 3
    assert record["candidate"] == {"model_id": "synth", "model_version": "1"}
    assert [q["verdict"] for q in record["quarters"]] == ["PASSED"] * 3 + ["NOT PASSED"]
    for entry, quarter in zip(record["quarters"], three.quarters, strict=True):
        assert entry["result"] == quarter.directory.name and quarter.directory.parent == tmp_path / "three"
        assert entry["verdict_sha256"] == tool._sha256_file(quarter.directory / "verdict.json")
    assert json.loads((two.directory / "decision.json").read_text())["decision"] == "NOT PROMOTED"


def test_a_missing_quarter_counts_as_not_passed(quarter_world: dict, locked: None, tmp_path: Path) -> None:
    three_given = decide(quarter_world, "ggg-", tmp_path / "three")
    assert passed(three_given) == LABELS[:3]
    last = three_given.quarters[3]
    assert last.reasons == ["no candidate store was given for this quarter"] and last.directory is None
    assert tool._report_promotion(three_given) == 0  # 3 of 4 passed

    two_given = decide(quarter_world, "-g-g", tmp_path / "two")
    assert passed(two_given) == [LABELS[1], LABELS[3]]
    assert tool._report_promotion(two_given) == 1  # two passing quarters are not enough
    record = json.loads((two_given.directory / "decision.json").read_text())
    assert record["quarters"][0]["result"] is None and record["quarters_passed"] == 2

    # a quarter given only in part is scored and does not pass: it is not the pinned quarter
    stores = year_stores(quarter_world["root"], "ggg-")[:-1]  # Q3 without its last store
    partial = decide(quarter_world, stores, tmp_path / "partial")
    assert passed(partial) == LABELS[:2]
    assert any("n_init_times against climatology differs" in line for line in partial.quarters[2].reasons)


def test_stores_of_another_candidate_or_another_harness_are_refused(
    quarter_world: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, data = quarter_world["root"], quarter_world["data"]
    good = year_stores(root, "ggg-")
    out = tmp_path / "out"

    # the synthetic root has no lock: nothing is decided on a tree that is not the locked harness
    with pytest.raises(tool.Refused, match="is not the locked harness"):
        decide(quarter_world, good, out)
    monkeypatch.setattr(tool, "_harness_lock_problems", lambda root: [])
    with pytest.raises(tool.Refused, match="the harness in use is not the one under"):
        decide(quarter_world, good, out)
    monkeypatch.setattr(tool, "_modules_outside_root", lambda root: [])
    # a scoring process that finds it is not running the harness under the root: nothing counts
    monkeypatch.setattr(tool, "_run_eval", lambda root, args: tool.EXIT_CHILD_REFUSED)
    with pytest.raises(tool.Refused, match="did not run the harness under"):
        decide(quarter_world, good, out)
    assert not list(out.glob("promotion_*"))
    monkeypatch.setattr(tool, "_run_eval", in_process_eval)

    # another candidate's passing quarter cannot stand in for one of this candidate's
    other = data / "candidates/other_2024-Q4.zarr"
    with pytest.raises(tool.Refused, match="not one model and version"):
        decide(quarter_world, [*good, other], out)
    version = open_zarr(good[-1]).load()
    version.attrs["model_version"] = "2"
    write_zarr(version, tmp_path / "version.zarr")
    with pytest.raises(tool.Refused, match="not one model and version"):
        decide(quarter_world, [*good[:-1], tmp_path / "version.zarr"], out)
    assert not list(out.glob("promotion_*"))
    # the other candidate, given all its own stores, is judged on its own
    own = [data / f"candidates/other_{label}.zarr" for label in LABELS]
    decision = decide(quarter_world, own, tmp_path / "own")
    assert decision.model_id == "someone-else" and passed(decision) == LABELS


def test_promote_trusts_no_file_supplied_by_the_caller(
    quarter_world: dict, locked: None, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """The verdict comes from this run's own scoring. Result directories that are already in the
    output directory, whoever wrote them, are not read as evidence."""
    honest = decide(quarter_world, "ggww", tmp_path / "honest")
    assert passed(honest) == LABELS[:2] and not honest.promoted

    # the caller plants passing results for the two failed quarters before the run: under invented
    # names, and as copies of this candidate's passing quarters with the period rewritten
    out = tmp_path / "planted"
    out.mkdir()
    for number, label in ((2, LABELS[2]), (3, LABELS[3])):
        forged = out / f"promotion_{label}_{'0' * 16}"
        shutil.copytree(honest.quarters[number - 2].directory, forged)
        verdict = json.loads((forged / "verdict.json").read_text())
        verdict["period"] = label
        (forged / "verdict.json").write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")
    record = json.loads((honest.directory / "decision.json").read_text())
    record["decision"], record["quarters_passed"] = "PROMOTED", 4
    (out / "promotion_2024_0000000000000000").mkdir()
    (out / "promotion_2024_0000000000000000/decision.json").write_text(json.dumps(record))
    planted = decide(quarter_world, "ggww", out)
    assert passed(planted) == LABELS[:2] and not planted.promoted
    assert (planted.directory / "decision.json").read_bytes() == (
        honest.directory / "decision.json"
    ).read_bytes()

    # planted under the very name this run will publish: refused, never adopted
    out = tmp_path / "squatted"
    target = out / honest.quarters[2].directory.name
    shutil.copytree(honest.quarters[0].directory, target)
    with pytest.raises(tool.Refused, match="is not the result this run produced"):
        decide(quarter_world, "ggww", out)

    # and the command line has no way to hand in a result directory
    with pytest.raises(SystemExit):
        tool.main(["promote", "--year", "2024", str(honest.quarters[0].directory)])
    capsys.readouterr()
    # given where a forecast store is expected, a result directory is not a forecast
    with pytest.raises(tool.Refused, match="cannot open|is refused|has no"):
        decide(quarter_world, [honest.quarters[0].directory], tmp_path / "as-store")


def test_promote_refuses_requests_it_cannot_decide(
    quarter_world: dict, base_world: dict, locked: None, tmp_path: Path
) -> None:
    root, data = quarter_world["root"], quarter_world["data"]
    good = year_stores(root, "gggg")
    out = tmp_path / "out"
    with pytest.raises(tool.Refused, match="no candidate forecast was given"):
        decide(quarter_world, [], out)
    with pytest.raises(tool.Refused, match="listed twice"):
        decide(quarter_world, [*good, good[0]], out)
    with pytest.raises(tool.Refused, match="no such forecast store"):
        decide(quarter_world, [*good, tmp_path / "nowhere.zarr"], out)
    # a store must lie inside one quarter of the year asked for
    whole = data / "candidates/whole_2024-Q1.zarr"
    shifted(whole, tmp_path / "spans.zarr", "2024-03-25T00:00")
    with pytest.raises(tool.Refused, match="holds init times of 2024-Q1, 2024-Q2: one quarter per store"):
        decide(quarter_world, [tmp_path / "spans.zarr"], out)
    shifted(whole, tmp_path / "earlier.zarr", "2023-01-01T00:00")
    with pytest.raises(tool.Refused, match="holds init times of 2023-Q1, not of 2024"):
        decide(quarter_world, [*good, tmp_path / "earlier.zarr"], out)
    with pytest.raises(tool.Refused, match="no pinned references for 2023-Q1"):
        tool.promote(good, data, 2023, out, root)
    # a tree that pins no quarters cannot promote anything
    with pytest.raises(tool.Refused, match="no pinned references for 2024-Q1, 2024-Q2, 2024-Q3, 2024-Q4"):
        tool.promote(good, data, 2024, out, base_world["root"])
    assert not out.exists()


def test_promote_command_line_and_who_may_say_promoted(
    quarter_world: dict,
    locked: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    tmp_path: Path,
) -> None:
    root, data = quarter_world["root"], quarter_world["data"]
    monkeypatch.setattr(tool, "ROOT", root)
    common = ["--year", "2024", "--data-dir", str(data), "--out", str(tmp_path / "year")]

    def stores(kinds: str) -> list[str]:
        return [str(path) for path in year_stores(root, kinds)]

    assert tool.main(["promote", *common, "--forecast", *stores("gggw")]) == 0
    out = capsys.readouterr().out
    assert "PROMOTED: synth version 1. 3 of 4 quarters of 2024 passed (rule: at least 3)" in out
    assert "NOT PROMOTED" not in out and "2024-Q4: NOT PASSED" in out and "2024-Q1: PASSED" in out
    assert "  - gate not passed against climatology" in out

    assert tool.main(["promote", *common, "--forecast", *stores("ggw-")]) == 1
    out = capsys.readouterr().out
    assert "NOT PROMOTED: synth version 1. 2 of 4 quarters of 2024 passed" in out
    assert "2024-Q4: NOT PASSED\n  - no candidate store was given for this quarter" in out
    assert "PROMOTED" not in out.replace("NOT PROMOTED", "")

    assert tool.main(["promote", *common, "--forecast", *stores("g---"), stores("g---")[0]]) == 2
    assert "REFUSED: a candidate store is listed twice" in capsys.readouterr().out
    assert tool.main(["promote", *common, "--forecast", str(tmp_path / "nowhere")]) == 2
    assert "REFUSED: candidate: no such forecast store" in capsys.readouterr().out

    # a single quarter, even a passing one, never says PROMOTED
    parts = [str(path) for path in good_parts(root, LABELS[0])]
    args = ["--data-dir", str(data), "--period", LABELS[0], "--out", str(tmp_path / "out")]
    assert tool.main(["evaluate", "--forecast", *parts, *args]) == 0
    out = capsys.readouterr().out
    assert "PASSED for 2024-Q1" in out and "One quarter is not a promotion" in out and "PROMOTED" not in out
    assert tool.main(["evaluate", "--forecast", str(data / f"candidates/weak_{LABELS[0]}.zarr"), *args]) == 1
    out = capsys.readouterr().out
    assert "NOT PASSED for 2024-Q1" in out and "PROMOTED" not in out


def test_committed_quarter_references_are_never_a_promotion(capsys: pytest.CaptureFixture) -> None:
    # reference scorecards are not a candidate's: `verify` says so and never says PROMOTED
    cards = [str(ROOT / "experiments" / f"{name}_2024-Q1.json") for name in ("gefs", "persistence")]
    assert tool.main(["verify", *cards, "--period", "2024-Q1"]) == 1
    out = capsys.readouterr().out
    assert "INCONSISTENT" in out and "PROMOTED" not in out
