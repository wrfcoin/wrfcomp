"""The promotion check: evaluation against both pinned references, and every refusal.

The world is synthetic and seeded (tests/conftest.py): a "climatology" and a "noaa-gefs" reference
forecast, their reference scorecards, an inventory that pins them, and a candidate that beats both.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import shutil
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


def _eval(root: Path, forecast: Path, out: Path, model_id: str, baseline: tuple[Path, str] | None) -> None:
    args = ["--forecast", str(forecast), "--truth", str(root / "data/truth/truth.nc")]
    args += ["--thresholds", str(root / "data/thresholds/tail.nc"), "--out", str(out)]
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
    # one published directory, named by the program from the period and the forecast's content hash
    directory = base_world["directory"]
    digest = base_world["cards"][0]["forecast_sha256"]
    assert [p.name for p in base_world["out"].iterdir()] == [directory.name]
    assert directory.name == f"promotion_{PERIOD}_{digest[:16]}"
    assert sorted(p.name for p in directory.iterdir()) == [
        "candidate_vs_climatology.json",
        "candidate_vs_noaa-gefs.json",
        "verdict.json",
    ]
    verdict = json.loads((directory / "verdict.json").read_text())
    assert verdict["verdict"] == "PROMOTED" and verdict["reasons"] == [] and verdict["period"] == PERIOD
    assert verdict["candidate"] == {"model_id": "synth", "model_version": "1", "forecast_sha256": digest}
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
    assert f"PROMOTED for {PERIOD}" in out and "NOT PROMOTED" not in out

    assert tool.main(["evaluate", "--forecast", str(data / "candidates/weak.zarr"), *args]) == 1
    out = capsys.readouterr().out
    assert "NOT PROMOTED" in out and "gate not passed against" in out

    # verify: the same good scorecards are only ever CONSISTENT
    paths = [str(base_world["directory"] / name) for name in tool.STAGED_NAMES.values()]
    assert tool.main(["verify", *paths, "--period", PERIOD]) == 0
    out = capsys.readouterr().out
    assert "CONSISTENT" in out and "proves nothing about where the scorecards came from" in out
    assert "This is not a promotion." in out
    assert "PROMOTED" not in out.replace("NOT PROMOTED", "")
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
    assert verdict["verdict"] == "NOT PROMOTED" and verdict["reasons"] == outcome.reasons
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
    assert outcome.directory.name.startswith(f"promotion_{PERIOD}_unscored-")
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
    assert "wrfcomp is loaded from" in str(refusal.value)
    assert not (tmp_path / "a").exists()


def test_origin_and_lock_checks_look_at_the_given_root(base_world: dict, tmp_path: Path) -> None:
    # this checkout's harness modules are files of this checkout
    assert tool._modules_outside_root(ROOT) == []
    outside = tool._modules_outside_root(base_world["root"])
    assert len(outside) == len(tool.HARNESS_MODULES)
    assert any(line.startswith("wrfcomp.harness.metrics is loaded from") for line in outside)
    expected = {"wrfcomp", "wrfcomp.harness.metrics", "wrfcomp.harness.evaluate", "wrfcomp.harness.schemas"}
    assert expected <= set(tool.HARNESS_MODULES)
    # the lock check runs that root's own script on that root
    assert tool._harness_lock_problems(ROOT) == []
    assert tool._harness_lock_problems(tmp_path) == [f"{tmp_path} has no scripts/harness_lock.py"]


def test_evaluations_run_through_uv_locked_on_the_checked_tree(tmp_path: Path) -> None:
    command = tool._eval_command(ROOT, ["--help"])
    assert Path(command[0]).name == "uv"
    script = str(ROOT / "scripts" / "check_promotion.py")
    assert command[1:] == [
        "run",
        "--locked",
        "--project",
        str(ROOT),
        "python",
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
    # stand-in evaluator that would write a card. The child must refuse before running it.
    shadow = tmp_path / "shadow"
    (shadow / "wrfcomp" / "harness").mkdir(parents=True)
    marker = tmp_path / "stand-in-ran"
    for name in ("align", "config", "hashing", "metrics", "schemas", "truth"):
        (shadow / "wrfcomp" / "harness" / f"{name}.py").write_text("")
    (shadow / "wrfcomp" / "__init__.py").write_text('__version__ = "0.1.0"\n')
    (shadow / "wrfcomp" / "harness" / "__init__.py").write_text("")
    (shadow / "wrfcomp" / "harness" / "evaluate.py").write_text(
        f"from pathlib import Path\n\n\ndef app(args=None, standalone_mode=True):\n"
        f"    Path({str(marker)!r}).write_text('ran')\n    return 0\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(shadow))
    assert tool._run_eval(ROOT, ["--forecast", "x.zarr"]) == tool.EXIT_CHILD_REFUSED
    assert not marker.exists()
    err = capfd.readouterr().err
    assert "the scoring process is not running the harness under" in err
    assert f"wrfcomp.harness.evaluate is loaded from {shadow}" in err
    monkeypatch.delenv("PYTHONPATH")

    # the child checks against the root it is given, and `evaluate` turns its refusal into a refusal
    assert tool.score_one(tmp_path, ["--help"]) == tool.EXIT_CHILD_REFUSED
    assert tool.main(["score-one", str(tmp_path), "--help"]) == tool.EXIT_CHILD_REFUSED
    capfd.readouterr()


# --------------------------------------------------------------------------- the real repository


def test_committed_inventory_pins_the_committed_reference_scorecards() -> None:
    inventory = tool.load_inventory(ROOT)
    assert sorted(inventory["periods"]) == ["2024-01", "2024-07"]
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
        for label in ("2024-01", "2024-07")
    }
    assert tool.build_inventory(ROOT, periods) == json.loads((ROOT / tool.INVENTORY).read_text())
    for label, entry in inventory["periods"].items():
        references = entry["references"]
        assert references["climatology"]["leads_scored"] == [1, 3, 6, 12, 24, 48]  # all v1 leads
        assert references["noaa-gefs"]["leads_scored"] == [3, 6, 12, 24, 48]  # 3 h and longer
        for record in references.values():
            assert (record["n_init_times"], record["n_stations"], record["init_hours_scored"]) == (
                62,
                500,
                [0, 12],
            )
            assert record["truth_sha256"] == references["climatology"]["truth_sha256"]
        assert len(entry["other_reference_forecasts"]) == 1, label
    assert len(tool.reference_forecast_hashes(inventory)) == 6  # no forecast is shared between periods


def test_real_reference_scorecards_are_never_a_promotion(capsys: pytest.CaptureFixture) -> None:
    # GEFS and persistence were both scored against climatology in January: not a candidate pair.
    cards = [str(ROOT / "experiments" / f"{name}_2024-01.json") for name in ("gefs", "persistence")]
    assert tool.main(["verify", *cards, "--period", "2024-01"]) == 1
    out = capsys.readouterr().out
    assert "INCONSISTENT" in out and "two scorecards against the same baseline: climatology" in out
    assert "PROMOTED" not in out
    # a period that is not pinned
    assert tool.main(["verify", *cards, "--period", "2024-02"]) == 2
    assert "REFUSED" in capsys.readouterr().out
