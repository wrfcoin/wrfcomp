"""The harness lock: the manifest tool, the owner-approval gate and the workflow that runs it."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


tool = _load("harness_lock", ROOT / "scripts" / "harness_lock.py")
gate = _load("harness_lock_gate", ROOT / ".github" / "harness_lock_gate.py")

# =============================================================================== the manifest tool


def test_the_repository_tree_matches_the_manifest() -> None:
    assert tool.check() == []


def test_manifest_records_tree_entries_version_scope_and_dependencies() -> None:
    manifest = json.loads((ROOT / "HARNESS_LOCK.json").read_text())
    assert manifest["manifest_version"] == 2
    assert manifest["harness_version"] == tool.package_version() == "0.1.0"
    assert manifest["scope"] == {"dirs": list(tool.LOCKED_DIRS), "files": list(tool.LOCKED_FILES)}
    entries = manifest["entries"]
    for required in (
        "src/wrfcomp/harness/metrics.py",
        "src/wrfcomp/harness/evaluate.py",
        "src/wrfcomp/harness/schemas.py",
        "src/wrfcomp/harness/config.py",
        "src/wrfcomp/harness/truth.py",
        "configs/splits.yaml",
        "configs/stations_conus.csv",
        "docs/VERIFICATION_PROTOCOL.md",
        "scripts/check_promotion.py",
        "experiments/REFERENCES.json",
        *(
            f"experiments/{m}_{p}.json"
            for m in ("climatology", "gefs", "persistence")
            for p in ("2024-01", "2024-07")
        ),
    ):
        assert required in entries, required
    for path, entry in entries.items():
        assert entry["mode"] == "100644", path
        assert len(entry["blob"]) == 40 and int(entry["blob"], 16) >= 0, path
    pins = manifest["scoring_dependencies"]
    assert set(pins) == {"scores", "numpy", "xarray", "scipy", "pandas"}
    assert all(isinstance(v, list) and v for v in pins.values())
    assert pins == tool.resolved_dependencies()


GIT_ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@example.invalid",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@example.invalid",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_SYSTEM": os.devnull,
}
UV_LOCK = """version = 1

[[package]]
name = "numpy"
version = "2.4.6"

[[package]]
name = "numpy"
version = "2.5.5"

[[package]]
name = "pandas"
version = "3.1.1"

[[package]]
name = "scipy"
version = "1.18.1"

[[package]]
name = "scores"
version = "2.7.0"

[[package]]
name = "xarray"
version = "2026.9.0"

[[package]]
name = "typer"
version = "0.12.0"
"""


def git(repo: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, env=GIT_ENV, check=True
    )
    return done.stdout


def commit(repo: Path, message: str = "change") -> None:
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", message)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A small repository with the real locked scope, a manifest, and a clean tree."""
    root = tmp_path / "repo"
    files = {
        "src/wrfcomp/__init__.py": '__version__ = "9.9.9"\n',
        "src/wrfcomp/harness/__init__.py": "",
        "src/wrfcomp/harness/metrics.py": "a = 1\n",
        "src/wrfcomp/other.py": "free = True\n",
        "configs/splits.yaml": "x: 1\n",
        "uv.lock": UV_LOCK,
        ".gitignore": "__pycache__/\n*.tmp\n",
        **{path: f"{path}\n" for path in tool.LOCKED_FILES},
    }
    for path, text in files.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(text)
    git(root.parent, "init", "-q", "-b", "main", str(root))
    commit(root, "base")
    tool.update("9.9.9", root)
    commit(root, "manifest")
    assert tool.check(root) == []
    return root


def test_update_then_check_is_clean_and_bytecode_caches_are_ignored(repo: Path) -> None:
    cache = repo / "src/wrfcomp/harness/__pycache__"
    cache.mkdir()
    (cache / "metrics.cpython-313.pyc").write_bytes(b"\0")
    assert tool.check(repo) == []
    # files outside the locked scope are free
    (repo / "src/wrfcomp/other.py").write_text("free = False\n")
    (repo / "notes.txt").write_text("hello\n")
    assert tool.check(repo) == []
    commit(repo)
    assert tool.check(repo) == []


def _edit_committed(repo: Path) -> None:
    (repo / "configs/splits.yaml").write_text("x: 2\n")
    commit(repo)


def _edit_uncommitted(repo: Path) -> None:
    (repo / "src/wrfcomp/harness/metrics.py").write_text("a = 2\n")


def _delete_uncommitted(repo: Path) -> None:
    (repo / "configs/splits.yaml").unlink()


def _untracked(repo: Path) -> None:
    (repo / "src/wrfcomp/harness/extra.py").write_text("b = 2\n")


def _ignored(repo: Path) -> None:
    (repo / "configs/override.tmp").write_text("x: 3\n")


def _added(repo: Path) -> None:
    (repo / "src/wrfcomp/harness/extra.py").write_text("b = 2\n")
    commit(repo)


def _removed(repo: Path) -> None:
    git(repo, "rm", "-q", "src/wrfcomp/harness/metrics.py")
    commit(repo)


def _renamed_out(repo: Path) -> None:
    git(repo, "mv", "src/wrfcomp/harness/metrics.py", "src/wrfcomp/metrics.py")
    commit(repo)


def _renamed_in(repo: Path) -> None:
    git(repo, "mv", "src/wrfcomp/other.py", "src/wrfcomp/harness/other.py")
    commit(repo)


def _renamed_inside(repo: Path) -> None:
    git(repo, "mv", "src/wrfcomp/harness/metrics.py", "src/wrfcomp/harness/scoring.py")
    commit(repo)


def _executable(repo: Path) -> None:
    git(repo, "update-index", "--chmod=+x", "configs/splits.yaml")
    git(repo, "commit", "-q", "-m", "mode")
    (repo / "configs/splits.yaml").chmod(0o755)


def _symlink(repo: Path) -> None:
    (repo / "elsewhere.yaml").write_text("x: 1\n")  # same bytes as the file it replaces
    (repo / "configs/splits.yaml").unlink()
    (repo / "configs/splits.yaml").symlink_to("../elsewhere.yaml")
    commit(repo)


def _submodule(repo: Path) -> None:
    head = git(repo, "rev-parse", "HEAD").strip()
    git(repo, "update-index", "--add", "--cacheinfo", f"160000,{head},configs/vendored")
    git(repo, "commit", "-q", "-m", "gitlink")


def _locked_file_removed(repo: Path) -> None:
    git(repo, "rm", "-q", "experiments/REFERENCES.json")
    commit(repo)


def _version_bumped(repo: Path) -> None:
    (repo / "src/wrfcomp/__init__.py").write_text('__version__ = "9.9.10"\n')
    commit(repo)


def _manifest(repo: Path, change) -> None:
    path = repo / "HARNESS_LOCK.json"
    manifest = json.loads(path.read_text())
    change(manifest)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    commit(repo)


def _manifest_version(repo: Path) -> None:
    _manifest(repo, lambda m: m.update(harness_version="10.0.0"))


def _manifest_scope(repo: Path) -> None:
    _manifest(repo, lambda m: m["scope"]["dirs"].remove("configs"))


def _manifest_entry_dropped(repo: Path) -> None:
    _manifest(repo, lambda m: m["entries"].pop("configs/splits.yaml"))


def _manifest_mode(repo: Path) -> None:
    _manifest(repo, lambda m: m["entries"]["configs/splits.yaml"].update(mode="100755"))


def _scoring_dependency(repo: Path) -> None:
    (repo / "uv.lock").write_text(UV_LOCK.replace('version = "2.7.0"', 'version = "2.8.0"'))
    commit(repo)


def _scoring_dependency_dropped_variant(repo: Path) -> None:
    (repo / "uv.lock").write_text(UV_LOCK.replace('[[package]]\nname = "numpy"\nversion = "2.4.6"\n\n', ""))


def _scoring_dependency_missing(repo: Path) -> None:
    (repo / "uv.lock").write_text(UV_LOCK.replace('name = "scipy"', 'name = "scipy-fork"'))


VIOLATIONS = {
    "edited and committed": (_edit_committed, ["changed: configs/splits.yaml"]),
    "edited, not committed": (
        _edit_uncommitted,
        ["uncommitted change in the locked scope: src/wrfcomp/harness/metrics.py"],
    ),
    "deleted, not committed": (
        _delete_uncommitted,
        ["uncommitted change in the locked scope: configs/splits.yaml"],
    ),
    "untracked file": (_untracked, ["untracked file in the locked scope: src/wrfcomp/harness/extra.py"]),
    "ignored file": (_ignored, ["ignored file in the locked scope: configs/override.tmp"]),
    "file added": (_added, ["added: src/wrfcomp/harness/extra.py"]),
    "file removed": (_removed, ["removed: src/wrfcomp/harness/metrics.py"]),
    "renamed out of scope": (_renamed_out, ["removed: src/wrfcomp/harness/metrics.py"]),
    "renamed into scope": (_renamed_in, ["added: src/wrfcomp/harness/other.py"]),
    "renamed inside scope": (
        _renamed_inside,
        ["added: src/wrfcomp/harness/scoring.py", "removed: src/wrfcomp/harness/metrics.py"],
    ),
    "made executable": (_executable, ["unsupported entry: configs/splits.yaml is a blob with mode 100755"]),
    "replaced by a symlink": (
        _symlink,
        ["unsupported entry: configs/splits.yaml is a blob with mode 120000"],
    ),
    "submodule": (_submodule, ["unsupported entry: configs/vendored is a commit with mode 160000"]),
    "locked file removed": (
        _locked_file_removed,
        ["missing: locked file experiments/REFERENCES.json", "removed: experiments/REFERENCES.json"],
    ),
    "package version bumped": (_version_bumped, ["version: the manifest locks harness '9.9.9'"]),
    "manifest version edited": (_manifest_version, ["version: the manifest locks harness '10.0.0'"]),
    "manifest scope edited": (_manifest_scope, ["scope: the manifest's scope differs"]),
    "manifest entry dropped": (_manifest_entry_dropped, ["added: configs/splits.yaml"]),
    "manifest mode edited": (
        _manifest_mode,
        ["mode changed: configs/splits.yaml is 100644, locked as 100755"],
    ),
    "scoring dependency changed": (
        _scoring_dependency,
        ["dependency: uv.lock resolves scores to 2.8.0, locked at 2.7.0"],
    ),
    "scoring dependency variant dropped": (
        _scoring_dependency_dropped_variant,
        ["dependency: uv.lock resolves numpy to 2.5.5, locked at 2.4.6, 2.5.5"],
    ),
    "scoring dependency gone": (_scoring_dependency_missing, ["dependency: uv.lock does not resolve scipy"]),
}


@pytest.mark.parametrize("case", sorted(VIOLATIONS))
def test_every_violation_is_reported(repo: Path, case: str) -> None:
    mutate, expected = VIOLATIONS[case]
    mutate(repo)
    problems = tool.check(repo)
    for fragment in expected:
        assert any(line.startswith(fragment) for line in problems), (fragment, problems)


def test_a_symlink_to_identical_bytes_is_still_a_violation(repo: Path) -> None:
    # The old check hashed file contents through the link and saw nothing.
    _symlink(repo)
    assert (repo / "configs/splits.yaml").read_text() == "x: 1\n"
    assert any("120000" in line for line in tool.check(repo))


def test_other_dependency_changes_are_free(repo: Path) -> None:
    (repo / "uv.lock").write_text(
        UV_LOCK.replace('version = "0.12.0"', 'version = "0.99.0"')
        + '\n[[package]]\nname = "new"\nversion = "1"\n'
    )
    commit(repo)
    assert tool.check(repo) == []


def test_update_reads_the_committed_tree_not_the_working_tree(repo: Path) -> None:
    before = (repo / "HARNESS_LOCK.json").read_text()
    (repo / "configs/splits.yaml").write_text("x: 2\n")  # not committed
    tool.update("9.9.9", repo)
    assert (repo / "HARNESS_LOCK.json").read_text() == before
    assert any(line.startswith("uncommitted change") for line in tool.check(repo))


def test_unreadable_lock_is_an_error_not_a_pass(
    repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    (repo / "HARNESS_LOCK.json").write_text('{"files": {}}\n')  # the old manifest format
    with pytest.raises(tool.LockError, match="version 2 manifest"):
        tool.check(repo)
    with pytest.raises(tool.LockError, match="git"):
        tool.tree_entries(tmp_path)  # not a repository
    assert tool.main(["harness_lock.py"]) == 2
    assert tool.main(["harness_lock.py", "check"]) == 0
    assert "Harness lock OK." in capsys.readouterr().out


# =============================================================================== the approval gate

OWNER = gate.APPROVERS[0]
HEAD = "a" * 40


def pull(*labels: str, changed_files: int = 1, head: str = HEAD) -> dict:
    return {"head": {"sha": head}, "changed_files": changed_files, "labels": [{"name": n} for n in labels]}


def committed(when: str, sha: str = HEAD) -> dict:
    return {"event": "committed", "sha": sha, "committer": {"date": when}, "author": {"date": when}}


def labeled(when: str, actor: str = OWNER, name: str = gate.LABEL) -> dict:
    return {"event": "labeled", "label": {"name": name}, "actor": {"login": actor}, "created_at": when}


def decide(
    files: list[dict], timeline: list[dict], *labels: str, action: str = "labeled", **kwargs
) -> tuple[bool, str]:
    kwargs.setdefault("pull", pull(*labels, changed_files=len(files)))
    kwargs.setdefault("event_head_sha", HEAD)
    approved, lines = gate.decide(files=files, timeline=timeline, action=action, **kwargs)
    return approved, "\n".join(lines)


LOCKED_CHANGE = [{"filename": "src/wrfcomp/harness/metrics.py", "status": "modified"}]
T1, T2, T3 = "2026-11-24T10:00:00Z", "2026-11-24T11:00:00Z", "2026-11-24T12:00:00Z"


def test_gate_passes_a_pull_request_that_touches_nothing_locked() -> None:
    files = [
        {"filename": "src/wrfcomp/models/new.py"},
        {"filename": "docs/BASELINES.md"},
        {"filename": "uv.lock"},
    ]
    for action in ("opened", "synchronize", "labeled"):
        approved, text = decide(files, [committed(T1)], action=action)
        assert approved and "No locked path" in text


@pytest.mark.parametrize(
    "path",
    [
        "src/wrfcomp/harness/metrics.py",
        "src/wrfcomp/harness/new_module.py",
        "configs/splits.yaml",
        "configs/new.yaml",
        ".github/workflows/harness-lock.yml",
        ".github/workflows/another.yml",
        ".github/harness_lock_gate.py",
        ".github/CODEOWNERS",
        "HARNESS_LOCK.json",
        "scripts/harness_lock.py",
        "scripts/check_promotion.py",
        "tests/test_harness_lock.py",
        "tests/test_check_promotion.py",
        "docs/HARNESS_LOCK.md",
        "docs/VERIFICATION_PROTOCOL.md",
        "experiments/REFERENCES.json",
        "experiments/gefs_2024-07.json",
    ],
)
def test_gate_needs_the_label_for_every_locked_path(path: str) -> None:
    approved, text = decide([{"filename": path}], [committed(T1)], action="opened")
    assert not approved and "NOT APPROVED" in text and "is not on the pull request" in text
    approved, _ = decide([{"filename": path}], [committed(T1), labeled(T2)], gate.LABEL)
    assert approved


def test_gate_sees_renames_out_of_and_into_the_locked_set() -> None:
    out = [
        {
            "filename": "src/wrfcomp/metrics.py",
            "previous_filename": "src/wrfcomp/harness/metrics.py",
            "status": "renamed",
        }
    ]
    into = [{"filename": "configs/extra.yaml", "previous_filename": "notes/extra.yaml", "status": "renamed"}]
    for files in (out, into):
        approved, text = decide(files, [committed(T1)], action="opened")
        assert not approved and "Locked paths touched" in text
    assert gate.changed_paths(out) == ["src/wrfcomp/harness/metrics.py", "src/wrfcomp/metrics.py"]
    assert not gate.is_locked("src/wrfcomp/harness.py") and not gate.is_locked("experiments/README.md")
    assert not gate.is_locked("configs.yaml") and not gate.is_locked("docs/HARNESS_LOCK.md.bak")


def test_gate_approves_only_a_fresh_owner_label() -> None:
    approved, text = decide(LOCKED_CHANGE, [committed(T1), labeled(T2)], gate.LABEL)
    assert approved and f"applied by {OWNER}" in text and HEAD in text

    # applied by anyone else, including the Actions bot of a workflow inside the pull request
    for actor in ("github-actions[bot]", "someone-else", None):
        event = labeled(T2, actor=actor) if actor else {**labeled(T2), "actor": None}
        approved, text = decide(LOCKED_CHANGE, [committed(T1), event], gate.LABEL)
        assert not approved and "not an approver" in text
    # the owner's label, followed by a re-label from a bot: the latest event decides
    approved, _ = decide(
        LOCKED_CHANGE, [committed(T1), labeled(T2), labeled(T3, actor="github-actions[bot]")], gate.LABEL
    )
    assert not approved
    # another label does not count, and the right label must actually be on the pull request
    approved, _ = decide(LOCKED_CHANGE, [committed(T1), labeled(T2, name="bug")], "bug")
    assert not approved
    approved, text = decide(LOCKED_CHANGE, [committed(T1), labeled(T2)])
    assert not approved and "is not on the pull request" in text
    approved, text = decide(LOCKED_CHANGE, [committed(T1)], gate.LABEL)
    assert not approved and "no labeled event" in text


def test_gate_binds_the_approval_to_the_head() -> None:
    # a commit after the label
    approved, text = decide(LOCKED_CHANGE, [committed(T1), labeled(T2), committed(T3)], gate.LABEL)
    assert not approved and "not newer than the latest head change" in text
    # at the very same second: not newer
    approved, _ = decide(LOCKED_CHANGE, [committed(T2), labeled(T2)], gate.LABEL)
    assert not approved
    # a force push, a base change or a reopen after the label
    for kind in gate.HEAD_CHANGE_EVENTS:
        approved, text = decide(
            LOCKED_CHANGE, [committed(T1), labeled(T2), {"event": kind, "created_at": T3}], gate.LABEL
        )
        assert not approved and "not newer" in text, kind
    # a fresh label after the push approves again
    approved, _ = decide(LOCKED_CHANGE, [committed(T1), labeled(T2), committed(T2), labeled(T3)], gate.LABEL)
    assert approved
    # commit times with an offset are compared as instants
    approved, _ = decide(LOCKED_CHANGE, [committed("2026-11-24T13:30:00+02:00"), labeled(T3)], gate.LABEL)
    assert approved
    approved, _ = decide(LOCKED_CHANGE, [committed("2026-11-24T14:30:00+02:00"), labeled(T3)], gate.LABEL)
    assert not approved
    # the author date counts too
    backdated = {**committed(T1), "author": {"date": T3}}
    approved, _ = decide(LOCKED_CHANGE, [backdated, labeled(T2)], gate.LABEL)
    assert not approved


def test_gate_fails_a_run_started_by_a_push_and_a_head_that_moved() -> None:
    timeline = [committed(T1), labeled(T2)]
    approved, text = decide(LOCKED_CHANGE, timeline, gate.LABEL, action="synchronize")
    assert not approved and "every push needs a fresh owner label" in text
    approved, text = decide(LOCKED_CHANGE, timeline, gate.LABEL, event_head_sha="b" * 40)
    assert not approved and "head moved during this run" in text
    for result in ("failure", "cancelled", ""):
        approved, text = decide(LOCKED_CHANGE, timeline, gate.LABEL, withdraw_result=result)
        assert not approved and "withdraws approval" in text
    for result in ("success", "skipped"):
        assert decide(LOCKED_CHANGE, timeline, gate.LABEL, withdraw_result=result)[0]


def test_gate_fails_closed_on_incomplete_or_unreadable_facts(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    harmless = [{"filename": "docs/BASELINES.md"}]
    # GitHub says 5 files changed but listed 1: the locked change may be in the missing part
    approved, text = decide(harmless, [committed(T1)], pull=pull(changed_files=5), action="opened")
    assert not approved and "may be cut off" in text
    approved, _ = decide(
        harmless, [committed(T1)], pull={"head": {"sha": HEAD}, "labels": []}, action="opened"
    )
    assert not approved
    many = [{"filename": f"docs/{i}.md"} for i in range(gate.MAX_LISTED_FILES)]
    assert not decide(many, [committed(T1)], action="opened")[0]
    # unreadable times
    approved, text = decide(LOCKED_CHANGE, [{"event": "committed", "sha": HEAD}, labeled(T2)], gate.LABEL)
    assert not approved and "no readable date" in text
    approved, _ = decide(LOCKED_CHANGE, [committed(T1), {**labeled(T2), "created_at": "soon"}], gate.LABEL)
    assert not approved
    approved, text = decide(LOCKED_CHANGE, [labeled(T2)], gate.LABEL)
    assert not approved and "no commit" in text
    # the command itself: no environment, or an API that cannot be reached
    for name in ("GH_TOKEN", "PR_NUMBER", "EVENT_ACTION", "EVENT_HEAD_SHA"):
        monkeypatch.delenv(name, raising=False)
    assert gate.main() == 1
    assert "NOT APPROVED" in capsys.readouterr().out
    env = {"GH_TOKEN": "x", "GITHUB_REPOSITORY": "o/r", "PR_NUMBER": "1", "EVENT_ACTION": "labeled"}
    for name, value in {**env, "EVENT_HEAD_SHA": HEAD, "GITHUB_API_URL": "http://127.0.0.1:1"}.items():
        monkeypatch.setenv(name, value)
    assert gate.main() == 1
    assert "could not read the pull request" in capsys.readouterr().out


def test_gate_main_reads_the_api_and_follows_pagination(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    pages = {
        "/repos/o/r/pulls/7": (pull(gate.LABEL, changed_files=2), None),
        "/repos/o/r/pulls/7/files?per_page=100": ([{"filename": "docs/a.md"}], "/files-page-2"),
        "/files-page-2": (LOCKED_CHANGE, None),
        "/repos/o/r/issues/7/timeline?per_page=100": ([committed(T1)], "/timeline-page-2"),
        "/timeline-page-2": ([labeled(T2)], None),
    }
    seen: list[str] = []

    def fake_get(url: str, token: str):
        assert token == "secret-token"
        key = url.removeprefix("https://api.test")
        seen.append(key)
        body, nxt = pages[key]
        return body, (f"https://api.test{nxt}" if nxt else None)

    monkeypatch.setattr(gate, "_get", fake_get)
    env = {
        "GH_TOKEN": "secret-token",
        "GITHUB_REPOSITORY": "o/r",
        "PR_NUMBER": "7",
        "EVENT_ACTION": "labeled",
    }
    for name, value in {**env, "EVENT_HEAD_SHA": HEAD, "GITHUB_API_URL": "https://api.test"}.items():
        monkeypatch.setenv(name, value)
    assert gate.main() == 0
    out = capsys.readouterr().out
    assert "APPROVED" in out and "secret-token" not in out
    assert "/files-page-2" in seen and "/timeline-page-2" in seen  # the locked file was on page 2
    monkeypatch.setenv("EVENT_ACTION", "synchronize")
    assert gate.main() == 1


# =============================================================================== wiring


def test_gate_scope_covers_the_manifest_scope_and_the_lock_machinery() -> None:
    for directory in tool.LOCKED_DIRS:
        assert f"{directory}/" in gate.LOCKED_PREFIXES
    for path in tool.LOCKED_FILES:
        assert path in gate.LOCKED_FILES
    for path in (
        "HARNESS_LOCK.json",
        "scripts/harness_lock.py",
        "tests/test_harness_lock.py",
        "tests/test_check_promotion.py",
        "docs/HARNESS_LOCK.md",
        ".github/workflows/harness-lock.yml",
        ".github/workflows/ci.yml",
        ".github/harness_lock_gate.py",
        ".github/CODEOWNERS",
    ):
        assert gate.is_locked(path), path
    owners = (ROOT / ".github" / "CODEOWNERS").read_text()
    for path in (*gate.LOCKED_FILES, *gate.LOCKED_PREFIXES):
        assert f"/{path} " in owners or f"/{path}\t" in owners, path
    assert all(line.split()[-1] == f"@{OWNER}" for line in owners.splitlines() if line.startswith("/"))


def _workflow(name: str) -> tuple[dict, str]:
    text = (ROOT / ".github" / "workflows" / name).read_text()
    return yaml.safe_load(text), text


def test_approval_workflow_runs_from_the_base_branch_and_never_checks_out() -> None:
    workflow, text = _workflow("harness-lock.yml")
    triggers = workflow[True]  # YAML reads the key `on` as a boolean
    assert set(triggers) == {"pull_request_target"}
    assert {"opened", "reopened", "synchronize", "labeled", "unlabeled"} <= set(
        triggers["pull_request_target"]["types"]
    )
    assert workflow["permissions"] == {}
    assert (
        workflow["concurrency"]["cancel-in-progress"] is False
    )  # a queued run must still withdraw the label

    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    assert "actions/checkout" not in code and "uses:" not in code
    assert "git " not in code and "head.ref" not in code and "head.repo" not in code
    assert "pull_request.title" not in code and "pull_request.body" not in code
    # expressions appear only in `if:`, `env:` values, the concurrency group and job wiring, never in a script
    for job in workflow["jobs"].values():
        for step in job["steps"]:
            assert "${{" not in step["run"], step["name"]

    gate_job = workflow["jobs"]["harness-lock"]
    assert gate_job["name"] == "harness-lock"
    assert gate_job["permissions"] == {"contents": "read", "pull-requests": "read", "issues": "read"}
    assert gate_job["needs"] == "withdraw-approval" and "!cancelled()" in gate_job["if"]
    fetch, run = gate_job["steps"]
    assert "contents/.github/harness_lock_gate.py?ref=${WORKFLOW_SHA}" in fetch["run"]
    assert (
        fetch["env"]["WORKFLOW_SHA"] == "${{ github.sha }}"
    )  # the base commit the workflow itself came from
    assert run["env"]["EVENT_HEAD_SHA"] == "${{ github.event.pull_request.head.sha }}"
    assert run["env"]["WITHDRAW_RESULT"] == "${{ needs.withdraw-approval.result }}"

    withdraw = workflow["jobs"]["withdraw-approval"]
    assert withdraw["if"] == "github.event.action == 'synchronize'"
    assert withdraw["permissions"] == {"pull-requests": "write"}  # the only write permission in the file
    assert (
        f"/labels/{gate.LABEL}" in withdraw["steps"][0]["run"] and "-X DELETE" in withdraw["steps"][0]["run"]
    )
    assert text.count("write") == 1


def test_ci_workflow_keeps_only_the_hash_check_lint_and_tests() -> None:
    workflow, text = _workflow("ci.yml")
    assert set(workflow[True]) == {"push", "pull_request"}
    assert workflow["permissions"] == {"contents": "read"}
    assert list(workflow["jobs"]) == ["test"]
    runs = [step.get("run", "") for step in workflow["jobs"]["test"]["steps"]]
    assert "python3 scripts/harness_lock.py check" in runs
    assert "uv run ruff check" in runs and "uv run pytest -q" in runs
    assert gate.LABEL not in text and "pull_request_target" not in text
