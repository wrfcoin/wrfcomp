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
            f"experiments/{m}_2024-{k:02d}.json"
            for m in ("climatology", "gefs", "persistence")
            for k in range(1, 13)
        ),
    ):
        assert required in entries, required
    for path, entry in entries.items():
        assert entry["mode"] == "100644", path
        assert len(entry["blob"]) == 40 and int(entry["blob"], 16) >= 0, path
    pins = manifest["scoring_dependencies"]
    assert sum(path.startswith("experiments/") for path in entries) == 37  # 36 scorecards and the inventory
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
        **{path: f"{path}\n" for path in tool.LOCKED_FILES},
        "src/wrfcomp/__init__.py": '__version__ = "9.9.9"\n',
        "src/wrfcomp/harness/__init__.py": "",
        "src/wrfcomp/harness/metrics.py": "a = 1\n",
        "src/wrfcomp/other.py": "free = True\n",
        "scripts/harness_lock.py": (ROOT / "scripts" / "harness_lock.py").read_text(),
        "configs/splits.yaml": "x: 1\n",
        "uv.lock": UV_LOCK,
        ".gitignore": "__pycache__/\n*.tmp\n",
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


CACHE = "src/wrfcomp/harness/__pycache__"


def _moved_into_cache(repo: Path) -> None:
    # the review's case: the file leaves the tree, staged, into a directory the old check skipped
    (repo / CACHE).mkdir()
    git(repo, "mv", "-f", "src/wrfcomp/harness/metrics.py", f"{CACHE}/metrics.py")


def _force_added_in_cache(repo: Path) -> None:
    (repo / CACHE).mkdir()
    (repo / CACHE / "evil.py").write_text("b = 2\n")
    git(repo, "add", "-f", f"{CACHE}/evil.py")


def _committed_in_cache(repo: Path) -> None:
    _force_added_in_cache(repo)
    git(repo, "commit", "-q", "-m", "tracked file in a cache directory")


def _source_in_cache(repo: Path) -> None:
    (repo / CACHE).mkdir()
    (repo / CACHE / "evil.py").write_text("b = 2\n")  # ignored by .gitignore, but not bytecode


def _nested_in_cache(repo: Path) -> None:
    (repo / CACHE / "deeper").mkdir(parents=True)
    (repo / CACHE / "deeper" / "metrics.cpython-313.pyc").write_bytes(b"\0")


def _bytecode_outside_cache(repo: Path) -> None:
    (repo / "src/wrfcomp/harness/metrics.pyc").write_bytes(b"\0")


def _staged_bytecode(repo: Path) -> None:
    (repo / CACHE).mkdir()
    (repo / CACHE / "metrics.cpython-313.pyc").write_bytes(b"\0")
    git(repo, "add", "-f", f"{CACHE}/metrics.cpython-313.pyc")


VIOLATIONS = {
    "moved into a bytecode directory": (
        _moved_into_cache,
        [f"uncommitted change in the locked scope: {CACHE}/metrics.py (git status R)"],
    ),
    "force-added in a bytecode directory": (
        _force_added_in_cache,
        [f"uncommitted change in the locked scope: {CACHE}/evil.py (git status A)"],
    ),
    "committed in a bytecode directory": (_committed_in_cache, [f"added: {CACHE}/evil.py"]),
    "source file in a bytecode directory": (
        _source_in_cache,
        [f"ignored file in the locked scope: {CACHE}/evil.py"],
    ),
    "bytecode nested below a bytecode directory": (
        _nested_in_cache,
        [f"ignored file in the locked scope: {CACHE}/deeper/metrics.cpython-313.pyc"],
    ),
    "bytecode outside a bytecode directory": (
        _bytecode_outside_cache,
        ["untracked file in the locked scope: src/wrfcomp/harness/metrics.pyc"],
    ),
    "staged bytecode": (
        _staged_bytecode,
        [f"uncommitted change in the locked scope: {CACHE}/metrics.cpython-313.pyc (git status A)"],
    ),
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


def run_script(repo: Path, *flags: str, env: dict | None = None) -> subprocess.CompletedProcess:
    inherited = {k: v for k, v in GIT_ENV.items() if k != "PYTHONSAFEPATH"}
    environment = {**inherited, **(env or {})}
    command = [sys.executable, *flags, "scripts/harness_lock.py", "check"]
    return subprocess.run(command, cwd=repo, capture_output=True, text=True, env=environment, check=False)


def test_tomllib_planted_next_to_the_script_cannot_hide_a_dependency_change(
    repo: Path, tmp_path: Path
) -> None:
    # The review's case: a pull request changes a scoring dependency in uv.lock and adds
    # scripts/tomllib.py, which answers with the pinned versions. Neither file is locked.
    marker = tmp_path / "planted-tomllib-imported"
    pinned = json.loads((repo / "HARNESS_LOCK.json").read_text())["scoring_dependencies"]
    packages = [{"name": name, "version": v} for name, versions in pinned.items() for v in versions]
    (repo / "scripts" / "tomllib.py").write_text(
        "from pathlib import Path\n\n"
        f"Path({str(marker)!r}).write_text('imported')\n"
        "TOMLDecodeError = ValueError\n\n\n"
        f"def loads(text):\n    return {{'package': {packages!r}}}\n"
    )
    (repo / "uv.lock").write_text(UV_LOCK.replace('version = "2.7.0"', 'version = "9.9.9"'))

    # Run by path without -P, Python would import the planted file: the script refuses to start.
    refused = run_script(repo)
    assert refused.returncode != 0 and "Harness lock OK." not in refused.stdout
    assert "REFUSED: run this script with Python's -P option" in refused.stderr
    assert not marker.exists()

    # With -P (as CI runs it) the real tomllib reads the real uv.lock and the change is reported.
    for flags, env in ((("-P",), None), ((), {"PYTHONSAFEPATH": "1"})):
        checked = run_script(repo, *flags, env=env)
        assert checked.returncode == 1, checked.stderr
        assert "dependency: uv.lock resolves scores to 9.9.9, locked at 2.7.0" in checked.stdout
        assert "Harness lock violated." in checked.stdout
    assert not marker.exists()

    # and with the dependency change taken back, the same command passes: the planted file is inert
    (repo / "uv.lock").write_text(UV_LOCK)
    assert "Harness lock OK." in run_script(repo, "-P").stdout
    assert not marker.exists()


def test_the_approval_gate_script_also_insists_on_a_safe_path() -> None:
    script = str(ROOT / ".github" / "harness_lock_gate.py")
    environment = {k: v for k, v in os.environ.items() if k not in ("PYTHONSAFEPATH", "GH_TOKEN")}
    plain = subprocess.run(
        [sys.executable, script], capture_output=True, text=True, env=environment, check=False
    )
    assert plain.returncode != 0 and "REFUSED: run this script with Python's -P option" in plain.stderr
    safe = subprocess.run(
        [sys.executable, "-P", script], capture_output=True, text=True, env=environment, check=False
    )
    assert safe.returncode == 1 and "NOT APPROVED: the gate could not read the pull request" in safe.stdout


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
HEAD, OLD_HEAD, BASE = "a" * 40, "c" * 40, "b" * 40
T1, T2 = "2026-11-24T10:00:00Z", "2026-11-24T11:00:00Z"


def pull(*, changed_files: int = 1, head: str = HEAD, base: str = BASE) -> dict:
    return {"head": {"sha": head}, "base": {"sha": base}, "changed_files": changed_files, "labels": []}


def comment(
    body: str, *, login: str | None = OWNER, kind: str = "User", edited: bool = False, number: int = 1
) -> dict:
    """A conversation comment in the form `gate.fetch_comments` returns."""
    return {
        "url": f"https://github.test/pr/1#issuecomment-{number}",
        "body": body,
        "login": login,
        "type": kind,
        "last_edited_at": T2 if edited else None,
        "includes_created_edit": False,
    }


def approval(head: str = HEAD, **kwargs) -> dict:
    return comment(f"{gate.APPROVAL_WORD} {head}", **kwargs)


def decide(files: list[dict], comments: list[dict], **kwargs) -> tuple[bool, str]:
    kwargs.setdefault("pull_before", pull(changed_files=len(files)))
    kwargs.setdefault("pull_after", kwargs["pull_before"])
    kwargs.setdefault("event_head_sha", HEAD)
    kwargs.setdefault("policy_sha", BASE)
    passes, lines = gate.decide(files=files, comments=comments, **kwargs)
    return passes, "\n".join(lines)


LOCKED_CHANGE = [{"filename": "src/wrfcomp/harness/metrics.py", "status": "modified"}]
HARMLESS = [
    {"filename": "src/wrfcomp/models/new.py"},
    {"filename": "docs/BASELINES.md"},
    {"filename": "uv.lock"},
]


def test_gate_passes_a_pull_request_that_touches_nothing_locked() -> None:
    passes, text = decide(HARMLESS, [])
    assert passes and "No locked path" in text
    # the verdict names the commit it is about and the base whose policy judged it
    assert f"Evaluated head {HEAD} on base {BASE}, with the policy of that base." in text


@pytest.mark.parametrize(
    "path",
    [
        "src/wrfcomp/harness/metrics.py",
        "src/wrfcomp/harness/new_module.py",
        "src/wrfcomp/__init__.py",
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
def test_gate_needs_an_approval_of_the_head_for_every_locked_path(path: str) -> None:
    passes, text = decide([{"filename": path}], [])
    assert not passes and f"NOT APPROVED: no approval comment for head {HEAD}" in text
    assert f"    {gate.APPROVAL_WORD} {HEAD}" in text  # it says exactly what the owner has to write
    passes, text = decide([{"filename": path}], [approval()])
    assert passes and f"APPROVED: {OWNER} approved head {HEAD}" in text


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
        passes, text = decide(files, [])
        assert not passes and "Locked paths touched" in text
    assert gate.changed_paths(out) == ["src/wrfcomp/harness/metrics.py", "src/wrfcomp/metrics.py"]
    assert not gate.is_locked("src/wrfcomp/harness.py") and not gate.is_locked("experiments/README.md")
    assert not gate.is_locked("configs.yaml") and not gate.is_locked("docs/HARNESS_LOCK.md.bak")
    assert not gate.is_locked("src/wrfcomp/baselines/__init__.py")


def test_approval_is_bound_to_the_exact_head_commit() -> None:
    # An approval of an earlier head says nothing about this one: that is all a new push needs.
    passes, text = decide(LOCKED_CHANGE, [approval(OLD_HEAD)])
    assert not passes and "first line is not exactly the approval of this head" in text
    # several approvals over the life of a pull request: only the one for the current head counts
    assert decide(LOCKED_CHANGE, [approval(OLD_HEAD, number=1), approval(number=2)])[0]
    assert not decide(LOCKED_CHANGE, [approval(OLD_HEAD, number=1), approval("d" * 40, number=2)])[0]
    # the base commit's sha is not the head's
    assert not decide(LOCKED_CHANGE, [approval(BASE)])[0]


@pytest.mark.parametrize(
    "body",
    [
        f"{gate.APPROVAL_WORD} {HEAD[:39]}",  # 39 characters
        f"{gate.APPROVAL_WORD} {HEAD[:7]}",  # a short sha
        f"{gate.APPROVAL_WORD} {HEAD}0",  # 41 characters
        f"{gate.APPROVAL_WORD} {HEAD.upper()}",
        f"{gate.APPROVAL_WORD}  {HEAD}",  # two spaces
        f"{gate.APPROVAL_WORD} {HEAD} ",  # trailing space
        f" {gate.APPROVAL_WORD} {HEAD}",  # leading space
        f"{gate.APPROVAL_WORD} {HEAD} looks good",
        f"{gate.APPROVAL_WORD}: {HEAD}",
        f"{gate.APPROVAL_WORD.lower()} {HEAD}",
        f"> {gate.APPROVAL_WORD} {HEAD}",  # a quotation of someone else's comment
        f"`{gate.APPROVAL_WORD} {HEAD}`",
        f"I will not write\n{gate.APPROVAL_WORD} {HEAD}",  # not the first line
        f"\n{gate.APPROVAL_WORD} {HEAD}",
        gate.APPROVAL_WORD,
        HEAD,
        "",
    ],
)
def test_only_the_exact_first_line_approves(body: str) -> None:
    passes, text = decide(LOCKED_CHANGE, [comment(body)])
    assert not passes and "NOT APPROVED" in text


def test_exact_first_line_may_be_followed_by_anything() -> None:
    for body in (
        f"{gate.APPROVAL_WORD} {HEAD}\nReviewed with Codex's findings, go ahead.",
        f"{gate.APPROVAL_WORD} {HEAD}\r\nWindows line ending",
        f"{gate.APPROVAL_WORD} {HEAD}\n",
    ):
        assert decide(LOCKED_CHANGE, [comment(body)])[0], body


def test_only_an_unedited_comment_by_the_owner_account_approves() -> None:
    for login, kind in (
        ("github-actions[bot]", "Bot"),
        ("someone-else", "User"),
        (OWNER, "Bot"),
        (None, "User"),
    ):
        passes, text = decide(LOCKED_CHANGE, [approval(login=login, kind=kind)])
        assert not passes and "is not an approver" in text, (login, kind)
    passes, text = decide(LOCKED_CHANGE, [approval(edited=True)])
    assert not passes and "GitHub records an edit of this comment" in text
    # GitHub's edit record must be there and empty: a creation that was itself an edit, or a comment
    # record without the edit fields, does not approve
    assert not decide(LOCKED_CHANGE, [{**approval(), "includes_created_edit": True}])[0]
    assert not decide(LOCKED_CHANGE, [{**approval(), "includes_created_edit": None}])[0]
    without_record = {k: v for k, v in approval().items() if k != "last_edited_at"}
    assert not decide(LOCKED_CHANGE, [without_record])[0]
    assert not decide(LOCKED_CHANGE, [{**approval(), "last_edited_at": ""}])[0]
    # someone else's attempt does not spoil the owner's approval, in either order
    stranger = approval(login="someone-else", number=2)
    assert decide(LOCKED_CHANGE, [stranger, approval(number=3)])[0]
    assert decide(LOCKED_CHANGE, [approval(number=3), stranger])[0]
    # malformed comment records are skipped, not trusted
    broken = [
        {"body": None},
        {"body": f"{gate.APPROVAL_WORD} {HEAD}"},
        {"body": f"{gate.APPROVAL_WORD} {HEAD}", "login": OWNER},
    ]
    assert not decide(LOCKED_CHANGE, broken)[0]
    # a label plays no part: with or without one, the answer is the same
    labelled = {**pull(), "labels": [{"name": "owner-approved-harness-change"}]}
    assert not decide(LOCKED_CHANGE, [], pull_before=labelled, pull_after=labelled)[0]
    assert decide(LOCKED_CHANGE, [approval()], pull_before=labelled, pull_after=labelled)[0]


def test_gate_fails_if_the_pull_request_moved_while_it_was_read() -> None:
    moved_head = pull(head="d" * 40)
    moved_base = pull(base="e" * 40)
    for files, comments in (
        (HARMLESS, []),
        (LOCKED_CHANGE, [approval()]),
        (LOCKED_CHANGE, [approval("d" * 40)]),
    ):
        before = pull(changed_files=len(files))
        for after in (moved_head, moved_base):
            after = {**after, "changed_files": len(files)}
            passes, text = decide(files, comments, pull_before=before, pull_after=after)
            assert not passes and "moved while it was being read" in text
        # a run started for an older head never answers for the current one, locked paths or not
        passes, text = decide(files, comments, event_head_sha=OLD_HEAD)
        assert not passes and f"started for head {OLD_HEAD}" in text
    # unreadable commits are not a pass either
    for bad in ({}, {"head": {"sha": "not-a-sha"}, "base": {"sha": BASE}}, {"head": {"sha": HEAD}}):
        passes, text = decide(
            HARMLESS, [], pull_before=pull(changed_files=3), pull_after={**bad, "changed_files": 3}
        )
        assert not passes and "NOT APPROVED" in text


def test_gate_applies_only_the_policy_of_the_base_being_judged(
    monkeypatch: pytest.MonkeyPatch, api_env: None, capsys
) -> None:
    older, newer = "9" * 40, "8" * 40
    # A re-run of an old run carries the old policy commit while the pull request now sits on a newer
    # base (or the other way round). Neither way of passing is open: not "nothing locked", not "approved".
    for files, comments in ((HARMLESS, []), (LOCKED_CHANGE, [approval()])):
        assert decide(files, comments)[0]  # same base and policy: passes
        for policy in (older, newer, "", BASE[:39]):
            passes, text = decide(files, comments, policy_sha=policy)
            assert not passes and "Update the pull request onto the current base branch" in text, policy
            assert "No locked path" not in text
            assert not any(line.startswith("APPROVED:") for line in text.splitlines())
    # through the command: the policy commit comes from the environment the workflow sets
    api = FakeApi(files=[HARMLESS], comments=[[]], pulls=[pull(changed_files=3), pull(changed_files=3)])
    api.install(monkeypatch)
    assert gate.main() == 0
    monkeypatch.setenv("POLICY_SHA", older)
    api = FakeApi(files=[HARMLESS], comments=[[]], pulls=[pull(changed_files=3), pull(changed_files=3)])
    api.install(monkeypatch)
    assert gate.main() == 1
    assert f"policy comes from base-branch commit {older}" in capsys.readouterr().out
    monkeypatch.delenv("POLICY_SHA")
    assert gate.main() == 1  # no policy commit given: not approved


def test_comments_are_read_with_their_edit_record_and_bad_responses_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = FakeApi(files=[], comments=[[approval(number=1)], [approval(edited=True, number=2)]], pulls=[])
    api.install(monkeypatch)
    got = gate.fetch_comments("https://api.test/graphql", "o/r", 7, "secret-token")
    assert got == [approval(number=1), approval(edited=True, number=2)]
    assert got[0]["last_edited_at"] is None and got[1]["last_edited_at"] == T2
    deleted_author = FakeApi(files=[], comments=[[approval(login=None)]], pulls=[])
    deleted_author.install(monkeypatch)
    assert gate.fetch_comments("https://api.test/graphql", "o/r", 7, "secret-token")[0]["login"] is None
    for bad in ({}, {"repository": None}, {"repository": {"pullRequest": {"comments": {"nodes": [{}]}}}}):
        monkeypatch.setattr(gate, "_graphql", lambda *a, bad=bad: bad)
        with pytest.raises(ValueError, match="unexpected GraphQL response shape"):
            gate.fetch_comments("https://api.test/graphql", "o/r", 7, "secret-token")


def test_graphql_address_defaults_to_the_api_address(monkeypatch: pytest.MonkeyPatch, api_env: None) -> None:
    # Outside Actions there is no GITHUB_GRAPHQL_URL: the gate derives it from the API address.
    api = FakeApi(files=[LOCKED_CHANGE], comments=[[approval()]], pulls=[pull(), pull()])
    api.install(monkeypatch)
    monkeypatch.delenv("GITHUB_GRAPHQL_URL")
    assert gate.main() == 0
    assert "graphql comments page 0" in api.seen


def test_gate_fails_closed_on_incomplete_or_unreadable_facts(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    harmless = [{"filename": "docs/BASELINES.md"}]
    # GitHub says 5 files changed but listed 1: the locked change may be in the missing part
    cut = pull(changed_files=5)
    passes, text = decide(harmless, [], pull_before=cut, pull_after=cut)
    assert not passes and "may be cut off" in text
    assert decide(harmless, [approval()], pull_before=cut, pull_after=cut)[
        0
    ]  # the owner may still approve it
    unknown = {"head": {"sha": HEAD}, "base": {"sha": BASE}}
    assert not decide(harmless, [], pull_before=unknown, pull_after=unknown)[0]
    many = [{"filename": f"docs/{i}.md"} for i in range(gate.MAX_LISTED_FILES)]
    assert not decide(many, [])[0]
    # the command itself: no environment, or an API that cannot be reached
    for name in ("GH_TOKEN", "PR_NUMBER", "EVENT_HEAD_SHA", "POLICY_SHA"):
        monkeypatch.delenv(name, raising=False)
    assert gate.main() == 1
    assert "NOT APPROVED" in capsys.readouterr().out
    env = {"GH_TOKEN": "x", "GITHUB_REPOSITORY": "o/r", "PR_NUMBER": "1", "EVENT_HEAD_SHA": HEAD}
    env["POLICY_SHA"] = BASE
    env.update(GITHUB_API_URL="http://127.0.0.1:1", GITHUB_GRAPHQL_URL="http://127.0.0.1:1/graphql")
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    assert gate.main() == 1
    assert "could not read the pull request" in capsys.readouterr().out


class FakeApi:
    """Stands in for the GitHub API. The pull request can change between the first and second read."""

    def __init__(self, *, files: list[list[dict]], comments: list[list[dict]], pulls: list[dict]) -> None:
        self.pulls, self.seen, self.comments = list(pulls), [], comments
        self.pages = {"/repos/o/r/pulls/7/files?per_page=100": files}

    def get(self, url: str, token: str):
        assert token == "secret-token"
        key = url.removeprefix("https://api.test")
        self.seen.append(key)
        if key == "/repos/o/r/pulls/7":
            return self.pulls.pop(0), None
        base, _, page = key.partition("#page")
        pages = self.pages[base]
        index = int(page or 0)
        nxt = f"https://api.test{base}#page{index + 1}" if index + 1 < len(pages) else None
        return pages[index], nxt

    def graphql(self, url: str, token: str, query: str, variables: dict) -> dict:
        assert token == "secret-token" and url == "https://api.test/graphql"
        assert "lastEditedAt" in query and "includesCreatedEdit" in query and "mutation" not in query
        assert (variables["owner"], variables["name"], variables["number"]) == ("o", "r", 7)
        index = int(variables["cursor"] or 0)
        self.seen.append(f"graphql comments page {index}")
        nodes = [
            {
                "url": c["url"],
                "body": c["body"],
                "lastEditedAt": c["last_edited_at"],
                "includesCreatedEdit": c["includes_created_edit"],
                "author": {"login": c["login"], "__typename": c["type"]} if c["login"] else None,
            }
            for c in self.comments[index]
        ]
        more = index + 1 < len(self.comments)
        info = {"hasNextPage": more, "endCursor": str(index + 1) if more else None}
        return {"repository": {"pullRequest": {"comments": {"pageInfo": info, "nodes": nodes}}}}

    def install(self, patch: pytest.MonkeyPatch) -> None:
        patch.setattr(gate, "_get", self.get)
        patch.setattr(gate, "_graphql", self.graphql)


@pytest.fixture
def api_env(monkeypatch: pytest.MonkeyPatch) -> None:
    env = {"GH_TOKEN": "secret-token", "GITHUB_REPOSITORY": "o/r", "PR_NUMBER": "7", "EVENT_HEAD_SHA": HEAD}
    env["POLICY_SHA"] = BASE
    # Both API addresses are set: a GitHub Actions runner already has GITHUB_GRAPHQL_URL in its
    # environment, and a test must never be pointed at the real API.
    env.update(GITHUB_API_URL="https://api.test", GITHUB_GRAPHQL_URL="https://api.test/graphql")
    for name, value in env.items():
        monkeypatch.setenv(name, value)


def test_gate_main_pages_through_files_and_comments_and_reads_the_pull_twice(
    monkeypatch: pytest.MonkeyPatch, api_env: None, capsys
) -> None:
    # the locked file is on the second page of files, the approval on the second page of comments
    api = FakeApi(
        files=[[{"filename": "docs/a.md"}], LOCKED_CHANGE],
        comments=[[comment("nice work")], [approval(number=2)]],
        pulls=[pull(changed_files=2), pull(changed_files=2)],
    )
    api.install(monkeypatch)
    assert gate.main() == 0
    out = capsys.readouterr().out
    assert f"APPROVED: {OWNER} approved head {HEAD}" in out and "secret-token" not in out
    assert api.seen[0] == api.seen[-1] == "/repos/o/r/pulls/7"  # read before and after paging
    assert sum("#page1" in key for key in api.seen) == 1 and "graphql comments page 1" in api.seen
    assert api.pulls == []

    # the same pull request without the approval comment
    api = FakeApi(files=[LOCKED_CHANGE], comments=[[comment("nice work")]], pulls=[pull(), pull()])
    api.install(monkeypatch)
    assert gate.main() == 1
    assert f"{gate.APPROVAL_WORD} {HEAD}" in capsys.readouterr().out


def test_gate_main_fails_when_a_push_lands_between_the_two_reads(
    monkeypatch: pytest.MonkeyPatch, api_env: None, capsys
) -> None:
    # First read: a harmless head. While the files are paged, a push adds a locked change.
    # The file list already shows nothing locked, so only the second read can catch it.
    for second in (pull(head="d" * 40, changed_files=3), pull(base="e" * 40, changed_files=3)):
        api = FakeApi(files=[HARMLESS], comments=[[]], pulls=[pull(changed_files=3), second])
        api.install(monkeypatch)
        assert gate.main() == 1
        assert "moved while it was being read" in capsys.readouterr().out
    # and an approval of the old head does not carry over to a head pushed during the run
    api = FakeApi(files=[LOCKED_CHANGE], comments=[[approval()]], pulls=[pull(), pull(head="d" * 40)])
    api.install(monkeypatch)
    assert gate.main() == 1


# =============================================================================== wiring


def test_gate_scope_covers_the_manifest_scope_and_the_lock_machinery() -> None:
    for directory in tool.LOCKED_DIRS:
        assert f"{directory}/" in gate.LOCKED_PREFIXES
    for path in tool.LOCKED_FILES:
        assert path in gate.LOCKED_FILES
    assert "src/wrfcomp/__init__.py" in tool.LOCKED_FILES  # the package initializer is hashed too
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


def test_approval_workflow_runs_from_the_base_branch_reads_only_and_never_checks_out() -> None:
    workflow, text = _workflow("harness-lock.yml")
    triggers = workflow[True]  # YAML reads the key `on` as a boolean
    assert set(triggers) == {"pull_request_target"}
    assert {"opened", "reopened", "synchronize"} <= set(triggers["pull_request_target"]["types"])
    assert workflow["permissions"] == {}
    assert "concurrency" not in workflow  # no shared queue in which one run can displace another
    assert list(workflow["jobs"]) == ["harness-lock"]

    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    assert "write" not in code  # no write permission anywhere in the workflow
    assert "actions/checkout" not in code and "uses:" not in code
    assert "git " not in code and "head.ref" not in code and "head.repo" not in code
    assert "pull_request.title" not in code and "pull_request.body" not in code
    assert "-X DELETE" not in code and "-X POST" not in code and "labels/" not in code
    # expressions appear only in `env:` values, never inside a script
    job = workflow["jobs"]["harness-lock"]
    assert all("${{" not in step["run"] for step in job["steps"])
    assert "concurrency" not in job and "needs" not in job and "if" not in job

    assert job["name"] == "harness-lock"
    assert job["permissions"] == {"contents": "read", "pull-requests": "read", "issues": "read"}
    fetch, run = job["steps"]
    assert "contents/.github/harness_lock_gate.py?ref=${WORKFLOW_SHA}" in fetch["run"]
    assert fetch["env"]["WORKFLOW_SHA"] == "${{ github.sha }}"  # the base commit the workflow came from
    assert run["env"]["EVENT_HEAD_SHA"] == "${{ github.event.pull_request.head.sha }}"
    assert run["run"] == 'python3 -P "$RUNNER_TEMP/harness_lock_gate.py"'  # never without -P
    assert run["env"]["POLICY_SHA"] == "${{ github.sha }}"  # the same commit the policy was fetched from
    assert set(run["env"]) == {"GH_TOKEN", "PR_NUMBER", "EVENT_HEAD_SHA", "POLICY_SHA"}


def test_ci_workflow_keeps_only_the_hash_check_lint_and_tests() -> None:
    workflow, text = _workflow("ci.yml")
    assert set(workflow[True]) == {"push", "pull_request"}
    assert workflow["permissions"] == {"contents": "read"}
    assert list(workflow["jobs"]) == ["test"]
    runs = [step.get("run", "") for step in workflow["jobs"]["test"]["steps"]]
    assert "python3 -P scripts/harness_lock.py check" in runs  # never without -P
    assert "uv run ruff check" in runs and "uv run pytest -q" in runs
    assert gate.APPROVAL_WORD not in text and "pull_request_target" not in text
