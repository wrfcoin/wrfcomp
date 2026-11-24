"""Harness lock: a manifest of the locked Git tree, the harness version and the scoring dependencies.

    python3 scripts/harness_lock.py check
    python3 scripts/harness_lock.py update <harness_version>

`check` compares the committed tree (`git ls-tree -r HEAD`) with `HARNESS_LOCK.json` and fails if

* a locked path was changed, added, removed or renamed (a rename shows as one removed and one added);
* a locked entry is anything but a regular, non-executable file (mode 100644): a symlink, an
  executable, a submodule;
* the working tree differs from the commit inside the locked scope (edited, deleted or untracked
  files, also ignored ones except Python bytecode caches);
* the manifest's harness version differs from the package version, or its scope from the scope
  written in this script;
* `uv.lock` resolves a scoring dependency (`scores`, `numpy`, `xarray`, `scipy`, `pandas`) to a
  version other than the recorded one. Other dependency changes are free.

`update` rewrites the manifest from the committed tree. Only do that in a pull request the project
owner has approved (docs/HARNESS_LOCK.md). Standard library and `git` only, so it runs before
dependencies are installed.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_NAME = "HARNESS_LOCK.json"
MANIFEST_VERSION = 2
REGULAR_FILE = "100644"

# The hashed scope. Changing it is itself a locked change: `check` compares it with the manifest.
LOCKED_DIRS = ("src/wrfcomp/harness", "configs")
LOCKED_FILES = (
    "docs/VERIFICATION_PROTOCOL.md",
    "scripts/check_promotion.py",
    "experiments/REFERENCES.json",
    "experiments/climatology_2024-01.json",
    "experiments/climatology_2024-07.json",
    "experiments/gefs_2024-01.json",
    "experiments/gefs_2024-07.json",
    "experiments/persistence_2024-01.json",
    "experiments/persistence_2024-07.json",
)
# Packages whose versions decide the numbers in a scorecard.
SCORING_DEPENDENCIES = ("numpy", "pandas", "scipy", "scores", "xarray")
VERSION_FILE = "src/wrfcomp/__init__.py"
IGNORED_PARTS = ("__pycache__",)


class LockError(RuntimeError):
    """The lock cannot be evaluated at all (no git, no manifest, unreadable lock file)."""


def _git(root: Path, *args: str) -> str:
    try:
        done = subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=True)
    except FileNotFoundError as exc:
        raise LockError("git is not installed; the harness lock is read from the Git tree") from exc
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.decode("utf-8", "replace").strip()
        raise LockError(f"git {' '.join(args[:2])} failed: {message}") from exc
    return done.stdout.decode("utf-8", "surrogateescape")


def scope_paths() -> list[str]:
    return [*LOCKED_DIRS, *LOCKED_FILES]


def tree_entries(root: Path = ROOT) -> dict[str, dict[str, str]]:
    """Committed entries inside the locked scope: path -> {"mode", "type", "blob"}."""
    out: dict[str, dict[str, str]] = {}
    listing = _git(root, "ls-tree", "-r", "-z", "HEAD", "--", *scope_paths())
    for record in listing.split("\0"):
        if not record:
            continue
        meta, path = record.split("\t", 1)
        mode, kind, object_id = meta.split(" ")
        out[path] = {"mode": mode, "type": kind, "blob": object_id}
    return out


def working_tree_changes(root: Path = ROOT) -> list[str]:
    """Paths inside the locked scope whose working tree state differs from the commit."""
    status = _git(
        root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        "--ignored=matching",
        "--",
        *scope_paths(),
    )
    problems = []
    records = status.split("\0")
    index = 0
    while index < len(records):
        record = records[index]
        index += 1
        if not record:
            continue
        code, path = record[:2], record[3:]
        if code[0] in "RC":  # a rename or copy carries its source path as the next record
            index += 1
        if any(part in IGNORED_PARTS for part in Path(path).parts):
            continue
        if code == "??":
            problems.append(f"untracked file in the locked scope: {path}")
        elif code == "!!":
            problems.append(f"ignored file in the locked scope: {path}")
        else:
            problems.append(f"uncommitted change in the locked scope: {path} (git status {code.strip()})")
    return sorted(problems)


def package_version(root: Path = ROOT) -> str:
    text = (root / VERSION_FILE).read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*"([^"]+)"\s*$', text, flags=re.MULTILINE)
    if not match:
        raise LockError(f"no __version__ in {VERSION_FILE}")
    return match.group(1)


def resolved_dependencies(root: Path = ROOT) -> dict[str, list[str]]:
    """Every version `uv.lock` resolves for each scoring dependency (one per Python range)."""
    try:
        lock = tomllib.loads((root / "uv.lock").read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise LockError(f"cannot read uv.lock: {exc}") from exc
    found: dict[str, set[str]] = {name: set() for name in SCORING_DEPENDENCIES}
    for package in lock.get("package", []):
        if package.get("name") in found and "version" in package:
            found[package["name"]].add(package["version"])
    return {name: sorted(versions) for name, versions in found.items()}


def build_manifest(harness_version: str, root: Path = ROOT) -> dict:
    entries = {
        path: {"mode": entry["mode"], "blob": entry["blob"]}
        for path, entry in sorted(tree_entries(root).items())
    }
    return {
        "manifest_version": MANIFEST_VERSION,
        "harness_version": harness_version,
        "scope": {"dirs": list(LOCKED_DIRS), "files": list(LOCKED_FILES)},
        "entries": entries,
        "scoring_dependencies": resolved_dependencies(root),
    }


def load_manifest(root: Path = ROOT) -> dict:
    try:
        manifest = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise LockError(f"cannot read {MANIFEST_NAME}: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("manifest_version") != MANIFEST_VERSION:
        raise LockError(f"{MANIFEST_NAME} is not a version {MANIFEST_VERSION} manifest")
    return manifest


def check(root: Path = ROOT) -> list[str]:
    """Every way the tree departs from the manifest. An empty list means the lock holds."""
    manifest = load_manifest(root)
    problems: list[str] = []

    version = package_version(root)
    if manifest.get("harness_version") != version:
        problems.append(
            f"version: the manifest locks harness {manifest.get('harness_version')!r} "
            f"but {VERSION_FILE} says {version!r}"
        )
    scope = {"dirs": list(LOCKED_DIRS), "files": list(LOCKED_FILES)}
    if manifest.get("scope") != scope:
        problems.append("scope: the manifest's scope differs from the scope in scripts/harness_lock.py")

    recorded: dict[str, dict[str, str]] = manifest.get("entries", {})
    actual = tree_entries(root)
    for path, entry in sorted(actual.items()):
        if entry["type"] != "blob" or entry["mode"] != REGULAR_FILE:
            problems.append(
                f"unsupported entry: {path} is a {entry['type']} with mode {entry['mode']}; "
                f"only regular files ({REGULAR_FILE}) may be in the locked scope"
            )
    for path in sorted(recorded):
        if path not in actual:
            problems.append(f"removed: {path}")
        elif actual[path]["blob"] != recorded[path].get("blob"):
            problems.append(f"changed: {path}")
        elif actual[path]["mode"] != recorded[path].get("mode"):
            problems.append(
                f"mode changed: {path} is {actual[path]['mode']}, locked as {recorded[path].get('mode')}"
            )
    problems += [f"added: {path}" for path in sorted(actual) if path not in recorded]
    for path in LOCKED_FILES:
        if path not in actual:
            problems.append(f"missing: locked file {path} is not in the committed tree")

    problems += working_tree_changes(root)

    pinned = manifest.get("scoring_dependencies", {})
    resolved = resolved_dependencies(root)
    for name in SCORING_DEPENDENCIES:
        if not resolved[name]:
            problems.append(f"dependency: uv.lock does not resolve {name}")
        elif resolved[name] != pinned.get(name):
            problems.append(
                f"dependency: uv.lock resolves {name} to {', '.join(resolved[name])}, "
                f"locked at {', '.join(pinned.get(name) or ['nothing'])}"
            )
    return sorted(set(problems))


def update(harness_version: str, root: Path = ROOT) -> None:
    manifest = build_manifest(harness_version, root)
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    (root / MANIFEST_NAME).write_text(text, encoding="utf-8")


def main(argv: list[str]) -> int:
    try:
        if len(argv) == 2 and argv[1] == "check":
            problems = check()
            for line in problems:
                print(line)
            if problems:
                print("Harness lock violated. See docs/HARNESS_LOCK.md.")
                return 1
            print("Harness lock OK.")
            return 0
        if len(argv) == 3 and argv[1] == "update":
            update(argv[2])
            print(f"Wrote {MANIFEST_NAME} for harness_version {argv[2]} from the committed tree.")
            return 0
    except LockError as exc:
        print(f"Harness lock cannot be checked: {exc}")
        return 1
    print("usage: harness_lock.py check | update <harness_version>")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
