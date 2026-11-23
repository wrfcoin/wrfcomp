"""Harness lock: a manifest of sha256 hashes for the locked scoring kernel and its config.

`check` fails if any locked file differs from the manifest, or if a file was added or removed.
`update` rewrites the manifest; only do that in a pull request the project owner has approved
(see docs/HARNESS_LOCK.md). Standard library only, so it runs before dependencies are installed.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "HARNESS_LOCK.json"
LOCKED_DIRS = ("src/wrfcomp/harness", "configs")


def locked_files() -> dict[str, str]:
    out: dict[str, str] = {}
    for rel in LOCKED_DIRS:
        for path in sorted((ROOT / rel).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                out[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def check() -> list[str]:
    manifest = json.loads(MANIFEST.read_text())
    recorded: dict[str, str] = manifest["files"]
    actual = locked_files()
    problems = [f"changed: {p}" for p in sorted(recorded) if p in actual and actual[p] != recorded[p]]
    problems += [f"removed: {p}" for p in sorted(recorded) if p not in actual]
    problems += [f"added: {p}" for p in sorted(actual) if p not in recorded]
    return problems


def update(harness_version: str) -> None:
    body = {"harness_version": harness_version, "locked_dirs": list(LOCKED_DIRS), "files": locked_files()}
    MANIFEST.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")


def main(argv: list[str]) -> int:
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
        print(f"Wrote {MANIFEST.name} for harness_version {argv[2]}.")
        return 0
    print("usage: harness_lock.py check | update <harness_version>")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
