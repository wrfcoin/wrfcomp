"""Owner-approval gate for the locked harness. Run by .github/workflows/harness-lock.yml.

The workflow is triggered by `pull_request_target`, so GitHub runs the workflow file of the BASE
branch, and the workflow fetches THIS file from that same base commit through the API. Nothing from
the pull request is checked out or executed: the change under review cannot alter its own gate.

Rule
    A pull request that changes, adds, removes or renames a locked path passes only if an approver
    listed below has posted a comment on the pull request whose whole first line is exactly

        APPROVE-HARNESS-CHANGE <full 40-character sha of the pull request's current head commit>

    and that comment has never been edited. The approval names the commit, so it needs no dates and
    no label: after a new push the head is another commit and the old comment no longer matches.
    A pull request that touches no locked path passes.

    The pull request is read twice, before and after the file list and the comments are paged
    through. If its head or base moved in between, or the head is not the one the run was started
    for, the run fails, whatever the file list says.

It fails closed: an API error or a file list GitHub cut off count as "not approved".

What this cannot do: tell the owner from an agent acting through the owner's GitHub account. See
docs/HARNESS_LOCK.md. Standard library only. This workflow needs no write permission.
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request

APPROVAL_WORD = "APPROVE-HARNESS-CHANGE"
APPROVERS = ("tonykoop",)  # GitHub logins whose approval comment counts. Bots and other accounts do not.

# The locked set: the hashed scope of scripts/harness_lock.py plus the lock machinery itself.
LOCKED_PREFIXES = ("src/wrfcomp/harness/", "configs/", ".github/")
LOCKED_FILES = (
    "HARNESS_LOCK.json",
    "src/wrfcomp/__init__.py",
    "scripts/harness_lock.py",
    "tests/test_harness_lock.py",
    "scripts/check_promotion.py",
    "tests/test_check_promotion.py",
    "docs/HARNESS_LOCK.md",
    "docs/VERIFICATION_PROTOCOL.md",
    "experiments/REFERENCES.json",
    "experiments/climatology_2024-01.json",
    "experiments/climatology_2024-07.json",
    "experiments/gefs_2024-01.json",
    "experiments/gefs_2024-07.json",
    "experiments/persistence_2024-01.json",
    "experiments/persistence_2024-07.json",
)
MAX_LISTED_FILES = 3000  # the pull request files API stops there
SHA = re.compile(r"[0-9a-f]{40}")


def is_locked(path: str) -> bool:
    return path in LOCKED_FILES or path.startswith(LOCKED_PREFIXES)


def changed_paths(files: list[dict]) -> list[str]:
    """Every path a pull request touches: new names and, for renames, the previous names."""
    paths: set[str] = set()
    for item in files:
        for key in ("filename", "previous_filename"):
            value = item.get(key)
            if isinstance(value, str) and value:
                paths.add(value)
    return sorted(paths)


def _sha(pull: dict, side: str) -> str | None:
    value = (pull.get(side) or {}).get("sha")
    return value if isinstance(value, str) and SHA.fullmatch(value) else None


def approval_line(head_sha: str) -> str:
    return f"{APPROVAL_WORD} {head_sha}"


def approving_comment(comments: list[dict], head_sha: str) -> tuple[dict | None, list[str]]:
    """The comment that approves `head_sha`, if any, and notes on comments that came close."""
    wanted = approval_line(head_sha)
    notes: list[str] = []
    for comment in comments:
        body = comment.get("body")
        if not isinstance(body, str):
            continue
        first_line = body.split("\n", 1)[0].removesuffix("\r")
        if not first_line.startswith(APPROVAL_WORD):
            continue
        user = comment.get("user") or {}
        login, where = user.get("login"), comment.get("html_url") or f"comment {comment.get('id')}"
        if first_line != wanted:
            notes.append(f"{where}: first line is not exactly the approval of this head")
        elif login not in APPROVERS or user.get("type") != "User":
            notes.append(f"{where}: written by {login!r}, who is not an approver")
        elif not comment.get("created_at") or comment.get("created_at") != comment.get("updated_at"):
            notes.append(f"{where}: the comment was edited after it was posted")
        else:
            return comment, notes
    return None, notes


def decide(
    *,
    pull_before: dict,
    pull_after: dict,
    files: list[dict],
    comments: list[dict],
    event_head_sha: str,
) -> tuple[bool, list[str]]:
    """Return (passes, lines to print). Pure: all GitHub data is passed in.

    `pull_before` is the pull request as read before the file list and the comments were fetched,
    `pull_after` as read after. The decision is about one commit: the head both of them name.
    """
    lines: list[str] = []
    head, base = _sha(pull_after, "head"), _sha(pull_after, "base")

    def refuse(reason: str) -> tuple[bool, list[str]]:
        lines.append(f"NOT APPROVED: {reason}")
        return False, lines

    # One snapshot: the same head and base before and after paging, and the head this run is for.
    if head is None or base is None:
        return refuse("the pull request has no readable head or base commit")
    if _sha(pull_before, "head") != head or _sha(pull_before, "base") != base:
        return refuse(
            f"the pull request moved while it was being read (head {_sha(pull_before, 'head')} -> {head}, "
            f"base {_sha(pull_before, 'base')} -> {base}); run the check again"
        )
    if event_head_sha != head:
        return refuse(f"this run was started for head {event_head_sha}, the pull request is now at {head}")
    lines.append(f"Evaluated head {head} on base {base}.")

    declared = pull_after.get("changed_files")
    paths = changed_paths(files)
    locked = [p for p in paths if is_locked(p)]
    if not isinstance(declared, int) or declared != len(files) or declared >= MAX_LISTED_FILES:
        lines.append(
            f"GitHub reports {declared!r} changed files but listed {len(files)}; the list may be cut off. "
            "Treating the pull request as touching the locked harness."
        )
        locked = locked or ["<file list incomplete>"]
    if not locked:
        lines.append(f"No locked path among the {len(paths)} changed paths. Nothing to approve.")
        return True, lines

    lines.append("Locked paths touched:")
    lines += [f"  {json.dumps(path)}" for path in locked]
    comment, notes = approving_comment(comments, head)
    lines += [f"  note: {note}" for note in notes]
    if comment is None:
        lines.append(
            "A change to the locked harness needs a comment on this pull request by the project owner "
            "whose whole first line is exactly:"
        )
        lines.append(f"    {approval_line(head)}")
        lines.append("Then run this check again. A new push needs a new comment. See docs/HARNESS_LOCK.md.")
        return refuse(f"no approval comment for head {head}")
    lines.append(
        f"APPROVED: {comment['user']['login']} approved head {head} in "
        f"{comment.get('html_url') or 'comment ' + str(comment.get('id'))}."
    )
    return True, lines


# --------------------------------------------------------------------------- GitHub API


def _get(url: str, token: str) -> tuple[object, str | None]:
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "harness-lock-gate",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        body = json.loads(response.read().decode("utf-8"))
        link = response.headers.get("Link") or ""
    next_url = None
    for part in link.split(","):
        if 'rel="next"' in part:
            next_url = part[part.find("<") + 1 : part.find(">")]
    return body, next_url


def get_all(url: str, token: str) -> list[dict]:
    """Follow pagination to the end. Anything that is not a list of objects is an error."""
    out: list[dict] = []
    next_url: str | None = url
    pages = 0
    while next_url:
        body, next_url = _get(next_url, token)
        pages += 1
        if not isinstance(body, list) or pages > 200:
            raise ValueError(f"unexpected response while paging {url}")
        out += body
    return out


def _pull(base_url: str, number: int, token: str) -> dict:
    pull, _ = _get(f"{base_url}/pulls/{number}", token)
    if not isinstance(pull, dict):
        raise ValueError("unexpected response for the pull request")
    return pull


def main() -> int:
    try:
        token = os.environ["GH_TOKEN"]
        repository = os.environ["GITHUB_REPOSITORY"]
        number = int(os.environ["PR_NUMBER"])
        event_head_sha = os.environ["EVENT_HEAD_SHA"]
        api = os.environ.get("GITHUB_API_URL", "https://api.github.com")
        base_url = f"{api}/repos/{repository}"
        pull_before = _pull(base_url, number, token)
        files = get_all(f"{base_url}/pulls/{number}/files?per_page=100", token)
        comments = get_all(f"{base_url}/issues/{number}/comments?per_page=100", token)
        pull_after = _pull(base_url, number, token)  # read again: the snapshot must not have moved
        passes, lines = decide(
            pull_before=pull_before,
            pull_after=pull_after,
            files=files,
            comments=comments,
            event_head_sha=event_head_sha,
        )
    except (KeyError, ValueError, OSError, urllib.error.URLError) as exc:
        # Fail closed: without the facts there is no approval.
        print(f"NOT APPROVED: the gate could not read the pull request ({type(exc).__name__}: {exc})")
        return 1
    for line in lines:
        print(line)
    return 0 if passes else 1


if __name__ == "__main__":
    sys.exit(main())
