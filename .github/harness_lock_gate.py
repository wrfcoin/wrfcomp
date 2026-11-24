"""Owner-approval gate for the locked harness. Run by .github/workflows/harness-lock.yml.

The workflow is triggered by `pull_request_target`, so GitHub runs the workflow file of the BASE
branch, and the workflow fetches THIS file from that same base commit through the API. Nothing from
the pull request is checked out or executed: the change under review cannot alter its own gate.

Rule
    A pull request that changes, adds, removes or renames a locked path passes only if
      1. it carries the label `owner-approved-harness-change`;
      2. the latest `labeled` event for that label was made by an approver listed below;
      3. that event is newer than the pull request's latest head change; and
      4. the run was not started by a new push (a push withdraws approval: a second job removes the
         label, and this gate fails that run outright).
    A pull request that touches no locked path passes.

It fails closed: an API error, a file list GitHub truncated, or a head that moved during the run all
count as "not approved".

What this cannot do: tell the owner from an agent acting through the owner's GitHub account. See
docs/HARNESS_LOCK.md. Standard library only.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from datetime import UTC, datetime

LABEL = "owner-approved-harness-change"
APPROVERS = ("tonykoop",)  # GitHub logins whose label counts. Bots and other accounts do not.

# The locked set: the hashed scope of scripts/harness_lock.py plus the lock machinery itself.
LOCKED_PREFIXES = ("src/wrfcomp/harness/", "configs/", ".github/")
LOCKED_FILES = (
    "HARNESS_LOCK.json",
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
# Timeline events that mean "the code under review may have changed".
HEAD_CHANGE_EVENTS = ("head_ref_force_pushed", "head_ref_restored", "base_ref_changed", "reopened")
MAX_LISTED_FILES = 3000  # the pull request files API stops there


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


def _time(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def latest_head_change(timeline: list[dict]) -> datetime | None:
    """The newest time at which the head (or the base) of the pull request changed.

    Commit times come from the commit itself, so the pusher sets them; force pushes and base
    changes carry GitHub's own time. A commit without a readable time is a reason to fail closed.
    """
    latest: datetime | None = None
    for event in timeline:
        kind = event.get("event")
        times: list[datetime | None] = []
        if kind == "committed":
            times = [_time((event.get(who) or {}).get("date")) for who in ("committer", "author")]
            if None in times:
                raise ValueError(f"commit {event.get('sha')} in the timeline has no readable date")
        elif kind in HEAD_CHANGE_EVENTS:
            times = [_time(event.get("created_at"))]
            if None in times:
                raise ValueError(f"timeline event {kind} has no readable time")
        for value in times:
            if value is not None and (latest is None or value > latest):
                latest = value
    return latest


def latest_label_event(timeline: list[dict]) -> dict | None:
    """The newest `labeled` event for the approval label, with its parsed time under "_time"."""
    found: dict | None = None
    for event in timeline:
        if event.get("event") != "labeled" or (event.get("label") or {}).get("name") != LABEL:
            continue
        when = _time(event.get("created_at"))
        if when is None:
            raise ValueError("a labeled event in the timeline has no readable time")
        if found is None or when >= found["_time"]:
            found = {**event, "_time": when}
    return found


def decide(
    *,
    pull: dict,
    files: list[dict],
    timeline: list[dict],
    action: str,
    event_head_sha: str,
    withdraw_result: str = "skipped",
) -> tuple[bool, list[str]]:
    """Return (approved, lines to print). Pure: all GitHub data is passed in."""
    lines: list[str] = []
    head_sha = (pull.get("head") or {}).get("sha")
    declared = pull.get("changed_files")
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

    def refuse(reason: str) -> tuple[bool, list[str]]:
        lines.append(f"NOT APPROVED: {reason}")
        lines.append(
            f"A change to the locked harness needs the label {LABEL}, applied by the project owner "
            "after the last push. See docs/HARNESS_LOCK.md."
        )
        return False, lines

    if withdraw_result not in ("success", "skipped"):
        return refuse(f"the job that withdraws approval on a new push ended as {withdraw_result!r}")
    if action == "synchronize":
        return refuse("this run was started by a new push; every push needs a fresh owner label")
    if not head_sha or head_sha != event_head_sha:
        return refuse(f"the head moved during this run (event {event_head_sha}, now {head_sha})")
    names = [(label or {}).get("name") for label in pull.get("labels") or []]
    if LABEL not in names:
        return refuse(f"the label {LABEL} is not on the pull request")
    try:
        label_event = latest_label_event(timeline)
        head_change = latest_head_change(timeline)
    except ValueError as exc:
        return refuse(str(exc))
    if label_event is None:
        return refuse("the label is present but the timeline has no labeled event for it")
    actor = (label_event.get("actor") or {}).get("login")
    if actor not in APPROVERS:
        return refuse(f"the label was applied by {actor!r}, who is not an approver ({', '.join(APPROVERS)})")
    if head_change is None:
        return refuse("the timeline shows no commit, so the approval cannot be tied to a head")
    if label_event["_time"] <= head_change:
        return refuse(
            f"the label ({label_event['_time'].isoformat()}) is not newer than the latest head change "
            f"({head_change.isoformat()})"
        )
    lines.append(
        f"APPROVED: {LABEL} applied by {actor} at {label_event['_time'].isoformat()}, after the latest "
        f"head change at {head_change.isoformat()}, for head {head_sha}."
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


def main() -> int:
    try:
        token = os.environ["GH_TOKEN"]
        repository = os.environ["GITHUB_REPOSITORY"]
        number = int(os.environ["PR_NUMBER"])
        action = os.environ["EVENT_ACTION"]
        event_head_sha = os.environ["EVENT_HEAD_SHA"]
        withdraw_result = os.environ.get("WITHDRAW_RESULT", "skipped")
        api = os.environ.get("GITHUB_API_URL", "https://api.github.com")
        base = f"{api}/repos/{repository}"
        pull, _ = _get(f"{base}/pulls/{number}", token)
        if not isinstance(pull, dict):
            raise ValueError("unexpected response for the pull request")
        files = get_all(f"{base}/pulls/{number}/files?per_page=100", token)
        timeline = get_all(f"{base}/issues/{number}/timeline?per_page=100", token)
        approved, lines = decide(
            pull=pull,
            files=files,
            timeline=timeline,
            action=action,
            event_head_sha=event_head_sha,
            withdraw_result=withdraw_result,
        )
    except (KeyError, ValueError, OSError, urllib.error.URLError) as exc:
        # Fail closed: without the facts there is no approval.
        print(f"NOT APPROVED: the gate could not read the pull request ({type(exc).__name__}: {exc})")
        return 1
    for line in lines:
        print(line)
    return 0 if approved else 1


if __name__ == "__main__":
    sys.exit(main())
