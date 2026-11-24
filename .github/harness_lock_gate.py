"""Owner-approval gate for the locked harness. Run by .github/workflows/harness-lock.yml.

The workflow is triggered by `pull_request_target`, so GitHub runs the workflow file of the BASE
branch, and the workflow fetches THIS file from that same base commit through the API. Nothing from
the pull request is checked out or executed: the change under review cannot alter its own gate.

Rule
    A pull request that changes, adds, removes or renames a locked path passes only if, at the time
    this check runs, there is a comment on the pull request by an approver listed below whose whole
    first line is exactly

        APPROVE-HARNESS-CHANGE <full 40-character sha of the pull request's current head commit>

    and GitHub's edit record for that comment is empty (`lastEditedAt` is null and the creation
    was not itself an edit). The approval names the commit, so it needs no dates and no label:
    after a new push the head is another commit and the old comment no longer matches.
    A pull request that touches no locked path passes.

    The decision is about one head on one base, under one policy:
      * the pull request is read twice, before and after the file list and the comments are paged
        through; if its head or base moved in between, the run fails;
      * the head must be the one the run was started for;
      * the base must be the commit this policy file was fetched from (`POLICY_SHA`). A run, or a
        re-run, whose policy is older or newer than the pull request's base fails: update the
        pull request onto the current base branch and run again.
    All three are checked before either way of passing.

    A file list that GitHub cut off is treated as touching the locked set: the pull request then
    needs the owner's approval of its head like any locked change.

    Approval is observed when the check runs. Editing or deleting the comment afterwards does not
    start a new run and does not take back a result already given.

An API error is "not approved".

What this cannot do: tell the owner from an agent acting through the owner's GitHub account. See
docs/HARNESS_LOCK.md. Standard library only. This workflow needs no write permission.
"""

import sys

# Before any other import. Running a script by path puts its directory (scripts/, which is not a
# locked directory) first on sys.path, ahead of the standard library and of every installed
# package. A file planted there (scripts/tomllib.py, scripts/scores/) would then be imported in
# place of the real module. `python -P` leaves the script directory off the path; without it this
# script does not run. `sys` is built in and cannot be shadowed.
if __name__ == "__main__" and not getattr(sys.flags, "safe_path", False):
    sys.exit(
        f"REFUSED: run this script with Python's -P option (python -P {sys.argv[0]} ...), so that files "
        "next to it cannot shadow the standard library or the scoring packages. See docs/HARNESS_LOCK.md."
    )

import json
import os
import re
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
    # the Gate 1 reference scorecards: three models, every month of the 2024 validation split
    *(
        f"experiments/{model}_2024-{month:02d}.json"
        for month in range(1, 13)
        for model in ("climatology", "gefs", "persistence")
    ),
    # the quarterly reference scorecards (promotion rule 2): three models, the four quarters of 2024
    *(
        f"experiments/{model}_2024-Q{quarter}.json"
        for quarter in range(1, 5)
        for model in ("climatology", "gefs", "persistence")
    ),
    # the 2023 confirmation references (owner ruling 2026-11-24): three models, the four quarters of
    # 2023, built from truth at or before 2022-12-17T18Z only
    *(
        f"experiments/{model}_2023-Q{quarter}.json"
        for quarter in range(1, 5)
        for model in ("climatology", "gefs", "persistence")
    ),
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
    """The comment that approves `head_sha`, if any, and notes on comments that came close.

    `comments` are in the form `fetch_comments` returns: body, url, login, type (GitHub's
    `__typename` of the author), last_edited_at and includes_created_edit.
    """
    wanted = approval_line(head_sha)
    notes: list[str] = []
    for comment in comments:
        body = comment.get("body")
        if not isinstance(body, str):
            continue
        first_line = body.split("\n", 1)[0].removesuffix("\r")
        if not first_line.startswith(APPROVAL_WORD):
            continue
        login, where = comment.get("login"), comment.get("url") or "a comment"
        never_edited = (
            "last_edited_at" in comment
            and comment["last_edited_at"] is None
            and comment.get("includes_created_edit") is False
        )
        if first_line != wanted:
            notes.append(f"{where}: first line is not exactly the approval of this head")
        elif login not in APPROVERS or comment.get("type") != "User":
            notes.append(f"{where}: written by {login!r}, who is not an approver")
        elif not never_edited:
            notes.append(f"{where}: GitHub records an edit of this comment (or no edit record at all)")
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
    policy_sha: str,
) -> tuple[bool, list[str]]:
    """Return (passes, lines to print). Pure: all GitHub data is passed in.

    `pull_before` is the pull request as read before the file list and the comments were fetched,
    `pull_after` as read after. `policy_sha` is the base-branch commit this policy was fetched
    from. The decision is about one head on one base under that policy.
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
    if policy_sha != base:
        return refuse(
            f"this run's policy comes from base-branch commit {policy_sha}, but the pull request's base is "
            f"{base}. Update the pull request onto the current base branch, then run the check again"
        )
    lines.append(f"Evaluated head {head} on base {base}, with the policy of that base.")

    declared = pull_after.get("changed_files")
    paths = changed_paths(files)
    locked = [p for p in paths if is_locked(p)]
    if not isinstance(declared, int) or declared != len(files) or declared >= MAX_LISTED_FILES:
        lines.append(
            f"GitHub reports {declared!r} changed files but listed {len(files)}; the list may be cut off. "
            "It is treated as touching the locked harness: it needs the owner's approval of this head."
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
        f"APPROVED: {comment['login']} approved head {head} in {comment.get('url') or 'a comment'} "
        "(as seen now; a later edit or deletion of the comment does not re-run this check)."
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


COMMENTS_QUERY = """
query($owner: String!, $name: String!, $number: Int!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      comments(first: 100, after: $cursor) {
        pageInfo { hasNextPage endCursor }
        nodes { url body lastEditedAt includesCreatedEdit author { __typename login } }
      }
    }
  }
}
"""


def _graphql(url: str, token: str, query: str, variables: dict) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps({"query": query, "variables": variables}).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "harness-lock-gate",
        },
        method="POST",  # GraphQL reads are POST requests; this query changes nothing
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        body = json.loads(response.read().decode("utf-8"))
    if not isinstance(body, dict) or body.get("errors") or not isinstance(body.get("data"), dict):
        raise ValueError(f"unexpected GraphQL response: {str(body)[:200]}")
    return body["data"]


def fetch_comments(graphql_url: str, repository: str, number: int, token: str) -> list[dict]:
    """Every conversation comment of the pull request, with GitHub's own edit record for each."""
    owner, name = repository.split("/", 1)
    out: list[dict] = []
    cursor: str | None = None
    for _ in range(200):
        variables = {"owner": owner, "name": name, "number": number, "cursor": cursor}
        data = _graphql(graphql_url, token, COMMENTS_QUERY, variables)
        try:
            page = data["repository"]["pullRequest"]["comments"]
            nodes, info = page["nodes"], page["pageInfo"]
            for node in nodes:
                author = node.get("author") or {}
                out.append(
                    {
                        "url": node["url"],
                        "body": node["body"],
                        "login": author.get("login"),
                        "type": author.get("__typename"),
                        "last_edited_at": node["lastEditedAt"],
                        "includes_created_edit": node["includesCreatedEdit"],
                    }
                )
            if not info["hasNextPage"]:
                return out
            cursor = info["endCursor"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"unexpected GraphQL response shape: {exc}") from exc
    raise ValueError("too many pages of comments")


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
        policy_sha = os.environ["POLICY_SHA"]
        api = os.environ.get("GITHUB_API_URL", "https://api.github.com")
        graphql_url = os.environ.get("GITHUB_GRAPHQL_URL", f"{api}/graphql")
        base_url = f"{api}/repos/{repository}"
        pull_before = _pull(base_url, number, token)
        files = get_all(f"{base_url}/pulls/{number}/files?per_page=100", token)
        comments = fetch_comments(graphql_url, repository, number, token)
        pull_after = _pull(base_url, number, token)  # read again: the snapshot must not have moved
        passes, lines = decide(
            pull_before=pull_before,
            pull_after=pull_after,
            files=files,
            comments=comments,
            event_head_sha=event_head_sha,
            policy_sha=policy_sha,
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
