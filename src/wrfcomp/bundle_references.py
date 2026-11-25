"""Reference inventory binding for the public per-year scoring data bundles."""

import hashlib
import json
import re
from typing import Any

MANIFEST_VERSION = 2
_PERIOD = re.compile(r"(?P<year>[0-9]{4})-(?:0[1-9]|1[0-2]|Q[1-4])")


def year_periods(inventory: dict[str, Any], year: int) -> dict[str, Any]:
    """Select whole monthly and quarterly records from the trusted inventory."""
    return {
        period: inventory["periods"][period]
        for period in sorted(inventory["periods"])
        if (match := _PERIOD.fullmatch(period)) and int(match["year"]) == year
    }


def periods_sha256(periods: dict[str, Any]) -> str:
    canonical = json.dumps(periods, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
