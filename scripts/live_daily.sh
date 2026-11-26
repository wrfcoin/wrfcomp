#!/usr/bin/env bash
# The daily live v4 run (lane I2, deliverable 5). The operator runs it, or a timer the owner approves;
# nothing installs a timer.
#
#   bash scripts/live_daily.sh 00 [YYYY-MM-DD]   the 00Z issue: fetch, build, commit/reveal (submit.sh),
#                                                publish (GitHub release asset), verify
#   bash scripts/live_daily.sh 12 [YYYY-MM-DD]   the 12Z cycle: fetch and build only (it keeps the learned
#                                                state current; 12Z is not committed)
#
# The date defaults to today (UTC). Run 00Z after about 04:30 UTC and 12Z after about 16:30 UTC, when
# the GEFS cycle is complete. Every step can be re-run: a kept cycle is not fetched again, a published
# issue is re-verified rather than rebuilt, submit.sh resumes, an uploaded asset is not uploaded again.
#
# Needs, for 00Z, in this shell:
#   WRFCOMP_FORECASTER    the throwaway forecaster address (0x..., from session-key keygen)
#   WRFCOIN_CLI           the wrfcoin CLI binary (built on the build host or from a release)
#   WRFCOIN_CLI_SHA256    its sha256
#   WRFCOIN_OPERATOR_TOKEN  the bearer the CLI uses (only the CLI and the record step see it)
#   gh, logged in with permission to create releases on wrfcoin/wrfcomp
#
# Transfer per day (both runs): GEFS ~184 MB per cycle x 2, METAR < 1 MB x 2, the archive ~2 MB up and
# ~2 MB back down for the check: about 372 MB/day. Node reads are local.
set -euo pipefail
umask 077
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy NO_PROXY no_proxy

CYCLE="${1:-}"
case "$CYCLE" in 00|12) ;; *) echo "usage: bash scripts/live_daily.sh 00|12 [YYYY-MM-DD]" >&2; exit 2;; esac
DAY="${2:-$(date -u +%F)}"
if ! [[ "$DAY" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
    echo "the date must be YYYY-MM-DD" >&2
    exit 2
fi
INIT="${DAY}T${CYCLE}"
ISSUE="${DAY}T${CYCLE}Z"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
LIVE="${WRFCOMP_DATA_DIR:-$HOME/wrfcomp-data}/live"
GH_REPO="wrfcoin/wrfcomp"

# the helper steps never see a bearer
py() {
    env -u WRFCOIN_OPERATOR_TOKEN -u WRFCOIN_READER_TOKEN uv run --frozen python -P "$@"
}

echo "== $ISSUE: airport readings (the last 30 hours)"
py scripts/live_v4.py fetch-metar --hours 30 --network
echo "== $ISSUE: GEFS cycle (skipped if kept)"
py scripts/live_v4.py fetch-gefs --init "$INIT" --network
echo "== $ISSUE: v4 store"
py scripts/live_v4.py build --init "$INIT"
if [ "$CYCLE" = 12 ]; then
    echo "done: the 12Z cycle is built (not committed)"
    exit 0
fi

: "${WRFCOMP_FORECASTER:?set WRFCOMP_FORECASTER (the throwaway forecaster address)}"
: "${WRFCOIN_CLI:?set WRFCOIN_CLI (the wrfcoin CLI binary)}"
: "${WRFCOIN_CLI_SHA256:?set WRFCOIN_CLI_SHA256 (its sha256)}"
CHAIN_DIR="$LIVE/chain/$ISSUE"
if [ ! -d "$CHAIN_DIR" ]; then
    echo "== $ISSUE: prepare the commit (nothing is sent)"
    py scripts/live_commit.py prepare --init "$INIT" --forecaster "$WRFCOMP_FORECASTER" \
        --signer-key "$LIVE/keys/forecaster.key" --cli "$WRFCOIN_CLI" --cli-sha256 "$WRFCOIN_CLI_SHA256"
fi
if [ ! -f "$CHAIN_DIR/receipt.json" ]; then
    echo "== $ISSUE: commit and reveal on testnet 888"
    if [ -e "$CHAIN_DIR/commit.claim" ]; then
        bash "$CHAIN_DIR/submit.sh" resume
    else
        bash "$CHAIN_DIR/submit.sh"
    fi
fi

echo "== $ISSUE: publish"
NAMES="$(py scripts/live_publish.py names --init "$INIT")"
read -r TAG ASSET ARCHIVE <<< "$NAMES"
if [ -z "${ARCHIVE:-}" ]; then
    echo "could not name the release asset" >&2
    exit 1
fi
py scripts/live_publish.py pack --init "$INIT"
if ! gh release view "$TAG" --repo "$GH_REPO" > /dev/null 2>&1; then
    gh release create "$TAG" --repo "$GH_REPO" --title "Live v4 forecasts ${DAY:0:7}" \
        --notes "Daily live v4 forecast stores on testnet 888, one asset per issue. Check one with scripts/live_publish.py check-reveal."
fi
if ! gh release view "$TAG" --repo "$GH_REPO" --json assets --jq '.assets[].name' | grep -qxF "$ASSET"; then
    gh release upload "$TAG" "$ARCHIVE" "$ARCHIVE.sha256" --repo "$GH_REPO"
fi

echo "== $ISSUE: verify the published copy against the chain"
DOWNLOAD="$LIVE/verify/download"
mkdir -p "$DOWNLOAD"
gh release download "$TAG" --repo "$GH_REPO" --pattern "$ASSET" --dir "$DOWNLOAD" --clobber
py scripts/live_publish.py check-reveal --init "$INIT" --archive "$DOWNLOAD/$ASSET"
echo "done: $LIVE/verify/$ISSUE.json"
