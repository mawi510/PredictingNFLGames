#!/bin/sh
# Starts a GitHub Actions workflow on main now. GitHub's own cron starts our jobs
# 4 to 11 hours late; a dispatch starts within seconds.
# Usage: dispatch-workflow.sh <workflow file name>, e.g. daily-results.yml
set -eu

workflow="$1"
token_file="${DISPATCH_TOKEN_FILE:-$HOME/.config/promatchpredict/gh-dispatch-token}"

curl -fsS -X POST \
  -H "Authorization: Bearer $(cat "$token_file")" \
  -H "Accept: application/vnd.github+json" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  -d '{"ref":"main"}' \
  "https://api.github.com/repos/mawi510/PredictingNFLGames/actions/workflows/$workflow/dispatches"

echo "$(date -u +%FT%TZ) dispatched $workflow"
