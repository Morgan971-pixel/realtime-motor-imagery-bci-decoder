#!/usr/bin/env bash
# CI data guard: fails if any tracked file matches the data policy in .gitignore. It catches what
# the pre-commit hook cannot: files committed with --no-verify or added with `git add -f`.
# Matching ignores letter case on every OS, like the hook (rec.CSV is data too).
# Run from the repository root: scripts/check_tracked.sh
set -euo pipefail
bad=$(git -c core.ignorecase=true -c core.quotepath=off ls-files --cached --ignored --exclude-standard)
if [ -n "$bad" ]; then
  echo "Blocked: tracked files match the data policy in .gitignore:" >&2
  echo "$bad" >&2
  exit 1
fi
echo "no tracked file matches the data policy"
