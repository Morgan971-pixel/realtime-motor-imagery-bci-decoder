#!/usr/bin/env bash
# Exercises scripts/hooks/ in a throwaway repo. Run from anywhere: scripts/test_hooks.sh
set -euo pipefail
hook="$(cd "$(dirname "$0")" && pwd)/hooks/pre-commit"
guard="$(cd "$(dirname "$0")" && pwd)/check_tracked.sh"
policy="$(cd "$(dirname "$0")/.." && pwd)/.gitignore"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
cd "$tmp"
git init -q
git config user.name test
git config user.email test@example.com

expect() {  # usage: expect allow|block "description"
  if "$hook" >/dev/null 2>&1; then got=allow; else got=block; fi
  if [ "$got" != "$1" ]; then echo "FAIL  expected $1: $2"; exit 1; fi
  echo "ok    $1  $2"
  git rm -rq --cached . >/dev/null 2>&1 || true
}

mkdir -p golden/sub data
echo a > golden/ok.csv;     git add golden/ok.csv;     expect allow "golden/ok.csv"
echo j > golden/model.json; git add golden/model.json; expect allow "golden/model.json"
echo b > rec.CSV;           git add rec.CSV;           expect block "rec.CSV (uppercase extension)"
echo c > golden/sub/x.csv;  git add golden/sub/x.csv;  expect block "golden/sub/x.csv (nested)"
echo d > data/notes.txt;    git add data/notes.txt;    expect block "data/notes.txt (any file under data/)"
mkdir -p public-data/eegmmidb
echo s > public-data/eegmmidb/SHA256SUMS.txt; git add public-data/eegmmidb/SHA256SUMS.txt; expect block "public-data/eegmmidb/SHA256SUMS.txt (any file under public-data/)"
echo e > s01.mat;           git add s01.mat;           expect block "s01.mat"
echo f > séance.csv;        git add séance.csv;        expect block "séance.csv (non-ASCII name)"
mkdir -p models results
echo m > models/s01.json;      git add models/s01.json;      expect block "models/s01.json (any file under models/)"
echo m > results/s01_model.json; git add results/s01_model.json; expect block "results/s01_model.json (trained model JSON)"
echo p > CMakePresets.json;    git add CMakePresets.json;    expect allow "CMakePresets.json (ordinary JSON)"
mkdir -p golden/invalid
echo i > golden/invalid/model_bad.json; git add golden/invalid/model_bad.json; expect allow "golden/invalid/model_bad.json (shared invalid fixture)"

git add golden/ok.csv
git commit -qm base --no-verify
git mv golden/ok.csv leaked.csv;                       expect block "rename out of golden/"

# scripts/check_tracked.sh is the CI guard for files that skipped the hook (--no-verify, git add -f).
# core.ignorecase=false here, as on Linux, so the uppercase case proves the guard forces it on.
tracked() {  # usage: tracked allow|block "description"
  if "$guard" >/dev/null 2>&1; then got=allow; else got=block; fi
  if [ "$got" != "$1" ]; then echo "FAIL  expected guard $1: $2"; exit 1; fi
  echo "ok    guard $1  $2"
  git rm -rq --cached . >/dev/null 2>&1 || true
}
git config core.ignorecase false
cp "$policy" .gitignore
echo k > leak.csv
git add .gitignore golden/model.json CMakePresets.json; tracked allow "golden/model.json and CMakePresets.json"
git add -f leak.csv;               tracked block "leak.csv added with -f"
git add -f rec.CSV;                tracked block "rec.CSV added with -f (uppercase extension)"
git add -f golden/sub/x.csv;       tracked block "golden/sub/x.csv added with -f (nested)"
git add -f data/notes.txt;         tracked block "data/notes.txt added with -f"
git add -f public-data/eegmmidb/SHA256SUMS.txt; tracked block "public-data/eegmmidb/SHA256SUMS.txt added with -f"
git add -f results/s01_model.json; tracked block "results/s01_model.json added with -f"

# commit-msg and pre-push hand off to user-wide hooks (git config --global core.hooksPath).
# GIT_CONFIG_GLOBAL points git at a throwaway global config so the real one is never read.
hooks_dir="$(cd "$(dirname "$hook")" && pwd)"
handoff() {  # usage: handoff allow|block "description" hook-name [stdin]
  if printf '%s' "${4:-}" | GIT_CONFIG_GLOBAL="$tmp/gitconfig" "$hooks_dir/$3" msg >/dev/null 2>&1; then got=allow; else got=block; fi
  if [ "$got" != "$1" ]; then echo "FAIL  expected handoff $1: $2"; exit 1; fi
  echo "ok    handoff $1  $2"
}
: > "$tmp/gitconfig"
handoff allow "commit-msg with no user-wide hooks" commit-msg
mkdir -p "$tmp/user-hooks"
printf '#!/bin/sh\nexit 1\n' > "$tmp/user-hooks/commit-msg"
printf '#!/bin/sh\ngrep -q refs/heads/main\n' > "$tmp/user-hooks/pre-push"
chmod +x "$tmp/user-hooks/commit-msg" "$tmp/user-hooks/pre-push"
git config --file "$tmp/gitconfig" core.hooksPath "$tmp/user-hooks"
handoff block "commit-msg runs a rejecting user-wide hook" commit-msg
handoff allow "pre-push passes its stdin to the user-wide hook" pre-push "refs/heads/main abc refs/heads/main def"
handoff block "pre-push with stdin the user-wide hook rejects" pre-push "refs/heads/other abc refs/heads/other def"
git config --file "$tmp/gitconfig" core.hooksPath "$hooks_dir"
handoff allow "user-wide path pointing back at scripts/hooks (no loop)" commit-msg
echo "all hook checks passed"
