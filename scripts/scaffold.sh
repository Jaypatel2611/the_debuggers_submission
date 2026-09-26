#!/usr/bin/env bash
# Creates the PRD §21 folder structure inside the repo root. Idempotent —
# safe to re-run; only creates what's missing, never overwrites existing files.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE="$ROOT/code/business_entity_resolution"

mkdir -p \
  "$CODE/src/data" \
  "$CODE/src/preprocessing/vocab" \
  "$CODE/src/blocking" \
  "$CODE/src/features" \
  "$CODE/src/models" \
  "$CODE/src/validation" \
  "$CODE/src/inference" \
  "$CODE/src/evaluation" \
  "$CODE/tests" \
  "$ROOT/output"

# __init__.py per package so `src` is importable as a normal package tree.
for d in src src/data src/preprocessing src/blocking src/features src/models src/validation src/inference src/evaluation; do
  f="$CODE/$d/__init__.py"
  [ -f "$f" ] || : > "$f"
done

[ -f "$CODE/README.md" ]        || : > "$CODE/README.md"
[ -f "$CODE/requirements.txt" ] || : > "$CODE/requirements.txt"

echo "Scaffolded $CODE (nests inside <team_name>_submission.zip per PRD §2 — do not rename/move)."
