#!/usr/bin/env bash
# Refresh deploy-only branches from main:
#   frontend  — git subtree split of web/ (files at repo root)
#   backend   — main minus web/
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -f web/package.json ]]; then
  echo "Run from repository root (web/package.json missing)." >&2
  exit 1
fi

git checkout main
git pull origin main

echo "==> frontend branch (subtree split web/)"
git branch -D frontend 2>/dev/null || true
git subtree split --prefix=web -b frontend
git push origin frontend --force-with-lease

echo "==> backend branch (no web/)"
git checkout -B backend main
git rm -rf web/
git commit -m "chore(deploy): backend-only — no web/ (SPA on frontend branch)"
git push origin backend --force-with-lease

git checkout main
echo "Done. Pushed frontend + backend."
