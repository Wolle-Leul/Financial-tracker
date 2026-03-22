# Refresh deploy-only branches from main:
#   frontend  — git subtree split of web/ (files at repo root)
#   backend   — main minus web/
$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')

if (-not (Test-Path 'web/package.json')) {
  throw 'Run from repository root (web/package.json missing).'
}

git checkout main
git pull origin main

Write-Host '==> frontend branch (subtree split web/)'
git branch -D frontend 2>$null
git subtree split --prefix=web -b frontend
git push origin frontend --force-with-lease

Write-Host '==> backend branch (no web/)'
git checkout -B backend main
git rm -rf web/
git commit -m 'chore(deploy): backend-only — no web/ (SPA on frontend branch)'
git push origin backend --force-with-lease

git checkout main
Write-Host 'Done. Pushed frontend + backend.'
