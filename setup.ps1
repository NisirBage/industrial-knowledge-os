# One-command bootstrap: creates a backend virtualenv, installs both
# dependency sets, and copies .env templates without overwriting anything
# you've already configured. Safe to re-run.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "==> Checking prerequisites"
if (-not (Get-Command python -ErrorAction SilentlyContinue)) { throw "python not found. Install Python 3.13+." }
if (-not (Get-Command node -ErrorAction SilentlyContinue)) { throw "node not found. Install Node.js 20+." }

Write-Host "==> Backend: creating virtualenv (backend\.venv) and installing dependencies"
Set-Location backend
if (-not (Test-Path .venv)) {
    python -m venv .venv
}
& .\.venv\Scripts\pip.exe install --quiet --upgrade pip
& .\.venv\Scripts\pip.exe install --quiet -r requirements.txt
if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
    Write-Host "    created backend\.env (edit it to set LLM_PROVIDER=anthropic + ANTHROPIC_API_KEY for real narration)"
} else {
    Write-Host "    backend\.env already exists, left untouched"
}
Set-Location ..

Write-Host "==> Frontend: installing dependencies"
Set-Location frontend
npm install --silent
if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
    Write-Host "    created frontend\.env"
} else {
    Write-Host "    frontend\.env already exists, left untouched"
}
Set-Location ..

Write-Host ""
Write-Host "Setup complete. Next steps:"
Write-Host "  .\run-dev.ps1          # start backend (:8000) + frontend (:5173)"
Write-Host "  docker compose up -d   # optional: real Qdrant + Neo4j (falls back automatically without it)"
Write-Host "  cd backend; .\.venv\Scripts\python.exe scripts\seed_demo_data.py   # optional: seed demo data"
