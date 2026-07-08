# Production-style run: no auto-reload, multiple backend workers, frontend
# built once and served as static assets. This is "production-lite" for
# demo/portfolio purposes, not a hardened deployment — see
# docs/architecture/deployment.md for what a real production setup adds
# (Postgres, managed Qdrant/Neo4j, secrets management, load balancing).
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$backendPython = if (Test-Path "backend\.venv\Scripts\python.exe") { "backend\.venv\Scripts\python.exe" } else { "python" }
$workers = if ($env:WORKERS) { $env:WORKERS } else { "4" }

Write-Host "==> Building frontend"
Set-Location frontend
npm run build
Set-Location ..

Write-Host "==> Starting backend on :8000 ($workers workers, no reload)"
$backend = Start-Process -PassThru -NoNewWindow -FilePath $backendPython `
    -ArgumentList "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", $workers `
    -WorkingDirectory "backend"

try {
    Start-Sleep -Seconds 1
    Write-Host "==> Serving frontend/dist on :4173 (Ctrl+C stops both)"
    Set-Location frontend
    npx vite preview --port 4173 --host
} finally {
    Write-Host ""
    Write-Host "Stopping backend..."
    Stop-Process -Id $backend.Id -Force -ErrorAction SilentlyContinue
}
