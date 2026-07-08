# Starts backend (:8000, auto-reload) and frontend (:5173, HMR) together.
# Ctrl+C stops both.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$backendPython = if (Test-Path "backend\.venv\Scripts\python.exe") { "backend\.venv\Scripts\python.exe" } else { "python" }

Write-Host "==> Starting backend on :8000"
$backend = Start-Process -PassThru -NoNewWindow -FilePath $backendPython `
    -ArgumentList "-m", "uvicorn", "app.main:app", "--reload", "--port", "8000" `
    -WorkingDirectory "backend"

try {
    Start-Sleep -Seconds 1
    Write-Host "==> Starting frontend on :5173 (Ctrl+C stops both)"
    Set-Location frontend
    npm run dev
} finally {
    Write-Host ""
    Write-Host "Stopping backend..."
    Stop-Process -Id $backend.Id -Force -ErrorAction SilentlyContinue
}
