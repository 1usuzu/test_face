# DeepTrust — local setup (Windows, repo root = parent of this script's folder)
# Run:  powershell -ExecutionPolicy Bypass -File .\scripts\setup-local.ps1
# Skip Node installs:  .\scripts\setup-local.ps1 -SkipNpm

param(
    [switch]$SkipNpm
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot

function Write-Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }

Write-Step "Repo root: $RepoRoot"

$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) {
    Write-Host "Python not found on PATH. Install Python 3.11 or 3.12 and retry." -ForegroundColor Red
    exit 1
}

$ver = & python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
Write-Host "Python version: $ver (recommended: 3.11 or 3.12)"

$venvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$venvPip = Join-Path $RepoRoot ".venv\Scripts\pip.exe"

if (-not (Test-Path $venvPython)) {
    Write-Step "Creating .venv"
    python -m venv .venv
}

Write-Step "Installing backend dependencies (this may take a few minutes)"
& $venvPip install --upgrade pip
& $venvPip install -r (Join-Path $RepoRoot "backend\requirements.txt")

Write-Step "Verifying PyTorch in venv"
& $venvPython -c "import torch; print('torch OK:', torch.__version__)"

$backendEnv = Join-Path $RepoRoot "backend\.env"
$backendExample = Join-Path $RepoRoot "backend\.env.example"
if (-not (Test-Path $backendEnv) -and (Test-Path $backendExample)) {
    Write-Step "Copying backend\.env.example -> backend\.env (edit SERVER_PRIVATE_KEY before starting API)"
    Copy-Item $backendExample $backendEnv
    Write-Host "Created backend\.env — set SERVER_PRIVATE_KEY (and dev flags if needed)." -ForegroundColor Yellow
}

Write-Step "npm install (blockchain + frontend)"
if (-not $SkipNpm) {
    Push-Location (Join-Path $RepoRoot "blockchain")
    npm install
    Pop-Location
    Push-Location (Join-Path $RepoRoot "frontend")
    npm install
    Pop-Location
} else {
    Write-Host "Skipped npm (pass -SkipNpm only if you already ran npm install)." -ForegroundColor DarkYellow
}

Write-Step "Done"
Write-Host @"

Next steps (backend):
  1. Edit backend\.env — set SERVER_PRIVATE_KEY (use Hardhat account #0 key for local demo).
     For dev fallback you may set ALLOW_INSECURE_DEV_KEY=true and INSECURE_DEV_PRIVATE_KEY=<same key>.
  2. Start API (from repo root):
       cd backend
       ..\.venv\Scripts\python.exe -m uvicorn api:app --reload --port 8000
  3. Check: http://127.0.0.1:8000/api/health  (detector should be true)

Frontend (after backend works):
  4. Copy frontend\.env.example to frontend\.env and set VITE_* (contract address after deploy).
  5. cd frontend; npm run dev

Skip Node next time:  .\scripts\setup-local.ps1 -SkipNpm
"@ -ForegroundColor Green
