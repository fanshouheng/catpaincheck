param([string]$Python = '')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot
if (!$Python) {
    $bundledPython = Join-Path $env:USERPROFILE '.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
    if (Test-Path -LiteralPath $bundledPython) { $Python = $bundledPython }
    else { $Python = (Get-Command python -ErrorAction Stop).Source }
}
if (!(Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
    & $Python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
}
$projectPython = Join-Path $projectRoot '.venv/Scripts/python.exe'
& $projectPython -m pip install -r backend/requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Backend dependencies failed.' }
& $projectPython scripts/setup-localization.py
if ($LASTEXITCODE -ne 0) { throw 'Localization model download or verification failed.' }
& $projectPython scripts/setup-kittycheck.py
if ($LASTEXITCODE -ne 0) { throw 'Automatic FGS model download or verification failed.' }
& $projectPython scripts/export-yolo-model.py --check
if ($LASTEXITCODE -ne 0) {
    # Export tools are local-only. Inference uses OpenCV and LiteRT.
    & $projectPython -m pip install -r backend/requirements-model.txt
    if ($LASTEXITCODE -ne 0) { throw 'CPU export dependencies failed.' }
    & $projectPython -m pip install -r backend/requirements-export.txt
    if ($LASTEXITCODE -ne 0) { throw 'ONNX export dependencies failed.' }
    # The headless OpenCV wheel is already installed; do not install a second cv2 wheel.
    & $projectPython -m pip install --no-deps ultralytics==8.3.221
    if ($LASTEXITCODE -ne 0) { throw 'Ultralytics export installation failed.' }
    & $projectPython scripts/export-yolo-model.py
    if ($LASTEXITCODE -ne 0) { throw 'YOLO download or export verification failed.' }
}
Push-Location frontend
try {
    npm.cmd ci --no-audit --no-fund
    if ($LASTEXITCODE -ne 0) { throw 'Frontend dependency installation failed.' }
    npm.cmd run build
    if ($LASTEXITCODE -ne 0) { throw 'Frontend build failed.' }
} finally { Pop-Location }
Write-Host 'Setup finished. Run scripts/start.ps1 or double-click the launcher.'
