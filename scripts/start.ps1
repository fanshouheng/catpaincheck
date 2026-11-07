param([int]$Port = 8765, [switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv/Scripts/python.exe'
$distPath = Join-Path $projectRoot 'frontend/dist/index.html'
if (!(Test-Path -LiteralPath $pythonPath) -or !(Test-Path -LiteralPath $distPath)) {
    throw 'Dependencies or frontend build are missing. Run scripts/setup.ps1 first.'
}
$baseUrl = "http://127.0.0.1:$Port"
try {
    $existing = Invoke-RestMethod "$baseUrl/api/health" -TimeoutSec 2
    if ($existing.service -ne 'catpaincheck') { throw 'Port is occupied by another application.' }
    if (!$NoBrowser) { Start-Process $baseUrl }
    Write-Host "CatPainCheck is already running: $baseUrl"
    exit 0
} catch {
    if ($_.Exception.Message -eq 'Port is occupied by another application.') { throw }
}
$dataPath = Join-Path $projectRoot 'data'
New-Item -ItemType Directory -Force -Path $dataPath | Out-Null
$serverProcess = Start-Process -FilePath $pythonPath -WorkingDirectory $projectRoot `
    -ArgumentList @('-m', 'uvicorn', 'app.main:app', '--app-dir', 'backend', '--host', '127.0.0.1', '--port', "$Port") `
    -WindowStyle Hidden -RedirectStandardOutput (Join-Path $dataPath 'server.log') `
    -RedirectStandardError (Join-Path $dataPath 'server-error.log') -PassThru
Set-Content -LiteralPath (Join-Path $dataPath 'server.pid') -Value $serverProcess.Id -Encoding ascii
for ($attempt = 0; $attempt -lt 60; $attempt++) {
    if ($serverProcess.HasExited) {
        throw "Server exited. See $(Join-Path $dataPath 'server-error.log')"
    }
    try {
        $health = Invoke-RestMethod "$baseUrl/api/health" -TimeoutSec 1
        if ($health.service -eq 'catpaincheck') {
            Write-Host "CatPainCheck: $baseUrl"
            if (!$NoBrowser) { Start-Process $baseUrl }
            exit 0
        }
    } catch { }
    Start-Sleep -Milliseconds 500
}
throw "Server did not become ready. See $(Join-Path $dataPath 'server-error.log')"
