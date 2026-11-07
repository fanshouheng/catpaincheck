$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pidFile = Join-Path $projectRoot 'data/server.pid'
if (!(Test-Path -LiteralPath $pidFile)) {
    Write-Host 'No recorded CatPainCheck process.'
    exit 0
}
$recordedPid = [int](Get-Content -LiteralPath $pidFile -Raw).Trim()
$processInfo = Get-CimInstance Win32_Process -Filter "ProcessId=$recordedPid" -ErrorAction SilentlyContinue
if ($null -ne $processInfo) {
    $expectedPython = (Join-Path $projectRoot '.venv/Scripts/python.exe').Replace('/', '\')
    if ($processInfo.ExecutablePath -ne $expectedPython -or $processInfo.CommandLine -notmatch 'uvicorn app.main:app') {
        throw 'The recorded PID belongs to a different process; no process was stopped.'
    }
    # A Windows venv launcher can own a second Python process. Stop the verified
    # project process tree so the web server cannot remain behind on its port.
    & taskkill.exe /PID "$recordedPid" /T /F | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'The project process tree could not be stopped.' }
    Write-Host 'CatPainCheck stopped.'
}
Remove-Item -LiteralPath $pidFile
