$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$logDirectory = Join-Path $projectRoot "private_data\logs"
$statusPath = Join-Path $logDirectory "per-stock-kronos-2025.status"
$pidPath = Join-Path $logDirectory "per-stock-kronos-2025.pid"
$logPath = Join-Path $logDirectory "per-stock-kronos-2025.log"

if (Test-Path $statusPath) { Get-Content $statusPath } else { Write-Host "NOT STARTED" }
if (Test-Path $pidPath) {
    $processId = [int](Get-Content $pidPath)
    $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
    if ($process) { Write-Host "PROCESS ACTIVE PID=$processId" }
    else { Write-Host "PROCESS NOT ACTIVE" }
}
if (Test-Path $logPath) {
    Write-Host "`nLatest progress:"
    Get-Content $logPath -Tail 25
}
