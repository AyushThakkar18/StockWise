$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $projectRoot

Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class StockWiseKronosPower {
    [DllImport("kernel32.dll")]
    public static extern uint SetThreadExecutionState(uint flags);
}
"@

$continuous = [Convert]::ToUInt32("80000000", 16)
$systemRequired = [uint32]1
$logDirectory = Join-Path $projectRoot "private_data\logs"
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
$logPath = Join-Path $logDirectory "per-stock-kronos-2025.log"
$statusPath = Join-Path $logDirectory "per-stock-kronos-2025.status"
$pidPath = Join-Path $logDirectory "per-stock-kronos-2025.pid"
Set-Content $pidPath $PID
Set-Content $statusPath "RUNNING $(Get-Date -Format o) PID=$PID"

try {
    [void][StockWiseKronosPower]::SetThreadExecutionState($continuous -bor $systemRequired)
    Start-Transcript -Path $logPath -Append | Out-Null
    python -m portfoliopilot.historical_council_backtest `
        --start 2024-01-01 `
        --evaluation-start 2025-01-02 `
        --end 2025-12-31 `
        --per-stock-scoring `
        --kronos `
        --kronos-device cuda `
        --workers 4 `
        --output private_data/results/per-stock-kronos-2025.json `
        --report private_data/results/per-stock-kronos-2025.md
    if ($LASTEXITCODE -ne 0) { throw "Backtest exited with code $LASTEXITCODE" }
    Set-Content $statusPath "COMPLETED $(Get-Date -Format o) PID=$PID"
}
catch {
    Set-Content $statusPath "FAILED $(Get-Date -Format o) PID=$PID $($_.Exception.Message)"
    throw
}
finally {
    try { Stop-Transcript | Out-Null } catch {}
    [void][StockWiseKronosPower]::SetThreadExecutionState($continuous)
}
