$ErrorActionPreference = "Stop"

Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class StockWisePower {
    [DllImport("kernel32.dll")]
    public static extern uint SetThreadExecutionState(uint flags);
}
"@

$continuous = [Convert]::ToUInt32("80000000", 16)
$systemRequired = [uint32]0x00000001
$logDirectory = "private_data/logs"
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
$logPath = Join-Path $logDirectory "comparative-top100-top20-2025.log"

try {
  [void][StockWisePower]::SetThreadExecutionState($continuous -bor $systemRequired)
  Start-Transcript -Path $logPath -Append | Out-Null
  python -m portfoliopilot.historical_council_backtest `
    --start 2024-01-01 `
    --evaluation-start 2025-01-02 `
    --end 2025-12-31 `
    --council-cache private_data/council-top100-v1-2025 `
    --selection-cache private_data/comparative-selections-v1-2025 `
    --output private_data/results/comparative-top100-top20-2025.json `
    --report private_data/results/comparative-top100-top20-2025.md `
    --comparative-top100 `
    --workers 4
  if ($LASTEXITCODE -ne 0) {
    throw "Comparative backtest failed with exit code $LASTEXITCODE"
  }
}
finally {
  try { Stop-Transcript | Out-Null } catch {}
  [void][StockWisePower]::SetThreadExecutionState($continuous)
}
