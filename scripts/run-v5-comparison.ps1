param(
    [int]$IntervalSeconds = 900
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

Add-Type @"
using System.Runtime.InteropServices;
public static class StockWiseV5Power {
    [DllImport("kernel32.dll", CharSet = CharSet.Auto, SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint flags);
}
"@

$continuous = [uint32]2147483648
$systemRequired = [uint32]1
[void][StockWiseV5Power]::SetThreadExecutionState($continuous -bor $systemRequired)

try {
    python -m portfoliopilot.live_service `
        --database private_data/live/v5-comparison.db `
        --initial-cash 1027.242295324810314071695432 `
        --device cuda `
        --interval $IntervalSeconds
}
finally {
    [void][StockWiseV5Power]::SetThreadExecutionState($continuous)
}
