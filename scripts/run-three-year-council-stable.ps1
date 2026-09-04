param(
    [ValidateSet('cuda', 'cpu')]
    [string]$Device = 'cuda'
)

$ErrorActionPreference = 'Stop'
$project = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$results = Join-Path $project 'private_data\results'
New-Item -ItemType Directory -Path $results -Force | Out-Null

if ($Device -eq 'cuda') {
    & nvidia-smi --query-gpu=name,driver_version,memory.free --format=csv,noheader
    if ($LASTEXITCODE -ne 0) {
        throw 'NVIDIA driver is unavailable. Reboot or repair the driver before running.'
    }
    & python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"
    if ($LASTEXITCODE -ne 0) {
        throw 'PyTorch cannot access CUDA.'
    }
}

Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class StockWisePower {
    [DllImport("kernel32.dll")]
    public static extern uint SetThreadExecutionState(uint flags);
}
'@

$continuous = [Convert]::ToUInt32('80000000', 16)
$systemRequired = [uint32]0x00000001
$awayModeRequired = [uint32]0x00000040
[StockWisePower]::SetThreadExecutionState($continuous -bor $systemRequired -bor $awayModeRequired) | Out-Null

try {
    Set-Location $project
    $ErrorActionPreference = 'Continue'
    & python -m portfoliopilot.kronos_llm_allocation_backtest `
        --start 2021-12-01 `
        --evaluation-start 2023-01-03 `
        --end 2025-12-31 `
        --kronos-device $Device `
        --kronos-recycle-every 10 `
        --kronos-cooldown-seconds 0.25 `
        --output private_data/results/kronos-llm-monthly-2023-2025.json `
        --report private_data/results/kronos-llm-monthly-2023-2025.md `
        *>&1 | Tee-Object -FilePath private_data/results/kronos-llm-monthly-2023-2025.log
    $pythonExit = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    if ($pythonExit -ne 0) {
        throw "Three-year backtest failed with exit code $pythonExit"
    }
}
finally {
    [StockWisePower]::SetThreadExecutionState($continuous) | Out-Null
}
