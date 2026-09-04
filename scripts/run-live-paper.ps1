param(
    [ValidateSet('auto', 'cuda', 'cpu')]
    [string]$Device = 'auto',
    [int]$IntervalSeconds = 900,
    [int]$Port = 8000,
    [switch]$InitializeFromLatestClose
)

$ErrorActionPreference = 'Stop'
if ($IntervalSeconds -lt 60) { throw 'IntervalSeconds must be at least 60.' }

Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class StockWisePower {
    [DllImport("kernel32.dll", CharSet = CharSet.Auto, SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint esFlags);
}
'@

$continuous = [Convert]::ToUInt32('80000000', 16)
$systemRequired = [Convert]::ToUInt32('00000001', 16)
[void][StockWisePower]::SetThreadExecutionState($continuous -bor $systemRequired)

$resultDirectory = Join-Path $PSScriptRoot '..\private_data\live'
New-Item -ItemType Directory -Force -Path $resultDirectory | Out-Null
$apiOut = Join-Path $resultDirectory 'api.log'
$apiError = Join-Path $resultDirectory 'api-error.log'
$serviceLog = Join-Path $resultDirectory 'service.log'
$priceLog = Join-Path $resultDirectory 'prices.log'
$priceError = Join-Path $resultDirectory 'prices-error.log'

$api = $null
$prices = $null
try {
    $health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 2
    if ($health.status -ne 'ok') { throw 'unhealthy existing service' }
    Write-Host "Using existing StockWise dashboard on port $Port."
}
catch {
    $api = Start-Process -FilePath python -ArgumentList @(
        '-m', 'uvicorn', 'portfoliopilot.api:app', '--host', '127.0.0.1', '--port', $Port
    ) -WorkingDirectory (Resolve-Path (Join-Path $PSScriptRoot '..')) `
      -WindowStyle Hidden -RedirectStandardOutput $apiOut -RedirectStandardError $apiError -PassThru
}

try {
    Write-Host "StockWise dashboard: http://127.0.0.1:$Port"
    Write-Host "Service log: $serviceLog"
    Write-Host "Price log: $priceLog"
    Write-Host 'Press Ctrl+C to stop both processes.'
    if ($InitializeFromLatestClose) {
        $savedErrorPreference = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        python -m portfoliopilot.live_service --device $Device --initialize-from-latest-close `
            2>&1 | Tee-Object -FilePath $serviceLog -Append
        $initializationExitCode = $LASTEXITCODE
        $ErrorActionPreference = $savedErrorPreference
        if ($initializationExitCode -ne 0) {
            throw "Live initialization failed with exit code $initializationExitCode."
        }
    }
    $prices = Start-Process -FilePath python -ArgumentList @(
        '-m', 'portfoliopilot.live_service', '--device', $Device, '--prices-only',
        '--interval', $IntervalSeconds
    ) -WorkingDirectory (Resolve-Path (Join-Path $PSScriptRoot '..')) `
      -WindowStyle Hidden -RedirectStandardOutput $priceLog `
      -RedirectStandardError $priceError -PassThru
    $savedErrorPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    python -m portfoliopilot.live_service --device $Device --research-only `
        --interval $IntervalSeconds `
        2>&1 | Tee-Object -FilePath $serviceLog -Append
    $serviceExitCode = $LASTEXITCODE
    $ErrorActionPreference = $savedErrorPreference
    if ($serviceExitCode -ne 0) { throw "Live service failed with exit code $serviceExitCode." }
}
finally {
    if ($prices -and -not $prices.HasExited) { Stop-Process -Id $prices.Id }
    if ($api -and -not $api.HasExited) { Stop-Process -Id $api.Id }
    [void][StockWisePower]::SetThreadExecutionState($continuous)
}
