$ErrorActionPreference = "Stop"

$tokenLine = Get-Content ".env" | Where-Object { $_ -match '^PORTFOLIOPILOT_API_TOKEN=' } | Select-Object -First 1
if (-not $tokenLine) {
    throw "PORTFOLIOPILOT_API_TOKEN is required for server deployment."
}
$token = ($tokenLine -split '=', 2)[1].Trim()
if ($token.Length -lt 32) {
    throw "PORTFOLIOPILOT_API_TOKEN must contain at least 32 characters."
}

Write-Output "Environment safety checks passed."

