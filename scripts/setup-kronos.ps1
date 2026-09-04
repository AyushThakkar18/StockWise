$ErrorActionPreference = "Stop"
$repository = Join-Path $PSScriptRoot "..\private_data\Kronos"
if (-not (Test-Path $repository)) {
    git clone --depth 1 https://github.com/shiyu-coder/Kronos.git $repository
}
python -m pip install -e ".[kronos]"
Write-Host "Kronos is ready. The 102M base checkpoint downloads on first use."
