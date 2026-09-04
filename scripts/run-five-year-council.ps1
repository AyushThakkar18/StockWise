$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "src"

python -m portfoliopilot.historical_council_backtest `
  --start 2020-01-01 `
  --evaluation-start 2021-01-04 `
  --end 2025-12-31 `
  --prices private_data/prices-yahoo `
  --rank-cache private_data/ai-rankings-pit-2021-2025 `
  --council-cache private_data/council-pit-2021-2025 `
  --output private_data/results/council-pit-2021-2025.json `
  --report private_data/results/council-pit-2021-2025.md
