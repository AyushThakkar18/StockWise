$ErrorActionPreference = "Stop"
python -m portfoliopilot.historical_council_backtest `
    --start 2024-01-01 `
    --evaluation-start 2025-01-02 `
    --end 2025-01-30 `
    --per-stock-scoring `
    --kronos `
    --kronos-device cuda `
    --output private_data/results/per-stock-kronos-pilot-2025-01.json `
    --report private_data/results/per-stock-kronos-pilot-2025-01.md
