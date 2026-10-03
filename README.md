# StockWise

**A multi-agent investment research and portfolio backtesting system.**

StockWise combines point-in-time quantitative features, Kronos time-series forecasts, structured
LLM reviews, deterministic portfolio rules, and next-open execution. It is research software: it
does not place live trades, provide investment advice, or guarantee profit.

## Results at a glance

The retained strategy was evaluated over both a 2025 window and a longer 2023-2025 window. Each
selected stock receives a 5% target weight, unused allocation remains invested in SPY, and both
simulations include 5 basis points of transaction costs.

| Evaluation window | Monthly decisions | Strategy return | Profit on $1K paper portfolio | Sharpe | Max drawdown |
|---|---:|---:|---:|---:|---:|
| **2025** | 12 | **17.49%** | **$174.94** | 0.87 | -20.53% |
| **2023-2025** | 36 | **68.34%** | **$683.44** | 1.20 | -20.10% |

These are simulated diagnostic results from a paper portfolio, not evidence of expected future
performance.

See the [readable decision report](results/kronos-llm-three-policies-2025.md) and
[machine-readable results](results/kronos-llm-three-policies-2025.json).

## How it works

```text
Point-in-time eligible S&P 500 members
                  |
       deterministic Top 100
  momentum | trend | growth | risk data
                  |
         Kronos Top 50
       relative to SPY forecast
                  |
     +------------+------------+
     |            |            |
 technical     quality       risk
 LLM agent     LLM agent     LLM agent
     +------------+------------+
                  |
        structured synthesis agent
           selects 0-20 stocks
                  |
       deterministic allocation
   5% stock slots; unused weight -> SPY
                  |
        next-session-open execution
```

The LLM receives anonymous, lagged numeric packets rather than company names. It cannot set
position weights or bypass portfolio constraints. Structured Pydantic outputs are validated and
hash-cached so interrupted runs resume without repeating completed API calls.

## Live paper-trading council

The prospective live path uses named stocks because it does not make historical decisions. Every
candidate is isolated in its own request and reviewed by four specialists—market/technical,
business/fundamentals, news/filings/catalysts, and risk/sentiment—followed by a fifth synthesis
agent. Ordinary code validates candidate identity and evidence citations, combines specialist
scores, ranks the candidates, and enforces portfolio constraints.

The live `v4` policy selects every candidate passing the 70-point quality gate, up to 20. If fewer
than 10 qualify, it fills the remaining target slots from the highest-ranked candidates that have
no hard safety blocker. If fewer than 10 safe candidates exist, the monthly decision still proceeds
with the safe subset: every stock keeps its ten-slot weight and unused slots remain in cash. With
10-20 selections, 99.8% of current portfolio equity is divided equally; 0.2% remains as an
execution-cost reserve. There is no automatic SPY fallback. The dossier labels quality-gate
selections separately from minimum-diversification top-ups.

Each immutable decision dossier contains the evidence available at the decision time, every agent's
score, confidence, supporting points, concerns, citations, synthesis discussion, deterministic
ranking, selection outcome, and a human-readable Markdown report. The same structured dossier is
designed to drive the future dashboard, preventing displayed explanations from diverging from the
record used to construct paper orders.

The default forward-testing path uses an internal, broker-free simulator. It freezes the after-close
council dossier and target weights, then executes once against supplied next-session opening prices
with 2 bps of slippage, 5 bps of transaction costs, and a small execution-cost reserve. Orders,
fills, cost basis, realized and unrealized P&L, positions, and portfolio snapshots are append-only.
The optional Alpaca adapter remains disabled and cannot address Alpaca's live-trading domain.

The published 2023-2025 and 2025 results above belong to the retained historical 5%-slot/SPY
allocation experiment. They do not represent a backtest of the newer live `v4` allocation policy.

## Backtest protocol

- Evaluations: January through December 2025 (12 decisions) and January 2023 through December 2025
  (36 decisions).
- Warm-up data begins January 2024 for the one-year run and December 2021 for the three-year run.
- Universe: point-in-time S&P 500 membership, reduced deterministically to 100 candidates.
- Forecasting: Kronos-base ranks the deterministic candidates to 50.
- Council: technical, quality, and risk specialists followed by structured synthesis.
- Portfolio: zero to 20 long positions, rebalanced every 21 trading sessions.
- Execution: decisions formed after the close and executed at the next session's open.
- Frictions: 5 basis points per traded notional; dividends and splits are incorporated.

## Reliability boundaries

- The strategy only consumes information available by each decision date.
- Missing execution prices fail closed rather than silently dropping trades.
- LLM outputs are schema-validated, candidate-bound, anonymous, and reproducibly cached.
- Portfolio weights are assigned by deterministic code rather than the LLM.
- Some unavailable historical securities leave residual survivorship bias.
- The Kronos checkpoint's training cutoff is not documented precisely enough for a causal claim.
- The project observed 2025 while developing the system, so this is not an untouched holdout.
- Even the three-year backtest cannot establish statistical significance or future profitability.

## Technology

Python, Pydantic, OpenAI structured outputs, Kronos/PyTorch, FastAPI, SQLite, Docker, pytest, Ruff,
and GitHub Actions.

## Run locally

Requirements: Python 3.11+, Docker optional, an OpenAI API key, SEC user-agent configuration, and
the private historical price cache described in `DATA_SOURCES.md`.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
python -m pytest -q
```

Run or resume the retained experiment:

```powershell
portfoliopilot-kronos-llm-allocation
```

Completed Kronos forecasts and LLM responses are cached locally. API keys and private datasets are
excluded from version control.

Run the same monthly strategy over 2023-2025 with GPU stability controls:

```powershell
.\scripts\run-three-year-council-stable.ps1
```

The launcher verifies CUDA, prevents Windows sleep while active, recycles Kronos GPU allocations
every ten uncached forecasts, inserts short cooling intervals, and writes separate restart-safe
three-year results and logs. Use `-Device cpu` when CUDA is unavailable.

## Project layout

```text
src/portfoliopilot/kronos_llm_allocation_backtest.py  experiment runner and report
src/portfoliopilot/kronos_llm_selection.py            selection and allocation rules
src/portfoliopilot/kronos_forecast.py                 Kronos inference and caching
src/portfoliopilot/openai_bounded_agents.py           specialist structured reviews
src/portfoliopilot/openai_council_selector.py         bounded synthesis selection
src/portfoliopilot/live_council.py                    live five-agent audit contracts and ranking
src/portfoliopilot/openai_live_council.py             isolated named-stock structured LLM calls
src/portfoliopilot/live_evidence.py                    timestamped, candidate-bound news evidence
src/portfoliopilot/alpaca_paper.py                     paper-only broker and news client
src/portfoliopilot/alpaca_execution.py                 deterministic 5% paper rebalancing
src/portfoliopilot/paper_dashboard.py                  dashboard decision and trade read models
src/portfoliopilot/point_in_time.py                    next-open portfolio simulator
tests/test_kronos_llm_selection.py                     allocation and council tests
```

Paper execution requires separate paper-account credentials and an explicit enable flag:

```dotenv
ALPACA_PAPER_ENABLED=false
ALPACA_PAPER_KEY_ID=
ALPACA_PAPER_SECRET_KEY=
```

Keep the flag false while validating evidence and decisions in shadow mode. The dashboard read
endpoints are `/dashboard/overview`, `/dashboard/decisions`, and `/dashboard/trades`.

The two broker-free operational endpoints are `/paper/council/freeze` after a session closes and
`/paper/council/execute-next-open` after the following session's opening prices are available. Both
are idempotent: a frozen decision cannot be changed and a completed session cannot execute twice.

Start the local API and dashboard:

```powershell
python -m uvicorn portfoliopilot.api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/dashboard`. The responsive dashboard shows portfolio equity, total,
realized and unrealized P&L, cash, current positions, buy/sell fills, and the complete contribution
from every council agent. `/operations/status` reports the durable job queue and execution mode.

`MonthlyCouncilScheduler` enqueues one idempotent council decision after the final closed exchange
session of each month and binds it to the next session's open. It never backfills a missed decision
with information that became available later.

Run the complete forward-paper service and dashboard with one command:

```powershell
.\scripts\run-live-paper.ps1 -Device auto
```

For the first portfolio only, initialize immediately from the latest completed market close:

```powershell
.\scripts\run-live-paper.ps1 -Device auto -InitializeFromLatestClose
```

The flag refuses to run if the month already has a frozen decision. After initialization, the same
process remains active to execute at the next open and continue normal monthly monitoring.

The launcher prevents Windows sleep, starts the dashboard, and checks every 15 minutes. The service
does nothing before the US close or when the current month already has a frozen decision. When a new
month is due, it refreshes current S&P 500 histories, ranks up to 200 data-eligible stocks,
passes the strongest 75 Kronos candidates to the council, and
the five-agent council. It freezes the result and executes it once the next session's opening prices
are available. Between rebalances it refreshes marks so the dashboard shows current simulated P&L.
Logs are written to `private_data/live/`.

Run a single readiness cycle without leaving the service open:

```powershell
python -m portfoliopilot.live_service --once --device auto
```

Before the close this reports `WAIT_FOR_AFTER_CLOSE` without invoking Kronos or the LLM. A monthly
research cycle requires `OPENAI_API_KEY`, `SEC_USER_AGENT`, the Kronos checkout under
`private_data/Kronos`, and network access to Yahoo Finance and SEC EDGAR. Seventy-five candidates receive
four isolated specialist reviews and one synthesis review; four candidates are processed in
parallel. Timestamped, ticker-bound Yahoo headlines provide broker-free news and sentiment evidence;
future headlines are rejected. All completed responses and Kronos forecasts are restart-safe cached.

