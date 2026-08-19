# StockWise

**An auditable five-agent investment research and paper-trading system.**

StockWise combines LangGraph/LangChain orchestration, point-in-time company evidence, an AI-assisted
quantitative ranker, realistic portfolio simulation, and a FastAPI operations layer. LLMs analyze
evidence; deterministic code controls citations, contradictions, abstention, risk, and execution.

> Research software only. It does not provide investment advice, guarantee profit, or trade live capital.

## Results at a glance

### Five-agent council — 2025

The quantitative LLM ranked 99 stocks using anonymous lagged features. Every monthly Top-20
shortlist was reviewed using SEC facts filed by the decision date and lagged Tiingo evidence. Only
a completed synthesis with a `SUPPORT` verdict could enter the next-open portfolio.

| Strategy | Return | Profit on $100K | Sharpe | Max drawdown |
|---|---:|---:|---:|---:|
| 21-day momentum | **42.44%** | $42,438 | 1.63 | -18.56% |
| Quantitative LLM ranker | 36.90% | $36,898 | 1.48 | -21.85% |
| Equal weight | 32.22% | $32,216 | 1.56 | -18.63% |
| **Five-agent council** | **23.11%** | **$23,109** | **0.96** | **-24.91%** |
| SPY | 17.88% | $17,885 | — | — |

The council beat SPY by **5.22 percentage points**, was positive in 9 of 12 holding periods, and
beat SPY in 7 of 12. Across 240 reviews it produced 64 `SUPPORT` decisions, 27 `CONCERN` decisions,
and 149 fail-safe abstentions. Its strict filtering underperformed the standalone ranker and simple
baselines; that negative comparison is reported rather than hidden.

### Quantitative LLM ranker — five-year experiment

The earlier ranker experiment covered 97 stocks and 1,255 sessions from January 2021 through
December 2025. It returned **128.6% versus 92.4% for SPY**, including 23.1% versus 16.3% in the 2025
holdout. The research council did not produce the five-year result.

All results include 5 bps simulated transaction costs. The frozen August 2026 universe creates
**survivorship bias**, so these numbers demonstrate the engineering workflow—not future profitability.

- [Five-agent decision output](data/current-top99-council-2025.json)
- [Five-year readable report](data/current-top100-ai-final.md)
- [Five-year complete metrics](data/current-top100-ai-final.json)
- [Frozen universe](data/current-top100-2026-08-19.json)

## Five-agent architecture

```text
Anonymous lagged features -> quantitative LLM -> monthly Top-20 shortlist
                                                  |
                                 point-in-time SEC + Tiingo evidence
                                                  |
                   +------------------------------+------------------+
                   |                 |                 |              |
                   v                 v                 v              v
             Business agent   Catalyst agent    Failure agent   Portfolio agent
                   +------------------------------+------------------+
                                                  |
                         deterministic evidence/contradiction audit
                                      /                           \
                                     v                             v
                           fifth synthesis agent             fail-safe abstention
                                     |
                              SUPPORT verdict only
                                     |
                         next-open portfolio execution
```

Four LangGraph specialists run concurrently. Deterministic code validates report identity, evidence
references, source diversity, citation coverage, agent failures, and unresolved high-severity
contradictions. Failed audits abstain; approved audits route to the fifth synthesis agent.

## Engineering highlights

- Checkpointed LangGraph fan-out/fan-in workflow with conditional routing and SQLite persistence.
- LangChain/OpenAI structured outputs validated through immutable Pydantic contracts.
- Point-in-time SEC company facts and lagged Tiingo evidence; future filings are rejected.
- Anonymous 21/63/126/252-session momentum and 63-session volatility features.
- Close-to-next-open execution with dividends, splits, turnover, and transaction costs.
- FastAPI, immutable event records, durable jobs, telemetry, backups, Docker, and CI.
- **99 automated tests** spanning orchestration, leakage controls, accounting, and security.

## 2025 protocol

- 99 stocks with complete 2025 coverage; SNDK was excluded because it began trading in February.
- Rebalanced every 21 sessions into up to 20 equal-weight positions.
- Signals formed at the close and executed at the next open with 5 bps costs.
- SEC facts had to be filed no later than the decision date.
- Council entry required a valid four-agent audit, completed synthesis, and `SUPPORT` verdict.
- SPY total return served as the benchmark.

## Quick start

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
python -m pytest -q
python -m uvicorn portfoliopilot.api:app --reload
```

Configure `.env` and never commit it:

```dotenv
OPENAI_API_KEY=your_key
OPENAI_MODEL=gpt-4o-mini
TIINGO_API_KEY=your_key
SEC_USER_AGENT=StockWise your-email@example.com
PORTFOLIOPILOT_API_TOKEN=a_long_random_secret
```

The SEC contact is sent only in EDGAR request headers. API docs run at
`http://127.0.0.1:8000/docs`.

## Reproduce the quantitative experiment

```powershell
python -m portfoliopilot.current_universe_download `
  --universe data/current-top100-2026-08-19.json `
  --start 2020-01-01 --end 2025-12-31 --limit 30

python -m portfoliopilot.current_universe_backtest `
  --universe data/current-top100-2026-08-19.json `
  --start 2020-01-01 --end 2025-12-31 --ai `
  --output data/current-top100-ai-final.json
```

Downloads and LLM decisions are resumable and privately cached. Raw provider payloads, model
responses, databases, and secrets are excluded from Git.

## Repository map

```text
src/portfoliopilot/
  langgraph_research.py       checkpointed graph and routing
  openai_research.py          specialist and synthesis agents
  research.py                 evidence and contradiction audits
  sec_edgar.py                point-in-time SEC facts
  ai_ranking.py               anonymous quantitative ranking
  point_in_time.py            dynamic universe and council gate
  multi_backtest.py           multi-asset simulator
  api.py / worker.py          API and durable processing
tests/                        99 automated tests
data/                         published results and frozen universe
```

## Docker

```powershell
docker compose up --build -d
```

See [deployment guidance](docs/DEPLOYMENT.md) for the server overlay and security requirements.

## Limitations

- The published current-constituent universe contains survivorship bias.
- A historical-membership simulator exists, but a fully unbiased run still needs complete
  former/delisted-security prices and terminal returns.
- Council evidence currently includes SEC facts and market data, not licensed historical news.
- One year is a single market regime and does not establish future profitability.
- Forward paper trading is required before considering real-money use.

More detail: [architecture](docs/ARCHITECTURE.md) · [product](docs/PRODUCT.md) ·
[data sources](DATA_SOURCES.md) · [deployment](docs/DEPLOYMENT.md) · [roadmap](docs/ROADMAP.md)
