# StockWise

**An evidence-grounded, five-agent investment research council built with LangGraph and LangChain.**

StockWise coordinates specialized LLM agents to evaluate equities using point-in-time SEC financial
facts and lagged market evidence. The agents do not place trades directly: deterministic software
validates their citations, checks for contradictions, enforces abstention, and controls portfolio
construction and next-session execution.

> Research and educational software only. It does not provide investment advice, guarantee profit,
> or place live trades.

## Multi-agent council

The council separates research into five explicit responsibilities:

1. **Business-change agent** - identifies material changes in financial and operating performance.
2. **Catalyst agent** - analyzes dated events and potential drivers without presenting uncertainty as fact.
3. **Failure-mode agent** - independently investigates leverage, cyclicality, concentration, and downside risk.
4. **Portfolio-context agent** - evaluates volatility, factor exposure, and portfolio-level implications.
5. **Synthesis agent** - reconciles audited specialist reports into `SUPPORT`, `CONCERN`, or `ABSTAIN`.

The first four agents run concurrently. The fifth agent is invoked only after deterministic evidence
and contradiction checks pass.

## System architecture

```text
100-stock research universe
          |
          v
Anonymous lagged features
          |
          v
Candidate-ranking LLM -> monthly Top-20 research queue
                                  |
                   point-in-time evidence packet
                      / SEC facts + Tiingo data
                                  |
          +-----------------------+-----------------------+
          |                       |                       |
          v                       v                       v
 Business-change agent      Catalyst agent        Failure-mode agent
          |                                               |
          +--------------- Portfolio-context agent -------+
                                  |
                      deterministic audit layer
             identity | citations | source diversity | coverage
                  failures | high-severity contradictions
                         /                        \
                        v                          v
              fifth synthesis agent         fail-safe abstention
                        |
             SUPPORT / CONCERN / ABSTAIN
                        |
                 SUPPORT verdict only
                        |
          risk-controlled next-open simulation
```

### Why the audit layer matters

Every specialist must return a schema-validated Pydantic object and cite supplied evidence IDs.
Deterministic code then verifies:

- all four specialist roles completed;
- symbol and decision timestamps match;
- every factual finding cites an allowed evidence record;
- at least two distinct data sources are present;
- at least 75% of curated evidence is referenced;
- no unresolved high-severity contradiction remains; and
- no specialist or synthesis call failed.

If any requirement fails, the graph routes to abstention. This prevents a fluent but unsupported LLM
response from reaching portfolio construction.

## 2025 council results

The council was evaluated over calendar year 2025. Each month, a candidate-ranking LLM reviewed
anonymous lagged features across 100 stocks and produced a Top-20 research queue. The multi-agent
council then analyzed those candidates using only evidence available by that decision date.

| Metric | Five-agent council | SPY benchmark |
|---|---:|---:|
| Simulated return | **23.11%** | 17.88% |
| Profit on $100,000 | **$23,109** | $17,885 |
| Excess return | **+5.22 percentage points** | - |
| Positive 21-session periods | **9 of 12** | - |
| Periods outperforming SPY | **7 of 12** | - |
| Sharpe ratio | 0.96 | - |
| Maximum drawdown | -24.91% | - |
| Simulated transaction costs | $749 | - |

### Council activity

| Decision outcome | Count |
|---|---:|
| Candidate reviews | 240 |
| `SUPPORT` | 64 |
| `CONCERN` | 27 |
| Fail-safe abstentions | 149 |
| Decisions reaching synthesis | 91 |
| Execution errors | 0 |

The council beat SPY in this simulation, but its strict filtering produced a concentrated portfolio
and did not eliminate drawdown risk. The result demonstrates orchestrated analysis and controlled
abstention, not guaranteed investment performance.

See the [complete council output](data/current-top100-council-2025.json) and
[frozen research universe](data/current-top100-2026-08-19.json).

## Backtest protocol

- Evaluation window: January 2 through December 31, 2025.
- Universe: 100-stock research universe.
- Research frequency: every 21 trading sessions.
- Evidence: SEC company facts filed by the decision date and lagged Tiingo end-of-day prices.
- Execution: close-derived decisions execute at the following session's open.
- Portfolio: long-only, cash-funded, equal weight across council-supported candidates.
- Costs: 5 basis points per traded notional.
- Benchmark: SPY total return including dividends and splits.
- Reliability: cached LLM decisions, immutable evidence records, and deterministic audit routes.

### Important limitation

The 100-stock universe was derived from a current Top-100 snapshot dated August 2026. The test
therefore contains survivorship bias even though the SEC and market evidence supplied to the agents
was point-in-time constrained. The result must be described as a simulated, survivorship-biased
research experiment.

## Engineering highlights

- LangGraph `StateGraph` with parallel specialist fan-out and conditional synthesis/abstention paths.
- LangChain structured outputs backed by immutable Pydantic research contracts.
- SQLite graph checkpoints and hash-cached model decisions for restart-safe execution.
- Prompt-injection-resistant evidence handling: source text is treated as untrusted data.
- Point-in-time SEC adapter that excludes filings submitted after the decision date.
- Deterministic citation, coverage, identity, source-diversity, and contradiction audits.
- Multi-asset next-open simulator with splits, dividends, transaction costs, and SPY-relative metrics.
- FastAPI, durable background jobs, health telemetry, backups, Docker, and GitHub Actions.
- **99 automated tests** covering orchestration, evidence integrity, accounting, execution, and security.

## Quick start

Requirements: Python 3.11+ and optionally Docker.

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

The SEC contact is sent only in EDGAR request headers. Interactive API documentation is available at
`http://127.0.0.1:8000/docs`.

## Repository map

```text
src/portfoliopilot/
  langgraph_research.py       five-agent graph, checkpoints, and routing
  openai_research.py          structured specialist and synthesis adapters
  research.py                 deterministic evidence and contradiction audits
  research_contracts.py       typed agent reports, findings, audits, and verdicts
  sec_edgar.py                point-in-time SEC company-facts evidence
  evidence.py                 evidence curation and temporal validation
  ai_ranking.py               anonymous candidate-ranking stage
  point_in_time.py            council gate and dynamic-universe controls
  multi_backtest.py           next-open multi-asset simulator
  api.py / worker.py          API and durable background processing
tests/                        99 automated tests
data/                         published council result and frozen universe
```

## Docker

```powershell
docker compose up --build -d
```

See [deployment guidance](docs/DEPLOYMENT.md) for server topology, secret handling, persistence, and
API-token enforcement.

## Current limitations and next steps

- Replace the current-constituent universe with complete historical membership, delisting prices,
  and terminal returns to remove survivorship bias.
- Add licensed point-in-time news, management commentary, estimates, and industry-specific evidence.
- Evaluate the council across additional market regimes and pre-register model/prompt changes.
- Forward-test council decisions through paper trading before considering real-money use.

More detail: [architecture](docs/ARCHITECTURE.md) | [product](docs/PRODUCT.md) |
[data sources](DATA_SOURCES.md) | [deployment](docs/DEPLOYMENT.md) | [roadmap](docs/ROADMAP.md)
