# Final Current-Top-100 AI Backtest

## Protocol

- Data requested from 2020-01-01 through 2025-12-31; 253 sessions reserved for feature warm-up.
- Evaluated January 2021 through December 2025 across 1,255 sessions.
- 97 securities had complete-window histories; GEV, PLTR, and SNDK were excluded.
- Current-constituent universe dated 2026-08-19; results therefore contain survivorship bias.
- Benchmark: SPY total return with splits and cash dividends.
- Monthly next-session execution, 20 equal-weight positions, and 5 bps transaction costs.
- Development ended 2023-12-29, validation ended 2024-12-30, holdout ended 2025-12-31.
- The AI received anonymous lagged momentum and volatility features, never ticker identities or dates.

## AI candidate results

| Stage | AI return | SPY return | Excess | Sharpe | Max drawdown |
|---|---:|---:|---:|---:|---:|
| Development | 52.40% | 33.43% | +18.97% | 0.82 | -23.11% |
| Validation | 21.88% | 24.01% | -2.13% | 1.06 | -19.00% |
| Untouched holdout | 23.08% | 16.30% | +6.78% | 1.08 | -23.47% |
| Full evaluated period | 128.61% | 92.43% | +36.18% | 0.92 | -24.29% |

The AI candidate annualized at 18.03%. It generated positive returns in 40 of 60 non-overlapping
21-session periods (66.7%) and beat SPY in 34 of 60 (56.7%). In the untouched holdout it was
positive in 8 of 12 periods and beat SPY in 5 of 12. Its full-period information ratio was 0.35.

## Selected strategy and interpretation

The development-only selection rule chose equal weight, not AI. Equal weight returned 162.76%
versus SPY's 92.43%, and returned 31.32% versus SPY's 16.30% in holdout. Consequently, the final
experiment supports the claim that the AI candidate beat SPY in this simulation, but not that AI
was the best tested strategy or that it added value over equal weighting.

The current-universe construction is survivorship-biased. These results are appropriate for an
exploratory project metric if described as simulated, but not as evidence of future profitability.
The separate five-agent LangGraph research council is implemented and tested but has not yet been
historically evaluated with point-in-time filings and news.

## Resume-safe metrics

- 97 stocks and 1,255 evaluated trading sessions.
- 60 cached monthly AI decisions with schema validation and deterministic risk controls.
- 128.6% simulated AI return versus 92.4% for SPY; +36.2 percentage-point excess.
- 23.1% untouched-holdout return versus 16.3% for SPY; +6.8 percentage-point excess.
- 18.0% annualized return, 0.92 Sharpe ratio, and -24.3% maximum drawdown.
- Positive in 40 of 60 monthly holding periods; beat SPY in 34 of 60.

Always qualify these as results from a survivorship-biased current-universe simulation.
