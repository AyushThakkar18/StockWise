# Data sources and rights register

No external market dataset is bundled in this milestone.

## Alpha Vantage

The default adapter requests the free `TIME_SERIES_DAILY` raw-price endpoint with `outputsize=compact` under the user's own API key. This currently limits backtests to roughly the latest 100 sessions. Source and retrieval vintage are retained. Usage, attribution, call frequency, storage, and redistribution must follow the user's Alpha Vantage plan and current terms. Dividends and splits are not synthesized, so results spanning corporate actions are incomplete. Full history and the adjusted endpoint currently require a provider plan and are not used by the free workflow. Delistings and historical index membership are not supplied; defensible stock-selection research still requires survivorship-safe datasets.

Forward snapshot jobs require every approved symbol to contain the expected completed session. A partial provider response blocks the entire snapshot. Exchange sessions are explicit versioned inputs; the application does not infer holidays or half-days from weekdays.

## FRED

An API key is configured but no adapter is enabled yet. Macro observations must use real-time/vintage dates (such as ALFRED semantics) before use in historical decisions.

## Tiingo end-of-day

The optional long-history adapter uses Tiingo's end-of-day API under the user's own Individual account. Terms reviewed 2026-08-19: API data is for internal consumption; individual users may use an Individual plan; redistribution requires separate permission and additional fees. The terms describe the free Starter plan as an evaluation plan, so continued free availability and limits are not guaranteed. Do not commit, publish, serve, or redistribute cached payloads.

The adapter stores raw OHLCV, `divCash`, and `splitFactor` with retrieval vintage. It deliberately does not use currently adjusted historical OHLC as point-in-time truth. The backtester applies split factors and cash dividends, although ex-date cash credit remains an approximation of payment timing. Tiingo also does not by itself provide historical universe membership sufficient to eliminate survivorship bias.

## Historical S&P 500 membership

The point-in-time universe adapter uses the MIT-licensed `fja05680/sp500` updated historical components dataset, retrieved at runtime from its public GitHub repository. It determines membership as of each rebalance date; future rows are never consulted. Coverage begins in 1996. The cached file's SHA-256 fingerprint is included in the dataset version recorded with each run. Historical ticker annotations are normalized for provider lookup, but corporate renames and provider-symbol mismatches can still occur and must be logged rather than silently substituted.

The intended top-100 universe is not today's top 100. On each rebalance date, the system intersects that date's historical S&P 500 members with Tiingo market-cap observations available by the decision time, then selects the largest 100. Missing or stale capitalization data blocks the rebalance.

Before an adapter is enabled it must record provider, license/terms URL and review date, attribution, redistribution restrictions, coverage, adjustment methodology, latency, timestamp semantics, revision policy, and known limitations. Raw licensed payloads must not be redistributed unless terms allow it. SEC EDGAR data, exchange calendars, prices, news, fundamentals, estimates, macro vintages, and delisting data each require separate review. Retrieval time is not a substitute for publication or strategy-availability time.
