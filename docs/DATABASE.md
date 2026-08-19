# Database entities

Production persistence uses PostgreSQL with append-only event tables. Foreign keys shown as arrows.

| Entity | Essential fields |
|---|---|
| instruments | id, symbol, name, sector, exchange, valid_from/to |
| universes | id, name, version; members -> instruments with effective dates |
| market_observations | instrument, source, observed/published/available/retrieved times, vintage, quality, OHLCV |
| evidence_records | instrument, kind, claim, source URI, four timestamps, content hash, quality |
| filing_versions / news_articles | evidence, accession/provider ID, immutable body hash |
| feature_snapshots | instrument, as_of, feature version, input hash, values JSON |
| model_versions / forecasts | immutable artifact metadata; forecast -> model, feature snapshot |
| research_reports | decision time, typed report, evidence links, model/prompt version |
| portfolios / portfolio_snapshots | configuration; immutable cash, marks, weights, hash |
| positions | portfolio, instrument, quantity, average cost, opened/closed timestamps |
| decisions / policy_checks | action, reason codes, versions, snapshot hash; pass/fail trace |
| orders / fills | decision, side, quantity, timing; execution details and costs |
| transactions / cash_ledger | fill/action reference, signed amount, running balance |
| corporate_actions | instrument, effective/pay dates, type, terms, source |
| experiments / protocol_locks | split dates, hashes, seeds, results, holdout-opened flag |
| performance_snapshots | portfolio and benchmark equity, returns, risk and attribution |
| audit_events | actor, event type, entity reference, timestamp, canonical payload hash |

Rows used by decisions are immutable. Corrections append a superseding version. Database constraints prohibit negative quantities/cash and reopening a protocol lock.

