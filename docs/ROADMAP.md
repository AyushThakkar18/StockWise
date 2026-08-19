# Phased implementation plan

1. **Specification and ledger (current):** contracts, integrity gates, accounting, execution simulator, API, invariant tests.
2. **Data and backtest (in progress):** Alpha Vantage daily adapter, immutable SQLite event boundary, next-open event loop, SPY/cash/momentum baselines, costs, and core performance metrics are implemented. Exchange calendars, point-in-time corporate actions, delistings, universe history, PostgreSQL, multi-asset strategies, and statistical intervals remain.
3. **Candidate models:** deterministic price features, cross-sectional excess-return labels, purged walk-forward evaluation, an interpretable ridge baseline, immutable hash-verified artifacts/evaluations, embargoed protocols, one-way holdouts, constrained target weights, multi-asset simulation, clustered intervals, and development-only hyperparameter selection are implemented.
4. **AI research council (in progress):** typed independent reports, point-in-time evidence curation, citation validation, contradiction detection, deterministic auditing, and an optional structured-output OpenAI adapter are implemented. Durable research persistence, source adapters, and ablation evaluation remain.
5. **Forward paper core (in progress):** frozen allocation sessions, append-only order lifecycle events, next-session execution, restart recovery, durable idempotent jobs, immutable snapshots, retries/dead letters, and health summaries are implemented. Calendar-aware scheduling, notifications, and the dashboard remain.
4. **Research council:** structured evidence extraction, independent failure review, contradiction checks, deterministic auditor, ablations.
5. **Forward paper mode:** durable scheduler, frozen daily snapshots, monitoring, accessible Next.js UI, operational telemetry.

Promotion between phases requires passing leakage, accounting, reproducibility, and benchmark-relative risk gates. A positive return alone is insufficient.
