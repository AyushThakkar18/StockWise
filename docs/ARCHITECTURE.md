# Clean-room architecture

This design was derived from the product requirements and general event-sourcing and quantitative-research principles. It does not reproduce another trading repository's graph, roles, prompts, or orchestration.

```text
timestamped sources -> point-in-time gate -> feature snapshots -> versioned forecasts
                                                        |              |
                                                        v              v
immutable journal <- paper broker <- order policy <- target-weight optimizer
       |                 |                ^              ^
       v                 v                |              |
account snapshots -> performance      decision auditor <- typed research evidence
       |
       +-> API / dashboard / experiment registry
```

The AI research council produces typed evidence and may block a proposal. It cannot size positions or create/submit orders. The deterministic path owns policy gates, target weights, costs, fills, and accounting.

## Repository structure

```text
src/portfoliopilot/contracts.py  domain contracts and enums
src/portfoliopilot/integrity.py  point-in-time policy
src/portfoliopilot/ledger.py     accounting and snapshots
src/portfoliopilot/broker.py     next-session execution simulator
src/portfoliopilot/state.py      deterministic lifecycle
src/portfoliopilot/features.py   versioned point-in-time features
src/portfoliopilot/validation.py purged walk-forward date splits
src/portfoliopilot/experiments.py immutable protocols and holdouts
src/portfoliopilot/dataset.py    cross-sectional labels and examples
src/portfoliopilot/models.py     interpretable model artifacts
src/portfoliopilot/model_evaluation.py walk-forward predictions
src/portfoliopilot/promotion.py  deterministic promotion gates
src/portfoliopilot/model_registry.py immutable artifacts/evaluations
src/portfoliopilot/optimizer.py   constrained deterministic allocation
src/portfoliopilot/multi_backtest.py multi-asset next-open simulation
src/portfoliopilot/model_selection.py development-only tuning
src/portfoliopilot/statistics.py date-clustered inference
src/portfoliopilot/allocation_pipeline.py model-to-policy bridge
src/portfoliopilot/paper_session.py frozen forward sessions
src/portfoliopilot/operations.py operational orders and performance
src/portfoliopilot/evidence.py   point-in-time evidence curation
src/portfoliopilot/research_contracts.py typed council records
src/portfoliopilot/research.py   independent roles and deterministic audit
src/portfoliopilot/openai_research.py optional structured-output transport
src/portfoliopilot/jobs.py       durable local queue and worker
src/portfoliopilot/snapshots.py  immutable data/research snapshots
src/portfoliopilot/monitoring.py operational health summaries
src/portfoliopilot/scheduling.py explicit calendar and after-close jobs
src/portfoliopilot/workflows.py  fail-closed snapshot workflow handlers
src/portfoliopilot/api.py        minimal paper-only API
tests/                            invariant tests
docs/                             specification, schema, decisions, roadmap
```
