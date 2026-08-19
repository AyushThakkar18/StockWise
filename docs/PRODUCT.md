# Product specification

## Outcome

PortfolioPilot AI runs an auditable forward paper portfolio against SPY. It freezes the data available at each decision time, ranks an approved universe, computes target weights deterministically, simulates execution no earlier than the next session, and reports funded-account performance.

## Users and claims

The primary user is an individual researcher. Screens must distinguish forecasts, historical simulations, live paper results, and benchmark results. The product may report measured historical or paper performance with dates and assumptions. It may not claim assured profit, calibrated probability without calibration evidence, suitability, or live execution.

## Safety invariants

1. Paper trading is the only execution mode.
2. A decision cannot consume evidence whose `available_to_strategy_at` is after its decision time.
3. Missing, stale, conflicted, or failed-quality evidence blocks opening or adding.
4. Numerical forecasts, sizing, constraints, accounting, and fills are deterministic.
5. Orders execute after the decision timestamp and never at the price used to decide.
6. Cash cannot become negative; long-only positions cannot become negative.
7. Every mutation appends an immutable journal record and carries policy/model versions.
8. Historical decisions and opened holdouts are never rewritten.
9. Unallocated capital remains cash (or SPY only when explicitly configured).
10. Profitability is never guaranteed; model promotion requires prospective evidence.

## Accounting rules

- Money, prices, quantities, and fees use `Decimal`; currency rounds to cents only at ledger boundaries.
- A buy debits cash by gross consideration plus costs and adds shares/cost basis.
- A sell credits cash by gross consideration minus costs and realizes proceeds minus relieved cost basis.
- Average cost is used in this slice. Splits preserve total basis; dividends are cash ledger entries.
- Equity equals cash plus marked position value. A snapshot hash covers its canonical inputs.
- Rejected/partial orders never fabricate holdings; partial fills leave an explicit residual.

## Acceptance for this milestone

The ledger reconciles, future evidence is rejected, same-session execution is rejected, risk constraints block invalid fills, transitions are valid, and identical inputs produce identical outputs.

