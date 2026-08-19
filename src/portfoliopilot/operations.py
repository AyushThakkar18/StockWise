from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date, datetime
from decimal import Decimal

from .contracts import Order, Side
from .ledger import PortfolioLedger
from .optimizer import AllocationResult
from .store import EventStore


@dataclass(frozen=True)
class PerformanceSnapshot:
    session: date
    portfolio_value: Decimal
    benchmark_value: Decimal
    cash: Decimal
    gross_exposure: Decimal
    total_return: Decimal
    benchmark_return: Decimal
    excess_return: Decimal
    drawdown: Decimal
    snapshot_hash: str


def orders_from_targets(
    session_id: str,
    decision_at: datetime,
    earliest_execution_at: datetime,
    allocation: AllocationResult,
    ledger: PortfolioLedger,
    marks: dict[str, Decimal],
    maximum_position_weight: Decimal,
    policy_version: str,
    minimum_notional: Decimal = Decimal(10),
) -> tuple[Order, ...]:
    if earliest_execution_at <= decision_at:
        raise ValueError("orders must execute after their decision session")
    if any(weight < 0 for weight in allocation.target_weights.values()) or sum(
        allocation.target_weights.values()
    ) > 1 + 1e-12:
        raise ValueError("target weights must be long-only and fully funded")
    equity = ledger.equity(marks)
    symbols = sorted(set(allocation.target_weights) | set(ledger.positions))
    orders = []
    for symbol in symbols:
        position = ledger.positions.get(symbol)
        current_quantity = position.quantity if position else Decimal(0)
        if symbol not in marks:
            if current_quantity:
                raise ValueError(f"missing mark for held position: {symbol}")
            continue
        target_value = equity * Decimal(str(allocation.target_weights.get(symbol, 0.0)))
        target_quantity = target_value / marks[symbol]
        delta = target_quantity - current_quantity
        if abs(delta) * marks[symbol] < minimum_notional:
            continue
        side = Side.BUY if delta > 0 else Side.SELL
        orders.append(Order(
            id=f"{session_id}:{symbol}:{side.value}", decision_id=f"decision:{session_id}:{symbol}",
            symbol=symbol, side=side, quantity=abs(delta), submitted_at=decision_at,
            earliest_execution_at=earliest_execution_at, reference_price=marks[symbol],
            max_position_weight=maximum_position_weight, policy_version=policy_version,
        ))
    return tuple(orders)


class PerformanceTracker:
    def __init__(
        self, store: EventStore, initial_portfolio_value: Decimal,
        initial_benchmark_value: Decimal,
    ):
        if initial_portfolio_value <= 0 or initial_benchmark_value <= 0:
            raise ValueError("initial values must be positive")
        self.store = store
        self.initial_portfolio_value = initial_portfolio_value
        self.initial_benchmark_value = initial_benchmark_value
        self.peak = initial_portfolio_value
        self.snapshots: list[PerformanceSnapshot] = []

    def record(
        self, session: date, ledger: PortfolioLedger, marks: dict[str, Decimal],
        benchmark_value: Decimal,
    ) -> PerformanceSnapshot:
        if self.snapshots and session <= self.snapshots[-1].session:
            raise ValueError("performance sessions must be strictly increasing")
        portfolio_value = ledger.equity(marks)
        self.peak = max(self.peak, portfolio_value)
        gross_exposure = sum(
            (position.quantity * marks[symbol] for symbol, position in ledger.positions.items()),
            Decimal(0),
        )
        core = {
            "session": session.isoformat(), "portfolio_value": str(portfolio_value),
            "benchmark_value": str(benchmark_value), "cash": str(ledger.cash),
            "gross_exposure": str(gross_exposure),
            "total_return": str(portfolio_value / self.initial_portfolio_value - 1),
            "benchmark_return": str(benchmark_value / self.initial_benchmark_value - 1),
            "excess_return": str(
                portfolio_value / self.initial_portfolio_value
                - benchmark_value / self.initial_benchmark_value
            ),
            "drawdown": str(portfolio_value / self.peak - 1),
        }
        digest = hashlib.sha256(
            json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        snapshot = PerformanceSnapshot(
            session, portfolio_value, benchmark_value, ledger.cash, gross_exposure,
            Decimal(core["total_return"]), Decimal(core["benchmark_return"]),
            Decimal(core["excess_return"]), Decimal(core["drawdown"]), digest,
        )
        self.store.append(
            f"performance:{session.isoformat()}", "PERFORMANCE_SNAPSHOT", "portfolio",
            {**asdict(snapshot), "session": session.isoformat()},
        )
        self.snapshots.append(snapshot)
        return snapshot
