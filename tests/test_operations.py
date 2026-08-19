from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest

from portfoliopilot.broker import PaperBroker
from portfoliopilot.contracts import MarketQuote, Side
from portfoliopilot.ledger import PortfolioLedger
from portfoliopilot.operations import PerformanceTracker, orders_from_targets
from portfoliopilot.optimizer import AllocationResult
from portfoliopilot.paper_session import PaperSessionCoordinator
from portfoliopilot.store import EventStore


def test_target_deltas_create_buy_and_exit_orders() -> None:
    now = datetime(2025, 1, 1, tzinfo=UTC)
    ledger = PortfolioLedger(Decimal(10_000))
    allocation = AllocationResult({"A": 0.10}, 0.90, {}, 0.10)
    orders = orders_from_targets(
        "s1", now, now + timedelta(days=1), allocation, ledger,
        {"A": Decimal(100)}, Decimal("0.2"), "p1",
    )
    assert len(orders) == 1
    assert orders[0].side == Side.BUY
    assert orders[0].quantity == Decimal(10)


def test_performance_snapshots_reconcile_and_track_drawdown(tmp_path) -> None:
    tracker = PerformanceTracker(EventStore(tmp_path / "events.db"), Decimal(1000), Decimal(1000))
    ledger = PortfolioLedger(Decimal(1000))
    first = tracker.record(date(2025, 1, 1), ledger, {}, Decimal(1000))
    ledger.cash = Decimal(900)
    second = tracker.record(date(2025, 1, 2), ledger, {}, Decimal(1010))
    assert first.excess_return == 0
    assert second.drawdown == Decimal("-0.1")
    assert second.excess_return == Decimal("-0.11")
    with pytest.raises(ValueError, match="increasing"):
        tracker.record(date(2025, 1, 2), ledger, {}, Decimal(1010))


def test_restart_replays_orders_and_fills_exactly_once(tmp_path) -> None:
    path = tmp_path / "events.db"
    now = datetime(2025, 1, 1, tzinfo=UTC)
    first_ledger = PortfolioLedger(Decimal(10_000))
    coordinator = PaperSessionCoordinator(EventStore(path), first_ledger, PaperBroker())
    allocation = AllocationResult({"A": 0.1}, 0.9, {}, 0.1)
    coordinator.freeze("s1", now, "snapshot", allocation)
    order = orders_from_targets(
        "s1", now, now + timedelta(hours=12), allocation, first_ledger,
        {"A": Decimal(100)}, Decimal("0.2"), "p1",
    )[0]
    coordinator.queue("s1", order)
    execution = now + timedelta(days=1)
    coordinator.execute(order.id, MarketQuote(
        symbol="A", observed_at=execution, available_at=execution, mid=Decimal(100),
        spread_bps=Decimal(0), available_quantity=Decimal(100),
    ))

    recovered_ledger = PortfolioLedger(Decimal(10_000))
    recovered = PaperSessionCoordinator(EventStore(path), recovered_ledger, PaperBroker())
    recovered.recover()
    assert recovered_ledger.cash == first_ledger.cash
    assert recovered_ledger.positions["A"].quantity == first_ledger.positions["A"].quantity
    assert order.id in recovered.executed_orders

