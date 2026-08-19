from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from portfoliopilot.broker import PaperBroker
from portfoliopilot.contracts import MarketQuote, Order, OrderStatus, Side
from portfoliopilot.ledger import PortfolioLedger
from portfoliopilot.optimizer import AllocationResult
from portfoliopilot.paper_session import PaperSessionCoordinator
from portfoliopilot.store import EventStore


def test_frozen_session_executes_once_in_later_session(tmp_path) -> None:
    decision_at = datetime(2025, 1, 2, 21, tzinfo=UTC)
    ledger = PortfolioLedger(Decimal(10_000))
    coordinator = PaperSessionCoordinator(EventStore(tmp_path / "events.db"), ledger, PaperBroker())
    allocation = AllocationResult({"ABC": 0.1}, 0.9, {}, 0.1)
    coordinator.freeze("s1", decision_at, "snapshot", allocation)
    order = Order(
        id="o1", decision_id="d1", symbol="ABC", side=Side.BUY, quantity=Decimal(10),
        submitted_at=decision_at, earliest_execution_at=decision_at + timedelta(hours=12),
        reference_price=Decimal(100), max_position_weight=Decimal("0.2"), policy_version="p1",
    )
    coordinator.queue("s1", order)
    next_session = decision_at + timedelta(days=1)
    result = coordinator.execute("o1", MarketQuote(
        symbol="ABC", observed_at=next_session, available_at=next_session,
        mid=Decimal(100), spread_bps=Decimal(0), available_quantity=Decimal(100),
    ))
    assert result.status == OrderStatus.FILLED
    with pytest.raises(ValueError, match="already"):
        coordinator.execute("o1", result.fill and MarketQuote(
            symbol="ABC", observed_at=next_session, available_at=next_session,
            mid=Decimal(100), spread_bps=Decimal(0), available_quantity=Decimal(100),
        ))


def test_session_cannot_be_rewritten(tmp_path) -> None:
    coordinator = PaperSessionCoordinator(
        EventStore(tmp_path / "events.db"), PortfolioLedger(Decimal(1000)), PaperBroker()
    )
    now = datetime(2025, 1, 1, tzinfo=UTC)
    allocation = AllocationResult({}, 1.0, {}, 0.0)
    coordinator.freeze("s1", now, "snapshot", allocation)
    with pytest.raises(ValueError, match="already frozen"):
        coordinator.freeze("s1", now, "changed", allocation)

