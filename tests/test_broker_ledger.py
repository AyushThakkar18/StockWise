from datetime import datetime, timedelta, timezone
from decimal import Decimal

from portfoliopilot.broker import PaperBroker
from portfoliopilot.contracts import MarketQuote, Order, OrderStatus, Side
from portfoliopilot.ledger import PortfolioLedger

NOW = datetime(2025, 1, 2, 21, tzinfo=timezone.utc)


def order(side: Side, quantity: str = "10") -> Order:
    return Order(
        id=f"o-{side}", decision_id="d1", symbol="ABC", side=side,
        quantity=Decimal(quantity), submitted_at=NOW,
        earliest_execution_at=NOW + timedelta(hours=12), reference_price=Decimal("100"),
        max_position_weight=Decimal("0.5"), policy_version="policy-1",
    )


def quote(quantity: str = "100") -> MarketQuote:
    observed = NOW + timedelta(days=1)
    return MarketQuote(
        symbol="ABC", observed_at=observed, available_at=observed,
        mid=Decimal("100"), spread_bps=Decimal("10"),
        available_quantity=Decimal(quantity),
    )


def test_round_trip_reconciles_costs_and_realized_pnl() -> None:
    ledger = PortfolioLedger(Decimal("10000"))
    broker = PaperBroker(fee_per_order=Decimal("1"), slippage_bps=Decimal("5"))
    buy = broker.execute(order(Side.BUY), quote(), ledger)
    assert buy.status == OrderStatus.FILLED
    sell = broker.execute(order(Side.SELL), quote(), ledger)
    assert sell.status == OrderStatus.FILLED
    assert ledger.positions["ABC"].quantity == 0
    assert ledger.cash < ledger.initial_cash
    assert ledger.realized_pnl == ledger.cash - ledger.initial_cash


def test_partial_fill_is_explicit() -> None:
    ledger = PortfolioLedger(Decimal("10000"))
    result = PaperBroker().execute(order(Side.BUY), quote("4"), ledger)
    assert result.status == OrderStatus.PARTIAL
    assert result.residual_quantity == Decimal("6")
    assert ledger.positions["ABC"].quantity == Decimal("4")


def test_same_session_execution_is_rejected() -> None:
    ledger = PortfolioLedger(Decimal("10000"))
    early = quote().model_copy(update={"observed_at": NOW, "available_at": NOW})
    result = PaperBroker().execute(order(Side.BUY), early, ledger)
    assert result.status == OrderStatus.REJECTED
    assert ledger.cash == ledger.initial_cash


def test_position_limit_blocks_fill() -> None:
    ledger = PortfolioLedger(Decimal("1000"))
    result = PaperBroker().execute(order(Side.BUY, "10"), quote(), ledger)
    assert result.status == OrderStatus.REJECTED
    assert "position weight" in result.reason

