from datetime import UTC, date, datetime
from decimal import Decimal

from portfoliopilot.contracts import ExecutionResult, Fill, OrderStatus, Side
from portfoliopilot.paper_dashboard import PaperDashboard
from portfoliopilot.paper_store import PaperTradingStore, StrategyVersion
from portfoliopilot.store import EventStore


def test_dashboard_projects_decisions_snapshots_and_fills(tmp_path) -> None:
    path = tmp_path / "paper.db"
    paper, events = PaperTradingStore(path), EventStore(path)
    version = paper.register_strategy(StrategyVersion("live", {"slot": ".05"}, "source"))
    paper.record_decision(
        version, date(2026, 8, 28), date(2026, 8, 31),
        {"schema_version": "live-council-decision-v1", "selected": ["AAPL"]},
    )
    paper.record_snapshot(
        version, "live", date(2026, 8, 31), Decimal("1010"), Decimal("10"),
        {"AAPL": Decimal("4")}, "broker-snapshot-1",
    )
    result = ExecutionResult(
        order_id="o1", status=OrderStatus.FILLED, residual_quantity=Decimal(0),
        fill=Fill(
            id="f1", order_id="o1", symbol="AAPL", side=Side.BUY,
            quantity=Decimal(4), price=Decimal(250), spread_cost=Decimal(0),
            slippage_cost=Decimal(0), fee=Decimal(0), executed_at=datetime.now(UTC),
        ),
    )
    events.append("result:o1", "PAPER_ORDER_RESULT", "session-1", result.model_dump(mode="json"))

    dashboard = PaperDashboard(paper, events)
    assert dashboard.overview()["latest_snapshot"]["equity"] == "1010"
    assert dashboard.decisions()[0]["payload"]["selected"] == ["AAPL"]
    assert dashboard.trades()[0]["price"] == "250"
