from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest

from portfoliopilot.contracts import Quality
from portfoliopilot.market_data import DailyBar
from portfoliopilot.snapshots import ResearchStore
from portfoliopilot.store import EventStore
from portfoliopilot.workflows import MarketSnapshotHandler

DECISION = datetime(2025, 1, 2, 21, tzinfo=UTC)


def bar(symbol: str, session: date) -> DailyBar:
    stamp = datetime.combine(session, datetime.min.time(), tzinfo=UTC) + timedelta(hours=21)
    return DailyBar(
        symbol=symbol, session=session, open=Decimal(100), high=Decimal(100),
        low=Decimal(100), close=Decimal(100), adjusted_close=Decimal(100), volume=100,
        dividend=Decimal(0), split_coefficient=Decimal(1), source="fixture",
        observed_at=stamp, published_at=stamp, available_to_strategy_at=stamp,
        retrieved_at=stamp, vintage="v1", quality=Quality.PASS,
    )


class Client:
    def __init__(self, missing: str | None = None):
        self.missing = missing

    def daily(self, symbol, retrieved_at=None):
        if symbol == self.missing:
            return (bar(symbol, date(2025, 1, 1)),)
        return (bar(symbol, date(2025, 1, 1)), bar(symbol, date(2025, 1, 2)))


def payload():
    return {
        "session": "2025-01-02", "decision_at": DECISION.isoformat(),
        "next_execution_at": "2025-01-03T14:30:00+00:00", "universe_id": "core",
        "symbols": ["ABC", "SPY"], "calendar_version": "fixture-v1",
    }


def test_snapshot_handler_freezes_complete_universe_and_is_idempotent(tmp_path) -> None:
    snapshots = ResearchStore(tmp_path / "research.db")
    events = EventStore(tmp_path / "events.db")
    handler = MarketSnapshotHandler(
        lambda: Client(), snapshots, events, lambda: DECISION + timedelta(minutes=5)
    )
    handler(payload())
    handler(payload())
    rows = snapshots.connection.execute("SELECT * FROM data_snapshots").fetchall()
    assert len(rows) == 1
    assert len(events.events("core:2025-01-02")) == 1


def test_incomplete_universe_fails_closed_without_snapshot(tmp_path) -> None:
    snapshots = ResearchStore(tmp_path / "research.db")
    handler = MarketSnapshotHandler(
        lambda: Client(missing="ABC"), snapshots, EventStore(tmp_path / "events.db"),
        lambda: DECISION + timedelta(minutes=5),
    )
    with pytest.raises(RuntimeError, match="ABC"):
        handler(payload())
    assert not snapshots.connection.execute("SELECT * FROM data_snapshots").fetchall()


def test_handler_rejects_early_run(tmp_path) -> None:
    handler = MarketSnapshotHandler(
        lambda: Client(), ResearchStore(tmp_path / "research.db"),
        EventStore(tmp_path / "events.db"), lambda: DECISION - timedelta(minutes=1),
    )
    with pytest.raises(ValueError, match="before"):
        handler(payload())
