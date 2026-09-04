import sqlite3
from datetime import date
from decimal import Decimal

import pytest

from portfoliopilot.paper_store import PaperTradingStore, StrategyVersion


def version() -> StrategyVersion:
    return StrategyVersion("core-satellite", {"core": ".70"}, "source-1")


def test_decisions_are_idempotent_and_immutable(tmp_path) -> None:
    store = PaperTradingStore(tmp_path / "paper.db")
    version_hash = store.register_strategy(version())
    payload = {"targets": {"SPY": ".70", "A": ".30"}}
    first = store.record_decision(
        version_hash, date(2026, 1, 2), date(2026, 1, 5), payload,
    )
    assert store.record_decision(
        version_hash, date(2026, 1, 2), date(2026, 1, 5), payload,
    ) == first
    with pytest.raises(ValueError, match="different content"):
        store.record_decision(
            version_hash, date(2026, 1, 2), date(2026, 1, 5), {"targets": {"SPY": "1"}},
        )
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        store.connection.execute("UPDATE paper_decisions SET payload_json = '{}' WHERE id = 1")
    read_model = store.decision_payloads(version_hash)
    assert read_model[0]["decision_date"] == "2026-01-02"
    assert read_model[0]["payload"] == payload


def test_snapshots_cannot_be_rewritten(tmp_path) -> None:
    store = PaperTradingStore(tmp_path / "paper.db")
    version_hash = store.register_strategy(version())
    store.record_snapshot(
        version_hash, "blended", date(2026, 1, 5), Decimal("100000"),
        Decimal("0"), {"SPY": Decimal("10")}, "prices-1",
    )
    store.record_snapshot(
        version_hash, "blended", date(2026, 1, 5), Decimal("100000"),
        Decimal("0"), {"SPY": Decimal("10")}, "prices-1",
    )
    with pytest.raises(ValueError, match="different content"):
        store.record_snapshot(
            version_hash, "blended", date(2026, 1, 5), Decimal("99999"),
            Decimal("0"), {"SPY": Decimal("10")}, "prices-1",
        )
