import sqlite3
from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from portfoliopilot.contracts import PolicyCheck, Quality
from portfoliopilot.market_data import DailyBar
from portfoliopilot.research_contracts import CouncilAudit, CouncilResult
from portfoliopilot.snapshots import ResearchStore


def bar(available: datetime) -> DailyBar:
    return DailyBar(
        symbol="ABC", session=date(2025, 1, 1), open=Decimal(100), high=Decimal(100),
        low=Decimal(100), close=Decimal(100), adjusted_close=Decimal(100), volume=100,
        dividend=Decimal(0), split_coefficient=Decimal(1), source="fixture",
        observed_at=available, published_at=available, available_to_strategy_at=available,
        retrieved_at=available, vintage="v1", quality=Quality.PASS,
    )


def council(as_of: datetime) -> CouncilResult:
    return CouncilResult(
        symbol="ABC", as_of=as_of, evidence_ids=("e1",), reports=(),
        audit=CouncilAudit(
            approved=False,
            checks=(PolicyCheck(name="all_roles", passed=False, detail="fixture"),),
            contradictions=(), blocker_codes=("FIXTURE",), evidence_coverage=0,
        ),
    )


def test_snapshot_rejects_future_data_and_is_immutable(tmp_path) -> None:
    store = ResearchStore(tmp_path / "research.db")
    as_of = datetime(2025, 1, 2, tzinfo=UTC)
    digest = store.save_bars("s1", as_of, (bar(as_of),))
    assert len(digest) == 64
    with pytest.raises(ValueError, match="unavailable"):
        store.save_bars("future", as_of, (bar(as_of.replace(day=3)),))
    with pytest.raises(sqlite3.IntegrityError):
        store.connection.execute(
            "UPDATE data_snapshots SET payload_hash = 'changed' WHERE snapshot_id = 's1'"
        )


def test_council_result_round_trips_with_hash_verification(tmp_path) -> None:
    store = ResearchStore(tmp_path / "research.db")
    result = council(datetime(2025, 1, 2, tzinfo=UTC))
    store.save_council("r1", result)
    assert store.load_council("r1") == result

