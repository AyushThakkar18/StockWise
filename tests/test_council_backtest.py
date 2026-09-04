from datetime import UTC, date, datetime
from decimal import Decimal

from portfoliopilot.contracts import Quality
from portfoliopilot.council_backtest import price_evidence
from portfoliopilot.market_data import DailyBar


def test_price_evidence_is_lagged_and_source_identified() -> None:
    bars = tuple(
        DailyBar(
            symbol="ABC", session=date(2024, 1, 1).fromordinal(date(2024, 1, 1).toordinal() + i),
            open=Decimal(100 + i), high=Decimal(101 + i), low=Decimal(99 + i),
            close=Decimal(100 + i), adjusted_close=Decimal(100 + i), volume=10,
            dividend=Decimal(0), split_coefficient=Decimal(1), source="Yahoo Finance via yfinance",
            observed_at=datetime(2024, 1, 1, tzinfo=UTC),
            published_at=datetime(2024, 1, 1, tzinfo=UTC),
            available_to_strategy_at=datetime(2024, 1, 1, tzinfo=UTC),
            retrieved_at=datetime(2026, 1, 1, tzinfo=UTC), vintage="test", quality=Quality.PASS,
        ) for i in range(253)
    )
    evidence = price_evidence("ABC", bars[-1].session, bars)
    assert evidence.source == "Yahoo Finance via yfinance"
    assert "252-session return" in evidence.claim
