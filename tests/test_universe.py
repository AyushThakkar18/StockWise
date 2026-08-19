from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest

from portfoliopilot.universe import (
    MarketCapitalization,
    load_membership_history,
    select_top_market_cap,
)


def test_membership_uses_latest_record_not_future_record(tmp_path) -> None:
    payload = (
        b"date,tickers\n"
        b'2020-01-01,"A,B,OLD-202002,BF.B"\n'
        b'2020-02-01,"A,B,C,BF.B"\n'
    )
    history = load_membership_history(tmp_path / "members.csv", lambda _: payload)
    assert history.members_on(date(2020, 1, 15)) == ("A", "B", "BF-B", "OLD")
    assert "C" not in history.members_on(date(2020, 1, 15))


def test_top_market_cap_uses_only_fresh_available_observations() -> None:
    as_of = datetime(2020, 1, 10, 21, tzinfo=UTC)

    def item(symbol, value, available):
        return MarketCapitalization(symbol, available.date(), available, Decimal(value), "fixture", "v1")

    observations = {
        "A": (item("A", 100, as_of - timedelta(days=1)),),
        "B": (item("B", 300, as_of - timedelta(days=2)),),
        "C": (
            item("C", 1000, as_of + timedelta(minutes=1)),
            item("C", 200, as_of - timedelta(days=3)),
        ),
    }
    assert select_top_market_cap(as_of, ("A", "B", "C"), observations, count=2) == ("B", "C")


def test_incomplete_market_cap_coverage_blocks_selection() -> None:
    with pytest.raises(ValueError, match="only 0"):
        select_top_market_cap(datetime(2020, 1, 1, tzinfo=UTC), ("A", "B"), {}, count=2)
