import json
from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from portfoliopilot.tiingo import TiingoClient


def response() -> bytes:
    return json.dumps([{
        "date": "2020-01-02T00:00:00.000Z", "open": 100, "high": 103,
        "low": 99, "close": 102, "volume": 1000, "divCash": 0.25,
        "splitFactor": 1.0, "adjClose": 51,
    }]).encode()


def test_tiingo_normalizes_raw_prices_and_corporate_action_fields() -> None:
    observed = {}

    def fetch(url, headers):
        observed.update(url=url, headers=headers)
        return response()

    retrieved = datetime(2025, 1, 1, tzinfo=UTC)
    bar = TiingoClient("secret", fetch).daily(
        "abc", retrieved, date(2020, 1, 1), date(2020, 1, 3)
    )[0]
    assert bar.close == Decimal(102)
    assert bar.adjusted_close == bar.close
    assert bar.dividend == Decimal("0.25")
    assert bar.split_coefficient == 1
    assert "secret" not in observed["url"]
    assert observed["headers"]["Authorization"] == "Token secret"
    assert "startDate=2020-01-01" in observed["url"]


def test_tiingo_provider_error_fails_closed() -> None:
    client = TiingoClient("secret", lambda *_: b'{"detail":"rate limited"}')
    with pytest.raises(RuntimeError, match="rate limited"):
        client.daily("ABC")


def test_tiingo_fundamentals_normalize_market_cap() -> None:
    raw = json.dumps([{
        "date": "2020-01-02T00:00:00.000Z", "marketCap": 123456789,
    }]).encode()
    item = TiingoClient("secret", lambda *_: raw).market_capitalizations(
        "abc", date(2020, 1, 1), date(2020, 1, 3)
    )[0]
    assert item.symbol == "ABC"
    assert item.market_cap == Decimal(123456789)
    assert item.observed_date == date(2020, 1, 2)
