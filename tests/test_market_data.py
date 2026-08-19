import json
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from portfoliopilot.market_data import AlphaVantageClient


def payload() -> bytes:
    return json.dumps({"Time Series (Daily)": {
        "2025-01-03": {
            "1. open": "101", "2. high": "103", "3. low": "100", "4. close": "102",
            "5. adjusted close": "102", "6. volume": "1000",
            "7. dividend amount": "0", "8. split coefficient": "1",
        }
    }}).encode()


def test_alpha_vantage_normalization_has_point_in_time_fields() -> None:
    retrieved = datetime(2025, 2, 1, tzinfo=UTC)
    bar = AlphaVantageClient("test", fetch=lambda _: payload()).daily_adjusted("abc", retrieved)[0]
    assert bar.symbol == "ABC"
    assert bar.close == Decimal(102)
    assert bar.available_to_strategy_at < bar.retrieved_at
    assert bar.vintage == f"retrieved:{retrieved.isoformat()}"


def test_provider_error_is_not_silently_treated_as_data() -> None:
    client = AlphaVantageClient("test", fetch=lambda _: b'{"Note":"rate limit"}')
    with pytest.raises(RuntimeError, match="rate limit"):
        client.daily_adjusted("ABC")


def test_free_daily_endpoint_uses_raw_close_without_inventing_actions() -> None:
    raw = json.dumps({"Time Series (Daily)": {
        "2025-01-03": {
            "1. open": "101", "2. high": "103", "3. low": "100",
            "4. close": "102", "5. volume": "1000",
        }
    }}).encode()
    bar = AlphaVantageClient("test", fetch=lambda _: raw).daily("ABC")[0]
    assert bar.adjusted_close == bar.close
    assert bar.dividend == 0
    assert bar.split_coefficient == 1
    assert bar.vintage.startswith("raw:")
