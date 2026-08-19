from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest

from portfoliopilot.backtest_cli import run_backtest
from portfoliopilot.contracts import Quality
from portfoliopilot.market_data import DailyBar


def bars(symbol: str, values: tuple[int, ...]) -> tuple[DailyBar, ...]:
    output = []
    for index, value in enumerate(values):
        session = date(2020, 1, 1) + timedelta(days=index)
        stamp = datetime.combine(session, datetime.min.time(), tzinfo=UTC)
        price = Decimal(value)
        output.append(DailyBar(
            symbol=symbol, session=session, open=price, high=price, low=price, close=price,
            adjusted_close=price, volume=100, dividend=Decimal(0),
            split_coefficient=Decimal(1), source="fixture", observed_at=stamp,
            published_at=stamp, available_to_strategy_at=stamp, retrieved_at=stamp,
            vintage="v1", quality=Quality.PASS,
        ))
    return tuple(output)


class Client:
    def daily(self, symbol):
        if symbol == "SPY":
            return bars(symbol, (100, 100, 100, 100))
        return bars(symbol, (100, 100, 110, 121))


def test_cli_workflow_returns_reproducible_protocol_and_metrics() -> None:
    arguments = (
        Client(), "ABC", "SPY", date(2020, 1, 1), date(2020, 1, 4),
        "buy_and_hold", 1, Decimal(100_000), Decimal(0),
    )
    first = run_backtest(*arguments)
    second = run_backtest(*arguments)
    assert first == second
    assert first["ending_value"] == "121000"
    assert first["metrics"]["excess_return"] > 0


def test_cli_rejects_invalid_date_range() -> None:
    with pytest.raises(ValueError, match="precede"):
        run_backtest(
            Client(), "ABC", "SPY", date(2020, 1, 4), date(2020, 1, 1),
            "cash", 1, Decimal(1000), Decimal(0),
        )
