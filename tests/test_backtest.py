from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest

from portfoliopilot.backtest import (
    Backtester,
    BacktestPoint,
    BuyAndHold,
    Cash,
    Momentum,
    period_hit_metrics,
)
from portfoliopilot.contracts import Quality
from portfoliopilot.market_data import DailyBar


def bars(prices: list[int], symbol: str = "ABC") -> tuple[DailyBar, ...]:
    start = date(2025, 1, 1)
    result = []
    for index, price in enumerate(prices):
        session = start + timedelta(days=index)
        stamp = datetime.combine(session, datetime.min.time(), tzinfo=UTC) + timedelta(hours=21)
        result.append(DailyBar(
            symbol=symbol, session=session, open=Decimal(price), high=Decimal(price),
            low=Decimal(price), close=Decimal(price), adjusted_close=Decimal(price),
            volume=1_000_000, dividend=Decimal(0), split_coefficient=Decimal(1),
            source="fixture", observed_at=stamp, published_at=stamp,
            available_to_strategy_at=stamp, retrieved_at=stamp, vintage="fixture",
            quality=Quality.PASS,
        ))
    return tuple(result)


def test_signal_from_close_executes_only_at_next_open() -> None:
    result = Backtester(cost_bps=Decimal(0)).run(
        bars([100, 200, 200]), bars([100, 100, 100], "SPY"), Momentum(lookback=1)
    )
    # The day-2 increase creates a signal after its close; it cannot capture that increase.
    assert result.points[-1].equity == Decimal(100_000)


def test_buy_and_hold_matches_simple_price_return_after_entry() -> None:
    result = Backtester(cost_bps=Decimal(0)).run(
        bars([100, 100, 110]), bars([100, 100, 100], "SPY"), BuyAndHold()
    )
    assert result.points[-1].equity == Decimal(110_000)
    assert result.metrics["total_return"] == pytest.approx(0.1)


def test_cash_has_zero_return_and_zero_drawdown() -> None:
    result = Backtester().run(bars([100, 80, 120]), bars([100, 100, 100], "SPY"), Cash())
    assert result.metrics["total_return"] == 0
    assert result.metrics["max_drawdown"] == 0


def test_repeated_runs_are_reproducible() -> None:
    engine = Backtester()
    inputs = bars([100, 101, 102, 99])
    benchmark = bars([100, 100, 101, 101], "SPY")
    assert engine.run(inputs, benchmark, Momentum(1)) == engine.run(inputs, benchmark, Momentum(1))


def test_split_changes_share_count_without_creating_a_loss() -> None:
    prices = list(bars([100, 50, 55]))
    prices[1] = prices[1].model_copy(update={"split_coefficient": Decimal(2)})
    result = Backtester(cost_bps=Decimal(0)).run(
        tuple(prices), bars([100, 100, 100], "SPY"), BuyAndHold()
    )
    assert result.points[1].equity == Decimal(100_000)
    assert result.points[2].equity == Decimal(110_000)


def test_dividend_is_credited_to_cash() -> None:
    prices = list(bars([100, 100, 100]))
    prices[2] = prices[2].model_copy(update={"dividend": Decimal(1)})
    result = Backtester(cost_bps=Decimal(0)).run(
        tuple(prices), bars([100, 100, 100], "SPY"), BuyAndHold()
    )
    assert result.points[-1].equity == Decimal(101_000)


def test_period_hit_metrics_measure_strategy_direction_and_benchmark_wins() -> None:
    points = tuple(
        BacktestPoint(date(2025, 1, 1) + timedelta(days=index), Decimal(equity),
                      Decimal(benchmark), Decimal(0), Decimal(0))
        for index, (equity, benchmark) in enumerate(((100, 100), (110, 105), (99, 100)))
    )
    metrics = period_hit_metrics(points, horizon=1)
    assert metrics["positive_1_session_rate"] == 0.5
    assert metrics["benchmark_win_1_session_rate"] == 0.5
    assert metrics["evaluated_1_session_periods"] == 2
