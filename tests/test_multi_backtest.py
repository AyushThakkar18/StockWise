from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

from portfoliopilot.contracts import Quality
from portfoliopilot.market_data import DailyBar
from portfoliopilot.multi_backtest import EqualWeight, MultiAssetBacktester, RankedMomentum


def bars(symbol: str, prices: list[int]) -> tuple[DailyBar, ...]:
    output = []
    for index, price in enumerate(prices):
        session = date(2020, 1, 1) + timedelta(days=index)
        stamp = datetime.combine(session, datetime.min.time(), tzinfo=UTC)
        value = Decimal(price)
        output.append(DailyBar(
            symbol=symbol, session=session, open=value, high=value, low=value, close=value,
            adjusted_close=value, volume=1_000_000, dividend=Decimal(0),
            split_coefficient=Decimal(1), source="fixture", observed_at=stamp,
            published_at=stamp, available_to_strategy_at=stamp, retrieved_at=stamp,
            vintage="v1", quality=Quality.PASS,
        ))
    return tuple(output)


def test_equal_weight_accounting_and_next_open_execution() -> None:
    universe = {"A": bars("A", [100, 100, 110]), "B": bars("B", [100, 100, 90])}
    benchmark = bars("SPY", [100, 100, 100])
    result = MultiAssetBacktester(cost_bps=Decimal(0), rebalance_every=10).run(
        universe, benchmark, EqualWeight()
    )
    assert result.points[0].weights == {}
    assert result.points[1].weights == {"A": Decimal("0.5"), "B": Decimal("0.5")}
    assert result.points[-1].equity == Decimal(100_000)
    assert result.metrics["annual_turnover"] > 0


def test_ranked_momentum_does_not_capture_signal_day_move() -> None:
    universe = {"A": bars("A", [100, 200, 200]), "B": bars("B", [100, 100, 100])}
    benchmark = bars("SPY", [100, 100, 100])
    result = MultiAssetBacktester(cost_bps=Decimal(0), rebalance_every=1).run(
        universe, benchmark, RankedMomentum(lookback=1, top_n=1)
    )
    assert result.points[-1].equity == Decimal(100_000)


def test_transaction_costs_reduce_equity_and_cash_never_goes_negative() -> None:
    universe = {"A": bars("A", [100, 100, 100]), "B": bars("B", [100, 100, 100])}
    result = MultiAssetBacktester(cost_bps=Decimal(10)).run(
        universe, bars("SPY", [100, 100, 100]), EqualWeight()
    )
    assert result.points[-1].equity < Decimal(100_000)
    assert all(point.cash >= 0 for point in result.points)


def test_window_uses_prior_bars_for_features_but_resets_capital() -> None:
    universe = {"A": bars("A", [100, 110, 120, 130]), "B": bars("B", [100, 100, 100, 100])}
    result = MultiAssetBacktester(cost_bps=Decimal(0), rebalance_every=1).run_window(
        universe, bars("SPY", [100, 100, 100, 100]),
        RankedMomentum(lookback=1, top_n=1), date(2020, 1, 3),
    )
    assert result.points[0].equity == Decimal(100_000)
    assert result.points[1].weights == {"A": Decimal(1)}
