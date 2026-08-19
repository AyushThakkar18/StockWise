from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

from portfoliopilot.contracts import Quality
from portfoliopilot.market_data import DailyBar
from portfoliopilot.strategy_compare import CandidateStrategy, compare_strategies


def bars(symbol: str, multiplier: Decimal) -> tuple[DailyBar, ...]:
    output = []
    price = Decimal(100)
    for index in range(100):
        session = date(2020, 1, 1) + timedelta(days=index)
        stamp = datetime.combine(session, datetime.min.time(), tzinfo=UTC)
        price *= multiplier
        output.append(DailyBar(
            symbol=symbol, session=session, open=price, high=price, low=price, close=price,
            adjusted_close=price, volume=100, dividend=Decimal(0),
            split_coefficient=Decimal(1), source="fixture", observed_at=stamp,
            published_at=stamp, available_to_strategy_at=stamp, retrieved_at=stamp,
            vintage="v1", quality=Quality.PASS,
        ))
    return tuple(output)


def test_comparison_uses_fixed_boundaries_and_reports_holdout() -> None:
    result = compare_strategies(
        {"A": bars("A", Decimal("1.002")), "B": bars("B", Decimal("1.001"))},
        bars("SPY", Decimal("1.0005")),
        (CandidateStrategy("buy_and_hold", None), CandidateStrategy("momentum", 5)),
        cost_bps=Decimal(0),
    )
    assert result["sessions"] == 100
    assert result["development_end"] < result["validation_end"] < result["holdout_end"]
    assert set(result["selected_results"]) == {"development", "validation", "holdout"}
    assert not result["eligible_for_paper"]
    assert not result["promotion_gates"]["minimum_holdout_dates"]
    assert not result["promotion_gates"]["survivorship_safe_universe"]
