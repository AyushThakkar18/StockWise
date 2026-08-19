from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

from portfoliopilot.contracts import Quality
from portfoliopilot.features import FEATURE_VERSION, build_price_features
from portfoliopilot.market_data import DailyBar


def bars(count: int, symbol: str, future_last: bool = False) -> tuple[DailyBar, ...]:
    start = date(2024, 1, 1)
    result = []
    for index in range(count):
        session = start + timedelta(days=index)
        available = datetime.combine(session, datetime.min.time(), tzinfo=UTC) + timedelta(hours=21)
        if future_last and index == count - 1:
            available += timedelta(days=100)
        price = Decimal(100 + index)
        result.append(DailyBar(
            symbol=symbol, session=session, open=price, high=price, low=price, close=price,
            adjusted_close=price, volume=1_000, dividend=Decimal(0),
            split_coefficient=Decimal(1), source="fixture", observed_at=available,
            published_at=available, available_to_strategy_at=available,
            retrieved_at=available, vintage="v1", quality=Quality.PASS,
        ))
    return tuple(result)


def test_features_exclude_records_unavailable_at_decision_time() -> None:
    prices = bars(65, "ABC", future_last=True)
    benchmark = bars(65, "SPY")
    decision_at = prices[-2].available_to_strategy_at
    snapshot = build_price_features(prices, benchmark, decision_at)
    assert snapshot.as_of == prices[-2].session
    assert snapshot.values["momentum_63d"] is not None
    assert snapshot.version == FEATURE_VERSION


def test_feature_hash_is_reproducible() -> None:
    prices, benchmark = bars(30, "ABC"), bars(30, "SPY")
    decision_at = prices[-1].available_to_strategy_at
    left = build_price_features(prices, benchmark, decision_at)
    right = build_price_features(prices, benchmark, decision_at)
    assert left == right

