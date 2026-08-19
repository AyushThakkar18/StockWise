from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

from portfoliopilot.contracts import Quality
from portfoliopilot.dataset import build_training_rows
from portfoliopilot.features import FeatureSnapshot
from portfoliopilot.market_data import DailyBar


def bars(symbol: str) -> tuple[DailyBar, ...]:
    output = []
    for index, price in enumerate((100, 101, 103, 106)):
        session = date(2020, 1, 1) + timedelta(days=index)
        stamp = datetime.combine(session, datetime.min.time(), tzinfo=UTC)
        output.append(DailyBar(
            symbol=symbol, session=session, open=Decimal(price), high=Decimal(price),
            low=Decimal(price), close=Decimal(price), adjusted_close=Decimal(price),
            volume=100, dividend=Decimal(0), split_coefficient=Decimal(1), source="fixture",
            observed_at=stamp, published_at=stamp, available_to_strategy_at=stamp,
            retrieved_at=stamp, vintage="v1", quality=Quality.PASS,
        ))
    return tuple(output)


def test_labels_respect_cutoff_and_are_not_features() -> None:
    stock, benchmark = bars("ABC"), bars("SPY")
    snapshots = tuple(
        FeatureSnapshot("ABC", bar.session, bar.available_to_strategy_at, "f1", str(index), {"x": 1.0})
        for index, bar in enumerate(stock[:3])
    )
    rows = build_training_rows(
        snapshots, {"ABC": stock}, benchmark, horizon_sessions=1, label_cutoff=stock[1].session
    )
    assert len(rows) == 1
    assert rows[0].label_end_date == stock[1].session
    assert "excess_return" not in rows[0].features

