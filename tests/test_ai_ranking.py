from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest

from portfoliopilot.ai_ranking import AIRankedPortfolio, feature_packet
from portfoliopilot.contracts import Quality
from portfoliopilot.market_data import DailyBar


def bars(symbol: str, growth: Decimal) -> tuple[DailyBar, ...]:
    start = date(2020, 1, 1)
    output = []
    for index in range(254):
        session = start + timedelta(days=index)
        timestamp = datetime.combine(session, datetime.min.time(), tzinfo=UTC)
        price = Decimal(100) * growth ** index
        output.append(DailyBar(
            symbol=symbol, session=session, open=price, high=price, low=price, close=price,
            adjusted_close=price, volume=100, dividend=Decimal(0), split_coefficient=Decimal(1),
            source="fixture", observed_at=timestamp, published_at=timestamp,
            available_to_strategy_at=timestamp, retrieved_at=timestamp, vintage="v1",
            quality=Quality.PASS,
        ))
    return tuple(output)


def test_feature_packet_anonymizes_symbols_and_contains_only_lagged_features() -> None:
    packet, aliases = feature_packet({"SECRET": bars("SECRET", Decimal("1.001"))})
    rendered = str(packet)
    assert "SECRET" not in rendered
    assert aliases == {"asset_001": "SECRET"}


def test_ai_portfolio_maps_validated_anonymous_ranking_to_weights() -> None:
    history = {"A": bars("A", Decimal("1.001")), "B": bars("B", Decimal("1.002"))}
    strategy = AIRankedPortfolio(lambda packet: ("asset_002", "asset_001"), top_n=1)
    assert strategy.targets(history) == {"B": Decimal(1)}


def test_ai_portfolio_rejects_unknown_alias() -> None:
    strategy = AIRankedPortfolio(lambda packet: ("unknown",), top_n=1)
    with pytest.raises(KeyError):
        strategy.targets({"A": bars("A", Decimal("1.001"))})
