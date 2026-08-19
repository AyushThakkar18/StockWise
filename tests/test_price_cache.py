from datetime import UTC, date, datetime
from decimal import Decimal

from portfoliopilot.contracts import Quality
from portfoliopilot.market_data import DailyBar
from portfoliopilot.price_cache import PriceCache


def test_price_cache_round_trip_avoids_second_provider_call(tmp_path) -> None:
    timestamp = datetime(2020, 1, 2, 21, tzinfo=UTC)
    bar = DailyBar(
        symbol="ABC", session=date(2020, 1, 2), open=Decimal(1), high=Decimal(2),
        low=Decimal(1), close=Decimal(2), adjusted_close=Decimal(2), volume=10,
        dividend=Decimal("0.1"), split_coefficient=Decimal(1), source="fixture",
        observed_at=timestamp, published_at=timestamp, available_to_strategy_at=timestamp,
        retrieved_at=timestamp, vintage="v1", quality=Quality.PASS,
    )

    class Client:
        calls = 0

        def daily(self, *args, **kwargs):
            self.calls += 1
            return (bar,)

    client = Client()
    cache = PriceCache(tmp_path)
    assert cache.daily(client, "ABC", date(2020, 1, 1), date(2020, 1, 3)) == (bar,)
    assert cache.daily(client, "ABC", date(2020, 1, 1), date(2020, 1, 3)) == (bar,)
    assert client.calls == 1


def test_price_cache_reuses_and_slices_covering_range(tmp_path) -> None:
    cache = PriceCache(tmp_path)
    broad_start, broad_end = date(2020, 1, 1), date(2020, 1, 4)
    requested_start, requested_end = date(2020, 1, 2), date(2020, 1, 3)
    timestamp = datetime(2020, 1, 1, 21, tzinfo=UTC)
    source = tuple(DailyBar(
        symbol="BRK-B", session=date(2020, 1, day), open=Decimal(1), high=Decimal(1),
        low=Decimal(1), close=Decimal(1), adjusted_close=Decimal(1), volume=10,
        dividend=Decimal(0), split_coefficient=Decimal(1), source="fixture",
        observed_at=timestamp, published_at=timestamp, available_to_strategy_at=timestamp,
        retrieved_at=timestamp, vintage="v1", quality=Quality.PASS,
    ) for day in range(1, 5))

    class Client:
        def daily(self, *args, **kwargs):
            return source

    cache.daily(Client(), "BRK-B", broad_start, broad_end)
    result = cache.daily(None, "BRK-B", requested_start, requested_end)  # type: ignore[arg-type]
    assert tuple(bar.session for bar in result) == (requested_start, requested_end)
