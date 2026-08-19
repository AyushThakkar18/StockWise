from datetime import date

import pytest

from portfoliopilot.universe_cache import UniverseDataCache


class EmptyClient:
    def market_capitalizations(self, symbol, start, end):
        return ()


def test_empty_provider_response_is_not_cached(tmp_path) -> None:
    cache = UniverseDataCache(tmp_path)
    with pytest.raises(ValueError, match="no market-cap history"):
        cache.market_caps(EmptyClient(), "ABC", date(2020, 1, 1), date(2021, 1, 1))
    assert not (tmp_path / "ABC-market-cap.json").exists()
