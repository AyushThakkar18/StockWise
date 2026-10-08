from datetime import UTC, date, datetime
from decimal import Decimal

from portfoliopilot.yahoo_download import _ohlc, _session_close, download_quotes, yahoo_symbol


def test_yahoo_symbol_translates_class_shares() -> None:
    assert yahoo_symbol("BRK.B") == "BRK-B"
    assert yahoo_symbol("AAPL") == "AAPL"


def test_ohlc_repairs_provider_rounding_inversion() -> None:
    opened, high, low, close = _ohlc({
        "Open": 10.01, "High": 10.0, "Low": 10.02, "Close": 10.015,
    })
    assert (opened, high, low, close) == (
        Decimal("10.01"), Decimal("10.015"), Decimal("10.01"), Decimal("10.015"),
    )


def test_download_quotes_uses_last_price(monkeypatch) -> None:
    class Ticker:
        def __init__(self) -> None:
            self.fast_info = {"lastPrice": 516.39}

    monkeypatch.setattr("portfoliopilot.yahoo_download.yf.Ticker", lambda symbol: Ticker())
    assert download_quotes(("DELL",), attempts=1) == {"DELL": Decimal("516.39")}


def test_session_close_tracks_new_york_daylight_saving_time() -> None:
    assert _session_close(date(2026, 7, 1)) == datetime(2026, 7, 1, 20, tzinfo=UTC)
    assert _session_close(date(2026, 1, 2)) == datetime(2026, 1, 2, 21, tzinfo=UTC)
