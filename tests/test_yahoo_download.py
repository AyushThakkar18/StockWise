from decimal import Decimal

from portfoliopilot.yahoo_download import _ohlc, download_quotes, yahoo_symbol


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
