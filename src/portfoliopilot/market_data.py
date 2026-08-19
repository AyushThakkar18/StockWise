from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, date, datetime, time
from decimal import Decimal
from urllib.parse import urlencode
from urllib.request import urlopen

from pydantic import Field, model_validator

from .contracts import FrozenModel, Quality


class DailyBar(FrozenModel):
    symbol: str
    session: date
    open: Decimal = Field(gt=0)
    high: Decimal = Field(gt=0)
    low: Decimal = Field(gt=0)
    close: Decimal = Field(gt=0)
    adjusted_close: Decimal = Field(gt=0)
    volume: int = Field(ge=0)
    dividend: Decimal = Field(ge=0)
    split_coefficient: Decimal = Field(gt=0)
    source: str
    observed_at: datetime
    published_at: datetime
    available_to_strategy_at: datetime
    retrieved_at: datetime
    vintage: str
    quality: Quality = Quality.PASS

    @model_validator(mode="after")
    def validate_prices_and_time(self) -> DailyBar:
        if self.low > min(self.open, self.close) or self.high < max(self.open, self.close):
            raise ValueError("OHLC range is inconsistent")
        if self.published_at > self.available_to_strategy_at:
            raise ValueError("availability cannot precede publication")
        return self


class AlphaVantageClient:
    endpoint = "https://www.alphavantage.co/query"

    def __init__(self, api_key: str, fetch: Callable[[str], bytes] | None = None):
        if not api_key:
            raise ValueError("ALPHA_VANTAGE_API_KEY is required")
        self.api_key = api_key
        self.fetch = fetch or self._fetch

    @staticmethod
    def _fetch(url: str) -> bytes:
        with urlopen(url, timeout=30) as response:
            return response.read()

    def daily_adjusted(self, symbol: str, retrieved_at: datetime | None = None) -> tuple[DailyBar, ...]:
        retrieved = retrieved_at or datetime.now(UTC)
        query = urlencode({
            "function": "TIME_SERIES_DAILY_ADJUSTED", "symbol": symbol,
            "outputsize": "full", "apikey": self.api_key,
        })
        payload = json.loads(self.fetch(f"{self.endpoint}?{query}"))
        if "Error Message" in payload or "Note" in payload or "Information" in payload:
            message = payload.get("Error Message") or payload.get("Note") or payload.get("Information")
            raise RuntimeError(f"Alpha Vantage response: {message}")
        series = payload.get("Time Series (Daily)")
        if not isinstance(series, dict):
            raise TypeError("Alpha Vantage daily series missing")
        bars = []
        for raw_date, row in series.items():
            session = date.fromisoformat(raw_date)
            # Daily prices are usable after the represented US session. Retrieval is recorded
            # separately; current adjusted history is not historical corporate-action truth.
            session_close = datetime.combine(session, time(21), tzinfo=UTC)
            bars.append(DailyBar(
                symbol=symbol.upper(), session=session, open=Decimal(row["1. open"]),
                high=Decimal(row["2. high"]), low=Decimal(row["3. low"]),
                close=Decimal(row["4. close"]), adjusted_close=Decimal(row["5. adjusted close"]),
                volume=int(row["6. volume"]), dividend=Decimal(row["7. dividend amount"]),
                split_coefficient=Decimal(row["8. split coefficient"]), source="Alpha Vantage",
                observed_at=session_close, published_at=session_close,
                available_to_strategy_at=session_close, retrieved_at=retrieved,
                vintage=f"retrieved:{retrieved.isoformat()}",
            ))
        return tuple(sorted(bars, key=lambda bar: bar.session))

    def daily(self, symbol: str, retrieved_at: datetime | None = None) -> tuple[DailyBar, ...]:
        """Free raw daily endpoint; corporate actions are intentionally not synthesized."""
        retrieved = retrieved_at or datetime.now(UTC)
        query = urlencode({
            "function": "TIME_SERIES_DAILY", "symbol": symbol,
            "outputsize": "compact", "apikey": self.api_key,
        })
        payload = json.loads(self.fetch(f"{self.endpoint}?{query}"))
        if "Error Message" in payload or "Note" in payload or "Information" in payload:
            message = payload.get("Error Message") or payload.get("Note") or payload.get("Information")
            raise RuntimeError(f"Alpha Vantage response: {message}")
        series = payload.get("Time Series (Daily)")
        if not isinstance(series, dict):
            raise TypeError("Alpha Vantage daily series missing")
        bars = []
        for raw_date, row in series.items():
            session = date.fromisoformat(raw_date)
            session_close = datetime.combine(session, time(21), tzinfo=UTC)
            close = Decimal(row["4. close"])
            bars.append(DailyBar(
                symbol=symbol.upper(), session=session, open=Decimal(row["1. open"]),
                high=Decimal(row["2. high"]), low=Decimal(row["3. low"]), close=close,
                adjusted_close=close, volume=int(row["5. volume"]), dividend=Decimal(0),
                split_coefficient=Decimal(1), source="Alpha Vantage",
                observed_at=session_close, published_at=session_close,
                available_to_strategy_at=session_close, retrieved_at=retrieved,
                vintage=f"raw:retrieved:{retrieved.isoformat()}",
            ))
        return tuple(sorted(bars, key=lambda bar: bar.session))
