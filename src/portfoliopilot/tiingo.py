from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, date, datetime, time
from decimal import Decimal
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .contracts import Quality
from .market_data import DailyBar
from .universe import MarketCapitalization


class TiingoClient:
    endpoint = "https://api.tiingo.com/tiingo/daily"

    def __init__(
        self, api_key: str,
        fetch: Callable[[str, dict[str, str]], bytes] | None = None,
    ):
        if not api_key:
            raise ValueError("TIINGO_API_KEY is required")
        self.api_key = api_key
        self.fetch = fetch or self._fetch

    @staticmethod
    def _fetch(url: str, headers: dict[str, str]) -> bytes:
        request = Request(url, headers=headers)
        with urlopen(request, timeout=30) as response:
            return response.read()

    def daily(
        self, symbol: str, retrieved_at: datetime | None = None,
        start_date: date | None = None, end_date: date | None = None,
    ) -> tuple[DailyBar, ...]:
        retrieved = retrieved_at or datetime.now(UTC)
        parameters = {"resampleFreq": "daily", "format": "json"}
        if start_date:
            parameters["startDate"] = start_date.isoformat()
        if end_date:
            parameters["endDate"] = end_date.isoformat()
        url = f"{self.endpoint}/{symbol.upper()}/prices?{urlencode(parameters)}"
        payload = json.loads(self.fetch(url, {"Authorization": f"Token {self.api_key}"}))
        if isinstance(payload, dict):
            message = payload.get("detail") or payload.get("message") or payload
            raise RuntimeError(f"Tiingo response: {message}")  # noqa: TRY004 - provider error payload
        if not isinstance(payload, list):
            raise TypeError("Tiingo daily response must be a list")
        bars = []
        for row in payload:
            session = date.fromisoformat(row["date"][:10])
            session_close = datetime.combine(session, time(21), tzinfo=UTC)
            bars.append(DailyBar(
                symbol=symbol.upper(), session=session, open=Decimal(str(row["open"])),
                high=Decimal(str(row["high"])), low=Decimal(str(row["low"])),
                close=Decimal(str(row["close"])), adjusted_close=Decimal(str(row["close"])),
                volume=int(row.get("volume") or 0),
                dividend=Decimal(str(row.get("divCash") or 0)),
                split_coefficient=Decimal(str(row.get("splitFactor") or 1)),
                source="Tiingo", observed_at=session_close, published_at=session_close,
                available_to_strategy_at=session_close, retrieved_at=retrieved,
                vintage=f"raw:retrieved:{retrieved.isoformat()}", quality=Quality.PASS,
            ))
        return tuple(sorted(bars, key=lambda bar: bar.session))

    def market_capitalizations(
        self, symbol: str, start_date: date, end_date: date,
        retrieved_at: datetime | None = None,
    ) -> tuple[MarketCapitalization, ...]:
        retrieved = retrieved_at or datetime.now(UTC)
        parameters = urlencode({
            "startDate": start_date.isoformat(), "endDate": end_date.isoformat(),
        })
        url = f"https://api.tiingo.com/tiingo/fundamentals/{symbol.upper()}/daily?{parameters}"
        payload = json.loads(self.fetch(url, {"Authorization": f"Token {self.api_key}"}))
        if isinstance(payload, dict):
            message = payload.get("detail") or payload.get("message") or payload
            raise RuntimeError(f"Tiingo response: {message}")  # noqa: TRY004
        if not isinstance(payload, list):
            raise TypeError("Tiingo fundamentals response must be a list")
        observations = []
        for row in payload:
            if row.get("marketCap") is None:
                continue
            observed = date.fromisoformat(row["date"][:10])
            available = datetime.combine(observed, time(21), tzinfo=UTC)
            observations.append(MarketCapitalization(
                symbol=symbol.upper(), observed_date=observed,
                available_to_strategy_at=available,
                market_cap=Decimal(str(row["marketCap"])), source="Tiingo Fundamentals",
                vintage=f"retrieved:{retrieved.isoformat()}",
            ))
        return tuple(sorted(observations, key=lambda item: item.observed_date))
