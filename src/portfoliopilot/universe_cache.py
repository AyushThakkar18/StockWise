from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

from .tiingo import TiingoClient
from .universe import MarketCapitalization


class UniverseDataCache:
    """Private checkpoint cache; contents must not be committed or redistributed."""

    def __init__(self, directory: Path):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)

    def market_caps(
        self, client: TiingoClient, symbol: str, start: date, end: date,
    ) -> tuple[MarketCapitalization, ...]:
        path = self.directory / f"{symbol.upper()}-market-cap.json"
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            return tuple(MarketCapitalization(
                symbol=item["symbol"], observed_date=date.fromisoformat(item["observed_date"]),
                available_to_strategy_at=datetime.fromisoformat(item["available_to_strategy_at"]),
                market_cap=Decimal(item["market_cap"]), source=item["source"], vintage=item["vintage"],
            ) for item in payload)
        observations = client.market_capitalizations(symbol, start, end)
        if not observations:
            raise ValueError(f"Tiingo returned no market-cap history for {symbol}")
        payload = [
            {**asdict(item), "observed_date": item.observed_date.isoformat(),
             "available_to_strategy_at": item.available_to_strategy_at.isoformat(),
             "market_cap": str(item.market_cap)}
            for item in observations
        ]
        path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        return observations
