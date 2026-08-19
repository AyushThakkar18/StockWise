from __future__ import annotations

import json
import re
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

from .contracts import Quality
from .market_data import DailyBar
from .tiingo import TiingoClient


class PriceCache:
    """Private checkpoint cache for raw provider bars."""

    def __init__(self, directory: Path):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)

    def daily(
        self, client: TiingoClient, symbol: str, start: date, end: date,
    ) -> tuple[DailyBar, ...]:
        path = self.covering_path(symbol, start, end)
        if path is not None:
            bars = tuple(
                self._decode(item) for item in json.loads(path.read_text(encoding="utf-8"))
            )
            return tuple(bar for bar in bars if start <= bar.session <= end)
        path = self.directory / f"{symbol.upper()}-{start}-{end}.json"
        bars = client.daily(symbol, start_date=start, end_date=end)
        if not bars:
            raise ValueError(f"Tiingo returned no prices for {symbol}")
        path.write_text(json.dumps([self._encode(bar) for bar in bars], separators=(",", ":")),
                        encoding="utf-8")
        return bars

    def covering_path(self, symbol: str, start: date, end: date) -> Path | None:
        exact = self.directory / f"{symbol.upper()}-{start}-{end}.json"
        if exact.exists():
            return exact
        pattern = re.compile(
            rf"^{re.escape(symbol.upper())}-(\d{{4}}-\d{{2}}-\d{{2}})-(\d{{4}}-\d{{2}}-\d{{2}})\.json$"
        )
        candidates = []
        for candidate in self.directory.glob(f"{symbol.upper()}-*.json"):
            match = pattern.match(candidate.name)
            if not match:
                continue
            cached_start, cached_end = map(date.fromisoformat, match.groups())
            if cached_start <= start and cached_end >= end:
                candidates.append((cached_end - cached_start, candidate.name, candidate))
        return min(candidates)[2] if candidates else None

    @staticmethod
    def _encode(bar: DailyBar) -> dict[str, object]:
        item = bar.model_dump()
        for key in ("session", "observed_at", "published_at", "available_to_strategy_at", "retrieved_at"):
            item[key] = item[key].isoformat()
        for key in ("open", "high", "low", "close", "adjusted_close", "dividend", "split_coefficient"):
            item[key] = str(item[key])
        item["quality"] = bar.quality.value
        return item

    @staticmethod
    def _decode(item: dict[str, object]) -> DailyBar:
        values = dict(item)
        values["session"] = date.fromisoformat(str(values["session"]))
        for key in ("observed_at", "published_at", "available_to_strategy_at", "retrieved_at"):
            values[key] = datetime.fromisoformat(str(values[key]))
        for key in ("open", "high", "low", "close", "adjusted_close", "dividend", "split_coefficient"):
            values[key] = Decimal(str(values[key]))
        values["quality"] = Quality(str(values["quality"]))
        return DailyBar(**values)
