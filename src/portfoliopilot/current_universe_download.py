from __future__ import annotations

import argparse
import json
import time
from datetime import date
from pathlib import Path
from urllib.error import HTTPError

from .config import Settings
from .price_cache import PriceCache
from .tiingo import TiingoClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Checkpoint prices for a frozen current universe")
    parser.add_argument("--universe", type=Path, required=True)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--delay", type=float, default=0.25)
    parser.add_argument("--directory", type=Path, default=Path("private_data/prices"))
    arguments = parser.parse_args()
    if arguments.start >= arguments.end or arguments.limit <= 0 or arguments.delay < 0:
        parser.error("invalid date range, limit, or delay")
    snapshot = json.loads(arguments.universe.read_text(encoding="utf-8"))
    symbols = tuple(snapshot["symbols"]) + ("SPY",)
    if len(snapshot["symbols"]) != snapshot["count"] or len(set(symbols)) != len(symbols):
        parser.error("invalid universe snapshot")
    settings = Settings.from_env()
    if not settings.tiingo_api_key:
        parser.error("TIINGO_API_KEY is missing from .env")
    cache = PriceCache(arguments.directory)
    client = TiingoClient(settings.tiingo_api_key)
    calls, failures = 0, {}
    for symbol in symbols:
        path = cache.directory / f"{symbol}-{arguments.start}-{arguments.end}.json"
        if path.exists():
            continue
        if calls >= arguments.limit:
            break
        try:
            cache.daily(client, symbol, arguments.start, arguments.end)
        except HTTPError as exc:
            failures[symbol] = f"HTTP {exc.code}: {exc.read().decode(errors='replace')[:300]}"
            calls += 1
            if exc.code == 429:
                break
        except Exception as exc:  # noqa: BLE001 - report provider-specific failures
            failures[symbol] = f"{type(exc).__name__}: {exc}"
            calls += 1
        else:
            calls += 1
        time.sleep(arguments.delay)
    cached = sum(
        (cache.directory / f"{symbol}-{arguments.start}-{arguments.end}.json").exists()
        for symbol in symbols
    )
    print(json.dumps({
        "universe_as_of": snapshot["as_of"], "survivorship_safe": False,
        "required_series": len(symbols), "cached_series": cached,
        "remaining_series": len(symbols) - cached, "provider_calls_this_run": calls,
        "failures": failures,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
