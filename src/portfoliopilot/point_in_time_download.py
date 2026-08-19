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
from .universe import load_membership_history


def required_symbols(history, start: date, end: date) -> tuple[str, ...]:
    dated = {
        symbol for session in history.dates if start <= session <= end
        for symbol in history.records[session]
    }
    return tuple(sorted(dated | set(history.members_on(start))))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Checkpoint prices for every point-in-time S&P 500 constituent",
    )
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--delay", type=float, default=0.25)
    parser.add_argument(
        "--membership", type=Path,
        default=Path("private_data/universe/sp500-components-updated.csv"),
    )
    parser.add_argument("--directory", type=Path, default=Path("private_data/prices"))
    arguments = parser.parse_args()
    if arguments.start >= arguments.end or arguments.limit <= 0 or arguments.delay < 0:
        parser.error("invalid date range, limit, or delay")
    settings = Settings.from_env()
    if not settings.tiingo_api_key:
        parser.error("TIINGO_API_KEY is missing from .env")
    history = load_membership_history(arguments.membership)
    symbols = required_symbols(history, arguments.start, arguments.end) + ("SPY",)
    cache = PriceCache(arguments.directory)
    client = TiingoClient(settings.tiingo_api_key)
    calls, failures = 0, {}
    for symbol in symbols:
        if cache.covering_path(symbol, arguments.start, arguments.end) is not None:
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
        except Exception as exc:  # noqa: BLE001 - preserve symbol-specific provider failures
            failures[symbol] = f"{type(exc).__name__}: {exc}"
            calls += 1
        else:
            calls += 1
        time.sleep(arguments.delay)
    cached = sum(
        cache.covering_path(symbol, arguments.start, arguments.end) is not None
        for symbol in symbols
    )
    print(json.dumps({
        "dataset_version": history.version,
        "historical_membership": True,
        "required_series": len(symbols),
        "cached_series": cached,
        "remaining_series": len(symbols) - cached,
        "provider_calls_this_run": calls,
        "failures": failures,
        "note": "A valid run also requires delisting/terminal-return coverage.",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
