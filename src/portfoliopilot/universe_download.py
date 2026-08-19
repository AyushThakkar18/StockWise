from __future__ import annotations

import argparse
import json
import time
from datetime import date
from pathlib import Path
from urllib.error import HTTPError

from .config import Settings
from .tiingo import TiingoClient
from .universe import load_membership_history
from .universe_cache import UniverseDataCache


def main() -> None:
    parser = argparse.ArgumentParser(description="Checkpoint historical market caps for PIT universes")
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--limit", type=int, default=20, help="maximum provider calls this run")
    parser.add_argument("--delay", type=float, default=1.0, help="seconds between provider calls")
    parser.add_argument("--directory", type=Path, default=Path("private_data/universe"))
    arguments = parser.parse_args()
    if arguments.start >= arguments.end or arguments.limit <= 0 or arguments.delay < 0:
        parser.error("invalid date range or limit")
    settings = Settings.from_env()
    if not settings.tiingo_api_key:
        parser.error("TIINGO_API_KEY is missing from .env")
    history = load_membership_history(arguments.directory / "sp500-components-updated.csv")
    relevant_dates = [
        session for session in history.dates if arguments.start <= session <= arguments.end
    ]
    if not relevant_dates:
        parser.error("requested range is outside membership coverage")
    symbols = sorted({
        symbol for session in relevant_dates for symbol in history.records[session]
    } | set(history.members_on(arguments.start)))
    cache = UniverseDataCache(arguments.directory / "market_caps")
    client = TiingoClient(settings.tiingo_api_key)
    fetched, failures = 0, {}
    for symbol in symbols:
        path = cache.directory / f"{symbol}-market-cap.json"
        if path.exists():
            continue
        if fetched >= arguments.limit:
            break
        try:
            cache.market_caps(client, symbol, arguments.start, arguments.end)
        except HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:300]
            failures[symbol] = f"HTTP {exc.code}: {detail}"
        except Exception as exc:  # noqa: BLE001 - checkpoint records provider-specific failures
            failures[symbol] = f"{type(exc).__name__}: {exc}"
        fetched += 1
        time.sleep(arguments.delay)
    completed = sum((cache.directory / f"{symbol}-market-cap.json").exists() for symbol in symbols)
    print(json.dumps({
        "dataset_version": history.version, "required_symbols": len(symbols),
        "cached_symbols": completed, "provider_calls_this_run": fetched,
        "remaining_symbols": len(symbols) - completed, "failures": failures,
        "requested_range": [arguments.start.isoformat(), arguments.end.isoformat()],
        "redistribution": "private internal-use cache only",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
