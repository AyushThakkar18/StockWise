from __future__ import annotations

import argparse
import json
import time as sleep_time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path

import yfinance as yf

from .contracts import Quality
from .market_data import DailyBar
from .point_in_time_download import required_symbols
from .price_cache import PriceCache
from .universe import load_membership_history


def yahoo_symbol(symbol: str) -> str:
    """Translate the common class-share notation used by the project."""
    return symbol.replace(".", "-")


def download_quotes(symbols: tuple[str, ...], attempts: int = 3) -> dict[str, Decimal]:
    """Fetch current/last regular-market prices, including today's completed close."""
    def one(symbol: str) -> tuple[str, Decimal | None]:
        try:
            value = yf.Ticker(yahoo_symbol(symbol)).fast_info.get("lastPrice")
            return symbol, _decimal(value) if value is not None else None
        except Exception:  # noqa: BLE001 - retry individual transient provider failures
            return symbol, None

    output, pending = {}, tuple(dict.fromkeys(symbols))
    for attempt in range(attempts):
        if not pending:
            break
        with ThreadPoolExecutor(max_workers=min(8, len(pending))) as pool:
            for symbol, value in pool.map(one, pending):
                if value is not None and value > 0:
                    output[symbol] = value
        pending = tuple(symbol for symbol in pending if symbol not in output)
        if pending and attempt + 1 < attempts:
            sleep_time.sleep(2**attempt)
    return output


def _decimal(value: object) -> Decimal:
    return Decimal(str(float(value)))


def _ohlc(row) -> tuple[Decimal, Decimal, Decimal, Decimal]:
    opened, high, low, close = (_decimal(row[key]) for key in ("Open", "High", "Low", "Close"))
    # Yahoo occasionally returns sub-cent float/rounding inversions. Preserve the observations
    # while enforcing the price-bar invariant that high/low contain open and close.
    return opened, max(high, opened, close), min(low, opened, close), close


def download_batch(symbols: tuple[str, ...], start: date, end: date) -> dict[str, tuple[DailyBar, ...]]:
    aliases = {yahoo_symbol(symbol): symbol for symbol in symbols}
    retrieved = datetime.now(UTC)
    frame = yf.download(
        list(aliases), start=start.isoformat(), end=(end + timedelta(days=1)).isoformat(),
        auto_adjust=False, actions=True, group_by="ticker", threads=True, progress=False,
    )
    output: dict[str, tuple[DailyBar, ...]] = {}
    for provider_symbol, symbol in aliases.items():
        try:
            rows = frame[provider_symbol] if getattr(frame.columns, "nlevels", 1) > 1 else frame
        except KeyError:
            continue
        bars = []
        for timestamp, row in rows.dropna(subset=["Open", "High", "Low", "Close"]).iterrows():
            session = timestamp.date()
            session_close = datetime.combine(session, time(21), tzinfo=UTC)
            adjusted = row.get("Adj Close", row["Close"])
            opened, high, low, close = _ohlc(row)
            bars.append(DailyBar(
                symbol=symbol, session=session, open=opened, high=high, low=low,
                close=close, adjusted_close=_decimal(adjusted),
                volume=int(row.get("Volume", 0) or 0),
                dividend=_decimal(row.get("Dividends", 0) or 0),
                split_coefficient=_decimal(row.get("Stock Splits", 0) or 1),
                source="Yahoo Finance via yfinance", observed_at=session_close,
                published_at=session_close, available_to_strategy_at=session_close,
                retrieved_at=retrieved, vintage=f"retrieved:{retrieved.isoformat()}",
                quality=Quality.PASS,
            ))
        if bars:
            output[symbol] = tuple(bars)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-download point-in-time universe prices from Yahoo")
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--batch-size", type=int, default=75)
    parser.add_argument(
        "--symbols-file", type=Path,
        help="optionally include symbols from a JSON object containing a symbols list",
    )
    parser.add_argument("--membership", type=Path, default=Path("private_data/universe/sp500-components-updated.csv"))
    parser.add_argument("--directory", type=Path, default=Path("private_data/prices-yahoo"))
    arguments = parser.parse_args()
    if arguments.start >= arguments.end or arguments.batch_size <= 0:
        parser.error("invalid date range or batch size")

    history = load_membership_history(arguments.membership)
    symbols = set(required_symbols(history, arguments.start, arguments.end)) | {"SPY"}
    if arguments.symbols_file:
        extra = json.loads(arguments.symbols_file.read_text(encoding="utf-8")).get("symbols")
        if not isinstance(extra, list) or not all(isinstance(item, str) for item in extra):
            parser.error("symbols-file must contain a string symbols list")
        symbols.update(item.upper() for item in extra)
    symbols = tuple(sorted(symbols))
    cache = PriceCache(arguments.directory)
    pending = tuple(s for s in symbols if cache.covering_path(s, arguments.start, arguments.end) is None)
    failures: dict[str, str] = {}
    for offset in range(0, len(pending), arguments.batch_size):
        batch = pending[offset:offset + arguments.batch_size]
        try:
            downloaded = download_batch(batch, arguments.start, arguments.end)
        except Exception as exc:  # noqa: BLE001 - retain batch-specific provider error
            failures[",".join(batch)] = f"{type(exc).__name__}: {exc}"
            continue
        for symbol, bars in downloaded.items():
            cache.write(symbol, arguments.start, arguments.end, bars)
        for symbol in set(batch) - set(downloaded):
            failures[symbol] = "no usable Yahoo price rows"

    cached = sum(cache.covering_path(s, arguments.start, arguments.end) is not None for s in symbols)
    print(json.dumps({
        "provider": "Yahoo Finance via yfinance", "dataset_version": history.version,
        "required_series": len(symbols), "cached_series": cached,
        "remaining_series": len(symbols) - cached, "failures": failures,
        "directory": str(arguments.directory),
        "note": "Yahoo is fallback research data; validate corporate actions and delistings.",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
