from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

from .backtest import Backtester, BuyAndHold, Cash, Momentum
from .config import Settings
from .market_data import AlphaVantageClient, DailyBar
from .tiingo import TiingoClient


def run_backtest(
    client,
    symbol: str,
    benchmark_symbol: str,
    start: date,
    end: date,
    strategy_name: str,
    momentum_lookback: int,
    starting_cash: Decimal,
    cost_bps: Decimal,
) -> dict[str, Any]:
    if start >= end:
        raise ValueError("start date must precede end date")
    strategies = {
        "cash": Cash(), "buy_and_hold": BuyAndHold(),
        "momentum": Momentum(momentum_lookback),
    }
    if strategy_name not in strategies:
        raise ValueError(f"unknown strategy: {strategy_name}")
    prices = _between(_daily(client, symbol.upper(), start, end), start, end)
    if isinstance(client, AlphaVantageClient) and symbol.upper() != benchmark_symbol.upper():
        # Respect the provider's free-key burst guidance between symbol requests.
        time.sleep(1.1)
    benchmark = _between(_daily(client, benchmark_symbol.upper(), start, end), start, end)
    result = Backtester(starting_cash, cost_bps).run(prices, benchmark, strategies[strategy_name])
    protocol = {
        "symbol": symbol.upper(), "benchmark": benchmark_symbol.upper(),
        "start": start.isoformat(), "end": end.isoformat(), "strategy": strategy_name,
        "momentum_lookback": momentum_lookback, "starting_cash": str(starting_cash),
        "cost_bps": str(cost_bps), "sessions": len(result.points),
        "execution": "signal at close, trade at next open",
        "price_field": "raw close; corporate actions are not point-in-time reconstructed",
    }
    protocol_hash = hashlib.sha256(
        json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "protocol": protocol, "protocol_hash": protocol_hash, "metrics": result.metrics,
        "first_session": result.points[0].session.isoformat(),
        "last_session": result.points[-1].session.isoformat(),
        "ending_value": str(result.points[-1].equity),
        "benchmark_ending_value": str(result.points[-1].benchmark_equity),
        "warning": (
            "Research result only. This single-symbol test is not evidence of future profitability; "
            "Alpha Vantage current history is not survivorship-safe or vintage-adjusted."
        ),
    }


def _between(bars: tuple[DailyBar, ...], start: date, end: date) -> tuple[DailyBar, ...]:
    selected = tuple(bar for bar in bars if start <= bar.session <= end)
    if len(selected) < 2:
        raise ValueError("provider returned fewer than two sessions in the requested range")
    return selected


def _daily(client, symbol: str, start: date, end: date) -> tuple[DailyBar, ...]:
    if isinstance(client, TiingoClient):
        return client.daily(symbol, start_date=start, end_date=end)
    return client.daily(symbol)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a point-in-time execution-safe baseline backtest")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--benchmark", default="SPY")
    parser.add_argument("--provider", choices=("alpha", "tiingo"), default="alpha")
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--strategy", choices=("cash", "buy_and_hold", "momentum"), default="momentum")
    parser.add_argument("--lookback", type=int, default=21)
    parser.add_argument("--starting-cash", type=Decimal, default=Decimal(100_000))
    parser.add_argument("--cost-bps", type=Decimal, default=Decimal(5))
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    settings = Settings.from_env()
    if arguments.provider == "alpha" and not settings.alpha_vantage_api_key:
        parser.error("ALPHA_VANTAGE_API_KEY is missing from .env")
    if arguments.provider == "tiingo" and not settings.tiingo_api_key:
        parser.error("TIINGO_API_KEY is missing from .env")
    client = (
        AlphaVantageClient(settings.alpha_vantage_api_key or "")
        if arguments.provider == "alpha" else TiingoClient(settings.tiingo_api_key or "")
    )
    output = run_backtest(
        client, arguments.symbol,
        arguments.benchmark, arguments.start, arguments.end, arguments.strategy,
        arguments.lookback, arguments.starting_cash, arguments.cost_bps,
    )
    rendered = json.dumps(output, indent=2, sort_keys=True)
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
