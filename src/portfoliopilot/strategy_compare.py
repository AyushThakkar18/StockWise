from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from statistics import fmean
from typing import Any

from .backtest import Backtester, BuyAndHold, Momentum, performance_metrics
from .config import Settings
from .market_data import AlphaVantageClient, DailyBar
from .tiingo import TiingoClient


@dataclass(frozen=True)
class CandidateStrategy:
    name: str
    lookback: int | None


def compare_strategies(
    prices: dict[str, tuple[DailyBar, ...]], benchmark: tuple[DailyBar, ...],
    candidates: tuple[CandidateStrategy, ...], starting_cash: Decimal = Decimal(100_000),
    cost_bps: Decimal = Decimal(5), minimum_stage_dates: int = 60,
    survivorship_safe: bool = False, maximum_drawdown: float = -0.25,
) -> dict[str, Any]:
    if len(prices) < 2:
        raise ValueError("at least two stocks are required for comparison")
    results = {}
    minimum_points = None
    for candidate in candidates:
        strategy_results = {}
        for symbol, bars in prices.items():
            strategy = BuyAndHold() if candidate.lookback is None else Momentum(candidate.lookback)
            result = Backtester(starting_cash, cost_bps).run(bars, benchmark, strategy)
            strategy_results[symbol] = result
            minimum_points = len(result.points) if minimum_points is None else min(minimum_points, len(result.points))
        results[_key(candidate)] = strategy_results
    if minimum_points is None or minimum_points < 60:
        raise ValueError("at least 60 aligned sessions are required")
    development_end = max(30, int(minimum_points * 0.60))
    validation_end = max(development_end + 15, int(minimum_points * 0.80))
    if validation_end >= minimum_points:
        raise ValueError("not enough sessions for development, validation, and holdout")
    boundaries = {
        "development": (0, development_end),
        "validation": (development_end - 1, validation_end),
        "holdout": (validation_end - 1, minimum_points),
    }
    evaluations: dict[str, dict[str, Any]] = {}
    for key, symbol_results in results.items():
        evaluations[key] = {}
        for stage, (start, end) in boundaries.items():
            per_symbol = {
                symbol: performance_metrics(tuple(result.points[start:end]))
                for symbol, result in symbol_results.items()
            }
            evaluations[key][stage] = {
                "mean_information_ratio": fmean(item["information_ratio"] for item in per_symbol.values()),
                "mean_excess_return": fmean(item["excess_return"] for item in per_symbol.values()),
                "mean_sharpe": fmean(item["sharpe"] for item in per_symbol.values()),
                "worst_drawdown": min(item["max_drawdown"] for item in per_symbol.values()),
                "per_symbol": per_symbol,
            }
    selected = max(
        evaluations,
        key=lambda key: (
            evaluations[key]["development"]["mean_information_ratio"],
            evaluations[key]["development"]["mean_excess_return"], key,
        ),
    )
    development = evaluations[selected]["development"]
    validation = evaluations[selected]["validation"]
    holdout = evaluations[selected]["holdout"]
    holdout_dates = min(
        item["decision_dates"] for item in holdout["per_symbol"].values()
    )
    validation_dates = min(
        item["decision_dates"] for item in validation["per_symbol"].values()
    )
    gates = {
        "development_excess_positive": development["mean_excess_return"] > 0,
        "validation_excess_positive": validation["mean_excess_return"] > 0,
        "validation_information_ratio_positive": validation["mean_information_ratio"] > 0,
        "holdout_excess_positive": holdout["mean_excess_return"] > 0,
        "holdout_information_ratio_positive": holdout["mean_information_ratio"] > 0,
        "minimum_validation_dates": validation_dates >= minimum_stage_dates,
        "minimum_holdout_dates": holdout_dates >= minimum_stage_dates,
        "validation_drawdown_within_limit": validation["worst_drawdown"] >= maximum_drawdown,
        "holdout_drawdown_within_limit": holdout["worst_drawdown"] >= maximum_drawdown,
        "survivorship_safe_universe": survivorship_safe,
    }
    eligible = all(gates.values()) and (
        development["mean_excess_return"] > 0
        and validation["mean_excess_return"] > 0
        and validation["mean_information_ratio"] > 0
        and holdout["mean_excess_return"] > 0
        and holdout["mean_information_ratio"] > 0
    )
    sample_result = next(iter(results[selected].values()))
    return {
        "selected_strategy": selected, "eligible_for_paper": eligible,
        "promotion_gates": gates, "minimum_stage_dates": minimum_stage_dates,
        "maximum_drawdown": maximum_drawdown, "survivorship_safe": survivorship_safe,
        "selection_rule": "highest development mean information ratio; no validation/holdout retuning",
        "sessions": minimum_points, "symbols": sorted(prices),
        "development_end": sample_result.points[development_end - 1].session.isoformat(),
        "validation_end": sample_result.points[validation_end - 1].session.isoformat(),
        "holdout_end": sample_result.points[minimum_points - 1].session.isoformat(),
        "selected_results": {"development": development, "validation": validation, "holdout": holdout},
        "all_candidates": evaluations,
        "warning": "Short compact-history research; passing gates would not establish future profitability.",
    }


def fetch_universe(
    client, symbols: tuple[str, ...], benchmark: str,
    start: date, end: date,
) -> tuple[dict[str, tuple[DailyBar, ...]], tuple[DailyBar, ...]]:
    cache = {}
    for index, symbol in enumerate((*symbols, benchmark)):
        normalized = symbol.upper()
        if normalized in cache:
            continue
        if index:
            time.sleep(1.1)
        source = (
            client.daily(normalized, start_date=start, end_date=end)
            if isinstance(client, TiingoClient) else client.daily(normalized)
        )
        cache[normalized] = tuple(bar for bar in source if start <= bar.session <= end)
    return {symbol.upper(): cache[symbol.upper()] for symbol in symbols}, cache[benchmark.upper()]


def _key(candidate: CandidateStrategy) -> str:
    return candidate.name if candidate.lookback is None else f"{candidate.name}_{candidate.lookback}d"


def main() -> None:
    parser = argparse.ArgumentParser(description="Leakage-controlled multi-symbol strategy comparison")
    parser.add_argument("--symbols", default="AAPL,MSFT,NVDA,AMZN,GOOGL")
    parser.add_argument("--benchmark", default="SPY")
    parser.add_argument("--provider", choices=("alpha", "tiingo"), default="alpha")
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--lookbacks", default="5,10,21,42")
    parser.add_argument("--cost-bps", type=Decimal, default=Decimal(5))
    parser.add_argument("--output", type=Path, default=Path("data/strategy-comparison.json"))
    arguments = parser.parse_args()
    settings = Settings.from_env()
    if arguments.provider == "alpha" and not settings.alpha_vantage_api_key:
        parser.error("ALPHA_VANTAGE_API_KEY is missing from .env")
    if arguments.provider == "tiingo" and not settings.tiingo_api_key:
        parser.error("TIINGO_API_KEY is missing from .env")
    symbols = tuple(item.strip().upper() for item in arguments.symbols.split(",") if item.strip())
    lookbacks = tuple(int(item) for item in arguments.lookbacks.split(","))
    prices, benchmark = fetch_universe(
        (
            AlphaVantageClient(settings.alpha_vantage_api_key or "")
            if arguments.provider == "alpha" else TiingoClient(settings.tiingo_api_key or "")
        ), symbols,
        arguments.benchmark, arguments.start, arguments.end,
    )
    candidates = (CandidateStrategy("buy_and_hold", None),) + tuple(
        CandidateStrategy("momentum", lookback) for lookback in lookbacks
    )
    result = compare_strategies(prices, benchmark, candidates, cost_bps=arguments.cost_bps)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
