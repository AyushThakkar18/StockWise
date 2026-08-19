from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from statistics import fmean
from typing import Any, Protocol

from .ai_ranking import AIRankedPortfolio, OpenAIRanker
from .backtest import BacktestPoint, performance_metrics
from .market_data import DailyBar
from .multi_backtest import EqualWeight, MultiAssetBacktester, RankedMomentum
from .price_cache import PriceCache


class Ranker(Protocol):
    def __call__(self, packet: dict[str, object]) -> tuple[str, ...]: ...


def evaluate_current_universe(
    universe: dict[str, tuple[DailyBar, ...]], benchmark: tuple[DailyBar, ...],
    cost_bps: Decimal = Decimal(5), rebalance_every: int = 21,
    ai_ranker: Ranker | None = None, benchmark_label: str = "SPY",
) -> dict[str, Any]:
    if len(universe) < 20:
        raise ValueError("at least 20 complete stock histories are required")
    strategies = (EqualWeight(),) + tuple(
        RankedMomentum(lookback=lookback, top_n=20)
        for lookback in (21, 63, 126, 252)
    )
    if ai_ranker is not None:
        strategies += (AIRankedPortfolio(ai_ranker),)
    backtester = MultiAssetBacktester(cost_bps=cost_bps, rebalance_every=rebalance_every)
    results = {strategy.name + _suffix(strategy): backtester.run(universe, benchmark, strategy)
               for strategy in strategies}
    total_points = min(len(result.points) for result in results.values())
    warmup = 253
    points = total_points - warmup
    if points < 500:
        raise ValueError("at least 500 aligned sessions are required")
    development_end = warmup + int(points * 0.60)
    validation_end = warmup + int(points * 0.80)
    boundaries = {
        "development": (warmup, development_end),
        "validation": (development_end - 1, validation_end),
        "holdout": (validation_end - 1, total_points),
    }
    equal_weight = results["equal_weight"].points
    evaluations = {}
    for name, result in results.items():
        evaluations[name] = {
            stage: performance_metrics(tuple(
                BacktestPoint(point.session, point.equity,
                              equal_weight[index].equity if benchmark_label == "equal_weight" else point.benchmark_equity,
                              Decimal(0), point.cost)
                for index, point in enumerate(result.points[start:end], start)
            ))
            for stage, (start, end) in boundaries.items()
        }
        evaluations[name]["full_period"] = performance_metrics(tuple(
            BacktestPoint(point.session, point.equity,
                          equal_weight[index].equity if benchmark_label == "equal_weight" else point.benchmark_equity,
                          Decimal(0), point.cost)
            for index, point in enumerate(result.points[warmup:], warmup)
        ))
    selected = max(
        evaluations,
        key=lambda name: (
            evaluations[name]["development"]["information_ratio"],
            evaluations[name]["development"]["excess_return"], name,
        ),
    )
    selected_result = results[selected]
    return {
        "selected_strategy": selected,
        "selection_rule": "highest development information ratio; validation and holdout untouched",
        "sessions": points, "feature_warmup_sessions": warmup,
        "symbols_tested": sorted(universe),
        "symbol_count": len(universe),
        "development_end": selected_result.points[development_end - 1].session.isoformat(),
        "validation_end": selected_result.points[validation_end - 1].session.isoformat(),
        "holdout_end": selected_result.points[-1].session.isoformat(),
        "selected_results": evaluations[selected],
        "all_candidates": evaluations,
        "cost_bps": float(cost_bps),
        "benchmark": benchmark_label,
        "rebalance_every_sessions": rebalance_every,
        "survivorship_safe": False,
        "resume_eligible": False,
        "warning": "Current constituents were selected with future knowledge; exploratory only.",
    }


def load_complete_histories(
    snapshot: dict[str, Any], cache: PriceCache, start: date, end: date,
    maximum_edge_gap: timedelta = timedelta(days=10),
    allow_missing_benchmark: bool = False,
) -> tuple[dict[str, tuple[DailyBar, ...]], tuple[DailyBar, ...], dict[str, str], str]:
    histories, exclusions = {}, {}
    for symbol in (*snapshot["symbols"], "SPY"):
        path = cache.directory / f"{symbol}-{start}-{end}.json"
        if not path.exists():
            exclusions[symbol] = "not downloaded"
            continue
        bars = cache.daily(None, symbol, start, end)  # type: ignore[arg-type]
        if bars[0].session > start + maximum_edge_gap or bars[-1].session < end - maximum_edge_gap:
            exclusions[symbol] = f"incomplete coverage {bars[0].session}..{bars[-1].session}"
            continue
        histories[symbol] = bars
    if "SPY" not in histories:
        if not allow_missing_benchmark:
            raise ValueError(f"SPY benchmark unavailable: {exclusions.get('SPY', 'unknown')}")
        benchmark = next(iter(histories.values()))
        benchmark_label = "equal_weight"
    else:
        benchmark = histories.pop("SPY")
        benchmark_label = "SPY"
    return histories, benchmark, exclusions, benchmark_label


def _suffix(strategy: EqualWeight | RankedMomentum | AIRankedPortfolio) -> str:
    if isinstance(strategy, RankedMomentum):
        return f"_{strategy.lookback}d_top{strategy.top_n}"
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest a frozen current Top-100 universe")
    parser.add_argument("--universe", type=Path, required=True)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--cost-bps", type=Decimal, default=Decimal(5))
    parser.add_argument("--rebalance-every", type=int, default=21)
    parser.add_argument("--ai", action="store_true", help="include anonymous OpenAI feature reranking")
    parser.add_argument("--allow-missing-benchmark", action="store_true")
    parser.add_argument("--directory", type=Path, default=Path("private_data/prices"))
    parser.add_argument("--output", type=Path, default=Path("data/current-top100-backtest.json"))
    arguments = parser.parse_args()
    snapshot = json.loads(arguments.universe.read_text(encoding="utf-8"))
    histories, benchmark, exclusions, benchmark_label = load_complete_histories(
        snapshot, PriceCache(arguments.directory), arguments.start, arguments.end,
        allow_missing_benchmark=arguments.allow_missing_benchmark,
    )
    ai_ranker = None
    if arguments.ai:
        from .config import Settings

        settings = Settings.from_env()
        if not settings.openai_api_key:
            parser.error("OPENAI_API_KEY is missing from .env")
        ai_ranker = OpenAIRanker(
            settings.openai_api_key, settings.openai_model, Path("private_data/ai-rankings"),
        )
    result = evaluate_current_universe(
        histories, benchmark, arguments.cost_bps, arguments.rebalance_every, ai_ranker,
        benchmark_label,
    )
    result.update({
        "universe_as_of": snapshot["as_of"], "requested_symbol_count": snapshot["count"],
        "excluded_symbols": exclusions,
        "mean_available_sessions": fmean(len(bars) for bars in histories.values()),
    })
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    arguments.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
