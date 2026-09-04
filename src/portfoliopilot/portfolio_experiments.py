from __future__ import annotations

import argparse
import json
from datetime import date
from decimal import Decimal
from pathlib import Path

from .config import Settings
from .historical_council_backtest import prepare_universe
from .point_in_time import PointInTimeBacktester
from .price_cache import PriceCache
from .robust_research import WINDOWS
from .robust_strategy import SimpleMomentumTrendGrowthStrategy
from .sec_edgar import SECEdgarCache
from .universe import load_membership_history


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare portfolio construction without LLM calls")
    parser.add_argument("--start", type=date.fromisoformat, default=date(2020, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2025, 12, 31))
    parser.add_argument("--window", choices=tuple(WINDOWS), action="append")
    parser.add_argument("--position-count", type=int, action="append")
    parser.add_argument(
        "--weighting", choices=("equal", "score", "mild_volatility"), action="append",
    )
    parser.add_argument("--retention-bonus", type=float, action="append")
    parser.add_argument("--membership", type=Path, default=Path("private_data/universe/sp500-components-updated.csv"))
    parser.add_argument("--prices", type=Path, default=Path("private_data/prices-yahoo"))
    parser.add_argument("--output", type=Path, default=Path("private_data/results/portfolio-construction-experiments.json"))
    arguments = parser.parse_args()
    counts = arguments.position_count or [20, 30, 40, 50]
    weightings = arguments.weighting or ["equal", "score", "mild_volatility"]
    bonuses = arguments.retention_bonus or [0.0, .015]
    if any(count < 10 or count > 100 for count in counts):
        parser.error("position count must be between 10 and 100")
    if any(bonus < 0 or bonus > .10 for bonus in bonuses):
        parser.error("retention bonus must be between 0 and 0.10")
    settings = Settings.from_env()
    if not settings.sec_user_agent:
        parser.error("SEC_USER_AGENT is required")
    membership = load_membership_history(arguments.membership)
    universe, benchmark, filtered, exclusions, coverage = prepare_universe(
        membership, PriceCache(arguments.prices), arguments.start, arguments.end,
        date(2021, 1, 4),
    )
    sec = SECEdgarCache(settings.sec_user_agent, Path("private_data/sec-edgar"))
    cik = sec.ticker_map()
    results = {}
    for window_name, (window_start, window_end) in WINDOWS.items():
        if arguments.window and window_name not in arguments.window:
            continue
        clipped_end = min(window_end, arguments.end)
        window_universe = {
            symbol: tuple(bar for bar in bars if bar.session <= clipped_end)
            for symbol, bars in universe.items()
        }
        window_benchmark = tuple(bar for bar in benchmark if bar.session <= clipped_end)
        results[window_name] = {}
        for count in counts:
            for weighting in weightings:
                for bonus in bonuses:
                    name = f"n{count}_{weighting}_retain{bonus:.3f}"
                    print(f"[{window_name}] {name}", flush=True)
                    strategy = SimpleMomentumTrendGrowthStrategy(
                        window_benchmark, sec, cik, top_n=count,
                        buffer_rank=max(count + 10, int(count * 1.5)),
                        weighting_method=weighting, retention_bonus=bonus,
                        name=name,
                    )
                    result = PointInTimeBacktester(
                        cost_bps=Decimal(5), rebalance_every=21,
                    ).run(window_universe, window_benchmark, filtered, strategy, window_start)
                    results[window_name][name] = {
                        "parameters": {
                            "position_count": count, "weighting": weighting,
                            "retention_bonus": bonus,
                        },
                        "ending_equity": float(result.points[-1].equity),
                        "metrics": result.metrics,
                    }
    payload = {
        "frozen_signal_weights": {
            "momentum_12_1": .35, "momentum_6_1": .25, "trend": .25, "growth": .15,
        },
        "construction_grid": {
            "position_counts": counts, "weightings": weightings, "retention_bonuses": bonuses,
        },
        "results": results, "price_coverage": coverage.coverage,
        "excluded_symbols": exclusions,
        "selection_rule": (
            "Prefer configurations stable across development, validation, and test windows; do not "
            "select solely by highest return or use the already-observed 2025 diagnostic for tuning."
        ),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"saved {arguments.output}")


if __name__ == "__main__":
    main()
