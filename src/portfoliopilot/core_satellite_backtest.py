from __future__ import annotations

import argparse
import json
from datetime import date
from decimal import Decimal
from pathlib import Path

from .config import Settings
from .core_satellite import CoreSatelliteStrategy
from .historical_council_backtest import prepare_universe
from .point_in_time import PointInTimeBacktester
from .price_cache import PriceCache
from .robust_strategy import SimpleMomentumTrendGrowthStrategy
from .sec_edgar import SECEdgarCache
from .universe import load_membership_history


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare SPY, StockWise, and core-satellite")
    parser.add_argument("--start", type=date.fromisoformat, default=date(2024, 1, 1))
    parser.add_argument("--evaluation-start", type=date.fromisoformat, default=date(2025, 1, 2))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2025, 12, 31))
    parser.add_argument("--core-weight", type=Decimal, default=Decimal("0.70"))
    parser.add_argument("--membership", type=Path, default=Path("private_data/universe/sp500-components-updated.csv"))
    parser.add_argument("--prices", type=Path, default=Path("private_data/prices-yahoo"))
    parser.add_argument("--output", type=Path, default=Path("private_data/results/core-satellite-2025.json"))
    arguments = parser.parse_args()
    settings = Settings.from_env()
    if not settings.sec_user_agent:
        parser.error("SEC_USER_AGENT is required")
    membership = load_membership_history(arguments.membership)
    universe, benchmark, filtered, exclusions, coverage = prepare_universe(
        membership, PriceCache(arguments.prices), arguments.start, arguments.end,
        arguments.evaluation_start,
    )
    sec = SECEdgarCache(settings.sec_user_agent, Path("private_data/sec-edgar"))
    cik = sec.ticker_map()
    active = SimpleMomentumTrendGrowthStrategy(benchmark, sec, cik)
    active_result = PointInTimeBacktester(cost_bps=Decimal(5), rebalance_every=21).run(
        universe, benchmark, filtered, active, arguments.evaluation_start,
    )
    blended_active = SimpleMomentumTrendGrowthStrategy(benchmark, sec, cik)
    blended = CoreSatelliteStrategy(blended_active, core_weight=arguments.core_weight)
    blended_result = PointInTimeBacktester(cost_bps=Decimal(5), rebalance_every=21).run(
        {**universe, "SPY": benchmark}, benchmark, filtered, blended,
        arguments.evaluation_start,
    )
    payload = {
        "evaluation_start": arguments.evaluation_start.isoformat(),
        "evaluation_end": blended_result.points[-1].session.isoformat(),
        "core_weight": float(arguments.core_weight),
        "active_weight": float(Decimal(1) - arguments.core_weight),
        "benchmark": "SPY", "price_coverage": coverage.coverage,
        "excluded_symbols": exclusions,
        "results": {
            "active_only": active_result.metrics,
            "core_satellite": blended_result.metrics,
            "spy": {"total_return": blended_result.metrics["benchmark_total_return"]},
        },
        "monthly_targets": {
            day.isoformat(): {symbol: float(weight) for symbol, weight in targets.items()}
            for day, targets in blended.target_history.items()
        },
        "limitations": [
            "This historical period has already been observed and is diagnostic, not a holdout.",
            "The active sleeve has not demonstrated stable out-of-sample alpha.",
        ],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["results"], indent=2))


if __name__ == "__main__":
    main()
