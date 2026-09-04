from __future__ import annotations

import argparse
import json
from datetime import date
from decimal import Decimal
from pathlib import Path

from .config import Settings
from .historical_council_backtest import prepare_universe
from .multi_backtest import RankedMomentum
from .point_in_time import PointInTimeBacktester
from .price_cache import PriceCache
from .robust_strategy import RobustMultifactorStrategy, SimpleMomentumTrendGrowthStrategy
from .sec_edgar import SECEdgarCache
from .signal_diagnostics import forward_signal_diagnostics
from .universe import load_membership_history

WINDOWS = {
    "development_2021_2022": (date(2021, 1, 4), date(2022, 12, 30)),
    "validation_2023": (date(2023, 1, 3), date(2023, 12, 29)),
    "test_2024": (date(2024, 1, 2), date(2024, 12, 31)),
    "final_holdout_2025": (date(2025, 1, 2), date(2025, 12, 31)),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run chronological multifactor ablations")
    parser.add_argument("--start", type=date.fromisoformat, default=date(2020, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2025, 12, 31))
    parser.add_argument("--membership", type=Path, default=Path("private_data/universe/sp500-components-updated.csv"))
    parser.add_argument("--prices", type=Path, default=Path("private_data/prices-yahoo"))
    parser.add_argument("--window", choices=tuple(WINDOWS), action="append")
    parser.add_argument("--variant", choices=(
        "momentum_63d", "simple_momentum_trend_growth",
        "multifactor_equal_weight", "multifactor_risk_aware",
    ), action="append")
    parser.add_argument("--output", type=Path, default=Path("private_data/results/robust-walk-forward.json"))
    arguments = parser.parse_args()
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
        if window_start > arguments.end or window_end < arguments.start:
            continue
        clipped_end = min(window_end, arguments.end)
        window_universe = {
            symbol: tuple(bar for bar in bars if bar.session <= clipped_end)
            for symbol, bars in universe.items()
        }
        window_benchmark = tuple(bar for bar in benchmark if bar.session <= clipped_end)
        variants = {
            "momentum_63d": RankedMomentum(63, 20),
            "simple_momentum_trend_growth": SimpleMomentumTrendGrowthStrategy(
                window_benchmark, sec, cik,
            ),
            "multifactor_equal_weight": RobustMultifactorStrategy(
                window_benchmark, sec, cik, risk_aware=False, correlation_penalty=0,
                name="multifactor_equal_weight",
            ),
            "multifactor_risk_aware": RobustMultifactorStrategy(
                window_benchmark, sec, cik, name="multifactor_risk_aware",
            ),
        }
        results[window_name] = {}
        for variant_name, strategy in variants.items():
            if arguments.variant and variant_name not in arguments.variant:
                continue
            print(f"[{window_name}] running {variant_name}", flush=True)
            result = PointInTimeBacktester(cost_bps=Decimal(5), rebalance_every=21).run(
                window_universe, window_benchmark, filtered, strategy, window_start,
            )
            results[window_name][variant_name] = {
                "evaluation_end": result.points[-1].session.isoformat(),
                "ending_equity": float(result.points[-1].equity),
                "metrics": result.metrics,
                "signal_diagnostics": forward_signal_diagnostics(
                    getattr(strategy, "audits", {}), window_universe,
                ),
            }
    payload = {
        "architecture": "fixed factors; chronological windows; next-open execution; 5 bps costs",
        "windows": {name: [start.isoformat(), end.isoformat()] for name, (start, end) in WINDOWS.items()},
        "results": results,
        "price_coverage": coverage.coverage,
        "excluded_symbols": exclusions,
        "interpretation_rule": (
            "Do not tune on final_holdout_2025. Retain a layer only when it improves validation and "
            "test risk-adjusted performance, not merely total return."
        ),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"saved {arguments.output}")


if __name__ == "__main__":
    main()
