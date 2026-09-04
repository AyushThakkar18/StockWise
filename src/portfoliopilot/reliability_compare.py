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
from .reliable_strategy import ReliableCompositeStrategy
from .sec_edgar import SECEdgarCache
from .universe import load_membership_history


def main() -> None:
    parser = argparse.ArgumentParser(description="Run leakage-safe strategy ablations")
    parser.add_argument("--start", type=date.fromisoformat, default=date(2020, 1, 1))
    parser.add_argument("--evaluation-start", type=date.fromisoformat, default=date(2021, 1, 4))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2025, 12, 31))
    parser.add_argument("--output", type=Path, default=Path("private_data/results/reliability-ablation-2021-2025.json"))
    arguments = parser.parse_args()
    settings = Settings.from_env()
    if not settings.sec_user_agent:
        parser.error("SEC_USER_AGENT is required")
    membership = load_membership_history(Path("private_data/universe/sp500-components-updated.csv"))
    universe, benchmark, filtered, exclusions, _ = prepare_universe(
        membership, PriceCache(Path("private_data/prices-yahoo")),
        arguments.start, arguments.end, arguments.evaluation_start,
    )
    sec = SECEdgarCache(settings.sec_user_agent, Path("private_data/sec-edgar"))
    cik = sec.ticker_map()
    configurations = (
        ("price_momentum_monthly", RankedMomentum(63, 20), 21),
        ("growth_quality_monthly", ReliableCompositeStrategy(benchmark, sec, cik), 21),
        ("growth_quality_quarterly", ReliableCompositeStrategy(benchmark, sec, cik), 63),
    )
    results = {}
    for name, strategy, frequency in configurations:
        result = PointInTimeBacktester(
            cost_bps=Decimal(5), rebalance_every=frequency,
        ).run(universe, benchmark, filtered, strategy, arguments.evaluation_start)
        results[name] = {
            "metrics": result.metrics, "ending_equity": float(result.points[-1].equity),
            "rebalance_every_sessions": frequency,
            "decision_audits": {
                key.isoformat(): value
                for key, value in getattr(strategy, "audits", {}).items()
            },
        }
    payload = {
        "evaluation_start": arguments.evaluation_start.isoformat(),
        "evaluation_end": arguments.end.isoformat(), "benchmark": "SPY",
        "excluded_symbols": exclusions, "results": results,
        "selection_rule": "Compare pre-registered variants; do not select on this same period.",
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps({name: item["metrics"] for name, item in results.items()}, indent=2))


if __name__ == "__main__":
    main()
