from __future__ import annotations

import argparse
import json
from datetime import date
from decimal import Decimal
from pathlib import Path

from .bounded_agent_strategy import BoundedMultiAgentStrategy
from .config import Settings
from .council_backtest import position_attribution
from .historical_council_backtest import prepare_universe
from .openai_bounded_agents import ALLOWED_MODEL, OpenAIBoundedAgent
from .point_in_time import PointInTimeBacktester
from .price_cache import PriceCache
from .robust_strategy import RobustMultifactorStrategy
from .sec_edgar import SECEdgarCache
from .universe import load_membership_history


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest bounded OpenAI multi-agent features")
    parser.add_argument("--start", type=date.fromisoformat, default=date(2024, 1, 1))
    parser.add_argument("--evaluation-start", type=date.fromisoformat, default=date(2025, 1, 2))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2025, 12, 31))
    parser.add_argument("--membership", type=Path, default=Path("private_data/universe/sp500-components-updated.csv"))
    parser.add_argument("--prices", type=Path, default=Path("private_data/prices-yahoo"))
    parser.add_argument("--cache", type=Path, default=Path("private_data/openai-bounded-gpt-4o-mini-v1"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--candidate-count", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("private_data/results/bounded-openai-agents-2025.json"))
    arguments = parser.parse_args()

    settings = Settings.from_env()
    if not settings.openai_api_key:
        parser.error("OPENAI_API_KEY is required")
    history = load_membership_history(arguments.membership)
    histories, benchmark, filtered, exclusions, coverage = prepare_universe(
        history, PriceCache(arguments.prices), arguments.start, arguments.end,
        arguments.evaluation_start,
    )
    sec = SECEdgarCache(settings.sec_user_agent or "StockWise", Path("private_data/sec-edgar"))
    base = RobustMultifactorStrategy(
        benchmark, sec, sec.ticker_map(), top_n=arguments.candidate_count,
        buffer_rank=max(40, arguments.candidate_count),
    )
    agent = OpenAIBoundedAgent(settings.openai_api_key, arguments.cache, model=ALLOWED_MODEL)
    strategy = BoundedMultiAgentStrategy(
        base,
        agent,
        benchmark, candidate_count=arguments.candidate_count, workers=arguments.workers,
    )
    result = PointInTimeBacktester(cost_bps=Decimal(5), rebalance_every=21).run(
        histories, benchmark, filtered, strategy, arguments.evaluation_start,
    )
    payload = {
        "experiment": "bounded-openai-multi-agent-features",
        "model": ALLOWED_MODEL,
        "api_requests_made": agent.requests_made,
        "evaluation_start": arguments.evaluation_start.isoformat(),
        "evaluation_end": result.points[-1].session.isoformat(),
        "architecture": (
            "robust deterministic Top 30 with Top 10 locked; independent technical, quality and "
            "risk agents review ranks 11-30; capped 8% score adjustment; deterministic "
            "correlation, sector, volatility, regime and next-open execution controls"
        ),
        "ending_equity": float(result.points[-1].equity), "metrics": result.metrics,
        "position_attribution": position_attribution(result),
        "monthly_audits": {str(day): audit for day, audit in strategy.audits.items()},
        "excluded_symbols": exclusions, "price_coverage": coverage.coverage,
        "limitations": [
            "This strategy was designed after observing 2025 results and 2025 is therefore diagnostic.",
            "Unavailable historical securities leave residual survivorship bias.",
            "LLM features are bounded but may remain unstable across model-provider revisions.",
        ],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "ending_equity": payload["ending_equity"], "metrics": payload["metrics"],
        "output": str(arguments.output),
    }, indent=2))


if __name__ == "__main__":
    main()
