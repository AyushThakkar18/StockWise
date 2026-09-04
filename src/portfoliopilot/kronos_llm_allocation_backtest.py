from __future__ import annotations

import argparse
import json
from datetime import date
from decimal import Decimal
from pathlib import Path

from .config import Settings
from .council_backtest import position_attribution
from .historical_council_backtest import prepare_universe
from .kronos_forecast import KronosForecaster
from .kronos_llm_selection import CouncilAllocationStrategy, KronosCouncilDecisionEngine
from .openai_bounded_agents import ALLOWED_MODEL, OpenAIBoundedAgent
from .openai_council_selector import OpenAICouncilSelector
from .point_in_time import PointInTimeBacktester
from .price_cache import PriceCache
from .robust_strategy import SimpleMomentumTrendGrowthStrategy
from .sec_edgar import SECEdgarCache
from .universe import load_membership_history

POLICIES = (
    "equal_capped_cash",
    "five_percent_slots_spy",
    "half_spy_two_point_five_percent_slots",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Kronos Top-50 plus bounded LLM council")
    parser.add_argument("--start", type=date.fromisoformat, default=date(2024, 1, 1))
    parser.add_argument("--evaluation-start", type=date.fromisoformat, default=date(2025, 1, 2))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2025, 12, 31))
    parser.add_argument("--membership", type=Path, default=Path("private_data/universe/sp500-components-updated.csv"))
    parser.add_argument("--prices", type=Path, default=Path("private_data/prices-yahoo"))
    parser.add_argument("--kronos-repository", type=Path, default=Path("private_data/Kronos"))
    parser.add_argument("--kronos-cache", type=Path, default=Path("private_data/kronos-base"))
    parser.add_argument("--kronos-device", default="cuda")
    parser.add_argument("--kronos-recycle-every", type=int, default=20)
    parser.add_argument("--kronos-cooldown-seconds", type=float, default=.1)
    parser.add_argument("--llm-cache", type=Path, default=Path("private_data/openai-kronos-top50-v1"))
    parser.add_argument("--output", type=Path, default=Path("private_data/results/kronos-llm-three-policies-2025.json"))
    parser.add_argument("--report", type=Path, default=Path("private_data/results/kronos-llm-three-policies-2025.md"))
    arguments = parser.parse_args()
    settings = Settings.from_env()
    if not settings.openai_api_key:
        parser.error("OPENAI_API_KEY is required")
    if not settings.sec_user_agent:
        parser.error("SEC_USER_AGENT is required")
    membership = load_membership_history(arguments.membership)
    universe, benchmark, filtered, exclusions, coverage = prepare_universe(
        membership, PriceCache(arguments.prices), arguments.start, arguments.end,
        arguments.evaluation_start,
    )
    sec = SECEdgarCache(settings.sec_user_agent, Path("private_data/sec-edgar"))
    candidate_strategy = SimpleMomentumTrendGrowthStrategy(
        benchmark, sec, sec.ticker_map(), top_n=100, buffer_rank=120,
        maximum_position_weight=Decimal(1), maximum_sector_weight=Decimal(1),
    )
    agent = OpenAIBoundedAgent(
        settings.openai_api_key, arguments.llm_cache, model=ALLOWED_MODEL,
    )
    engine = KronosCouncilDecisionEngine(
        candidate_strategy,
        KronosForecaster(
            arguments.kronos_repository, arguments.kronos_cache,
            model_id="NeoQuasar/Kronos-base", device=arguments.kronos_device,
            recycle_every=arguments.kronos_recycle_every,
            cooldown_seconds=arguments.kronos_cooldown_seconds,
        ),
        agent,
        OpenAICouncilSelector(
            settings.openai_api_key, arguments.llm_cache / "synthesis", model=ALLOWED_MODEL,
        ),
        benchmark,
    )
    results, strategies = {}, {}
    simulation_universe = {**universe, "SPY": benchmark}
    for policy in POLICIES:
        print(f"running allocation policy: {policy}", flush=True)
        strategy = CouncilAllocationStrategy(engine, policy, name=policy)
        result = PointInTimeBacktester(cost_bps=Decimal(5), rebalance_every=21).run(
            simulation_universe, benchmark, filtered, strategy, arguments.evaluation_start,
        )
        strategies[policy] = strategy
        results[policy] = {
            "ending_equity": float(result.points[-1].equity),
            "metrics": result.metrics,
            "position_attribution": position_attribution(result),
            "monthly_allocations": {
                day.isoformat(): audit for day, audit in strategy.audits.items()
            },
        }
    monthly = []
    for day, decision in sorted(engine.decisions.items()):
        monthly.append({
            **decision,
            "allocations": {
                policy: strategies[policy].audits[day] for policy in POLICIES
            },
        })
    payload = {
        "experiment": "deterministic-top100-kronos-top50-bounded-llm-up-to20",
        "model": ALLOWED_MODEL,
        "new_specialist_requests": agent.requests_made,
        "new_synthesis_requests": engine.selector.requests_made,
        "evaluation_start": arguments.evaluation_start.isoformat(),
        "evaluation_end": arguments.end.isoformat(),
        "architecture": (
            "point-in-time deterministic Top 100; Kronos-base Top 50; three structured anonymous "
            "LLM reviews; code approves at most 20 and runs three deterministic allocation policies"
        ),
        "results": results, "monthly_decisions": monthly,
        "excluded_symbols": exclusions, "price_coverage": coverage.coverage,
        "limitations": [
            "The evaluated period has already been observed and is diagnostic, not an untouched holdout.",
            "Kronos checkpoint training cutoff is not documented precisely enough for a causal claim.",
            "LLM approval thresholds were chosen by design and are not evidence of predictive calibration.",
        ],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    arguments.report.write_text(render_report(payload), encoding="utf-8")
    print(json.dumps({
        policy: {"ending_equity": item["ending_equity"], **item["metrics"]}
        for policy, item in results.items()
    }, indent=2))


def render_report(payload) -> str:
    labels = {
        "equal_capped_cash": "Equal among approvals (10% cap; unused cash)",
        "five_percent_slots_spy": "5% slots; unused allocation to SPY",
        "half_spy_two_point_five_percent_slots": "50% SPY core; 2.5% slots",
    }
    lines = [
        "# Kronos Top-50 + bounded LLM council", "",
        ("Deterministic features rank the point-in-time universe to 100 candidates. Kronos-base "
         "ranks those candidates to 50. Three anonymous structured LLM specialists review them, "
         "then a synthesis agent selects zero to twenty stocks; the same selections feed all "
         "allocation policies."), "", "## Results", "",
        "| Policy | Return | Profit on $100K | SPY | Sharpe | Max drawdown |", "|---|---:|---:|---:|---:|---:|",
    ]
    for policy, item in payload["results"].items():
        metrics = item["metrics"]
        lines.append(
            f"| {labels[policy]} | {metrics['total_return']:.2%} | "
            f"${item['ending_equity'] - 100000:,.0f} | {metrics['benchmark_total_return']:.2%} | "
            f"{metrics['sharpe']:.2f} | {metrics['max_drawdown']:.2%} |"
        )
    lines.extend(("", "## Monthly decisions", ""))
    for month in payload["monthly_decisions"]:
        lines.extend((
            f"### {month['decision_date']}", "",
            (f"Approved **{month['approved_count']} of 20 maximum** from Kronos Top 50: "
             f"{', '.join(month['selected']) if month['selected'] else 'none'}."), "",
            f"Synthesis: {month['synthesis_rationale']}", "",
            "| Policy | Stocks bought | SPY | Cash |", "|---|---:|---:|---:|",
        ))
        for policy, audit in month["allocations"].items():
            lines.append(
                f"| {labels[policy]} | {audit['purchased_count']} | "
                f"{audit['spy_weight']:.2%} | {audit['cash_weight']:.2%} |"
            )
        lines.extend(("", "| Kronos rank | Symbol | Kronos score | Council score | Synthesis decision | Specialist objections |",
                      "|---:|---|---:|---:|---|---|"))
        for item in month["kronos_top_50"]:
            lines.append(
                f"| {item['kronos_rank']} | {item['symbol']} | {item['kronos_score']:.4f} | "
                f"{item['council_score']:.3f} | {'BUY' if item['selected'] else 'PASS'} | "
                f"{', '.join(item['rejection_reasons']) or 'none'} |"
            )
        lines.append("")
    lines.extend(("## Limitations", ""))
    lines.extend(f"- {item}" for item in payload["limitations"])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
