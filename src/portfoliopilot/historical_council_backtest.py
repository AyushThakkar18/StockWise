from __future__ import annotations

import argparse
import json
from datetime import date
from decimal import Decimal
from pathlib import Path

from .ai_ranking import AIRankedPortfolio, OpenAIRanker
from .comparative_council import ComparativeCouncilStrategy, OpenAIComparativeSelector
from .config import Settings
from .council_backtest import (
    CachedCouncilGate,
    monthly_decisions,
    position_attribution,
    render_report,
    summarize_decisions,
)
from .kronos_forecast import KronosForecaster
from .langgraph_research import LangGraphResearchCouncil
from .openai_research import OpenAICouncilSynthesizer, OpenAIResearchRunner
from .point_in_time import CouncilGatedStrategy, PointInTimeBacktester, audit_price_coverage
from .point_in_time_download import required_symbols
from .price_cache import PriceCache
from .reliable_strategy import ReliableCompositeStrategy
from .sec_edgar import SECEdgarCache
from .stock_scoring import OpenAIStockScorer, PerStockCouncilStrategy
from .universe import MembershipHistory, load_membership_history


def prepare_universe(history, cache, start, end, evaluation_start):
    required = required_symbols(history, evaluation_start, end)
    histories, exclusions = {}, {}
    for symbol in required:
        path = cache.covering_path(symbol, start, end)
        if path is None:
            exclusions[symbol] = "Yahoo history unavailable"
            continue
        histories[symbol] = cache.daily(None, symbol, start, end)  # type: ignore[arg-type]
    benchmark_path = cache.covering_path("SPY", start, end)
    if benchmark_path is None:
        raise ValueError("SPY history unavailable")
    benchmark = cache.daily(None, "SPY", start, end)  # type: ignore[arg-type]
    sessions = tuple(bar.session for bar in benchmark if bar.session >= evaluation_start)

    available = set(histories)
    filtered = MembershipHistory({
        session: tuple(symbol for symbol in symbols if symbol in available)
        for session, symbols in history.records.items()
    }, f"{history.version}:available-only")
    audit = audit_price_coverage(filtered, sessions, histories)
    coverage_exclusions = {symbol for _, symbol in audit.missing_symbol_sessions}
    for symbol in sorted(coverage_exclusions):
        exclusions[symbol] = "incomplete price coverage while an index member"
        histories.pop(symbol, None)
    if coverage_exclusions:
        available = set(histories)
        filtered = MembershipHistory({
            session: tuple(symbol for symbol in symbols if symbol in available)
            for session, symbols in history.records.items()
        }, f"{history.version}:available-complete-only")
    final_audit = audit_price_coverage(filtered, sessions, histories)
    if not final_audit.approved:
        raise ValueError("filtered historical universe still has incomplete price coverage")
    return histories, benchmark, filtered, exclusions, final_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a historical-membership five-agent backtest")
    parser.add_argument("--start", type=date.fromisoformat, default=date(2020, 1, 1))
    parser.add_argument("--evaluation-start", type=date.fromisoformat, default=date(2021, 1, 4))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2025, 12, 31))
    parser.add_argument("--membership", type=Path, default=Path("private_data/universe/sp500-components-updated.csv"))
    parser.add_argument("--prices", type=Path, default=Path("private_data/prices-yahoo"))
    parser.add_argument("--rank-cache", type=Path, default=Path("private_data/ai-rankings-pit-2021-2025"))
    parser.add_argument("--council-cache", type=Path, default=Path("private_data/council-pit-2021-2025"))
    parser.add_argument("--output", type=Path, default=Path("private_data/results/council-pit-2021-2025.json"))
    parser.add_argument("--report", type=Path, default=Path("private_data/results/council-pit-2021-2025.md"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument(
        "--reliable-composite", action="store_true",
        help="Use the deterministic growth/quality/risk composite before the LLM council",
    )
    parser.add_argument(
        "--comparative-top100", action="store_true",
        help="Research deterministic Top 100 and comparatively select exactly 20",
    )
    parser.add_argument(
        "--selection-cache", type=Path,
        default=Path("private_data/comparative-selections-v1"),
    )
    parser.add_argument(
        "--per-stock-scoring", action="store_true",
        help="Score each researched Top-100 stock independently, then sort and select Top 20",
    )
    parser.add_argument("--score-cache", type=Path, default=Path("private_data/stock-scores-v1"))
    parser.add_argument("--kronos", action="store_true", help="Add cached Kronos-base forecasts")
    parser.add_argument("--kronos-repository", type=Path, default=Path("private_data/Kronos"))
    parser.add_argument("--kronos-cache", type=Path, default=Path("private_data/kronos-base"))
    parser.add_argument("--kronos-device", default="auto")
    arguments = parser.parse_args()
    if arguments.comparative_top100 and arguments.per_stock_scoring:
        parser.error("choose either --comparative-top100 or --per-stock-scoring")

    history = load_membership_history(arguments.membership)
    histories, benchmark, filtered, exclusions, audit = prepare_universe(
        history, PriceCache(arguments.prices), arguments.start, arguments.end,
        arguments.evaluation_start,
    )
    readiness = {
        "dataset_version": filtered.version, "required_symbols": len(required_symbols(history, arguments.evaluation_start, arguments.end)),
        "available_symbols": len(histories), "excluded_symbol_count": len(exclusions),
        "excluded_symbols": exclusions, "eligible_symbol_sessions": audit.eligible_symbol_sessions,
        "price_coverage": audit.coverage,
    }
    if arguments.audit_only:
        print(json.dumps(readiness, indent=2, sort_keys=True))
        return

    settings = Settings.from_env()
    if not settings.openai_api_key or not settings.sec_user_agent:
        parser.error("OPENAI_API_KEY and SEC_USER_AGENT are required")
    research_model = settings.openai_research_model or settings.openai_model
    selection_model = settings.openai_selection_model or settings.openai_model
    scoring_model = settings.openai_scoring_model or selection_model
    council = LangGraphResearchCouncil(
        OpenAIResearchRunner(settings.openai_api_key, research_model),
        OpenAICouncilSynthesizer(settings.openai_api_key, research_model),
    )
    sec = SECEdgarCache(settings.sec_user_agent, Path("private_data/sec-edgar"))
    if arguments.comparative_top100 or arguments.per_stock_scoring:
        base = ReliableCompositeStrategy(
            benchmark, sec, sec.ticker_map(), top_n=100, buffer_rank=120,
            maximum_position_weight=Decimal("0.01"), maximum_sector_weight=Decimal(1),
            risk_off_invested_weight=Decimal(1),
        )
    elif arguments.reliable_composite:
        base = ReliableCompositeStrategy(benchmark, sec, sec.ticker_map())
    else:
        base = AIRankedPortfolio(OpenAIRanker(
            settings.openai_api_key, settings.openai_model, arguments.rank_cache,
        ))
    gate = CachedCouncilGate(
        council, histories, sec, arguments.council_cache, arguments.workers,
        factor_audits=base.audits if arguments.reliable_composite or arguments.comparative_top100
        or arguments.per_stock_scoring
        else None,
        factor_version=(
            "top100-comparative-factors-v1"
            if arguments.comparative_top100 or arguments.per_stock_scoring
            else "reliable-composite-v2"
        ),
    )
    if arguments.per_stock_scoring:
        forecaster = KronosForecaster(
            arguments.kronos_repository, arguments.kronos_cache,
            model_id="NeoQuasar/Kronos-base", device=arguments.kronos_device,
        ) if arguments.kronos else None
        strategy = PerStockCouncilStrategy(
            base, gate, OpenAIStockScorer(
                settings.openai_api_key, scoring_model, arguments.score_cache,
            ), forecaster=forecaster, benchmark=benchmark, workers=arguments.workers,
        )
    elif arguments.comparative_top100:
        strategy = ComparativeCouncilStrategy(
            base, gate, OpenAIComparativeSelector(
                settings.openai_api_key, selection_model, arguments.selection_cache,
            ),
        )
    else:
        strategy = CouncilGatedStrategy(
            base, gate, shortlist=20, top_n=20,
            maximum_position_weight=Decimal("0.10"),
            preserve_base_weights=arguments.reliable_composite,
        )
    result = PointInTimeBacktester(cost_bps=Decimal(5), rebalance_every=21).run(
        histories, benchmark, filtered, strategy, arguments.evaluation_start,
    )
    payload = {
        "experiment": (
            "five-agent-per-stock-top100-top20-kronos-base"
            if arguments.per_stock_scoring else "five-agent-comparative-top100-top20"
            if arguments.comparative_top100 else "five-agent-council-reliable-composite"
            if arguments.reliable_composite else "five-agent-council-historical-membership"
        ),
        "model": research_model, "selection_model": selection_model,
        "scoring_model": scoring_model if arguments.per_stock_scoring else None,
        "evaluation_start": arguments.evaluation_start.isoformat(),
        "evaluation_end": result.points[-1].session.isoformat(), "benchmark": "SPY",
        "universe_method": "point-in-time S&P 500 members with available complete Yahoo history",
        "readiness": readiness, "ending_equity": float(result.points[-1].equity),
        "metrics": result.metrics, "council_activity": summarize_decisions(gate.decisions),
        "position_attribution": position_attribution(result),
        "monthly_decisions": monthly_decisions(
            gate, histories, result.transactions, strategy.target_history,
            getattr(strategy, "selection_audits", None),
        ),
        "selection_audits": {
            str(session): audit for session, audit in getattr(
                strategy, "selection_audits", {},
            ).items()
        },
        "limitations": [
            "Unavailable and incomplete historical securities are excluded, leaving residual survivorship bias.",
            "The screen covers all available historical members, not a point-in-time market-cap Top 100.",
            "Yahoo adjusted history is not point-in-time corporate-action truth.",
        ],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    arguments.report.write_text(render_report(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
