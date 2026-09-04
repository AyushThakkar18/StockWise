from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, date, datetime, time
from decimal import Decimal
from pathlib import Path

from .ai_ranking import AIRankedPortfolio, OpenAIRanker
from .config import Settings
from .contracts import Evidence, Quality
from .langgraph_research import LangGraphResearchCouncil
from .multi_backtest import MultiAssetBacktester
from .openai_research import OpenAICouncilSynthesizer, OpenAIResearchRunner
from .point_in_time import CouncilGatedStrategy
from .price_cache import PriceCache
from .research_contracts import CouncilVerdict, OrchestratedCouncilResult
from .sec_edgar import SECEdgarCache


def load_evaluation_histories(snapshot, cache, start, end, evaluation_start):
    histories, exclusions = {}, {}
    for symbol in (*snapshot["symbols"], "SPY"):
        path = cache.covering_path(symbol, start, end)
        if path is None:
            exclusions[symbol] = "not downloaded"
            continue
        bars = cache.daily(None, symbol, start, end)  # type: ignore[arg-type]
        if bars[-1].session < end or bars[0].session > evaluation_start:
            exclusions[symbol] = f"incomplete evaluation coverage {bars[0].session}..{bars[-1].session}"
            continue
        histories[symbol] = bars
    if "SPY" not in histories:
        raise ValueError(f"SPY benchmark unavailable: {exclusions.get('SPY', 'unknown')}")
    return histories, histories.pop("SPY"), exclusions


def price_evidence(symbol: str, decision_on: date, bars) -> Evidence:
    available = tuple(bar for bar in bars if bar.session <= decision_on)
    if len(available) < 253:
        raise ValueError(f"{symbol} lacks 252-session feature history on {decision_on}")
    latest = available[-1]
    return Evidence(
        id=f"yahoo:{symbol}:{decision_on}", symbol=symbol,
        claim=(
            f"Close={latest.close}; adjusted_close={latest.adjusted_close}; "
            f"21-session return={latest.adjusted_close / available[-22].adjusted_close - 1:.6f}; "
            f"63-session return={latest.adjusted_close / available[-64].adjusted_close - 1:.6f}; "
            f"252-session return={latest.adjusted_close / available[-253].adjusted_close - 1:.6f}."
        ),
        source="Yahoo Finance via yfinance", observed_at=latest.observed_at,
        published_at=latest.published_at,
        available_to_strategy_at=latest.available_to_strategy_at,
        retrieved_at=latest.retrieved_at, vintage=latest.vintage, quality=Quality.PASS,
    )


def composite_evidence(
    symbol: str, decision_on: date, audits, version: str = "reliable-composite-v2",
) -> Evidence | None:
    audit = audits.get(decision_on)
    if audit is None:
        return None
    ranked = next(
        (item for item in audit.get("ranked", ()) if item.get("symbol") == symbol), None,
    )
    if ranked is None:
        return None
    available = datetime.combine(decision_on, time(21), tzinfo=UTC)
    fields = (
        "rank", "score", "momentum", "relative_strength", "low_volatility",
        "growth", "quality", "sector",
    )
    values = "; ".join(f"{field}={ranked.get(field)}" for field in fields)
    return Evidence(
        id=f"{version}:{symbol}:{decision_on}", symbol=symbol,
        claim=(
            f"Point-in-time deterministic composite: {values}; "
            f"market_regime={audit.get('market_regime')}; "
            f"portfolio_invested_target={audit.get('invested_target')}."
        ),
        source="StockWise deterministic growth-quality-risk model",
        observed_at=available, published_at=available,
        available_to_strategy_at=available, retrieved_at=datetime.now(UTC),
        vintage=version, quality=Quality.PASS,
    )


class CachedCouncilGate:
    def __init__(
        self, council: LangGraphResearchCouncil, histories, sec: SECEdgarCache,
        cache_directory: Path, workers: int = 4, factor_audits=None,
        factor_version: str = "reliable-composite-v2",
    ):
        self.council = council
        self.histories = histories
        self.sec = sec
        self.cache_directory = cache_directory
        self.workers = workers
        self.factor_audits = factor_audits
        self.factor_version = factor_version
        self.decisions: dict[str, OrchestratedCouncilResult] = {}
        self.shortlists: dict[date, tuple[str, ...]] = {}
        cache_directory.mkdir(parents=True, exist_ok=True)
        self.ticker_map = sec.ticker_map()

    def evaluate_many(self, symbols: tuple[str, ...], decision_on: date) -> dict[str, bool]:
        self.shortlists[decision_on] = symbols
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = {executor.submit(self._evaluate, symbol, decision_on): symbol for symbol in symbols}
            results = {futures[future]: future.result() for future in as_completed(futures)}
        return {
            symbol: result.route == "SYNTHESIS_COMPLETE"
            and result.synthesis is not None
            and result.synthesis.verdict == CouncilVerdict.SUPPORT
            for symbol, result in results.items()
        }

    def __call__(self, symbol: str, decision_on: date) -> bool:
        return self.evaluate_many((symbol,), decision_on)[symbol]

    def _evaluate(self, symbol: str, decision_on: date) -> OrchestratedCouncilResult:
        key = f"{decision_on}:{symbol}"
        path = self.cache_directory / f"{decision_on}-{symbol}.json"
        if path.exists():
            result = OrchestratedCouncilResult.model_validate_json(path.read_text(encoding="utf-8"))
        else:
            decision_at = datetime.combine(decision_on, time(22), tzinfo=UTC)
            evidence = [price_evidence(symbol, decision_on, self.histories[symbol])]
            if self.factor_audits is not None:
                factor_item = composite_evidence(
                    symbol, decision_on, self.factor_audits, self.factor_version,
                )
                if factor_item is not None:
                    evidence.append(factor_item)
            cik = self.ticker_map.get(symbol.replace(".", "-"))
            if cik is not None:
                sec_item = self.sec.evidence_on(symbol, cik, decision_on, datetime.now(UTC))
                if sec_item is not None:
                    evidence.append(sec_item)
            thread_id = f"{self.factor_version}:{key}" if self.factor_audits is not None else key
            result = self.council.evaluate(symbol, decision_at, tuple(evidence), thread_id)
            path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        self.decisions[key] = result
        return result


def summarize_decisions(decisions: dict[str, OrchestratedCouncilResult]) -> dict[str, object]:
    routes = Counter(result.route for result in decisions.values())
    verdicts = Counter(
        result.synthesis.verdict.value for result in decisions.values() if result.synthesis is not None
    )
    return {
        "candidate_reviews": len(decisions), "routes": dict(sorted(routes.items())),
        "verdicts": dict(sorted(verdicts.items())),
    }


def position_attribution(result) -> dict[str, object]:
    bought = {item.symbol for item in result.transactions if item.kind == "BUY"}
    ending_values = {
        symbol: float(weight * result.points[-1].equity)
        for symbol, weight in result.points[-1].weights.items()
    }
    pnl = {symbol: Decimal(str(ending_values.get(symbol, 0))) for symbol in bought}
    for item in result.transactions:
        if item.symbol not in pnl:
            continue
        if item.kind == "BUY":
            pnl[item.symbol] -= item.cash_amount + item.cost
        elif item.kind == "SELL":
            pnl[item.symbol] += item.cash_amount - item.cost
        elif item.kind == "DIVIDEND":
            pnl[item.symbol] += item.cash_amount
    profitable = sorted(symbol for symbol, value in pnl.items() if value > 0)
    losing = sorted(symbol for symbol, value in pnl.items() if value < 0)
    flat = sorted(symbol for symbol, value in pnl.items() if value == 0)
    closed = sorted(symbol for symbol in bought if symbol not in ending_values)
    open_symbols = sorted(bought - set(closed))
    closed_profitable = [symbol for symbol in closed if pnl[symbol] > 0]
    closed_losing = [symbol for symbol in closed if pnl[symbol] < 0]
    gains = [value for value in pnl.values() if value > 0]
    losses = [value for value in pnl.values() if value < 0]
    total = len(bought)
    return {
        "definition": (
            "Unique purchased-symbol P&L = sales + ending market value + dividends "
            "- purchases - allocated transaction costs."
        ),
        "symbols_bought": total,
        "profitable_symbols": len(profitable),
        "losing_symbols": len(losing),
        "break_even_symbols": len(flat),
        "profitable_symbol_rate": len(profitable) / total if total else 0,
        "losing_symbol_rate": len(losing) / total if total else 0,
        "fully_exited_symbols": len(closed), "open_symbols": len(open_symbols),
        "fully_exited_profitable": len(closed_profitable),
        "fully_exited_losing": len(closed_losing),
        "fully_exited_win_rate": len(closed_profitable) / len(closed) if closed else 0,
        "gross_attributed_gains": float(sum(gains, Decimal(0))),
        "gross_attributed_losses": float(sum(losses, Decimal(0))),
        "average_profit_per_profitable_symbol": (
            float(sum(gains, Decimal(0)) / len(gains)) if gains else 0
        ),
        "average_loss_per_losing_symbol": (
            float(sum(losses, Decimal(0)) / len(losses)) if losses else 0
        ),
        "profit_factor": (
            float(sum(gains, Decimal(0)) / abs(sum(losses, Decimal(0)))) if losses else None
        ),
        "best_symbol": max(pnl, key=pnl.get) if pnl else None,
        "worst_symbol": min(pnl, key=pnl.get) if pnl else None,
        "profitable": profitable, "losing": losing, "break_even": flat,
        "pnl_by_symbol": {symbol: float(value) for symbol, value in sorted(pnl.items())},
    }


def monthly_decisions(
    gate: CachedCouncilGate, histories, transactions, target_history=None,
    selection_audits=None,
) -> list[dict[str, object]]:
    trades_by_session_symbol = {}
    for item in transactions:
        if item.kind in {"BUY", "SELL"}:
            trades_by_session_symbol.setdefault((item.session, item.symbol), []).append(item)
    output = []
    for decision_on, shortlist in sorted(gate.shortlists.items()):
        selection = (selection_audits or {}).get(decision_on, {})
        scored = {item["symbol"]: item for item in selection.get("ranked", ())}
        supports = [symbol for symbol in shortlist if (
            gate.decisions[f"{decision_on}:{symbol}"].route == "SYNTHESIS_COMPLETE"
            and gate.decisions[f"{decision_on}:{symbol}"].synthesis is not None
            and gate.decisions[f"{decision_on}:{symbol}"].synthesis.verdict == CouncilVerdict.SUPPORT
        )]
        intended = target_history.get(decision_on, {}) if target_history is not None else None
        target_weight = 1 / len(supports) if supports else 0
        month = {"decision_date": decision_on.isoformat(), "top_20": []}
        for rank, symbol in enumerate(shortlist, 1):
            result = gate.decisions[f"{decision_on}:{symbol}"]
            bars = tuple(bar for bar in histories[symbol] if bar.session <= decision_on)
            following = next(bar for bar in histories[symbol] if bar.session > decision_on)
            synthesis = result.synthesis
            trades = trades_by_session_symbol.get((following.session, symbol), [])
            month["top_20"].append({
                "rank": scored.get(symbol, {}).get("rank", rank), "symbol": symbol,
                "decision_close": float(bars[-1].close),
                "return_21d": float(bars[-1].adjusted_close / bars[-22].adjusted_close - 1),
                "return_63d": float(bars[-1].adjusted_close / bars[-64].adjusted_close - 1),
                "return_252d": float(bars[-1].adjusted_close / bars[-253].adjusted_close - 1),
                "council_rating": synthesis.verdict.value if synthesis else result.route,
                "approved": symbol in supports,
                "target_weight": (
                    float(intended.get(symbol, 0)) if intended is not None
                    else target_weight if symbol in supports else 0
                ),
                "next_session": following.session.isoformat(), "next_open": float(following.open),
                "executions": [{
                    "action": trade.kind, "notional": float(trade.cash_amount),
                    "cost": float(trade.cost), "price": float(trade.price),
                } for trade in trades],
                "summary": synthesis.summary if synthesis else None,
                "risk_flags": list(synthesis.risk_flags) if synthesis else [],
                "evidence_reasons": [
                    {"role": report.role.value, "claim": finding.factual_claim,
                     "interpretation": finding.interpretation, "stance": finding.stance.value}
                    for report in result.council.reports for finding in report.findings[:1]
                ],
                "blockers": list(result.council.audit.blocker_codes),
                "llm_score": scored.get(symbol, {}).get("rating"),
                "score_components": scored.get(symbol, {}).get("assessment"),
                "kronos": scored.get(symbol, {}).get("kronos"),
            })
        month["top_20"].sort(key=lambda item: item["rank"])
        output.append(month)
    return output


def render_report(payload: dict[str, object]) -> str:
    metrics = payload["metrics"]
    attribution = payload["position_attribution"]
    lines = [
        "# Five-agent council backtest", "",
        f"- Evaluation: {payload['evaluation_start']} to {payload['evaluation_end']}",
        f"- Strategy return: {metrics['total_return']:.2%}",
        f"- SPY return: {metrics['benchmark_total_return']:.2%}",
        f"- Excess return: {metrics['excess_return']:.2%}",
        (f"- Profitable purchased symbols: {attribution['profitable_symbols']} of "
         f"{attribution['symbols_bought']} ({attribution['profitable_symbol_rate']:.2%})"),
        (f"- Losing purchased symbols: {attribution['losing_symbols']} of "
         f"{attribution['symbols_bought']} ({attribution['losing_symbol_rate']:.2%})"), "",
        f"- Average profit per profitable stock: ${attribution['average_profit_per_profitable_symbol']:,.2f}",
        f"- Average loss per losing stock: ${attribution['average_loss_per_losing_symbol']:,.2f}", "",
    ]
    for month in payload["monthly_decisions"]:
        lines.extend((f"## {month['decision_date']}", "", "| # | Stock | LLM score | Council | Close | Target | Next open |", "|---:|---|---:|---|---:|---:|---:|"))
        for item in month["top_20"]:
            lines.append(
                f"| {item['rank']} | {item['symbol']} | "
                f"{item['llm_score'] if item['llm_score'] is not None else '—'} | "
                f"{item['council_rating']} | ${item['decision_close']:.2f} | "
                f"{item['target_weight']:.2%} | ${item['next_open']:.2f} |"
            )
        lines.append("")
        lines.extend(("### Decision details", ""))
        for item in month["top_20"]:
            lines.extend((
                f"#### {item['rank']}. {item['symbol']} - {item['council_rating']}", "",
                (f"Decision close ${item['decision_close']:.2f}; 21d {item['return_21d']:.2%}; "
                 f"63d {item['return_63d']:.2%}; 252d {item['return_252d']:.2%}; "
                 f"target {item['target_weight']:.2%}; next open ${item['next_open']:.2f}."), "",
            ))
            if item["summary"]:
                lines.extend((f"Council summary: {item['summary']}", ""))
            if item["score_components"]:
                components = item["score_components"]
                lines.extend((
                    (f"LLM score: **{item['llm_score']:.2f}/100**. "
                     f"Business {components['business_quality']}; growth {components['growth_outlook']}; "
                     f"financial {components['financial_strength']}; catalyst {components['catalyst_strength']}; "
                     f"technical {components['technical_strength']}; forecast {components['forecast_strength']}; "
                     f"downside {components['downside_resilience']}; confidence {components['evidence_confidence']}."),
                    "", f"Scoring thesis: {components['thesis']}", "",
                ))
            if item["kronos"]:
                forecast = item["kronos"]
                lines.extend((
                    (f"Kronos: median {forecast['median_return']:.2%}; bear {forecast['bear_return']:.2%}; "
                     f"bull {forecast['bull_return']:.2%}; positive paths {forecast['probability_positive']:.1%}; "
                     f"forecast volatility {forecast['predicted_volatility']:.2%}."), "",
                ))
            if item["evidence_reasons"]:
                lines.append("Evidence highlights:")
                lines.append("")
                for reason in item["evidence_reasons"]:
                    lines.append(
                        f"- {reason['role']} ({reason['stance']}): {reason['claim']} "
                        f"{reason['interpretation']}"
                    )
                lines.append("")
            if item["risk_flags"]:
                lines.extend((f"Risk flags: {', '.join(item['risk_flags'])}", ""))
            if item["blockers"]:
                lines.extend((f"Audit blockers: {', '.join(item['blockers'])}", ""))
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the reproducible five-agent council backtest")
    parser.add_argument("--universe", type=Path, default=Path("data/current-top100-2026-08-19.json"))
    parser.add_argument("--start", type=date.fromisoformat, default=date(2024, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2025, 12, 31))
    parser.add_argument("--evaluation-start", type=date.fromisoformat, default=date(2025, 1, 2))
    parser.add_argument("--prices", type=Path, default=Path("private_data/prices-yahoo"))
    parser.add_argument("--rank-cache", type=Path, default=Path("private_data/ai-rankings-yahoo-2025"))
    parser.add_argument("--council-cache", type=Path, default=Path("private_data/council-yahoo-2025"))
    parser.add_argument("--output", type=Path, default=Path("private_data/results/council-yahoo-2025.json"))
    parser.add_argument("--report", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    arguments = parser.parse_args()
    if arguments.evaluation_start < arguments.start or arguments.evaluation_start > arguments.end:
        parser.error("evaluation-start must be inside the price window")

    settings = Settings.from_env()
    if not settings.openai_api_key:
        parser.error("OPENAI_API_KEY is missing from .env")
    if not settings.sec_user_agent:
        parser.error("SEC_USER_AGENT is missing from .env")
    snapshot = json.loads(arguments.universe.read_text(encoding="utf-8"))
    histories, benchmark, exclusions = load_evaluation_histories(
        snapshot, PriceCache(arguments.prices), arguments.start, arguments.end,
        arguments.evaluation_start,
    )

    runner = OpenAIResearchRunner(settings.openai_api_key, settings.openai_model)
    synthesizer = OpenAICouncilSynthesizer(settings.openai_api_key, settings.openai_model)
    council = LangGraphResearchCouncil(runner, synthesizer)
    gate = CachedCouncilGate(
        council, histories,
        SECEdgarCache(settings.sec_user_agent, Path("private_data/sec-edgar")),
        arguments.council_cache, arguments.workers,
    )
    strategy = CouncilGatedStrategy(
        AIRankedPortfolio(OpenAIRanker(
            settings.openai_api_key, settings.openai_model, arguments.rank_cache,
        )), gate, shortlist=20, top_n=20,
    )
    result = MultiAssetBacktester(cost_bps=Decimal(5), rebalance_every=21).run_window(
        histories, benchmark, strategy, arguments.evaluation_start,
    )
    decisions = monthly_decisions(gate, histories, result.transactions, strategy.target_history)
    payload = {
        "experiment": "five-agent-council-yahoo-2025", "model": settings.openai_model,
        "evaluation_start": arguments.evaluation_start.isoformat(),
        "evaluation_end": result.points[-1].session.isoformat(),
        "requested_symbol_count": snapshot["count"], "tested_symbol_count": len(histories),
        "excluded_symbols": exclusions,
        "price_source": "Yahoo Finance via yfinance", "benchmark": "SPY",
        "rebalance_every_sessions": 21, "cost_bps": 5,
        "ending_equity": float(result.points[-1].equity), "metrics": result.metrics,
        "council_activity": summarize_decisions(gate.decisions),
        "position_attribution": position_attribution(result),
        "monthly_decisions": decisions,
        "monthly_council": {
            key: {"route": value.route,
                  "verdict": value.synthesis.verdict.value if value.synthesis else None,
                  "blockers": list(value.council.audit.blocker_codes)}
            for key, value in sorted(gate.decisions.items())
        },
        "limitations": [
            "The frozen universe is based on current constituents and contains survivorship bias.",
            "Yahoo adjusted history is not point-in-time corporate-action truth.",
            "This is a research simulation, not evidence of future profitability.",
        ],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path = arguments.report or arguments.output.with_suffix(".md")
    report_path.write_text(render_report(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
