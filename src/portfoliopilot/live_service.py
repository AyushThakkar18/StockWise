from __future__ import annotations

import json
import time
from calendar import monthrange
from datetime import UTC, date, datetime, timedelta
from datetime import time as clock_time
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

from .bounded_agent_strategy import feature_packet
from .config import Settings
from .contracts import Evidence
from .forward_paper import ForwardPaperEngine
from .kronos_forecast import KronosForecaster
from .kronos_only_backtest import forecast_score
from .live_council import LiveCandidatePacket, LiveCouncil, LiveCouncilDecision
from .live_evidence import YahooNewsEvidence
from .openai_bounded_agents import ALLOWED_MODEL
from .openai_live_council import OpenAILiveCouncilAgents
from .robust_strategy import SimpleMomentumTrendGrowthStrategy
from .sec_edgar import SECEdgarCache
from .universe import load_membership_history
from .yahoo_download import download_batch, download_quotes

EASTERN = ZoneInfo("America/New_York")
LIVE_POLICY_VERSION = "live-council-v3-quality-floor-70"


def kronos_signal(forecast, benchmark, rank: int, population: int) -> dict[str, object]:
    """Convert unstable absolute forecasts into a bounded cross-sectional feature."""
    reasons = []
    if abs(benchmark.median_return) > .08:
        reasons.append("EXTREME_BENCHMARK_MEDIAN")
    if abs(forecast.median_return) > .15:
        reasons.append("EXTREME_CANDIDATE_MEDIAN")
    if forecast.forecast_dispersion > .15:
        reasons.append("HIGH_FORECAST_DISPERSION")
    percentile = 1.0 if population == 1 else 1 - (rank - 1) / (population - 1)
    signal = {
        "status": "UNRELIABLE" if reasons else "RELIABLE",
        "reliability_reasons": reasons,
        "cross_sectional_percentile": round(percentile, 6),
        "relative_to_benchmark": round(forecast.median_return - benchmark.median_return, 6),
        "horizon_sessions": forecast.horizon,
    }
    if not reasons:
        signal["candidate_forecast"] = forecast.model_dump(mode="json")
        signal["benchmark_forecast"] = benchmark.model_dump(mode="json")
    return signal


def select_reviewed_candidates(
    candidates: tuple[str, ...], ranked: list[tuple[str, float, object]], benchmark_forecast,
    count: int = 50,
) -> tuple[list[tuple[str, float, object]], str]:
    if abs(benchmark_forecast.median_return) <= .08:
        return ranked[:count], "KRONOS_CROSS_SECTIONAL_TOP_50"
    by_symbol = {item[0]: item for item in ranked}
    return (
        [by_symbol[symbol] for symbol in candidates[:count]],
        "DETERMINISTIC_TOP_50_KRONOS_REJECTED",
    )


def next_weekday_open(day: date) -> datetime:
    following = day + timedelta(days=1)
    while following.weekday() >= 5:
        following += timedelta(days=1)
    return datetime.combine(following, clock_time(9, 30), tzinfo=EASTERN).astimezone(UTC)


def add_calendar_month(day: date) -> date:
    year, month = day.year + (day.month == 12), day.month % 12 + 1
    return date(year, month, min(day.day, monthrange(year, month)[1]))


class YahooLivePrices:
    def histories(self, symbols: tuple[str, ...], end: date, attempts: int = 3) -> dict:
        output, pending = {}, tuple(dict.fromkeys(symbols))
        for attempt in range(attempts):
            if not pending:
                break
            output.update(download_batch(pending, end - timedelta(days=550), end))
            pending = tuple(symbol for symbol in pending if symbol not in output)
            if pending and attempt + 1 < attempts:
                time.sleep(2**attempt)
        return output

    def marks(self, symbols: tuple[str, ...]) -> dict[str, Decimal]:
        return download_quotes(symbols)


class ProductionCouncilRunner:
    def __init__(
        self, settings: Settings, *, device: str = "auto",
        membership_path: Path = Path("private_data/universe/sp500-components-updated.csv"),
    ) -> None:
        if not settings.openai_api_key or not settings.sec_user_agent:
            raise ValueError("OPENAI_API_KEY and SEC_USER_AGENT are required")
        self.settings, self.membership_path = settings, membership_path
        self.prices = YahooLivePrices()
        self.forecaster = KronosForecaster(
            Path("private_data/Kronos"), Path("private_data/kronos-live"),
            model_id="NeoQuasar/Kronos-base", device=device, recycle_every=10,
            cooldown_seconds=.25,
        )
        self.agents = OpenAILiveCouncilAgents(
            settings.openai_api_key, Path("private_data/openai-live-council"), model=ALLOWED_MODEL,
        )
        self.news = YahooNewsEvidence()

    def run(self, decision_at: datetime):
        decision_on = decision_at.astimezone(EASTERN).date()
        membership = load_membership_history(self.membership_path)
        members = membership.members_on(decision_on)
        histories = self.prices.histories(tuple(sorted(set(members) | {"SPY"})), decision_on)
        benchmark = histories.pop("SPY", None)
        if benchmark is None or len(histories) < 100:
            raise ValueError("insufficient current Yahoo coverage for live screening")
        sec = SECEdgarCache(
            self.settings.sec_user_agent or "", Path("private_data/sec-live") / decision_on.isoformat(),
        )
        ciks = sec.ticker_map()
        strategy = SimpleMomentumTrendGrowthStrategy(
            benchmark, sec, ciks, top_n=100, buffer_rank=120,
            maximum_position_weight=Decimal(1), maximum_sector_weight=Decimal(1),
        )
        candidates = tuple(strategy.targets(histories))[:100]
        if len(candidates) != 100:
            raise ValueError(f"deterministic screen produced only {len(candidates)} candidates")
        factors = {item["symbol"]: item for item in strategy.audits[decision_on]["ranked"]}
        benchmark_forecast = self.forecaster(benchmark)
        ranked = []
        for index, symbol in enumerate(candidates, 1):
            forecast = self.forecaster(histories[symbol])
            ranked.append((symbol, forecast_score(forecast, benchmark_forecast), forecast))
            if index % 10 == 0:
                print(f"[{decision_on}] live Kronos {index}/100", flush=True)
        ranked.sort(key=lambda item: (-item[1], int(factors[item[0]]["rank"]), item[0]))
        packets = []
        rank_by_symbol = {symbol: index for index, (symbol, _, _) in enumerate(ranked, 1)}
        # An invalid forecast remains in the audit but cannot control council admission.
        reviewed, screening_method = select_reviewed_candidates(
            candidates, ranked, benchmark_forecast,
        )
        for screen_rank, (symbol, relative_score, forecast) in enumerate(reviewed, 1):
            cik = ciks[symbol]
            metadata = sec.submission_metadata(symbol, cik)
            retrieved = datetime.now(UTC)
            evidence = tuple(filter(None, (
                sec.evidence_on(symbol, cik, decision_on, retrieved),
            ))) + self.news.collect(symbol, decision_at, retrieved)
            quantitative = feature_packet(symbol, histories[symbol], benchmark, factors[symbol])
            quantitative.update({
                "kronos_signal": kronos_signal(
                    forecast, benchmark_forecast, rank_by_symbol[symbol], len(ranked),
                ),
                "candidate_screening_method": screening_method,
            })
            packets.append(LiveCandidatePacket(
                candidate_id=f"{decision_on}:{symbol}", symbol=symbol,
                company_name=metadata.get("name") or symbol, sector=factors[symbol].get("sector"),
                decision_at=decision_at, deterministic_rank=int(factors[symbol]["rank"]),
                kronos_rank=screen_rank, quantitative_features=quantitative,
                evidence=tuple(item for item in evidence if isinstance(item, Evidence)),
            ))
        council = LiveCouncil(
            self.agents.specialist, self.agents.synthesize, model=ALLOWED_MODEL,
            prompt_version=LIVE_POLICY_VERSION, maximum_selections=20, minimum_score=70,
            minimum_specialist_score=55, minimum_specialists=3,
        )
        return council.decide(f"live-{decision_on.isoformat()}", packets), histories, benchmark


class LivePaperService:
    def __init__(
        self, settings: Settings, runner: ProductionCouncilRunner,
        database: Path | None = None,
    ) -> None:
        self.settings, self.runner = settings, runner
        self.engine = ForwardPaperEngine(database or settings.database_path)

    def cycle(
        self, now: datetime | None = None, *, update_prices: bool = True,
    ) -> dict[str, object]:
        current = (now or datetime.now(UTC)).astimezone(UTC)
        executed = self._execute_pending(current)
        valuation_updated = self._mark_to_market(current) if update_prices else False
        decisions = self.engine.paper.all_decision_payloads()
        local = current.astimezone(EASTERN)
        current_month = local.strftime("%Y-%m")
        next_due = (
            add_calendar_month(date.fromisoformat(decisions[-1]["earliest_execution_date"]))
            if decisions else local.date()
        )
        due_today = local.date() >= next_due
        after_close = local.time() >= clock_time(16, 15)
        created = None
        if due_today and after_close:
            decision, histories, benchmark = self.runner.run(current)
            marks = self._latest_marks(histories | {"SPY": benchmark})
            execution_at = next_weekday_open(current.astimezone(EASTERN).date())
            self.engine.freeze(decision, execution_at, marks)
            created = decision.decision_id
        result = {
            "checked_at": current.isoformat(), "month": current_month,
            "decision_due": created is not None,
            "decision_created": created, "executed_sessions": executed,
            "valuation_updated": valuation_updated,
            "next_research_date": next_due.isoformat(),
            "next_action": (
                "WAIT_FOR_NEXT_OPEN" if created else
                "WAIT_FOR_REBALANCE_DATE" if not due_today else "WAIT_FOR_AFTER_CLOSE"
            ),
        }
        event_id = f"live-service-cycle:{current.strftime('%Y%m%dT%H%M%S')}"
        if not any(event["event_id"] == event_id for event in self.engine.events.events()):
            self.engine.events.append(event_id, "LIVE_SERVICE_CYCLE", "live-service", result)
        return result

    def price_cycle(self, now: datetime | None = None) -> dict[str, object]:
        current = (now or datetime.now(UTC)).astimezone(UTC)
        updated = self._mark_to_market(current)
        return {
            "checked_at": current.isoformat(), "valuation_updated": updated,
            "mode": "PRICES_ONLY",
        }

    def initialize_from_latest_close(self, now: datetime | None = None) -> dict[str, object]:
        """Create the first monthly decision from the latest fully completed US session."""
        current = (now or datetime.now(UTC)).astimezone(UTC)
        local = current.astimezone(EASTERN)
        cutoff_day = local.date() if local.time() >= clock_time(16, 15) else local.date() - timedelta(1)
        while cutoff_day.weekday() >= 5:
            cutoff_day -= timedelta(1)
        cutoff = datetime.combine(cutoff_day, clock_time(16, 15), tzinfo=EASTERN).astimezone(UTC)
        month = cutoff.astimezone(EASTERN).strftime("%Y-%m")
        if any(item["decision_date"].startswith(month)
               for item in self.engine.paper.all_decision_payloads()):
            raise ValueError(f"a frozen paper decision already exists for {month}")
        decision, histories, benchmark = self.runner.run(cutoff)
        execution_at = next_weekday_open(cutoff_day)
        self.engine.freeze(
            decision, execution_at, self._latest_marks(histories | {"SPY": benchmark}),
        )
        result = {
            "decision_created": decision.decision_id,
            "decision_cutoff": cutoff.isoformat(),
            "execution_at": execution_at.isoformat(),
            "next_action": "WAIT_FOR_NEXT_OPEN",
        }
        self.engine.events.append(
            f"live-initialization:{decision.decision_id}", "LIVE_INITIALIZATION",
            "live-service", result,
        )
        return result

    def shadow_from_latest_close(self, now: datetime | None = None) -> LiveCouncilDecision:
        """Research the latest completed close without freezing or executing a portfolio."""
        current = (now or datetime.now(UTC)).astimezone(UTC)
        local = current.astimezone(EASTERN)
        cutoff_day = local.date() if local.time() >= clock_time(16, 15) else local.date() - timedelta(1)
        while cutoff_day.weekday() >= 5:
            cutoff_day -= timedelta(1)
        cutoff = datetime.combine(cutoff_day, clock_time(16, 15), tzinfo=EASTERN).astimezone(UTC)
        decision, _, _ = self.runner.run(cutoff)
        return decision.model_copy(update={"decision_id": f"shadow-{decision.decision_id}-v3"})

    def promote_shadow(self, path: Path, now: datetime | None = None) -> dict[str, object]:
        """Promote an audited shadow decision into a simulated replacement rebalance."""
        payload = json.loads(path.read_text(encoding="utf-8"))
        decision = LiveCouncilDecision.model_validate(payload["decision"])
        current = (now or datetime.now(UTC)).astimezone(UTC)
        execution_at = next_weekday_open(decision.decision_at.astimezone(EASTERN).date())
        if current < execution_at:
            raise ValueError("the decision's next market open has not arrived")
        completed = {
            event["entity_id"] for event in self.engine.events.events()
            if event["event_type"] == "PAPER_SESSION_EXECUTED"
        }
        if decision.decision_id in completed:
            snapshots = self.engine.paper.snapshots()
            held = set(snapshots[-1]["positions"]) if snapshots else set()
            missing = set(decision.selected_symbols) - held
            if not missing:
                return {"decision_id": decision.decision_id, "status": "ALREADY_EXECUTED"}
            decision = decision.model_copy(update={
                "decision_id": f"{decision.decision_id}-allocation-correction-v3",
            })
        snapshots = self.engine.paper.snapshots()
        held = set(snapshots[-1]["positions"]) if snapshots else set()
        symbols = tuple(sorted(set(decision.selected_symbols) | held | {"SPY"}))
        histories = self.runner.prices.histories(symbols, execution_at.astimezone(EASTERN).date())
        bars = {
            symbol: series[-1] for symbol, series in histories.items()
            if series and series[-1].session == execution_at.astimezone(EASTERN).date()
        }
        if set(bars) != set(symbols):
            missing = sorted(set(symbols) - set(bars))
            raise ValueError(f"missing execution-session prices: {', '.join(missing)}")
        marks = {symbol: bar.close for symbol, bar in bars.items()}
        opens = {symbol: bar.open for symbol, bar in bars.items()}
        self.engine.freeze(decision, execution_at, marks)
        results = self.engine.execute_next_open(decision.decision_id, execution_at, opens)
        self.engine.events.append(
            f"shadow-promotion:{decision.decision_id}", "SHADOW_DECISION_PROMOTED",
            decision.decision_id, {
                "source": str(path), "promoted_at": current.isoformat(),
                "execution_at": execution_at.isoformat(),
                "disclosure": "Replacement simulation generated after the historical open.",
            },
        )
        return {
            "decision_id": decision.decision_id, "status": "EXECUTED_REPLACEMENT_SIMULATION",
            "execution_at": execution_at.isoformat(), "orders": len(results),
            "selected_symbols": decision.selected_symbols,
        }

    def _execute_pending(self, now: datetime) -> list[str]:
        completed = {event["entity_id"] for event in self.engine.events.events()
                     if event["event_type"] == "PAPER_SESSION_EXECUTED"}
        output = []
        for item in self.engine.paper.all_decision_payloads():
            decision = item["payload"].get("decision", {})
            decision_id = decision.get("decision_id")
            if not decision_id or decision_id in completed:
                continue
            execution_at = datetime.fromisoformat(item["payload"]["earliest_execution_at"])
            execution_day = execution_at.astimezone(EASTERN).date()
            if now.astimezone(EASTERN).date() < execution_day:
                continue
            snapshots = self.engine.paper.snapshots()
            held = set(snapshots[-1]["positions"]) if snapshots else set()
            symbols = tuple(sorted(set(item["payload"]["target_weights"]) | held))
            histories = self.runner.prices.histories(symbols, now.astimezone(EASTERN).date())
            opens = {symbol: bars[-1].open for symbol, bars in histories.items()
                     if bars and bars[-1].session >= execution_day}
            if set(opens) != set(symbols):
                continue
            self.engine.execute_next_open(decision_id, execution_at, opens)
            output.append(decision_id)
        return output

    def _mark_to_market(self, now: datetime) -> bool:
        snapshots = self.engine.paper.snapshots()
        if not snapshots:
            return False
        positions = snapshots[-1]["positions"]
        if not positions:
            return False
        marks = self.runner.prices.marks(tuple(positions))
        stale = sorted(set(positions) - set(marks))
        if stale:
            valuations = [
                json.loads(event["payload"]) for event in self.engine.events.events()
                if event["event_type"] == "PORTFOLIO_VALUATION"
            ]
            previous = valuations[-1].get("positions", {}) if valuations else {}
            marks.update({
                symbol: Decimal(str(previous[symbol]["mark"])) for symbol in stale
                if symbol in previous and previous[symbol].get("mark") is not None
            })
        if set(marks) != set(positions):
            return False
        ledger, _ = self.engine._recover()
        payload = {
            "session": now.astimezone(EASTERN).date().isoformat(),
            "equity": str(ledger.equity(marks)), "cash": str(ledger.cash),
            "realized_pnl": str(ledger.realized_pnl),
            "stale_symbols": stale,
            "prices_retrieved_at": now.isoformat(),
            "price_source": "Yahoo Finance lastPrice",
            "positions": {
                symbol: {
                    "quantity": str(position.quantity), "average_cost": str(position.average_cost),
                    "mark": str(marks[symbol]), "market_value": str(position.quantity * marks[symbol]),
                    "unrealized_pnl": str(position.quantity * marks[symbol] - position.cost_basis),
                } for symbol, position in ledger.positions.items() if position.quantity
            },
        }
        payload["unrealized_pnl"] = str(sum(
            (Decimal(value["unrealized_pnl"]) for value in payload["positions"].values()), Decimal(0),
        ))
        event_id = f"mark:{now.strftime('%Y%m%dT%H%M')}"
        if not any(event["event_id"] == event_id for event in self.engine.events.events()):
            self.engine.events.append(event_id, "PORTFOLIO_VALUATION", "portfolio", payload)
        return True

    @staticmethod
    def _latest_marks(histories: dict) -> dict[str, Decimal]:
        return {symbol: bars[-1].close for symbol, bars in histories.items() if bars}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the restart-safe StockWise live paper service")
    parser.add_argument("--once", action="store_true", help="run one readiness/execution cycle")
    parser.add_argument(
        "--initialize-from-latest-close", action="store_true",
        help="create the first monthly decision from the latest completed US session",
    )
    parser.add_argument(
        "--shadow-latest-close", action="store_true",
        help="rerun latest-close research without changing the frozen paper portfolio",
    )
    parser.add_argument(
        "--shadow-output", type=Path,
        default=Path("private_data/results/live-shadow-latest-v2.json"),
    )
    parser.add_argument("--promote-shadow", type=Path)
    parser.add_argument("--interval", type=int, default=900, help="poll interval in seconds")
    parser.add_argument("--device", default="auto")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prices-only", action="store_true")
    mode.add_argument("--research-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.interval < 60:
        parser.error("interval must be at least 60 seconds")
    settings = Settings.from_env()
    service = LivePaperService(settings, ProductionCouncilRunner(settings, device=arguments.device))
    if arguments.initialize_from_latest_close:
        print(json.dumps(service.initialize_from_latest_close(), indent=2), flush=True)
        return
    if arguments.shadow_latest_close:
        decision = service.shadow_from_latest_close()
        arguments.shadow_output.parent.mkdir(parents=True, exist_ok=True)
        arguments.shadow_output.write_text(
            json.dumps(decision.audit_payload(), indent=2), encoding="utf-8",
        )
        print(json.dumps({
            "decision_id": decision.decision_id,
            "selected_symbols": decision.selected_symbols,
            "output": str(arguments.shadow_output),
        }, indent=2), flush=True)
        return
    if arguments.promote_shadow:
        print(json.dumps(service.promote_shadow(arguments.promote_shadow), indent=2), flush=True)
        return
    while True:
        try:
            result = (
                service.price_cycle() if arguments.prices_only
                else service.cycle(update_prices=not arguments.research_only)
            )
            print(json.dumps(result, indent=2), flush=True)
        except Exception as exc:
            print(json.dumps({"status": "ERROR", "error": f"{type(exc).__name__}: {exc}"}), flush=True)
            if arguments.once:
                raise
        if arguments.once:
            return
        time.sleep(arguments.interval)


if __name__ == "__main__":
    main()
