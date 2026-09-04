from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from pathlib import Path

from pydantic import Field

from .contracts import FrozenModel
from .kronos_forecast import KronosForecast
from .research_contracts import OrchestratedCouncilResult

PROMPT_VERSION = "anchored-stock-score-v1"
WEIGHTS = {
    "business_quality": .15, "growth_outlook": .15, "financial_strength": .15,
    "catalyst_strength": .10, "technical_strength": .15, "forecast_strength": .15,
    "downside_resilience": .10, "evidence_confidence": .05,
}


class StockAssessment(FrozenModel):
    business_quality: int = Field(ge=0, le=100)
    growth_outlook: int = Field(ge=0, le=100)
    financial_strength: int = Field(ge=0, le=100)
    catalyst_strength: int = Field(ge=0, le=100)
    technical_strength: int = Field(ge=0, le=100)
    forecast_strength: int = Field(ge=0, le=100)
    downside_resilience: int = Field(ge=0, le=100)
    evidence_confidence: int = Field(ge=0, le=100)
    thesis: str = Field(min_length=1, max_length=500)
    key_risks: tuple[str, ...] = Field(max_length=5)


def overall_score(assessment: StockAssessment) -> float:
    return round(sum(getattr(assessment, name) * weight for name, weight in WEIGHTS.items()), 2)


def stock_packet(
    symbol: str, factor: dict[str, object], result: OrchestratedCouncilResult,
    forecast: KronosForecast | None = None, benchmark_forecast: KronosForecast | None = None,
) -> dict[str, object]:
    findings = []
    for report in result.council.reports:
        for finding in report.findings:
            findings.append({
                "role": report.role.value, "topic": finding.topic,
                "claim": _anonymize(finding.factual_claim, symbol),
                "interpretation": _anonymize(finding.interpretation, symbol),
                "stance": finding.stance.value, "severity": finding.severity.value,
                "uncertainty": _anonymize(finding.uncertainty_notes, symbol),
            })
    synthesis = result.synthesis
    packet: dict[str, object] = {
        "prompt_version": PROMPT_VERSION,
        "objective": "Score expected risk-adjusted performance over the next 21 sessions.",
        "identity": "anonymous_candidate",
        "deterministic_factors": {
            name: factor.get(name) for name in (
                "rank", "score", "momentum", "relative_strength", "low_volatility",
                "growth", "quality", "sector",
            )
        },
        "research_findings": findings,
        "research_audit": {
            "route": result.route,
            "approved": result.council.audit.approved,
            "evidence_coverage": result.council.audit.evidence_coverage,
            "blocker_codes": result.council.audit.blocker_codes,
            "checks": [check.model_dump(mode="json") for check in result.council.audit.checks],
            "contradictions": [item.model_dump(mode="json") for item in result.council.audit.contradictions],
        },
        "synthesis": None if synthesis is None else {
            "verdict": synthesis.verdict.value,
            "summary": _anonymize(synthesis.summary, symbol),
            "risk_flags": [_anonymize(item, symbol) for item in synthesis.risk_flags],
        },
        "score_weights": WEIGHTS,
    }
    if forecast:
        packet["candidate_kronos_21_session_forecast"] = forecast.model_dump(mode="json")
    if benchmark_forecast:
        packet["benchmark_forecast"] = benchmark_forecast.model_dump(mode="json")
        if forecast:
            packet["candidate_forecast_excess_return"] = (
                forecast.median_return - benchmark_forecast.median_return
            )
    return packet


def _anonymize(value: str, symbol: str) -> str:
    return value.replace(symbol, "the company").replace(symbol.replace("-", "."), "the company")


class OpenAIStockScorer:
    def __init__(self, api_key: str, model: str, cache_directory: Path, maximum_attempts: int = 3):
        from openai import OpenAI

        self.client, self.model = OpenAI(api_key=api_key, timeout=180, max_retries=0), model
        self.cache_directory, self.maximum_attempts = cache_directory, maximum_attempts
        cache_directory.mkdir(parents=True, exist_ok=True)

    def __call__(self, packet: dict[str, object]) -> StockAssessment:
        from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

        encoded = json.dumps(packet, sort_keys=True, separators=(",", ":"))
        fingerprint = hashlib.sha256(
            f"{self.model}:{PROMPT_VERSION}:{encoded}".encode(),
        ).hexdigest()
        path = self.cache_directory / f"{fingerprint}.json"
        if path.exists():
            return StockAssessment.model_validate_json(path.read_text(encoding="utf-8"))
        for attempt in range(self.maximum_attempts):
            try:
                response = self.client.responses.parse(
                    model=self.model,
                    input=[{"role": "system", "content": (
                        "You score one anonymous stock using only supplied point-in-time evidence. "
                        "Use the full 0-100 range consistently: 90-100 exceptional, 75-89 strong, "
                        "60-74 favorable but mixed, 40-59 neutral/uncertain, 20-39 weak, 0-19 adverse. "
                        "Treat missing evidence, contradictions, low coverage, and forecast dispersion "
                        "as uncertainty; do not identify the company or use outside knowledge. Score "
                        "each component independently. The candidate forecast and benchmark forecast "
                        "are distinctly labeled: never describe benchmark values as candidate values; "
                        "evaluate the candidate forecast primarily relative to the benchmark forecast. "
                        "Code—not you—computes the final weighted score."
                    )}, {"role": "user", "content": encoded}],
                    text_format=StockAssessment,
                )
            except (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError):
                if attempt + 1 == self.maximum_attempts:
                    raise
                time.sleep(2 ** attempt)
                continue
            if response.output_parsed is not None:
                path.write_text(response.output_parsed.model_dump_json(indent=2), encoding="utf-8")
                return response.output_parsed
            time.sleep(2 ** attempt)
        raise ValueError("per-stock scorer returned no valid structured assessment")


def select_diversified(
    ranked: list[tuple[str, float, int, str]], count: int = 20, maximum_per_sector: int = 5,
) -> tuple[str, ...]:
    selected, sectors = [], Counter()
    for symbol, _, _, sector in sorted(ranked, key=lambda item: (-item[1], item[2], item[0])):
        if sectors[sector] >= maximum_per_sector:
            continue
        selected.append(symbol)
        sectors[sector] += 1
        if len(selected) == count:
            return tuple(selected)
    raise ValueError(f"sector constraints permit only {len(selected)} of {count} positions")


@dataclass
class PerStockCouncilStrategy:
    candidate_strategy: object
    council_gate: object
    scorer: object
    forecaster: object | None = None
    benchmark: tuple = ()
    workers: int = 4
    candidate_count: int = 100
    position_count: int = 20
    name: str = "top100_five_agent_per_stock_scoring_top20"
    target_history: dict[date, dict[str, Decimal]] = field(default_factory=dict)
    selection_audits: dict[date, dict[str, object]] = field(default_factory=dict)

    def targets(self, history) -> dict[str, Decimal]:
        decision_on = max(bars[-1].session for bars in history.values() if bars)
        candidates = tuple(self.candidate_strategy.targets(history))[:self.candidate_count]
        if len(candidates) != self.candidate_count:
            raise ValueError(f"expected {self.candidate_count} research candidates")
        print(f"[{decision_on}] researching {len(candidates)} candidates...", flush=True)
        self.council_gate.evaluate_many(candidates, decision_on)
        factors = {
            item["symbol"]: item for item in self.candidate_strategy.audits[decision_on]["ranked"]
        }
        forecasts = {}
        if self.forecaster:
            print(f"[{decision_on}] generating Kronos forecasts on configured device...", flush=True)
            for index, symbol in enumerate(candidates, 1):
                forecasts[symbol] = self.forecaster(history[symbol])
                if index % 10 == 0 or index == len(candidates):
                    print(f"[{decision_on}] Kronos {index}/{len(candidates)}", flush=True)
        benchmark_history = tuple(bar for bar in self.benchmark if bar.session <= decision_on)
        benchmark_forecast = self.forecaster(benchmark_history) if self.forecaster else None
        assessments = {}
        print(f"[{decision_on}] scoring candidates independently...", flush=True)
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {
                pool.submit(self.scorer, stock_packet(
                    symbol, factors[symbol],
                    self.council_gate.decisions[f"{decision_on}:{symbol}"],
                    forecasts.get(symbol), benchmark_forecast,
                )): symbol for symbol in candidates
            }
            for index, future in enumerate(as_completed(futures), 1):
                assessments[futures[future]] = future.result()
                if index % 10 == 0 or index == len(candidates):
                    print(f"[{decision_on}] LLM scores {index}/{len(candidates)}", flush=True)
        ranked = [(symbol, overall_score(assessments[symbol]), int(factors[symbol]["rank"]),
                   str(factors[symbol]["sector"])) for symbol in candidates]
        selected = select_diversified(ranked, self.position_count)
        targets = {symbol: Decimal(1) / Decimal(self.position_count) for symbol in selected}
        self.target_history[decision_on] = targets
        self.selection_audits[decision_on] = {
            "selected": list(selected),
            "ranked": [{"rank": index, "symbol": symbol, "rating": score,
                        "deterministic_rank": rank, "sector": sector,
                        "assessment": assessments[symbol].model_dump(mode="json"),
                        "kronos": forecasts[symbol].model_dump(mode="json") if symbol in forecasts else None}
                       for index, (symbol, score, rank, sector) in enumerate(
                           sorted(ranked, key=lambda item: (-item[1], item[2], item[0])), 1)],
        }
        print(f"[{decision_on}] selected Top {self.position_count}: {', '.join(selected)}", flush=True)
        return targets
