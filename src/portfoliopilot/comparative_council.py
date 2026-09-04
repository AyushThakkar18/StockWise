from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from pathlib import Path

from pydantic import Field

from .contracts import FrozenModel
from .research_contracts import FindingStance, OrchestratedCouncilResult

PROMPT_VERSION = "comparative-portfolio-v1"


class PortfolioSelection(FrozenModel):
    selected_ids: tuple[str, ...] = Field(min_length=20, max_length=20)
    selection_summary: str = Field(min_length=1, max_length=500)


def comparative_packet(
    candidates: tuple[str, ...], results: dict[str, OrchestratedCouncilResult],
    factor_audit: dict[str, object],
) -> tuple[dict[str, object], dict[str, str]]:
    aliases = {f"asset_{index:03d}": symbol for index, symbol in enumerate(candidates, 1)}
    factors = {item["symbol"]: item for item in factor_audit.get("ranked", ())}
    records = []
    for alias, symbol in aliases.items():
        result = results[symbol]
        findings = [finding for report in result.council.reports for finding in report.findings]
        stances = Counter(finding.stance.value for finding in findings)
        severe_concerns = sum(
            finding.stance == FindingStance.CONCERN and finding.severity.value == "HIGH"
            for finding in findings
        )
        synthesis = result.synthesis
        factor = factors[symbol]
        records.append({
            "id": alias,
            "deterministic_rank": factor["rank"],
            "deterministic_score": factor["score"],
            "momentum": factor["momentum"],
            "relative_strength": factor["relative_strength"],
            "low_volatility": factor["low_volatility"],
            "growth": factor["growth"],
            "quality": factor["quality"],
            "sector": factor["sector"],
            "council_route": result.route,
            "council_verdict": synthesis.verdict.value if synthesis else "ABSTAIN",
            "support_findings": stances["SUPPORT"],
            "concern_findings": stances["CONCERN"],
            "neutral_findings": stances["NEUTRAL"],
            "high_severity_concerns": severe_concerns,
            "evidence_coverage": result.council.audit.evidence_coverage,
            "audit_blocker_count": len(result.council.audit.blocker_codes),
            "risk_flag_count": len(synthesis.risk_flags) if synthesis else 0,
        })
    return {
        "prompt_version": PROMPT_VERSION,
        "objective": (
            "Select the strongest expected risk-adjusted return candidates for the next monthly "
            "holding period using only supplied point-in-time records."
        ),
        "constraints": {"exact_positions": 20, "maximum_per_sector": 5},
        "market_regime": factor_audit.get("market_regime"),
        "candidates": records,
    }, aliases


class OpenAIComparativeSelector:
    def __init__(
        self, api_key: str, model: str, cache_directory: Path,
        maximum_attempts: int = 3, request_timeout: float = 180,
    ):
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required")
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, timeout=request_timeout, max_retries=0)
        self.model = model
        self.cache_directory = cache_directory
        self.maximum_attempts = maximum_attempts
        cache_directory.mkdir(parents=True, exist_ok=True)

    def __call__(self, packet: dict[str, object]) -> PortfolioSelection:
        encoded = json.dumps(packet, sort_keys=True, separators=(",", ":"))
        fingerprint = hashlib.sha256(
            f"{self.model}:{PROMPT_VERSION}:{encoded}".encode(),
        ).hexdigest()
        path = self.cache_directory / f"{fingerprint}.json"
        if path.exists():
            return PortfolioSelection.model_validate_json(path.read_text(encoding="utf-8"))
        correction = ""
        for attempt in range(self.maximum_attempts):
            response = self.client.responses.parse(
                model=self.model,
                input=[
                    {"role": "system", "content": (
                        "You are the final comparative portfolio-selection agent. Compare all "
                        "anonymous candidates as one cross-section. Return exactly 20 unique IDs, "
                        "strongest first, with no more than five candidates from any sector. Use "
                        "only the supplied point-in-time data. Do not infer company identities, "
                        "invent facts, or relax constraints. " + correction
                    )},
                    {"role": "user", "content": encoded},
                ],
                text_format=PortfolioSelection,
            )
            selection = response.output_parsed
            if selection is not None:
                error = validate_selection(selection, packet)
                if error is None:
                    path.write_text(selection.model_dump_json(indent=2), encoding="utf-8")
                    return selection
                correction = f"Previous output was invalid: {error}. Correct it."
            time.sleep(2 ** attempt)
        raise ValueError("comparative selector failed deterministic validation")


def validate_selection(selection: PortfolioSelection, packet: dict[str, object]) -> str | None:
    candidates = {item["id"]: item for item in packet["candidates"]}  # type: ignore[index]
    selected = selection.selected_ids
    if len(selected) != 20 or len(set(selected)) != 20:
        return "selection must contain exactly 20 unique IDs"
    unknown = set(selected) - set(candidates)
    if unknown:
        return f"unknown candidate IDs: {sorted(unknown)}"
    sectors = Counter(candidates[item]["sector"] for item in selected)
    maximum = int(packet["constraints"]["maximum_per_sector"])  # type: ignore[index]
    if sectors and max(sectors.values()) > maximum:
        return f"sector limit exceeded: {dict(sectors)}"
    return None


@dataclass
class ComparativeCouncilStrategy:
    candidate_strategy: object
    council_gate: object
    selector: object
    candidate_count: int = 100
    position_count: int = 20
    name: str = "top100_five_agent_comparative_top20"
    target_history: dict[date, dict[str, Decimal]] = field(default_factory=dict)
    selection_audits: dict[date, dict[str, object]] = field(default_factory=dict)

    def targets(self, history) -> dict[str, Decimal]:
        decision_on = max(bars[-1].session for bars in history.values() if bars)
        ranked_targets = self.candidate_strategy.targets(history)
        candidates = tuple(ranked_targets)[:self.candidate_count]
        if len(candidates) != self.candidate_count:
            raise ValueError(f"expected {self.candidate_count} research candidates")
        self.council_gate.evaluate_many(candidates, decision_on)
        results = {
            symbol: self.council_gate.decisions[f"{decision_on}:{symbol}"]
            for symbol in candidates
        }
        factor_audit = self.candidate_strategy.audits[decision_on]
        packet, aliases = comparative_packet(candidates, results, factor_audit)
        selection = self.selector(packet)
        error = validate_selection(selection, packet)
        if error is not None:
            raise ValueError(error)
        selected = tuple(aliases[item] for item in selection.selected_ids)
        weight = Decimal(1) / Decimal(self.position_count)
        targets = {symbol: weight for symbol in selected}
        self.target_history[decision_on] = targets
        self.selection_audits[decision_on] = {
            "candidates": list(candidates), "selected": list(selected),
            "selection_summary": selection.selection_summary,
            "packet": packet,
        }
        return targets
