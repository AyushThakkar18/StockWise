from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from enum import StrEnum
from statistics import pstdev

from pydantic import Field, model_validator

from .contracts import Evidence, FrozenModel
from .portfolio_construction import CorrelationEstimate, pairwise_summary


class LiveCouncilRole(StrEnum):
    MARKET = "MARKET_TECHNICAL"
    BUSINESS = "BUSINESS_FUNDAMENTALS"
    CATALYST = "NEWS_FILINGS_CATALYSTS"
    RISK = "RISK_SENTIMENT"


ROLE_WEIGHTS: dict[LiveCouncilRole, float] = {
    LiveCouncilRole.MARKET: 0.30,
    LiveCouncilRole.BUSINESS: 0.30,
    LiveCouncilRole.CATALYST: 0.20,
    LiveCouncilRole.RISK: 0.20,
}


class LiveCandidatePacket(FrozenModel):
    candidate_id: str = Field(min_length=1)
    symbol: str = Field(min_length=1)
    company_name: str = Field(min_length=1)
    sector: str | None = None
    decision_at: datetime
    horizon_sessions: int = Field(default=21, ge=1)
    deterministic_rank: int = Field(ge=1)
    kronos_rank: int = Field(ge=1)
    quantitative_features: dict[str, object]
    evidence: tuple[Evidence, ...]

    @model_validator(mode="after")
    def evidence_is_candidate_bound_and_available(self) -> LiveCandidatePacket:
        ids = [item.id for item in self.evidence]
        if len(ids) != len(set(ids)):
            raise ValueError("evidence IDs must be unique")
        if any(item.symbol != self.symbol for item in self.evidence):
            raise ValueError("evidence symbol does not match candidate")
        if any(item.available_to_strategy_at > self.decision_at for item in self.evidence):
            raise ValueError("evidence was unavailable at the decision time")
        return self


class SpecialistContribution(FrozenModel):
    candidate_id: str
    symbol: str
    company_name: str
    role: LiveCouncilRole
    as_of: datetime
    score: int = Field(ge=0, le=100)
    confidence: int = Field(ge=0, le=100)
    summary: str = Field(min_length=1, max_length=1200)
    supporting_points: tuple[str, ...] = Field(max_length=5)
    concerns: tuple[str, ...] = Field(max_length=5)
    evidence_ids: tuple[str, ...]
    hard_blockers: tuple[str, ...] = Field(default=(), max_length=3)


class CandidateSynthesis(FrozenModel):
    candidate_id: str
    symbol: str
    company_name: str
    as_of: datetime
    investment_score: int = Field(ge=0, le=100)
    confidence: int = Field(ge=0, le=100)
    thesis: str = Field(min_length=1, max_length=1500)
    agreement: str = Field(min_length=1, max_length=800)
    disagreement: str = Field(min_length=1, max_length=800)
    risks: tuple[str, ...] = Field(max_length=6)
    evidence_ids: tuple[str, ...]
    hard_blockers: tuple[str, ...] = Field(default=(), max_length=3)


class CandidateCouncilRecord(FrozenModel):
    packet: LiveCandidatePacket
    contributions: tuple[SpecialistContribution, ...]
    synthesis: CandidateSynthesis
    deterministic_score: float = Field(ge=0, le=100)
    adjusted_score: float = Field(ge=0, le=100)
    specialist_score_dispersion: float = Field(default=0, ge=0, le=50)
    specialist_score_range: int = Field(default=0, ge=0, le=100)
    eligible: bool
    rejection_reasons: tuple[str, ...]

    @property
    def selection_score(self) -> float:
        """The single score used for reporting, ranking, and eligibility."""
        return self.adjusted_score


class LiveCouncilDecision(FrozenModel):
    decision_id: str
    decision_at: datetime
    model: str
    prompt_version: str
    maximum_selections: int = Field(ge=1, le=20)
    minimum_selections: int = Field(default=0, ge=0, le=20)
    minimum_score: float = Field(ge=0, le=100)
    minimum_specialist_score: int = Field(default=55, ge=0, le=100)
    minimum_specialists: int = Field(default=3, ge=1, le=4)
    candidates: tuple[CandidateCouncilRecord, ...]
    selected_symbols: tuple[str, ...]
    selection_basis: dict[str, str] = Field(default_factory=dict)
    portfolio_diagnostics: dict[str, object] = Field(default_factory=dict)

    def audit_payload(self) -> dict[str, object]:
        """Dashboard-ready structured data plus a report generated from that same record."""
        return {
            "schema_version": "live-council-decision-v2",
            "decision": self.model_dump(mode="json"),
            "human_report_markdown": self.human_report(),
        }

    def human_report(self) -> str:
        lines = [
            f"# Live council decision — {self.decision_at.isoformat()}", "",
            (
                f"Selected **{len(self.selected_symbols)} of {self.maximum_selections}**: "
                f"{', '.join(self.selected_symbols) if self.selected_symbols else 'none'}"
            ), "",
        ]
        if self.portfolio_diagnostics:
            average = self.portfolio_diagnostics.get("average_pairwise_correlation")
            maximum = self.portfolio_diagnostics.get("maximum_pairwise_correlation")
            lines.extend([
                "## Portfolio construction", "",
                f"Method: {self.portfolio_diagnostics.get('method', 'QUALITY_RANK_ONLY')}", "",
                (
                    "Selected pairwise correlation: "
                    f"average {_format_metric(average)}, maximum {_format_metric(maximum)}"
                ), "",
                (
                    "Sector counts: "
                    f"{self.portfolio_diagnostics.get('selected_sector_counts', {})}"
                ), "",
            ])
        for rank, record in enumerate(self.candidates, 1):
            packet, synthesis = record.packet, record.synthesis
            status = "SELECTED" if packet.symbol in self.selected_symbols else "NOT SELECTED"
            basis = self.selection_basis.get(packet.symbol)
            lines.extend([
                f"## {rank}. {packet.symbol} — {packet.company_name}", "",
                (
                    f"**{status} · score {record.adjusted_score:.2f}/100 · "
                    f"confidence {synthesis.confidence}/100**"
                ), "", synthesis.thesis, "",
                *([f"Selection basis: {basis}", ""] if basis else []),
                "### Agent contributions", "",
            ])
            for item in record.contributions:
                lines.extend([
                    (
                        f"#### {item.role.value.replace('_', ' ').title()} — "
                        f"{item.score}/100 ({item.confidence}% confidence)"
                    ), "",
                    item.summary, "",
                ])
                if item.supporting_points:
                    lines.extend(["Supporting evidence:", *[f"- {point}" for point in item.supporting_points], ""])
                if item.concerns:
                    lines.extend(["Concerns:", *[f"- {point}" for point in item.concerns], ""])
                lines.extend([f"Evidence: {', '.join(item.evidence_ids) or 'none'}", ""])
            lines.extend([
                "### Synthesis", "", f"Agreement: {synthesis.agreement}", "",
                f"Disagreement: {synthesis.disagreement}", "",
            ])
            if record.rejection_reasons:
                lines.extend([f"Decision constraints: {', '.join(record.rejection_reasons)}", ""])
        return "\n".join(lines).rstrip() + "\n"


SpecialistRunner = Callable[[LiveCouncilRole, LiveCandidatePacket], SpecialistContribution]
SynthesisRunner = Callable[
    [LiveCandidatePacket, tuple[SpecialistContribution, ...]], CandidateSynthesis
]


class LiveCouncil:
    """Five-agent, one-candidate-per-call council with deterministic ranking."""

    def __init__(
        self,
        specialist: SpecialistRunner,
        synthesizer: SynthesisRunner,
        *,
        model: str,
        prompt_version: str = "live-council-v3-quality-floor",
        maximum_selections: int = 20,
        minimum_selections: int = 0,
        minimum_score: float = 70,
        minimum_specialist_score: int = 55,
        minimum_specialists: int = 3,
        workers: int = 4,
        role_weights: Mapping[LiveCouncilRole, float] = ROLE_WEIGHTS,
    ) -> None:
        if set(role_weights) != set(LiveCouncilRole) or abs(sum(role_weights.values()) - 1) > 1e-9:
            raise ValueError("role weights must cover all specialist roles and sum to one")
        if not 1 <= maximum_selections <= 20:
            raise ValueError("maximum selections must be between one and twenty")
        if not 0 <= minimum_selections <= maximum_selections:
            raise ValueError("minimum selections must be between zero and the maximum")
        if workers < 1:
            raise ValueError("workers must be positive")
        if not 1 <= minimum_specialists <= 4:
            raise ValueError("minimum specialists must be between one and four")
        self.specialist, self.synthesizer = specialist, synthesizer
        self.model, self.prompt_version = model, prompt_version
        self.maximum_selections, self.minimum_selections = maximum_selections, minimum_selections
        self.minimum_score = minimum_score
        self.minimum_specialist_score = minimum_specialist_score
        self.minimum_specialists = minimum_specialists
        self.workers = workers
        self.role_weights = dict(role_weights)

    def decide(
        self,
        decision_id: str,
        packets: Sequence[LiveCandidatePacket],
        correlation_estimate: CorrelationEstimate | None = None,
    ) -> LiveCouncilDecision:
        if not packets:
            raise ValueError("at least one candidate is required")
        if len({item.candidate_id for item in packets}) != len(packets):
            raise ValueError("candidate IDs must be unique")
        decision_at = packets[0].decision_at
        if any(item.decision_at != decision_at for item in packets):
            raise ValueError("all candidates must share one decision time")
        with ThreadPoolExecutor(max_workers=min(self.workers, len(packets))) as pool:
            records = list(pool.map(self._evaluate, packets))
        records.sort(key=lambda item: (-item.adjusted_score, item.packet.kronos_rank, item.packet.symbol))
        qualified_records = self._portfolio_order(
            [item for item in records if item.eligible], correlation_estimate,
        )
        selected_records = qualified_records[: self.maximum_selections]
        selected = [item.packet.symbol for item in selected_records]
        selection_basis = {symbol: "PASSED_QUALITY_GATE" for symbol in selected}
        if len(selected) < self.minimum_selections:
            safe_top_ups = [
                item for item in records
                if item.packet.symbol not in selected and not self._has_hard_blocker(item)
            ]
            for item in self._portfolio_order(
                safe_top_ups, correlation_estimate, selected_records,
            ):
                selected.append(item.packet.symbol)
                selected_records.append(item)
                selection_basis[item.packet.symbol] = "MINIMUM_DIVERSIFICATION_TOP_UP"
                if len(selected) == self.minimum_selections:
                    break
        diagnostics = self._portfolio_diagnostics(
            records, selected_records, correlation_estimate,
        )
        return LiveCouncilDecision(
            decision_id=decision_id, decision_at=decision_at, model=self.model,
            prompt_version=self.prompt_version, maximum_selections=self.maximum_selections,
            minimum_selections=self.minimum_selections,
            minimum_score=self.minimum_score,
            minimum_specialist_score=self.minimum_specialist_score,
            minimum_specialists=self.minimum_specialists,
            candidates=tuple(records), selected_symbols=tuple(selected),
            selection_basis=selection_basis, portfolio_diagnostics=diagnostics,
        )

    @staticmethod
    def _portfolio_order(
        records: Sequence[CandidateCouncilRecord],
        estimate: CorrelationEstimate | None,
        initial: Sequence[CandidateCouncilRecord] = (),
    ) -> list[CandidateCouncilRecord]:
        """Preserve quality buckets; use diversification only inside whole-point near-ties."""
        if estimate is None:
            return list(records)
        buckets: dict[int, list[CandidateCouncilRecord]] = {}
        for item in records:
            bucket = math.floor(item.adjusted_score + .5)
            buckets.setdefault(bucket, []).append(item)
        chosen = list(initial)
        output: list[CandidateCouncilRecord] = []
        sector_counts = Counter(item.packet.sector or "Unknown" for item in chosen)
        for bucket in sorted(buckets, reverse=True):
            remaining = list(buckets[bucket])
            while remaining:
                def key(item: CandidateCouncilRecord) -> tuple[float, int, float, int, str]:
                    correlations = [
                        estimate.between(item.packet.symbol, held.packet.symbol)
                        for held in chosen
                    ]
                    available = [value for value in correlations if value is not None]
                    marginal = sum(available) / len(available) if available else 0.0
                    sector = item.packet.sector or "Unknown"
                    return (
                        marginal, sector_counts[sector], -item.adjusted_score,
                        item.packet.kronos_rank, item.packet.symbol,
                    )

                winner = min(remaining, key=key)
                remaining.remove(winner)
                output.append(winner)
                chosen.append(winner)
                sector_counts[winner.packet.sector or "Unknown"] += 1
        return output

    @staticmethod
    def _portfolio_diagnostics(
        records: Sequence[CandidateCouncilRecord],
        selected_records: Sequence[CandidateCouncilRecord],
        estimate: CorrelationEstimate | None,
    ) -> dict[str, object]:
        selected = tuple(item.packet.symbol for item in selected_records)
        average, maximum = pairwise_summary(selected, estimate)
        quality_ranks = {item.packet.symbol: index for index, item in enumerate(records, 1)}
        portfolio_ranks = {
            item.packet.symbol: index for index, item in enumerate(selected_records, 1)
        }
        sector_counts = Counter(item.packet.sector or "Unknown" for item in selected_records)
        candidate_metrics = {}
        for item in records:
            peers = tuple(symbol for symbol in selected if symbol != item.packet.symbol)
            values = [
                estimate.between(item.packet.symbol, peer) for peer in peers
            ] if estimate else []
            present = [value for value in values if value is not None]
            candidate_metrics[item.packet.symbol] = {
                "quality_rank": quality_ranks[item.packet.symbol],
                "portfolio_rank": portfolio_ranks.get(item.packet.symbol),
                "rounded_quality_bucket": math.floor(item.adjusted_score + .5),
                "selected": item.packet.symbol in portfolio_ranks,
                "average_correlation_to_selected": (
                    round(sum(present) / len(present), 6) if present else None
                ),
                "maximum_correlation_to_selected": (
                    round(max(present), 6) if present else None
                ),
            }
        quality_order = sorted(selected, key=quality_ranks.__getitem__)
        return {
            "method": (
                "QUALITY_BUCKET_THEN_SHRUNK_CORRELATION"
                if estimate else "QUALITY_RANK_ONLY_CORRELATION_UNAVAILABLE"
            ),
            "quality_score_controls_eligibility": True,
            "correlation_can_reject_candidate": False,
            "near_tie_definition": "same nearest-whole-point adjusted-score bucket",
            "correlation_estimator": estimate.method if estimate else None,
            "correlation_observations": estimate.observations if estimate else 0,
            "average_pairwise_correlation": round(average, 6) if average is not None else None,
            "maximum_pairwise_correlation": round(maximum, 6) if maximum is not None else None,
            "selected_sector_counts": dict(sorted(sector_counts.items())),
            "near_tie_reordered": list(selected) != quality_order,
            "candidate_metrics": candidate_metrics,
        }

    @staticmethod
    def _has_hard_blocker(record: CandidateCouncilRecord) -> bool:
        return bool(
            record.synthesis.hard_blockers
            or any(item.hard_blockers for item in record.contributions)
        )

    def _evaluate(self, packet: LiveCandidatePacket) -> CandidateCouncilRecord:
        contributions = tuple(self.specialist(role, packet) for role in LiveCouncilRole)
        allowed_evidence = {item.id for item in packet.evidence}
        for role, item in zip(LiveCouncilRole, contributions, strict=True):
            if (
                item.role != role or item.candidate_id != packet.candidate_id
                or item.symbol != packet.symbol or item.company_name != packet.company_name
                or item.as_of != packet.decision_at
            ):
                raise ValueError("specialist response identity mismatch")
            if not set(item.evidence_ids) <= allowed_evidence:
                raise ValueError("specialist cited unknown evidence")
        synthesis = self.synthesizer(packet, contributions)
        if (
            synthesis.candidate_id != packet.candidate_id or synthesis.symbol != packet.symbol
            or synthesis.company_name != packet.company_name or synthesis.as_of != packet.decision_at
        ):
            raise ValueError("synthesis response identity mismatch")
        if not set(synthesis.evidence_ids) <= allowed_evidence:
            raise ValueError("synthesis cited unknown evidence")
        specialist_score = sum(
            self.role_weights[item.role] * item.score for item in contributions
        )
        specialist_confidence = sum(
            self.role_weights[item.role] * item.confidence for item in contributions
        )
        # One documented score drives the report, ordering, threshold, and allocation eligibility.
        # The synthesis is deliberately bounded so it can reconcile specialists without overriding
        # their evidence-backed consensus.
        deterministic_score = 0.70 * specialist_score + 0.30 * synthesis.investment_score
        confidence = (0.70 * specialist_confidence + 0.30 * synthesis.confidence) / 100
        adjusted_score = deterministic_score * (0.80 + 0.20 * confidence)
        blockers = (
            blocker for item in contributions for blocker in item.hard_blockers
        )
        reasons = tuple(dict.fromkeys((*blockers, *synthesis.hard_blockers)))
        if adjusted_score < self.minimum_score:
            reasons += ("BELOW_MINIMUM_SCORE",)
        supporting_specialists = sum(
            item.score >= self.minimum_specialist_score for item in contributions
        )
        if supporting_specialists < self.minimum_specialists:
            reasons += ("INSUFFICIENT_SPECIALIST_SUPPORT",)
        return CandidateCouncilRecord(
            packet=packet, contributions=contributions, synthesis=synthesis,
            deterministic_score=deterministic_score, adjusted_score=adjusted_score,
            specialist_score_dispersion=pstdev(item.score for item in contributions),
            specialist_score_range=(
                max(item.score for item in contributions)
                - min(item.score for item in contributions)
            ),
            eligible=not reasons, rejection_reasons=reasons,
        )


def _format_metric(value: object) -> str:
    return "unavailable" if value is None else f"{float(value):.3f}"
