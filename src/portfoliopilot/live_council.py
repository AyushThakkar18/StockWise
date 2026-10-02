from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from enum import StrEnum
from statistics import pstdev

from pydantic import Field, model_validator

from .contracts import Evidence, FrozenModel


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

    def audit_payload(self) -> dict[str, object]:
        """Dashboard-ready structured data plus a report generated from that same record."""
        return {
            "schema_version": "live-council-decision-v1",
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

    def decide(self, decision_id: str, packets: Sequence[LiveCandidatePacket]) -> LiveCouncilDecision:
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
        qualified = tuple(
            item.packet.symbol for item in records if item.eligible
        )[: self.maximum_selections]
        selected = list(qualified)
        selection_basis = {symbol: "PASSED_QUALITY_GATE" for symbol in selected}
        if len(selected) < self.minimum_selections:
            for item in records:
                if item.packet.symbol in selected or self._has_hard_blocker(item):
                    continue
                selected.append(item.packet.symbol)
                selection_basis[item.packet.symbol] = "MINIMUM_DIVERSIFICATION_TOP_UP"
                if len(selected) == self.minimum_selections:
                    break
        return LiveCouncilDecision(
            decision_id=decision_id, decision_at=decision_at, model=self.model,
            prompt_version=self.prompt_version, maximum_selections=self.maximum_selections,
            minimum_selections=self.minimum_selections,
            minimum_score=self.minimum_score,
            minimum_specialist_score=self.minimum_specialist_score,
            minimum_specialists=self.minimum_specialists,
            candidates=tuple(records), selected_symbols=tuple(selected),
            selection_basis=selection_basis,
        )

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
