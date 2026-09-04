from datetime import UTC, datetime

import pytest

from portfoliopilot.contracts import Evidence, Quality
from portfoliopilot.live_council import (
    CandidateSynthesis,
    LiveCandidatePacket,
    LiveCouncil,
    LiveCouncilRole,
    SpecialistContribution,
)

NOW = datetime(2026, 8, 28, 20, tzinfo=UTC)


def packet(symbol: str = "AAPL", candidate_id: str = "candidate-1") -> LiveCandidatePacket:
    evidence = Evidence(
        id=f"{candidate_id}-e1", symbol=symbol, claim="Revenue increased year over year.",
        source="SEC", observed_at=NOW, published_at=NOW, available_to_strategy_at=NOW,
        retrieved_at=NOW, vintage="2026-Q2", quality=Quality.PASS,
    )
    return LiveCandidatePacket(
        candidate_id=candidate_id, symbol=symbol, company_name="Apple Inc.",
        sector="Technology", decision_at=NOW, deterministic_rank=2, kronos_rank=1,
        quantitative_features={"momentum_21d": 0.08}, evidence=(evidence,),
    )


def specialist(role: LiveCouncilRole, item: LiveCandidatePacket) -> SpecialistContribution:
    scores = {
        LiveCouncilRole.MARKET: 80,
        LiveCouncilRole.BUSINESS: 70,
        LiveCouncilRole.CATALYST: 60,
        LiveCouncilRole.RISK: 50,
    }
    return SpecialistContribution(
        candidate_id=item.candidate_id, symbol=item.symbol, company_name=item.company_name,
        role=role, as_of=item.decision_at, score=scores[role], confidence=80,
        summary=f"{role.value} contribution", supporting_points=("Positive evidence",),
        concerns=("One bounded concern",), evidence_ids=(item.evidence[0].id,),
    )


def synthesizer(
    item: LiveCandidatePacket, contributions: tuple[SpecialistContribution, ...],
) -> CandidateSynthesis:
    assert len(contributions) == 4
    return CandidateSynthesis(
        candidate_id=item.candidate_id, symbol=item.symbol, company_name=item.company_name,
        as_of=item.decision_at, investment_score=66, confidence=78,
        thesis="The evidence is favorable enough for ranking.",
        agreement="The agents agree that the setup is investable.",
        disagreement="They disagree about the strength of near-term catalysts.",
        risks=("Market reversal",), evidence_ids=(item.evidence[0].id,),
    )


def test_live_council_preserves_every_agent_contribution_and_ranks() -> None:
    decision = LiveCouncil(
        specialist, synthesizer, model="gpt-4o-mini", minimum_score=55,
    ).decide("decision-1", [packet()])

    assert decision.selected_symbols == ("AAPL",)
    assert len(decision.candidates[0].contributions) == 4
    assert decision.candidates[0].deterministic_score == pytest.approx(66.7)
    assert decision.candidates[0].adjusted_score == pytest.approx(63.95196)
    assert decision.candidates[0].specialist_score_range == 30
    report = decision.human_report()
    assert "AAPL — Apple Inc." in report
    assert "Market Technical — 80/100" in report
    assert "Agreement:" in report
    payload = decision.audit_payload()
    assert payload["schema_version"] == "live-council-decision-v1"
    assert payload["decision"]["selected_symbols"] == ["AAPL"]
    assert payload["human_report_markdown"] == report


def test_unknown_evidence_citation_fails_closed() -> None:
    def invalid(role: LiveCouncilRole, item: LiveCandidatePacket) -> SpecialistContribution:
        result = specialist(role, item)
        return result.model_copy(update={"evidence_ids": ("invented",)})

    with pytest.raises(ValueError, match="unknown evidence"):
        LiveCouncil(invalid, synthesizer, model="gpt-4o-mini").decide(
            "decision-1", [packet()],
        )


def test_future_evidence_is_rejected() -> None:
    item = packet()
    future = item.evidence[0].model_copy(
        update={"available_to_strategy_at": datetime(2026, 8, 29, tzinfo=UTC)},
    )
    with pytest.raises(ValueError, match="unavailable"):
        item.model_copy(update={"evidence": (future,)}).model_validate(
            item.model_copy(update={"evidence": (future,)}).model_dump(),
        )


def test_hard_blocker_prevents_selection_without_erasing_discussion() -> None:
    def blocked(role: LiveCouncilRole, item: LiveCandidatePacket) -> SpecialistContribution:
        result = specialist(role, item)
        if role == LiveCouncilRole.RISK:
            return result.model_copy(update={"hard_blockers": ("TRADING_HALTED",)})
        return result

    decision = LiveCouncil(blocked, synthesizer, model="gpt-4o-mini").decide(
        "decision-1", [packet()],
    )
    assert not decision.selected_symbols
    assert decision.candidates[0].rejection_reasons == (
        "TRADING_HALTED", "BELOW_MINIMUM_SCORE",
    )
    assert "Risk Sentiment" in decision.human_report()


def test_quality_floor_requires_three_supporting_specialists() -> None:
    def divided(role: LiveCouncilRole, item: LiveCandidatePacket) -> SpecialistContribution:
        result = specialist(role, item)
        score = 90 if role in (LiveCouncilRole.MARKET, LiveCouncilRole.BUSINESS) else 50
        return result.model_copy(update={"score": score, "confidence": 100})

    decision = LiveCouncil(
        divided, synthesizer, model="gpt-4o-mini", minimum_score=55,
    ).decide("decision-1", [packet()])
    assert not decision.selected_symbols
    assert "INSUFFICIENT_SPECIALIST_SUPPORT" in decision.candidates[0].rejection_reasons
