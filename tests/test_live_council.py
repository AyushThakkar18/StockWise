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
from portfoliopilot.portfolio_construction import CorrelationEstimate

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
    assert payload["schema_version"] == "live-council-decision-v2"
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


def test_minimum_portfolio_tops_up_by_rank_but_never_overrides_hard_blockers() -> None:
    packets = [packet(f"S{index}", f"candidate-{index}") for index in range(12)]

    def scored(role: LiveCouncilRole, item: LiveCandidatePacket) -> SpecialistContribution:
        result = specialist(role, item)
        index = int(item.symbol[1:])
        blockers = ("TRADING_HALTED",) if index == 0 and role == LiveCouncilRole.RISK else ()
        return result.model_copy(update={"score": 68 - index, "hard_blockers": blockers})

    decision = LiveCouncil(
        scored, synthesizer, model="gpt-4o-mini", minimum_score=70,
        minimum_selections=10,
    ).decide("decision-1", packets)

    assert len(decision.selected_symbols) == 10
    assert "S0" not in decision.selected_symbols
    assert decision.selected_symbols == tuple(f"S{index}" for index in range(1, 11))
    assert set(decision.selection_basis.values()) == {"MINIMUM_DIVERSIFICATION_TOP_UP"}
    assert "Selection basis: MINIMUM_DIVERSIFICATION_TOP_UP" in decision.human_report()


def test_minimum_portfolio_uses_every_safe_candidate_without_aborting() -> None:
    packets = [packet(f"S{index}", f"candidate-{index}") for index in range(10)]

    def blocked(role: LiveCouncilRole, item: LiveCandidatePacket) -> SpecialistContribution:
        result = specialist(role, item)
        if item.symbol == "S0" and role == LiveCouncilRole.RISK:
            return result.model_copy(update={"hard_blockers": ("TRADING_HALTED",)})
        return result

    decision = LiveCouncil(
        blocked, synthesizer, model="gpt-4o-mini", minimum_selections=10,
    ).decide("decision-1", packets)

    assert len(decision.selected_symbols) == 9
    assert "S0" not in decision.selected_symbols
    assert set(decision.selection_basis.values()) == {"MINIMUM_DIVERSIFICATION_TOP_UP"}


def test_shrunk_correlation_only_reorders_candidates_inside_quality_tie_bucket() -> None:
    packets = [packet(symbol, f"candidate-{symbol}") for symbol in ("A", "B", "C")]

    def scored(role: LiveCouncilRole, item: LiveCandidatePacket) -> SpecialistContribution:
        score = {"A": 80, "B": 79, "C": 79}[item.symbol]
        return specialist(role, item).model_copy(update={"score": score, "confidence": 100})

    def synthesized(
        item: LiveCandidatePacket, contributions: tuple[SpecialistContribution, ...],
    ) -> CandidateSynthesis:
        score = {"A": 80, "B": 79, "C": 79}[item.symbol]
        return synthesizer(item, contributions).model_copy(
            update={"investment_score": score, "confidence": 100},
        )

    estimate = CorrelationEstimate(
        {("A", "B"): .95, ("A", "C"): .10, ("B", "C"): .10}, 126,
    )
    decision = LiveCouncil(
        scored, synthesized, model="gpt-4o-mini", minimum_score=55,
        maximum_selections=2,
    ).decide("decision-1", packets, estimate)

    assert decision.selected_symbols == ("A", "C")
    assert all(decision.candidates[index].eligible for index in range(3))
    assert decision.portfolio_diagnostics["correlation_can_reject_candidate"] is False
    assert decision.portfolio_diagnostics["method"] == (
        "QUALITY_BUCKET_THEN_SHRUNK_CORRELATION"
    )
    assert decision.portfolio_diagnostics["candidate_metrics"]["B"]["quality_rank"] == 2
