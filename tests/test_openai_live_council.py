from datetime import UTC, datetime

from portfoliopilot.contracts import Evidence, Quality
from portfoliopilot.live_council import (
    CandidateSynthesis,
    LiveCandidatePacket,
    LiveCouncilRole,
    SpecialistContribution,
)
from portfoliopilot.openai_live_council import OpenAILiveCouncilAgents


def test_kronos_signal_is_visible_only_to_market_agent() -> None:
    now = datetime(2026, 8, 28, 20, tzinfo=UTC)
    evidence = Evidence(
        id="e1", symbol="AAPL", claim="Filed result", source="SEC", observed_at=now,
        published_at=now, available_to_strategy_at=now, retrieved_at=now,
        vintage="test", quality=Quality.PASS,
    )
    item = LiveCandidatePacket(
        candidate_id="c1", symbol="AAPL", company_name="Apple Inc.", decision_at=now,
        deterministic_rank=1, kronos_rank=1, evidence=(evidence,),
        quantitative_features={
            "deterministic_factors": {"momentum_12_1": .08, "growth": .12},
            "factor_percentiles": {"momentum_12_1": .7, "growth": .6},
            "price_features": {"return_latest_21d": .08},
            "market_regime": {"spy_return_21d": .02},
            "fundamental_completeness": 1.0,
            "signal_provenance": {"PRICE_MOMENTUM": ["price_features"]},
            "kronos_signal": {"status": "UNRELIABLE", "cross_sectional_percentile": .9},
        },
    )
    market = OpenAILiveCouncilAgents._packet_payload(item, LiveCouncilRole.MARKET)
    risk = OpenAILiveCouncilAgents._packet_payload(item, LiveCouncilRole.RISK)
    business = OpenAILiveCouncilAgents._packet_payload(item, LiveCouncilRole.BUSINESS)
    catalyst = OpenAILiveCouncilAgents._packet_payload(item, LiveCouncilRole.CATALYST)
    synthesis = OpenAILiveCouncilAgents._packet_payload(item, None)
    assert "kronos_signal" in market["quantitative_features"]
    assert "kronos_signal" not in risk["quantitative_features"]
    assert "kronos_signal" not in synthesis["quantitative_features"]
    assert "growth" not in market["quantitative_features"]["price_factors"]
    assert "price_features" not in business["quantitative_features"]
    assert business["quantitative_features"]["fundamental_factors"]["growth"] == .12
    assert "price_features" not in catalyst["quantitative_features"]
    assert "price_features" not in synthesis["quantitative_features"]


def test_orchestrator_normalizes_model_identity_fields() -> None:
    now = datetime(2026, 8, 28, 20, tzinfo=UTC)
    item = LiveCandidatePacket(
        candidate_id="c1", symbol="AAPL", company_name="Apple Inc.", decision_at=now,
        deterministic_rank=1, kronos_rank=1, evidence=(), quantitative_features={},
    )
    wrong_time = datetime(2026, 8, 28, tzinfo=UTC)
    contribution = SpecialistContribution(
        candidate_id="wrong", symbol="MSFT", company_name="Wrong", role=LiveCouncilRole.RISK,
        as_of=wrong_time, score=70, confidence=70, summary="Valid analysis",
        supporting_points=(), concerns=(), evidence_ids=(),
    )
    synthesis = CandidateSynthesis(
        candidate_id="wrong", symbol="MSFT", company_name="Wrong", as_of=wrong_time,
        investment_score=70, confidence=70, thesis="Valid synthesis", agreement="Agreement",
        disagreement="Disagreement", risks=(), evidence_ids=(),
    )
    agent = object.__new__(OpenAILiveCouncilAgents)
    responses = iter((contribution, synthesis))
    agent._cached_parse = lambda *args: next(responses)
    agent._fingerprint = lambda *args: "test"

    normalized = agent.specialist(LiveCouncilRole.MARKET, item)
    final = agent.synthesize(item, (normalized,))

    assert (normalized.candidate_id, normalized.symbol, normalized.company_name, normalized.as_of) == (
        "c1", "AAPL", "Apple Inc.", now,
    )
    assert normalized.role == LiveCouncilRole.MARKET
    assert (final.candidate_id, final.symbol, final.company_name, final.as_of) == (
        "c1", "AAPL", "Apple Inc.", now,
    )
