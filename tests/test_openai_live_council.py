from datetime import UTC, datetime

from portfoliopilot.contracts import Evidence, Quality
from portfoliopilot.live_council import LiveCandidatePacket, LiveCouncilRole
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
            "momentum_21d": .08,
            "kronos_signal": {"status": "UNRELIABLE", "cross_sectional_percentile": .9},
        },
    )
    market = OpenAILiveCouncilAgents._packet_payload(item, LiveCouncilRole.MARKET)
    risk = OpenAILiveCouncilAgents._packet_payload(item, LiveCouncilRole.RISK)
    synthesis = OpenAILiveCouncilAgents._packet_payload(item, None)
    assert "kronos_signal" in market["quantitative_features"]
    assert "kronos_signal" not in risk["quantitative_features"]
    assert "kronos_signal" not in synthesis["quantitative_features"]
