from portfoliopilot.bounded_agent_strategy import bounded_agent_adjustment
from portfoliopilot.bynara_agents import AgentFeature


def feature(role: str, signal: int, confidence: float = 1) -> AgentFeature:
    return AgentFeature(
        role=role, signal=signal, confidence=confidence, reason="Measured feature assessment",
        risk_flags=(),
    )


def test_agent_adjustment_is_bounded_and_directional() -> None:
    positive = {
        role: feature(role, 2) for role in ("TECHNICAL", "QUALITY", "RISK")
    }
    negative = {
        role: feature(role, -2) for role in ("TECHNICAL", "QUALITY", "RISK")
    }
    assert bounded_agent_adjustment(positive) == .08
    assert bounded_agent_adjustment(negative) == -.08


def test_low_confidence_shrinks_agent_influence() -> None:
    features = {
        role: feature(role, 2, 0) for role in ("TECHNICAL", "QUALITY", "RISK")
    }
    assert bounded_agent_adjustment(features) == 0
