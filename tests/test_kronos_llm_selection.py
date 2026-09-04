from decimal import Decimal

from portfoliopilot.bynara_agents import AgentFeature
from portfoliopilot.kronos_llm_selection import allocate, council_approval


def feature(role: str, signal: int, confidence: float = 1) -> AgentFeature:
    return AgentFeature(
        role=role, signal=signal, confidence=confidence,
        reason="bounded assessment", risk_flags=(),
    )


def test_council_requires_positive_majority_and_nonnegative_risk() -> None:
    supported = {
        "TECHNICAL": feature("TECHNICAL", 2),
        "QUALITY": feature("QUALITY", 1),
        "RISK": feature("RISK", 1),
    }
    assert council_approval(supported)[0] is True
    supported["RISK"] = feature("RISK", -1)
    approved, reasons = council_approval(supported)
    assert approved is False
    assert "NEGATIVE_RISK_REVIEW" in reasons


def test_equal_policy_caps_concentration_and_leaves_cash() -> None:
    assert allocate(["A", "B"], "equal_capped_cash") == {
        "A": Decimal("0.10"), "B": Decimal("0.10"),
    }
    targets = allocate([str(index) for index in range(20)], "equal_capped_cash")
    assert sum(targets.values()) == Decimal(1)


def test_five_percent_slots_route_unused_weight_to_spy() -> None:
    targets = allocate(["A", "B"], "five_percent_slots_spy")
    assert targets == {"SPY": Decimal("0.90"), "A": Decimal("0.05"), "B": Decimal("0.05")}


def test_half_spy_policy_preserves_core_and_unused_slots() -> None:
    targets = allocate(["A", "B"], "half_spy_two_point_five_percent_slots")
    assert targets == {
        "SPY": Decimal("0.950"), "A": Decimal("0.025"), "B": Decimal("0.025"),
    }
