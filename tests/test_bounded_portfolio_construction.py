from decimal import Decimal

from portfoliopilot.bounded_agent_strategy import BoundedMultiAgentStrategy


def test_strategy_defaults_to_twenty_equal_weight_positions() -> None:
    strategy = BoundedMultiAgentStrategy(object(), object(), ())
    assert strategy.position_count == 20
    assert Decimal(1) / Decimal(strategy.position_count) == Decimal("0.05")
