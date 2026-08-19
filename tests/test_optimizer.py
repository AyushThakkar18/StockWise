import pytest

from portfoliopilot.optimizer import (
    CandidateScore,
    PortfolioConstraints,
    construct_target_weights,
)


def candidate(symbol: str, sector: str, alpha: float, **updates) -> CandidateScore:
    values = {
        "symbol": symbol, "sector": sector, "expected_excess_return": alpha,
        "uncertainty": 0.05, "evidence_completeness": 0.9,
        "average_dollar_volume": 10_000_000, "approved": True,
    }
    values.update(updates)
    return CandidateScore(**values)


def test_allocation_respects_position_sector_and_turnover_limits() -> None:
    candidates = (
        candidate("A", "TECH", 0.08), candidate("B", "TECH", 0.06),
        candidate("C", "HEALTH", 0.05), candidate("D", "HEALTH", 0.04),
    )
    constraints = PortfolioConstraints(
        maximum_position_weight=0.10, maximum_sector_weight=0.15,
        maximum_turnover=0.25, maximum_invested_weight=0.50,
    )
    result = construct_target_weights(candidates, {}, constraints)
    assert max(result.target_weights.values()) <= 0.10
    assert result.target_weights["A"] + result.target_weights["B"] <= 0.15
    assert result.turnover <= 0.25
    assert sum(result.target_weights.values()) + result.cash_weight == pytest.approx(1.0)


def test_failed_candidate_gates_produce_abstention_trace() -> None:
    weak = candidate("A", "TECH", 0.0, evidence_completeness=0.2)
    result = construct_target_weights((weak,), {}, PortfolioConstraints())
    assert result.target_weights == {}
    assert result.cash_weight == 1
    assert {check.name for check in result.checks["A"] if not check.passed} == {
        "expected_excess_return", "evidence_completeness"
    }


def test_invalid_current_portfolio_is_rejected() -> None:
    with pytest.raises(ValueError, match="position weights"):
        construct_target_weights(
            (candidate("A", "TECH", 0.1),), {"A": 0.5}, PortfolioConstraints()
        )

