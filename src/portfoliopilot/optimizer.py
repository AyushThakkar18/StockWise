from __future__ import annotations

from dataclasses import dataclass

from .contracts import PolicyCheck


@dataclass(frozen=True)
class CandidateScore:
    symbol: str
    sector: str
    expected_excess_return: float
    uncertainty: float
    evidence_completeness: float
    average_dollar_volume: float
    approved: bool = True


@dataclass(frozen=True)
class PortfolioConstraints:
    maximum_position_weight: float = 0.10
    maximum_sector_weight: float = 0.25
    maximum_turnover: float = 0.30
    minimum_excess_return: float = 0.01
    maximum_uncertainty: float = 0.15
    minimum_evidence_completeness: float = 0.80
    minimum_average_dollar_volume: float = 5_000_000
    maximum_invested_weight: float = 1.0


@dataclass(frozen=True)
class AllocationResult:
    target_weights: dict[str, float]
    cash_weight: float
    checks: dict[str, tuple[PolicyCheck, ...]]
    turnover: float


def construct_target_weights(
    candidates: tuple[CandidateScore, ...],
    current_weights: dict[str, float],
    constraints: PortfolioConstraints,
) -> AllocationResult:
    _validate_constraints(constraints)
    if len({candidate.symbol for candidate in candidates}) != len(candidates):
        raise ValueError("candidate symbols must be unique")
    candidate_by_symbol = {candidate.symbol: candidate for candidate in candidates}
    if set(current_weights) - set(candidate_by_symbol):
        raise ValueError("current holdings require candidate sector metadata")
    if any(weight < 0 or weight > constraints.maximum_position_weight for weight in current_weights.values()):
        raise ValueError("current position weights violate constraints")
    current_sectors: dict[str, float] = {}
    for symbol, weight in current_weights.items():
        sector = candidate_by_symbol[symbol].sector
        current_sectors[sector] = current_sectors.get(sector, 0.0) + weight
    if sum(current_weights.values()) > 1 or any(
        weight > constraints.maximum_sector_weight for weight in current_sectors.values()
    ):
        raise ValueError("current portfolio is not fully funded or violates sector limits")
    checks = {candidate.symbol: _candidate_checks(candidate, constraints) for candidate in candidates}
    eligible = [candidate for candidate in candidates if all(check.passed for check in checks[candidate.symbol])]
    # Excess return is penalized by forecast dispersion; this is a rank score, not a probability.
    scores = {
        candidate.symbol: candidate.expected_excess_return / max(candidate.uncertainty, 1e-6)
        for candidate in eligible
    }
    desired: dict[str, float] = {symbol: 0.0 for symbol in current_weights}
    sector_weights: dict[str, float] = {}
    remaining = constraints.maximum_invested_weight
    for candidate in sorted(eligible, key=lambda item: (-scores[item.symbol], item.symbol)):
        sector_room = constraints.maximum_sector_weight - sector_weights.get(candidate.sector, 0.0)
        weight = max(0.0, min(constraints.maximum_position_weight, sector_room, remaining))
        if weight <= 0:
            continue
        desired[candidate.symbol] = weight
        sector_weights[candidate.sector] = sector_weights.get(candidate.sector, 0.0) + weight
        remaining -= weight
    all_symbols = set(current_weights) | set(desired)
    raw_turnover = sum(abs(desired.get(symbol, 0.0) - current_weights.get(symbol, 0.0)) for symbol in all_symbols)
    scale = min(1.0, constraints.maximum_turnover / raw_turnover) if raw_turnover else 1.0
    target = {
        symbol: current_weights.get(symbol, 0.0)
        + scale * (desired.get(symbol, 0.0) - current_weights.get(symbol, 0.0))
        for symbol in all_symbols
    }
    target = {symbol: weight for symbol, weight in sorted(target.items()) if weight > 1e-12}
    turnover = sum(abs(target.get(symbol, 0.0) - current_weights.get(symbol, 0.0)) for symbol in all_symbols)
    invested = sum(target.values())
    return AllocationResult(target, max(0.0, 1.0 - invested), checks, turnover)


def _candidate_checks(
    candidate: CandidateScore, constraints: PortfolioConstraints
) -> tuple[PolicyCheck, ...]:
    values = (
        ("approved", candidate.approved),
        ("expected_excess_return", candidate.expected_excess_return >= constraints.minimum_excess_return),
        ("uncertainty", 0 <= candidate.uncertainty <= constraints.maximum_uncertainty),
        ("evidence_completeness", candidate.evidence_completeness >= constraints.minimum_evidence_completeness),
        ("liquidity", candidate.average_dollar_volume >= constraints.minimum_average_dollar_volume),
    )
    return tuple(
        PolicyCheck(name=name, passed=passed, detail="deterministic allocation gate")
        for name, passed in values
    )


def _validate_constraints(constraints: PortfolioConstraints) -> None:
    weights = (
        constraints.maximum_position_weight, constraints.maximum_sector_weight,
        constraints.maximum_turnover, constraints.maximum_invested_weight,
    )
    if any(value < 0 or value > 1 for value in weights):
        raise ValueError("weight and turnover constraints must be between zero and one")
    if constraints.maximum_position_weight > constraints.maximum_sector_weight:
        raise ValueError("position limit cannot exceed sector limit")
