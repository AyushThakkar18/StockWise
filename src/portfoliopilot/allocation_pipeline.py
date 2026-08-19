from __future__ import annotations

from dataclasses import dataclass

from .features import FeatureSnapshot
from .models import LinearModelArtifact
from .optimizer import (
    AllocationResult,
    CandidateScore,
    PortfolioConstraints,
    construct_target_weights,
)


@dataclass(frozen=True)
class CandidateContext:
    snapshot: FeatureSnapshot
    sector: str
    evidence_completeness: float
    average_dollar_volume: float
    approved: bool = True


def allocate_from_model(
    artifact: LinearModelArtifact,
    candidates: tuple[CandidateContext, ...],
    residual_uncertainty: float,
    current_weights: dict[str, float],
    constraints: PortfolioConstraints,
) -> AllocationResult:
    if residual_uncertainty < 0:
        raise ValueError("residual uncertainty cannot be negative")
    scores = []
    for candidate in candidates:
        if candidate.snapshot.version != artifact.feature_version:
            raise ValueError("feature/model version mismatch")
        scores.append(CandidateScore(
            symbol=candidate.snapshot.symbol, sector=candidate.sector,
            expected_excess_return=artifact.predict({
                name: _required(candidate.snapshot, name) for name in artifact.feature_names
            }),
            uncertainty=residual_uncertainty,
            evidence_completeness=candidate.evidence_completeness,
            average_dollar_volume=candidate.average_dollar_volume,
            approved=candidate.approved,
        ))
    return construct_target_weights(tuple(scores), current_weights, constraints)


def _required(snapshot: FeatureSnapshot, name: str) -> float:
    value = snapshot.values.get(name)
    if value is None:
        raise ValueError(f"feature unavailable: {name}")
    return value

