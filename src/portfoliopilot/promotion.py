from __future__ import annotations

from dataclasses import dataclass

from .contracts import PolicyCheck


@dataclass(frozen=True)
class PromotionPolicy:
    minimum_independent_dates: int = 60
    minimum_rank_correlation: float = 0.02
    minimum_information_ratio: float = 0.0
    maximum_drawdown: float = -0.25
    maximum_turnover: float = 12.0


def evaluate_promotion(metrics: dict[str, float], policy: PromotionPolicy) -> tuple[PolicyCheck, ...]:
    checks = (
        ("independent_dates", metrics.get("independent_dates", 0) >= policy.minimum_independent_dates),
        ("rank_correlation", metrics.get("rank_correlation", float("-inf")) >= policy.minimum_rank_correlation),
        ("information_ratio", metrics.get("information_ratio", float("-inf")) >= policy.minimum_information_ratio),
        ("maximum_drawdown", metrics.get("max_drawdown", float("-inf")) >= policy.maximum_drawdown),
        ("turnover", metrics.get("annual_turnover", float("inf")) <= policy.maximum_turnover),
        ("holdout_opened", bool(metrics.get("holdout_opened", 0))),
    )
    return tuple(
        PolicyCheck(name=name, passed=passed, detail="deterministic paper-promotion gate")
        for name, passed in checks
    )
