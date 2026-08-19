from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from .dataset import TrainingRow
from .model_evaluation import WalkForwardResult, evaluate_walk_forward
from .models import LinearModelArtifact, train_ridge
from .validation import TimeFold


@dataclass(frozen=True)
class SelectionResult:
    selected_alpha: float
    development_score: float
    artifact: LinearModelArtifact
    candidates: dict[float, float]


def select_ridge_alpha(
    rows: tuple[TrainingRow, ...], folds: tuple[TimeFold, ...], candidate_alphas: tuple[float, ...],
    development_end: date, model_version: str,
) -> SelectionResult:
    if not candidate_alphas or any(alpha < 0 for alpha in candidate_alphas):
        raise ValueError("candidate alphas must be non-negative")
    if any(row.decision_date > development_end or row.label_end_date > development_end for row in rows):
        raise ValueError("selection rows must be fully contained in development data")
    if any(date_value > development_end for fold in folds for date_value in (*fold.train, *fold.validate)):
        raise ValueError("selection folds cannot access validation or holdout dates")
    evaluations: dict[float, WalkForwardResult] = {
        alpha: evaluate_walk_forward(rows, folds, alpha, f"{model_version}-alpha-{alpha:g}")
        for alpha in sorted(set(candidate_alphas))
    }
    # Prefer stronger date-level rank performance; smaller alpha wins exact ties.
    selected = max(evaluations, key=lambda alpha: (evaluations[alpha].mean_rank_correlation, -alpha))
    artifact = train_ridge(rows, selected, model_version)
    return SelectionResult(
        selected, evaluations[selected].mean_rank_correlation, artifact,
        {alpha: result.mean_rank_correlation for alpha, result in evaluations.items()},
    )

