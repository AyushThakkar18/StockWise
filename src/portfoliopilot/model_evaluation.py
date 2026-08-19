from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from statistics import fmean

from .dataset import TrainingRow
from .models import train_ridge
from .validation import TimeFold


@dataclass(frozen=True)
class Prediction:
    symbol: str
    decision_date: date
    predicted_excess: float
    realized_excess: float
    model_version: str


@dataclass(frozen=True)
class WalkForwardResult:
    predictions: tuple[Prediction, ...]
    mean_rank_correlation: float
    independent_dates: int


def evaluate_walk_forward(
    rows: tuple[TrainingRow, ...], folds: tuple[TimeFold, ...], alpha: float, model_prefix: str
) -> WalkForwardResult:
    predictions = []
    for fold_number, fold in enumerate(folds, start=1):
        training = tuple(row for row in rows if row.decision_date in fold.train)
        validation = tuple(row for row in rows if row.decision_date in fold.validate)
        if not training or not validation:
            continue
        if max(row.label_end_date for row in training) >= min(fold.validate):
            raise ValueError("training labels overlap validation decisions")
        version = f"{model_prefix}-fold-{fold_number}"
        model = train_ridge(training, alpha, version)
        predictions.extend(
            Prediction(row.symbol, row.decision_date, model.predict(row.features), row.excess_return, version)
            for row in validation
        )
    if not predictions:
        raise ValueError("walk-forward evaluation produced no predictions")
    dates = sorted({prediction.decision_date for prediction in predictions})
    correlations = []
    for decision_date in dates:
        group = [prediction for prediction in predictions if prediction.decision_date == decision_date]
        correlation = _rank_correlation(group)
        if correlation is not None:
            correlations.append(correlation)
    return WalkForwardResult(
        tuple(predictions), fmean(correlations) if correlations else 0.0, len(dates)
    )


def _rank_correlation(predictions: list[Prediction]) -> float | None:
    if len(predictions) < 2:
        return None
    predicted_order = {item.symbol: rank for rank, item in enumerate(sorted(predictions, key=lambda x: x.predicted_excess))}
    realized_order = {item.symbol: rank for rank, item in enumerate(sorted(predictions, key=lambda x: x.realized_excess))}
    size = len(predictions)
    squared_difference = sum(
        (predicted_order[item.symbol] - realized_order[item.symbol]) ** 2 for item in predictions
    )
    return 1 - 6 * squared_difference / (size * (size * size - 1))

