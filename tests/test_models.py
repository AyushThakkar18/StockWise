from datetime import date, timedelta

import pytest

from portfoliopilot.dataset import TrainingRow
from portfoliopilot.model_evaluation import evaluate_walk_forward
from portfoliopilot.models import LinearModelArtifact, train_ridge
from portfoliopilot.validation import TimeFold


def rows() -> tuple[TrainingRow, ...]:
    output = []
    start = date(2020, 1, 1)
    for day in range(8):
        decision_date = start + timedelta(days=day * 3)
        for symbol, value in (("A", -1.0), ("B", 0.0), ("C", 1.0)):
            output.append(TrainingRow(
                symbol=symbol, decision_date=decision_date, feature_version="f1",
                input_hash=f"{day}-{symbol}", features={"momentum": value, "quality": value / 2},
                forward_return=value * 0.01, benchmark_forward_return=0.0,
                excess_return=value * 0.01, label_end_date=decision_date + timedelta(days=1),
            ))
    return tuple(output)


def test_ridge_artifact_round_trip_and_tamper_detection() -> None:
    artifact = train_ridge(rows()[:12], alpha=1.0, model_version="m1")
    restored = LinearModelArtifact.from_json(artifact.to_json())
    assert restored == artifact
    assert restored.predict({"momentum": 1.0, "quality": 0.5}) > 0
    with pytest.raises(ValueError, match="hash mismatch"):
        LinearModelArtifact.from_json(artifact.to_json().replace('"alpha":1.0', '"alpha":2.0'))


def test_walk_forward_trains_only_before_validation() -> None:
    data = rows()
    dates = tuple(sorted({row.decision_date for row in data}))
    folds = (
        TimeFold(train=dates[:4], validate=dates[4:6]),
        TimeFold(train=dates[:6], validate=dates[6:]),
    )
    result = evaluate_walk_forward(data, folds, alpha=1.0, model_prefix="ridge")
    assert result.independent_dates == 4
    assert result.mean_rank_correlation == pytest.approx(1.0)
    assert {prediction.model_version for prediction in result.predictions} == {
        "ridge-fold-1", "ridge-fold-2"
    }


def test_walk_forward_rejects_overlapping_training_labels() -> None:
    data = list(rows())
    dates = tuple(sorted({row.decision_date for row in data}))
    data[0] = TrainingRow(**{
        **data[0].__dict__, "label_end_date": dates[4]
    })
    with pytest.raises(ValueError, match="overlap"):
        evaluate_walk_forward(tuple(data), (TimeFold(dates[:4], dates[4:6]),), 1.0, "m")

