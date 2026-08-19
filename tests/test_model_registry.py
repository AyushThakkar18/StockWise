import sqlite3
from datetime import date

import pytest

from portfoliopilot.dataset import TrainingRow
from portfoliopilot.model_registry import ModelRegistry
from portfoliopilot.models import train_ridge


def rows() -> tuple[TrainingRow, ...]:
    return tuple(
        TrainingRow(
            symbol=str(index), decision_date=date(2020, 1, 1), feature_version="f1",
            input_hash=str(index), features={"x": float(index)}, forward_return=index / 100,
            benchmark_forward_return=0, excess_return=index / 100,
            label_end_date=date(2020, 1, 2),
        )
        for index in range(4)
    )


def test_registry_preserves_artifact_and_evaluation_history(tmp_path) -> None:
    registry = ModelRegistry(tmp_path / "models.db")
    artifact = train_ridge(rows()[:12], 1.0, "m1")
    registry.register(artifact)
    assert registry.artifact("m1") == artifact
    registry.record_evaluation("m1", "exp-1", "VALIDATION", {"rank": 0.1}, False)
    registry.record_evaluation("m1", "exp-1", "HOLDOUT", {"rank": 0.05}, True)
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        registry.connection.execute(
            "UPDATE model_artifacts SET artifact_hash = 'changed' WHERE model_version = 'm1'"
        )


def test_validation_cannot_enable_paper_trading(tmp_path) -> None:
    registry = ModelRegistry(tmp_path / "models.db")
    artifact = train_ridge(rows()[:12], 1.0, "m1")
    registry.register(artifact)
    with pytest.raises(ValueError, match="holdout"):
        registry.record_evaluation("m1", "exp-1", "VALIDATION", {}, True)
