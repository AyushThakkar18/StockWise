import json
import sqlite3
from datetime import date

import pytest

from portfoliopilot.experiments import ExperimentProtocol, ExperimentRegistry
from portfoliopilot.validation import ProtocolDates


def protocol() -> ExperimentProtocol:
    return ExperimentProtocol(
        experiment_id="exp-1", protocol_version="p1", model_version="m1",
        universe=("ABC", "SPY"),
        dates=ProtocolDates(
            date(2020, 1, 1), date(2021, 12, 31),
            date(2022, 1, 3), date(2022, 12, 30),
            date(2023, 1, 2), date(2023, 12, 29),
        ),
        embargo_days=1, transaction_cost_bps=5, seed=7,
    )


def test_protocol_and_opened_holdout_are_immutable(tmp_path) -> None:
    registry = ExperimentRegistry(tmp_path / "experiments.db")
    expected_hash = protocol().protocol_hash
    assert registry.register(protocol()) == expected_hash
    registry.open_holdout("exp-1", {"sharpe": 0.4}, eligible=False)
    stored = registry.get("exp-1")
    assert stored["holdout_opened_at"] is not None
    assert json.loads(stored["holdout_result_json"]) == {"sharpe": 0.4}
    with pytest.raises(ValueError, match="already opened"):
        registry.open_holdout("exp-1", {"sharpe": 2.0}, eligible=True)
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        registry.connection.execute(
            "UPDATE experiments SET protocol_hash = 'changed' WHERE experiment_id = 'exp-1'"
        )


def test_duplicate_experiment_is_rejected(tmp_path) -> None:
    registry = ExperimentRegistry(tmp_path / "experiments.db")
    registry.register(protocol())
    with pytest.raises(sqlite3.IntegrityError):
        registry.register(protocol())

