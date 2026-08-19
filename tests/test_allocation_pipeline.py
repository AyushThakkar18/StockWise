from datetime import UTC, date, datetime

from portfoliopilot.allocation_pipeline import CandidateContext, allocate_from_model
from portfoliopilot.dataset import TrainingRow
from portfoliopilot.features import FeatureSnapshot
from portfoliopilot.models import train_ridge
from portfoliopilot.optimizer import PortfolioConstraints


def artifact():
    rows = tuple(
        TrainingRow(
            symbol, date(2020, 1, 1), "f1", symbol, {"x": value}, value / 10, 0,
            value / 10, date(2020, 1, 2),
        )
        for symbol, value in (("A", -1.0), ("B", 0.0), ("C", 1.0))
    )
    return train_ridge(rows, 0.1, "m1")


def test_model_predictions_flow_through_policy_gates() -> None:
    contexts = tuple(
        CandidateContext(
            FeatureSnapshot(symbol, date(2020, 2, 1), datetime(2020, 2, 1, tzinfo=UTC), "f1", symbol, {"x": value}),
            sector="TECH", evidence_completeness=1.0, average_dollar_volume=20_000_000,
        )
        for symbol, value in (("A", -1.0), ("C", 1.0))
    )
    result = allocate_from_model(
        artifact(), contexts, residual_uncertainty=0.05, current_weights={},
        constraints=PortfolioConstraints(minimum_excess_return=0.01),
    )
    assert "C" in result.target_weights
    assert "A" not in result.target_weights
    assert not next(check for check in result.checks["A"] if check.name == "expected_excess_return").passed

