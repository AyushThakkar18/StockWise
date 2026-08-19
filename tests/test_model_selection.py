from datetime import date, timedelta

import pytest

from portfoliopilot.dataset import TrainingRow
from portfoliopilot.model_selection import select_ridge_alpha
from portfoliopilot.validation import TimeFold


def rows() -> tuple[TrainingRow, ...]:
    start = date(2020, 1, 1)
    output = []
    for day in range(6):
        decision = start + timedelta(days=day * 3)
        for symbol, value in (("A", -1.0), ("B", 1.0)):
            output.append(TrainingRow(
                symbol, decision, "f1", f"{day}:{symbol}", {"x": value},
                value / 100, 0, value / 100, decision + timedelta(days=1),
            ))
    return tuple(output)


def test_selection_is_confined_to_development_dates() -> None:
    data = rows()
    dates = tuple(sorted({row.decision_date for row in data}))
    result = select_ridge_alpha(
        data, (TimeFold(dates[:3], dates[3:]),), (10.0, 0.1, 1.0),
        development_end=max(row.label_end_date for row in data), model_version="ridge-v1",
    )
    assert result.selected_alpha == 0.1
    assert result.artifact.model_version == "ridge-v1"


def test_selection_rejects_rows_past_development_boundary() -> None:
    data = rows()
    dates = tuple(sorted({row.decision_date for row in data}))
    with pytest.raises(ValueError, match="development"):
        select_ridge_alpha(data, (TimeFold(dates[:3], dates[3:]),), (1.0,), dates[-2], "m")

