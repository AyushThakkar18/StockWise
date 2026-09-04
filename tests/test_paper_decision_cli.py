from datetime import date

import pytest

from portfoliopilot.paper_decision_cli import validate_prospective_date


def test_paper_decision_cannot_be_backfilled_or_predated() -> None:
    today = date(2026, 8, 25)
    validate_prospective_date(today, today)
    with pytest.raises(ValueError, match="cannot be backfilled"):
        validate_prospective_date(date(2026, 8, 24), today)
    with pytest.raises(ValueError, match="cannot be backfilled"):
        validate_prospective_date(date(2026, 8, 26), today)
