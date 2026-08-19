from datetime import date, timedelta

import pytest

from portfoliopilot.validation import ProtocolDates, purged_walk_forward


def test_purged_walk_forward_has_gap_and_no_overlap() -> None:
    dates = tuple(date(2020, 1, 1) + timedelta(days=index) for index in range(30))
    folds = purged_walk_forward(dates, minimum_train=10, validation_size=5, purge_size=2, embargo_size=1)
    assert folds[0].train[-1] == dates[9]
    assert folds[0].validate[0] == dates[12]
    assert set(folds[0].train).isdisjoint(folds[0].validate)
    assert folds[1].validate[0] == dates[18]


def test_protocol_rejects_insufficient_embargo() -> None:
    protocol = ProtocolDates(
        date(2020, 1, 1), date(2020, 1, 10), date(2020, 1, 11),
        date(2020, 1, 20), date(2020, 1, 22), date(2020, 1, 30),
    )
    with pytest.raises(ValueError, match="embargo"):
        protocol.validate(1)

