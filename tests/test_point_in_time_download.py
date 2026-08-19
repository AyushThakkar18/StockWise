from datetime import date

from portfoliopilot.point_in_time_download import required_symbols
from portfoliopilot.universe import MembershipHistory


def test_required_symbols_include_removed_and_added_constituents() -> None:
    history = MembershipHistory({
        date(2020, 1, 1): ("A", "REMOVED"),
        date(2020, 2, 1): ("A", "ADDED"),
    }, "fixture")
    assert required_symbols(history, date(2020, 1, 15), date(2020, 2, 15)) == (
        "A", "ADDED", "REMOVED",
    )
