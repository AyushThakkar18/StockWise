from datetime import date
from decimal import Decimal

import pytest
from test_multi_backtest import bars

from portfoliopilot.multi_backtest import EqualWeight
from portfoliopilot.point_in_time import (
    CouncilGatedStrategy,
    PointInTimeBacktester,
    audit_price_coverage,
)
from portfoliopilot.universe import MembershipHistory


def test_coverage_audit_uses_membership_on_each_session() -> None:
    sessions = (date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3))
    membership = MembershipHistory({sessions[0]: ("A",), sessions[2]: ("B",)}, "fixture")
    universe = {"A": bars("A", [100, 100]), "B": bars("B", [100, 100, 100])}
    audit = audit_price_coverage(membership, sessions, universe)
    assert audit.approved
    assert audit.eligible_symbol_sessions == 3


def test_incomplete_historical_constituent_data_fails_closed() -> None:
    membership = MembershipHistory({date(2020, 1, 1): ("A", "FORMER")}, "fixture")
    with pytest.raises(ValueError, match="point-in-time price coverage"):
        PointInTimeBacktester().run(
            {"A": bars("A", [100, 100, 100])}, bars("SPY", [100, 100, 100]),
            membership, EqualWeight(),
        )


def test_dynamic_membership_exits_removed_stock_at_next_rebalance() -> None:
    membership = MembershipHistory({
        date(2020, 1, 1): ("A",), date(2020, 1, 2): ("B",),
    }, "fixture")
    result = PointInTimeBacktester(cost_bps=Decimal(0), rebalance_every=1).run(
        {"A": bars("A", [100, 100, 100]), "B": bars("B", [100, 100, 100])},
        bars("SPY", [100, 100, 100]), membership, EqualWeight(),
    )
    assert result.points[1].weights == {"A": Decimal(1)}
    assert result.points[2].weights == {"B": Decimal(1)}


def test_council_gate_abstains_instead_of_inventing_approval() -> None:
    strategy = CouncilGatedStrategy(EqualWeight(), lambda symbol, _: symbol == "A", top_n=2)
    targets = strategy.targets({"A": bars("A", [100]), "B": bars("B", [100])}, date(2020, 1, 1))
    assert targets == {"A": Decimal(1)}
