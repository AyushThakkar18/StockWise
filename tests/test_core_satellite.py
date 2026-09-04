from datetime import date
from decimal import Decimal

from test_multi_backtest import bars

from portfoliopilot.core_satellite import CoreSatelliteStrategy


class Active:
    name = "active"

    def targets(self, history):
        return {"A": Decimal("0.60"), "B": Decimal("0.40")}


def test_core_satellite_scales_active_sleeve() -> None:
    strategy = CoreSatelliteStrategy(Active(), Decimal("0.70"))
    targets = strategy.targets({"A": bars("A", [100, 101])})
    assert targets == {
        "SPY": Decimal("0.70"), "A": Decimal("0.18"), "B": Decimal("0.12"),
    }
    assert strategy.target_history[date(2020, 1, 2)] == targets
    assert sum(targets.values()) <= Decimal(1)


def test_empty_active_sleeve_remains_cash() -> None:
    active = Active()
    active.targets = lambda history: {}
    targets = CoreSatelliteStrategy(active).targets({"A": bars("A", [100, 101])})
    assert targets == {"SPY": Decimal("0.70")}
