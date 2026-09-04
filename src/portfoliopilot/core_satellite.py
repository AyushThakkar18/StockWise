from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal


@dataclass
class CoreSatelliteStrategy:
    """Keep a passive benchmark core and scale an active strategy into a bounded sleeve."""

    active: object
    core_weight: Decimal = Decimal("0.70")
    core_symbol: str = "SPY"
    name: str = "spy_core_stockwise_satellite"
    target_history: dict[date, dict[str, Decimal]] = field(default_factory=dict)
    audits: dict[date, dict[str, object]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not Decimal(0) <= self.core_weight <= Decimal(1):
            raise ValueError("core weight must be between zero and one")

    def targets(self, history) -> dict[str, Decimal]:
        decision_on = max(bars[-1].session for bars in history.values() if bars)
        active_targets = self.active.targets(history)
        sleeve = Decimal(1) - self.core_weight
        active_total = sum(active_targets.values(), Decimal(0))
        scaled = {
            symbol: sleeve * weight / active_total
            for symbol, weight in active_targets.items()
            if active_total > 0 and weight > 0
        }
        targets = {self.core_symbol: self.core_weight, **scaled}
        total = sum(targets.values(), Decimal(0))
        if total > Decimal(1):
            last = next(reversed(targets))
            targets[last] -= total - Decimal(1)
        self.target_history[decision_on] = targets
        self.audits[decision_on] = {
            "core_symbol": self.core_symbol,
            "core_weight": float(self.core_weight),
            "active_sleeve_weight": float(sum(scaled.values(), Decimal(0))),
            "active_strategy": getattr(self.active, "name", type(self.active).__name__),
            "active_targets": {symbol: float(weight) for symbol, weight in scaled.items()},
        }
        return targets
