from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal

from .contracts import AuditEvent, Fill, Side

ZERO = Decimal("0")


@dataclass
class Position:
    symbol: str
    quantity: Decimal = ZERO
    cost_basis: Decimal = ZERO

    @property
    def average_cost(self) -> Decimal:
        return self.cost_basis / self.quantity if self.quantity else ZERO


@dataclass
class PortfolioLedger:
    initial_cash: Decimal
    cash: Decimal = field(init=False)
    positions: dict[str, Position] = field(default_factory=dict)
    realized_pnl: Decimal = ZERO
    audit: list[AuditEvent] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.initial_cash < 0:
            raise ValueError("initial cash cannot be negative")
        self.cash = self.initial_cash
        self._record("CASH_DEPOSIT", "portfolio", {"amount": str(self.initial_cash)})

    def apply_fill(self, fill: Fill) -> None:
        position = self.positions.setdefault(fill.symbol, Position(fill.symbol))
        gross = fill.quantity * fill.price
        # The execution price already includes spread and slippage; those fields
        # are attribution, while only the explicit fee is charged separately.
        costs = fill.fee
        if fill.side == Side.BUY:
            debit = gross + costs
            if debit > self.cash:
                raise ValueError("insufficient cash")
            self.cash -= debit
            position.quantity += fill.quantity
            position.cost_basis += debit
        else:
            if fill.quantity > position.quantity:
                raise ValueError("cannot sell more than held")
            relieved_basis = position.average_cost * fill.quantity
            proceeds = gross - costs
            self.cash += proceeds
            position.quantity -= fill.quantity
            position.cost_basis -= relieved_basis
            self.realized_pnl += proceeds - relieved_basis
            if position.quantity == 0:
                position.cost_basis = ZERO
        self._record("FILL_APPLIED", fill.id, fill.model_dump(mode="json"))

    def equity(self, marks: dict[str, Decimal]) -> Decimal:
        missing = set(self.positions) - set(marks)
        missing = {s for s in missing if self.positions[s].quantity != 0}
        if missing:
            raise ValueError(f"missing marks: {sorted(missing)}")
        return self.cash + sum(
            (position.quantity * marks.get(symbol, ZERO) for symbol, position in self.positions.items()),
            ZERO,
        )

    def snapshot_hash(self, marks: dict[str, Decimal]) -> str:
        payload = {
            "cash": str(self.cash),
            "positions": {
                symbol: {"quantity": str(p.quantity), "cost_basis": str(p.cost_basis)}
                for symbol, p in sorted(self.positions.items())
            },
            "marks": {symbol: str(value) for symbol, value in sorted(marks.items())},
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def _record(self, event_type: str, entity_id: str, payload: dict) -> None:
        self.audit.append(
            AuditEvent(
                sequence=len(self.audit) + 1,
                occurred_at=datetime.now(timezone.utc),
                event_type=event_type,
                entity_id=entity_id,
                payload=payload,
            )
        )
