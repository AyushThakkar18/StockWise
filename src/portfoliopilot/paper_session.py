from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime

from .broker import PaperBroker
from .contracts import ExecutionResult, MarketQuote, Order
from .ledger import PortfolioLedger
from .optimizer import AllocationResult
from .store import EventStore


@dataclass(frozen=True)
class FrozenSession:
    session_id: str
    decision_at: datetime
    portfolio_snapshot_hash: str
    allocation_hash: str


class PaperSessionCoordinator:
    """Freezes allocations and permits their orders to execute only in a later session."""

    def __init__(self, store: EventStore, ledger: PortfolioLedger, broker: PaperBroker):
        self.store, self.ledger, self.broker = store, ledger, broker
        self.sessions: dict[str, FrozenSession] = {}
        self.orders: dict[str, tuple[str, Order]] = {}
        self.executed_orders: set[str] = set()

    def recover(self, apply_fills: bool = True) -> None:
        """Replay immutable events into a fresh coordinator and ledger."""
        if self.sessions or self.orders or self.executed_orders:
            raise ValueError("recovery requires an empty coordinator")
        if apply_fills and len(self.ledger.audit) != 1:
            raise ValueError("fill recovery requires a fresh ledger")
        for event in self.store.events():
            payload = json.loads(event["payload"])
            session_id = event["entity_id"]
            if event["event_type"] == "PAPER_SESSION_FROZEN":
                allocation_hash = hashlib.sha256(
                    json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
                self.sessions[session_id] = FrozenSession(
                    session_id, datetime.fromisoformat(payload["decision_at"]),
                    payload["portfolio_snapshot_hash"], allocation_hash,
                )
            elif event["event_type"] == "PAPER_ORDER_QUEUED":
                order = Order.model_validate(payload)
                self.orders[order.id] = (session_id, order)
            elif event["event_type"] == "PAPER_ORDER_RESULT":
                result = ExecutionResult.model_validate(payload)
                self.executed_orders.add(result.order_id)
                if apply_fills and result.fill is not None:
                    self.ledger.apply_fill(result.fill)

    def freeze(
        self, session_id: str, decision_at: datetime, portfolio_snapshot_hash: str,
        allocation: AllocationResult,
    ) -> FrozenSession:
        if session_id in self.sessions:
            raise ValueError("paper session is already frozen")
        payload = {
            "decision_at": decision_at.isoformat(), "portfolio_snapshot_hash": portfolio_snapshot_hash,
            "target_weights": allocation.target_weights, "cash_weight": allocation.cash_weight,
            "turnover": allocation.turnover,
        }
        allocation_hash = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        session = FrozenSession(session_id, decision_at, portfolio_snapshot_hash, allocation_hash)
        self.store.append(f"session:{session_id}:frozen", "PAPER_SESSION_FROZEN", session_id, payload)
        self.sessions[session_id] = session
        return session

    def queue(self, session_id: str, order: Order) -> None:
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError("paper session must be frozen before orders are queued")
        if order.id in self.orders or order.submitted_at != session.decision_at:
            raise ValueError("order is duplicate or not tied to the frozen decision time")
        self.store.append(
            f"session:{session_id}:order:{order.id}", "PAPER_ORDER_QUEUED", session_id,
            order.model_dump(mode="json"),
        )
        self.orders[order.id] = (session_id, order)

    def execute(self, order_id: str, quote: MarketQuote) -> ExecutionResult:
        if order_id in self.executed_orders:
            raise ValueError("paper order has already been executed")
        if order_id not in self.orders:
            raise KeyError(order_id)
        session_id, order = self.orders[order_id]
        result = self.broker.execute(order, quote, self.ledger)
        self.store.append(
            f"session:{session_id}:result:{order.id}", "PAPER_ORDER_RESULT", session_id,
            result.model_dump(mode="json"),
        )
        self.executed_orders.add(order_id)
        return result
