from __future__ import annotations

import json
from decimal import Decimal
from typing import Any

from .paper_store import PaperTradingStore
from .store import EventStore


class PaperDashboard:
    """Read-only projections built from immutable decision and execution records."""

    def __init__(self, paper: PaperTradingStore, events: EventStore) -> None:
        self.paper, self.events = paper, events

    def overview(self) -> dict[str, Any]:
        snapshots = self.paper.snapshots()
        latest = snapshots[-1] if snapshots else None
        decisions = self.paper.all_decision_payloads()
        trades = self.trades()
        valuations = [
            {**json.loads(event["payload"]), "occurred_at": event["occurred_at"]}
            for event in self.events.events() if event["event_type"] == "PORTFOLIO_VALUATION"
        ]
        configured = [
            json.loads(event["payload"]) for event in self.events.events("portfolio")
            if event["event_type"] == "PAPER_PORTFOLIO_CONFIGURED"
        ]
        return {
            "initial_cash": configured[0]["initial_cash"] if configured else str(Decimal("1000")),
            "latest_snapshot": latest,
            "portfolio": valuations[-1] if valuations else None,
            "decision_count": len(decisions),
            "trade_count": len(trades),
            "latest_decision": decisions[-1] if decisions else None,
        }

    def decisions(self) -> tuple[dict[str, Any], ...]:
        return self.paper.all_decision_payloads()

    def trades(self) -> tuple[dict[str, Any], ...]:
        output = []
        for event in self.events.events():
            if event["event_type"] != "PAPER_ORDER_RESULT":
                continue
            payload = json.loads(event["payload"])
            fill = payload.get("fill")
            output.append({
                "sequence": event["sequence"], "occurred_at": event["occurred_at"],
                "session_id": event["entity_id"], "order_id": payload["order_id"],
                "status": payload["status"], "reason": payload.get("reason"),
                "symbol": fill.get("symbol") if fill else None,
                "side": fill.get("side") if fill else None,
                "quantity": fill.get("quantity") if fill else None,
                "price": fill.get("price") if fill else None,
                "executed_at": fill.get("executed_at") if fill else None,
            })
        return tuple(output)
