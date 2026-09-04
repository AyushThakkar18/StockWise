from __future__ import annotations

from decimal import Decimal

from .alpaca_paper import AlpacaPaperClient
from .store import EventStore


class AlpacaPaperExecutor:
    """Converts frozen 5% targets to idempotent paper orders and records broker responses."""

    def __init__(self, client: AlpacaPaperClient, store: EventStore, enabled: bool = False) -> None:
        self.client, self.store, self.enabled = client, store, enabled

    def rebalance(
        self,
        decision_id: str,
        selected: tuple[str, ...],
        equity: Decimal,
        current_values: dict[str, Decimal],
        minimum_notional: Decimal = Decimal("10"),
    ) -> tuple[dict, ...]:
        if not self.enabled:
            raise ValueError("paper order submission is disabled")
        if equity <= 0 or len(selected) > 20 or len(set(selected)) != len(selected):
            raise ValueError("invalid paper rebalance")
        targets = {symbol: equity * Decimal("0.05") for symbol in selected}
        targets["SPY"] = equity * (Decimal(1) - Decimal("0.05") * len(selected))
        existing_ids = {event["event_id"] for event in self.store.events()}
        orders = []
        deltas = {
            symbol: targets.get(symbol, Decimal(0)) - current_values.get(symbol, Decimal(0))
            for symbol in set(targets) | set(current_values)
        }
        for symbol, delta in sorted(deltas.items(), key=lambda item: (item[1] > 0, item[0])):
            if abs(delta) < minimum_notional:
                continue
            side = "buy" if delta > 0 else "sell"
            client_order_id = f"{decision_id}:{symbol}:{side}"[:48]
            event_id = f"alpaca-paper-order:{client_order_id}"
            if event_id in existing_ids:
                continue
            response = self.client.submit_notional_order(
                symbol=symbol, notional=str(abs(delta).quantize(Decimal("0.01"))),
                side=side, client_order_id=client_order_id,
            )
            self.store.append(event_id, "ALPACA_PAPER_ORDER_SUBMITTED", decision_id, {
                "client_order_id": client_order_id, "symbol": symbol, "side": side,
                "target_value": str(targets.get(symbol, Decimal(0))),
                "previous_value": str(current_values.get(symbol, Decimal(0))),
                "requested_notional": str(abs(delta).quantize(Decimal("0.01"))),
                "broker_response": response,
            })
            orders.append(response)
        return tuple(orders)
