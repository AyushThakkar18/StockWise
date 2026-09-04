from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from pathlib import Path

from .broker import PaperBroker
from .contracts import MarketQuote
from .ledger import PortfolioLedger
from .live_council import LiveCouncilDecision
from .operations import orders_from_targets
from .optimizer import AllocationResult
from .paper_session import PaperSessionCoordinator
from .paper_store import PaperTradingStore, StrategyVersion
from .store import EventStore


class ForwardPaperEngine:
    """Durable close-decision/next-open simulator; never connects to a brokerage."""

    def __init__(
        self, database: Path, initial_cash: Decimal = Decimal("1000"),
        slippage_bps: Decimal = Decimal("2"), commission_bps: Decimal = Decimal("5"),
    ) -> None:
        if initial_cash <= 0:
            raise ValueError("initial cash must be positive")
        self.database, self.initial_cash = database, initial_cash
        self.paper, self.events = PaperTradingStore(database), EventStore(database)
        self.broker = PaperBroker(slippage_bps=slippage_bps, commission_bps=commission_bps)
        self.version = StrategyVersion(
            "live-five-agent-v3-quality-floor-70",
            {"slot_weight": "0.05", "fallback": "SPY", "execution": "SIMULATED_NEXT_OPEN",
             "execution_cost_reserve": "0.002",
             "minimum_order_notional": "0.01",
             "slippage_bps": str(slippage_bps), "commission_bps": str(commission_bps)},
            "live-council-v3-quality-floor-70",
        )
        self.version_hash = self.paper.register_strategy(self.version)

    def freeze(
        self, decision: LiveCouncilDecision, earliest_execution_at: datetime,
        portfolio_marks: dict[str, Decimal],
    ) -> str:
        if earliest_execution_at <= decision.decision_at:
            raise ValueError("execution must be in a later session")
        ledger, coordinator = self._recover()
        session_id = decision.decision_id
        targets = self._targets(decision.selected_symbols)
        allocation = AllocationResult(
            {symbol: float(weight) for symbol, weight in targets.items()}, 0.0, {}, 0.0,
        )
        snapshot_hash = ledger.snapshot_hash(portfolio_marks)
        coordinator.freeze(session_id, decision.decision_at, snapshot_hash, allocation)
        payload = decision.audit_payload() | {
            "execution_mode": "SIMULATED_NEXT_OPEN",
            "earliest_execution_at": earliest_execution_at.isoformat(),
            "target_weights": {symbol: str(weight) for symbol, weight in targets.items()},
        }
        return self.paper.record_decision(
            self.version_hash, decision.decision_at.date(), earliest_execution_at.date(),
            payload, decision.decision_at,
        )

    def execute_next_open(
        self, decision_id: str, execution_at: datetime, opening_prices: dict[str, Decimal],
    ) -> tuple[dict[str, object], ...]:
        ledger, coordinator = self._recover()
        completion_id = f"session:{decision_id}:execution-complete"
        if any(event["event_id"] == completion_id for event in self.events.events(decision_id)):
            return ()
        session = coordinator.sessions.get(decision_id)
        if session is None:
            raise KeyError(decision_id)
        decisions = [
            item for item in self.paper.all_decision_payloads()
            if item["payload"]["decision"]["decision_id"] == decision_id
        ]
        if len(decisions) != 1:
            raise ValueError("frozen council decision is missing or ambiguous")
        raw_targets = decisions[0]["payload"]["target_weights"]
        allocation = AllocationResult(
            {symbol: float(value) for symbol, value in raw_targets.items()}, 0.0, {}, 0.0,
        )
        orders = orders_from_targets(
            decision_id, session.decision_at, execution_at, allocation, ledger, opening_prices,
            Decimal(1), "live-council-v3-quality-floor-70",
            minimum_notional=Decimal("0.01"),
        )
        orders = tuple(sorted(orders, key=lambda order: (order.side.value == "BUY", order.symbol)))
        for order in orders:
            if order.id not in coordinator.orders:
                coordinator.queue(decision_id, order)
        results = []
        for order in orders:
            if order.id in coordinator.executed_orders:
                continue
            result = coordinator.execute(order.id, MarketQuote(
                symbol=order.symbol, observed_at=execution_at, available_at=execution_at,
                mid=opening_prices[order.symbol], spread_bps=Decimal(0),
                available_quantity=Decimal("1000000000"),
            ))
            results.append(result.model_dump(mode="json"))
        self.paper.record_snapshot(
            self.version_hash, "live-five-agent", execution_at.date(),
            ledger.equity(opening_prices), ledger.cash,
            {symbol: position.quantity for symbol, position in ledger.positions.items()
             if position.quantity},
            self._price_hash(opening_prices),
        )
        positions = {
            symbol: {
                "quantity": str(position.quantity),
                "average_cost": str(position.average_cost),
                "mark": str(opening_prices[symbol]),
                "market_value": str(position.quantity * opening_prices[symbol]),
                "unrealized_pnl": str(
                    position.quantity * opening_prices[symbol] - position.cost_basis
                ),
            }
            for symbol, position in ledger.positions.items() if position.quantity
        }
        self.events.append(f"session:{decision_id}:valuation", "PORTFOLIO_VALUATION", decision_id, {
            "session": execution_at.date().isoformat(),
            "equity": str(ledger.equity(opening_prices)), "cash": str(ledger.cash),
            "realized_pnl": str(ledger.realized_pnl), "positions": positions,
            "unrealized_pnl": str(sum(
                (Decimal(item["unrealized_pnl"]) for item in positions.values()), Decimal(0),
            )),
        })
        self.events.append(completion_id, "PAPER_SESSION_EXECUTED", decision_id, {
            "execution_at": execution_at.isoformat(), "result_count": len(results),
            "price_source_hash": self._price_hash(opening_prices),
        })
        return tuple(results)

    def _recover(self) -> tuple[PortfolioLedger, PaperSessionCoordinator]:
        ledger = PortfolioLedger(self.initial_cash)
        coordinator = PaperSessionCoordinator(self.events, ledger, self.broker)
        coordinator.recover()
        return ledger, coordinator

    @staticmethod
    def _targets(selected: tuple[str, ...]) -> dict[str, Decimal]:
        if len(selected) > 20:
            raise ValueError("at most twenty stocks may be selected")
        slot = Decimal("0.05")
        # A full rebalance can incur costs on both liquidation and purchases. Keep 0.20% idle so
        # the final alphabetically ordered buy is not rejected after earlier fills consume costs.
        reserve = Decimal("0.002")
        fallback = Decimal(1) - slot * len(selected)
        if fallback >= reserve:
            return {"SPY": fallback - reserve, **{symbol: slot for symbol in selected}}
        adjusted = (Decimal(1) - reserve) / Decimal(len(selected))
        return {symbol: adjusted for symbol in selected}

    @staticmethod
    def _price_hash(prices: dict[str, Decimal]) -> str:
        from .paper_store import canonical_hash
        return canonical_hash({symbol: str(value) for symbol, value in sorted(prices.items())})
