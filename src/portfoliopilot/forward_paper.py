from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

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
            "live-five-agent-v4-top200-kronos75-no-spy-quality-floor-70",
            {"weighting": "equal_above_10_capped_below_10", "target_minimum_positions": 10,
             "maximum_positions": 20, "fallback": "cash",
             "deterministic_candidates": 200, "kronos_candidates": 75,
             "execution": "SIMULATED_NEXT_OPEN",
             "execution_cost_reserve": "0.002",
             "minimum_order_notional": "0.01",
             "slippage_bps": str(slippage_bps), "commission_bps": str(commission_bps)},
            "live-council-v4-top200-kronos75-no-spy-quality-floor-70",
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
        payload = decision.audit_payload() | {
            "execution_mode": "SIMULATED_NEXT_OPEN",
            "earliest_execution_at": earliest_execution_at.isoformat(),
            "target_weights": {symbol: str(weight) for symbol, weight in targets.items()},
        }
        existing = coordinator.sessions.get(session_id)
        if (
            existing is not None
            and (existing.decision_at != decision.decision_at
                 or existing.portfolio_snapshot_hash != snapshot_hash)
        ):
            raise ValueError("existing paper session does not match the resumed decision")
        payload_hash = self.paper.record_decision(
            self.version_hash,
            decision.decision_at.astimezone(ZoneInfo("America/New_York")).date(),
            earliest_execution_at.astimezone(ZoneInfo("America/New_York")).date(),
            payload, decision.decision_at,
        )
        if existing is None:
            coordinator.freeze(session_id, decision.decision_at, snapshot_hash, allocation)
        return payload_hash

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
            Decimal(1),
            "live-council-v4-top200-kronos75-no-spy-quality-floor-70",
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

    def correct_incomplete_execution(
        self, decision_id: str, execution_at: datetime,
        opening_prices: dict[str, Decimal],
    ) -> tuple[dict[str, object], ...]:
        """Append target-restoring fills after a disclosed partial paper execution."""
        correction_id = f"{decision_id}-fill-correction-v1"
        completion_id = f"session:{correction_id}:execution-complete"
        if any(event["event_id"] == completion_id for event in self.events.events(correction_id)):
            return ()
        decisions = [
            item for item in self.paper.all_decision_payloads()
            if item["payload"]["decision"]["decision_id"] == decision_id
        ]
        if len(decisions) != 1:
            raise ValueError("frozen council decision is missing or ambiguous")
        ledger, coordinator = self._recover()
        raw_targets = decisions[0]["payload"]["target_weights"]
        allocation = AllocationResult(
            {symbol: float(value) for symbol, value in raw_targets.items()}, 0.0, {}, 0.0,
        )
        required = set(raw_targets) | {
            symbol for symbol, position in ledger.positions.items() if position.quantity
        }
        missing_prices = sorted(required - set(opening_prices))
        if missing_prices:
            raise ValueError(f"missing correction prices: {', '.join(missing_prices)}")
        decision_at = datetime.fromisoformat(decisions[0]["decided_at"])
        if correction_id not in coordinator.sessions:
            coordinator.freeze(
                correction_id, decision_at, ledger.snapshot_hash(opening_prices), allocation,
            )
        orders = orders_from_targets(
            correction_id, decision_at, execution_at, allocation, ledger, opening_prices,
            Decimal(1), "live-council-v4-execution-correction-v1",
            minimum_notional=Decimal("0.01"),
        )
        results = []
        for order in sorted(orders, key=lambda item: (item.side.value == "BUY", item.symbol)):
            if order.id not in coordinator.orders:
                coordinator.queue(correction_id, order)
            if order.id in coordinator.executed_orders:
                continue
            result = coordinator.execute(order.id, MarketQuote(
                symbol=order.symbol, observed_at=execution_at, available_at=execution_at,
                mid=opening_prices[order.symbol], spread_bps=Decimal(0),
                available_quantity=Decimal("1000000000"),
            ))
            results.append(result.model_dump(mode="json"))
        if any(result["status"] != "FILLED" for result in results):
            raise RuntimeError("paper execution correction did not fill every target")
        self.paper.record_snapshot(
            self.version_hash, "live-five-agent-correction", execution_at.date(),
            ledger.equity(opening_prices), ledger.cash,
            {symbol: position.quantity for symbol, position in ledger.positions.items()
             if position.quantity},
            self._price_hash(opening_prices),
        )
        positions = {
            symbol: {
                "quantity": str(position.quantity), "average_cost": str(position.average_cost),
                "mark": str(opening_prices[symbol]),
                "market_value": str(position.quantity * opening_prices[symbol]),
                "unrealized_pnl": str(
                    position.quantity * opening_prices[symbol] - position.cost_basis
                ),
            }
            for symbol, position in ledger.positions.items() if position.quantity
        }
        self.events.append(
            f"session:{correction_id}:disclosure", "PAPER_EXECUTION_CORRECTED", correction_id,
            {
                "original_decision_id": decision_id,
                "disclosure": "Append-only correction of falsely rejected simulated orders.",
                "execution_at": execution_at.isoformat(), "orders": len(results),
            },
        )
        self.events.append(
            f"session:{correction_id}:valuation", "PORTFOLIO_VALUATION", correction_id,
            {
                "session": execution_at.date().isoformat(),
                "equity": str(ledger.equity(opening_prices)), "cash": str(ledger.cash),
                "realized_pnl": str(ledger.realized_pnl), "positions": positions,
                "unrealized_pnl": str(sum(
                    (Decimal(item["unrealized_pnl"]) for item in positions.values()), Decimal(0),
                )),
                "execution_correction": True,
            },
        )
        self.events.append(completion_id, "PAPER_SESSION_EXECUTED", correction_id, {
            "execution_at": execution_at.isoformat(), "result_count": len(results),
            "price_source_hash": self._price_hash(opening_prices), "correction": True,
        })
        return tuple(results)

    def reconstruct_liquidation_at_close(
        self, session_id: str, session_close: datetime,
        closing_prices: dict[str, Decimal],
    ) -> dict[str, object]:
        """Append an explicitly disclosed missed-rebalance liquidation using historical closes."""
        completion_id = f"session:{session_id}:execution-complete"
        if any(event["event_id"] == completion_id for event in self.events.events(session_id)):
            return {"session_id": session_id, "status": "ALREADY_RECONSTRUCTED", "orders": 0}
        ledger, coordinator = self._recover()
        held = tuple(sorted(
            symbol for symbol, position in ledger.positions.items() if position.quantity
        ))
        if not held:
            return {"session_id": session_id, "status": "ALREADY_FLAT", "orders": 0}
        missing = sorted(set(held) - set(closing_prices))
        if missing:
            raise ValueError(f"missing historical close prices: {', '.join(missing)}")
        marks = {symbol: closing_prices[symbol] for symbol in held}
        decision_at = session_close - timedelta(minutes=1)
        allocation = AllocationResult({}, 1.0, {}, 0.0)
        coordinator.freeze(session_id, decision_at, ledger.snapshot_hash(marks), allocation)
        orders = orders_from_targets(
            session_id, decision_at, session_close, allocation, ledger, marks,
            Decimal("0.10"), "reconstructed-missed-rebalance-v1",
            minimum_notional=Decimal("0.01"),
        )
        results = []
        for order in orders:
            coordinator.queue(session_id, order)
            result = coordinator.execute(order.id, MarketQuote(
                symbol=order.symbol, observed_at=session_close, available_at=session_close,
                mid=marks[order.symbol], spread_bps=Decimal(0),
                available_quantity=Decimal("1000000000"),
            ))
            results.append(result.model_dump(mode="json"))
        if any(result["status"] != "FILLED" for result in results):
            raise RuntimeError("reconstructed liquidation did not fill every held position")
        self.paper.record_snapshot(
            self.version_hash, "live-five-agent", session_close.date(),
            ledger.equity({}), ledger.cash, {}, self._price_hash(marks),
        )
        valuation = {
            "session": session_close.date().isoformat(), "equity": str(ledger.cash),
            "cash": str(ledger.cash), "realized_pnl": str(ledger.realized_pnl),
            "positions": {}, "unrealized_pnl": "0",
            "price_source": "Yahoo Finance historical daily close",
            "reconstructed": True,
        }
        self.events.append(
            f"session:{session_id}:valuation", "PORTFOLIO_VALUATION", session_id, valuation,
        )
        self.events.append(
            f"session:{session_id}:reconstruction", "MISSED_REBALANCE_RECONSTRUCTED",
            session_id, {
                "session_close": session_close.isoformat(),
                "pricing_basis": "official historical daily close",
                "disclosure": "Backdated paper correction created after the stated session.",
                "symbols": held,
            },
        )
        self.events.append(completion_id, "PAPER_SESSION_EXECUTED", session_id, {
            "execution_at": session_close.isoformat(), "result_count": len(results),
            "price_source_hash": self._price_hash(marks), "reconstructed": True,
        })
        return {
            "session_id": session_id, "status": "RECONSTRUCTED", "orders": len(results),
            "cash": str(ledger.cash), "realized_pnl": str(ledger.realized_pnl),
        }

    def _recover(self) -> tuple[PortfolioLedger, PaperSessionCoordinator]:
        ledger = PortfolioLedger(self.initial_cash)
        coordinator = PaperSessionCoordinator(self.events, ledger, self.broker)
        coordinator.recover()
        return ledger, coordinator

    @staticmethod
    def _targets(selected: tuple[str, ...]) -> dict[str, Decimal]:
        if len(selected) > 20:
            raise ValueError("at most twenty stocks may be selected")
        if not selected:
            return {}
        # A full rebalance can incur costs on both liquidation and purchases. Keep 0.20% idle so
        # the final alphabetically ordered buy is not rejected after earlier fills consume costs.
        reserve = Decimal("0.002")
        # Below ten safe names, preserve the ten-slot cap and leave unused slots in cash rather
        # than concentrating the portfolio or falling back to SPY.
        weight = (Decimal(1) - reserve) / Decimal(max(10, len(selected)))
        return {symbol: weight for symbol in selected}

    @staticmethod
    def _price_hash(prices: dict[str, Decimal]) -> str:
        from .paper_store import canonical_hash
        return canonical_hash({symbol: str(value) for symbol, value in sorted(prices.items())})
