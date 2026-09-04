from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Protocol

from .backtest import BacktestPoint, performance_metrics
from .market_data import DailyBar, effective_split_coefficient
from .multi_backtest import CrossSectionalStrategy, MultiAssetPoint, MultiAssetResult, Transaction
from .universe import MembershipHistory


class CouncilGate(Protocol):
    """Return True to retain a candidate; False or None means abstain."""

    def __call__(self, symbol: str, decision_on: date) -> bool | None: ...


class BatchCouncilGate(CouncilGate, Protocol):
    def evaluate_many(self, symbols: tuple[str, ...], decision_on: date) -> dict[str, bool]: ...


@dataclass(frozen=True)
class CouncilGatedStrategy:
    base: CrossSectionalStrategy
    gate: CouncilGate
    shortlist: int = 40
    top_n: int = 20
    maximum_position_weight: Decimal = Decimal(1)
    maximum_invested_weight: Decimal = Decimal(1)
    preserve_base_weights: bool = False
    target_history: dict[date, dict[str, Decimal]] = field(
        default_factory=dict, compare=False, repr=False,
    )
    name: str = "ai_ranker_plus_research_council"

    def targets(
        self, history: dict[str, tuple[DailyBar, ...]], decision_on: date | None = None,
    ) -> dict[str, Decimal]:
        if decision_on is None:
            decision_on = max(bars[-1].session for bars in history.values() if bars)
        candidates = self.base.targets(history)
        # Ranking strategies return insertion-ordered targets, strongest first.
        ranked = list(candidates)
        shortlisted = tuple(ranked[: self.shortlist])
        if hasattr(self.gate, "evaluate_many"):
            decisions = self.gate.evaluate_many(shortlisted, decision_on)  # type: ignore[attr-defined]
            approved = [symbol for symbol in shortlisted if decisions.get(symbol, False)]
        else:
            approved = [symbol for symbol in shortlisted if self.gate(symbol, decision_on)]
        selected = approved[: self.top_n]
        if not selected:
            self.target_history[decision_on] = {}
            return {}
        if self.preserve_base_weights:
            targets = {
                symbol: min(candidates[symbol], self.maximum_position_weight)
                for symbol in selected
            }
            self.target_history[decision_on] = targets
            return targets
        weight = min(
            self.maximum_position_weight,
            self.maximum_invested_weight / Decimal(len(selected)),
        )
        targets = {symbol: weight for symbol in selected}
        self.target_history[decision_on] = targets
        return targets


@dataclass(frozen=True)
class CoverageAudit:
    eligible_symbol_sessions: int
    covered_symbol_sessions: int
    missing_symbol_sessions: tuple[tuple[date, str], ...]

    @property
    def coverage(self) -> float:
        return self.covered_symbol_sessions / self.eligible_symbol_sessions

    @property
    def approved(self) -> bool:
        return not self.missing_symbol_sessions


def audit_price_coverage(
    history: MembershipHistory, sessions: tuple[date, ...],
    universe: dict[str, tuple[DailyBar, ...]],
) -> CoverageAudit:
    available = {symbol: {bar.session for bar in bars} for symbol, bars in universe.items()}
    missing: list[tuple[date, str]] = []
    eligible = covered = 0
    for session in sessions:
        for symbol in history.members_on(session):
            eligible += 1
            if session in available.get(symbol, set()):
                covered += 1
            else:
                missing.append((session, symbol))
    return CoverageAudit(eligible, covered, tuple(missing))


class PointInTimeBacktester:
    """Dynamic-membership simulator that fails closed on incomplete constituent prices."""

    def __init__(
        self, starting_cash: Decimal = Decimal(100_000), cost_bps: Decimal = Decimal(5),
        rebalance_every: int = 21,
    ):
        if starting_cash <= 0 or cost_bps < 0 or rebalance_every <= 0:
            raise ValueError("invalid point-in-time backtest assumptions")
        self.starting_cash = starting_cash
        self.cost_bps = cost_bps
        self.rebalance_every = rebalance_every

    def run(
        self, universe: dict[str, tuple[DailyBar, ...]], benchmark: tuple[DailyBar, ...],
        membership: MembershipHistory, strategy: CrossSectionalStrategy,
        evaluation_start: date | None = None,
    ) -> MultiAssetResult:
        sessions = tuple(
            bar.session for bar in benchmark
            if evaluation_start is None or bar.session >= evaluation_start
        )
        audit = audit_price_coverage(membership, sessions, universe)
        if not audit.approved:
            sample = ", ".join(f"{day}:{symbol}" for day, symbol in audit.missing_symbol_sessions[:3])
            raise ValueError(
                f"point-in-time price coverage is {audit.coverage:.2%}; missing {sample}"
            )
        indexed = {symbol: {bar.session: bar for bar in bars} for symbol, bars in universe.items()}
        benchmark_index = {bar.session: bar for bar in benchmark}
        cash = self.starting_cash
        shares = {symbol: Decimal(0) for symbol in universe}
        histories: dict[str, list[DailyBar]] = {
            symbol: [bar for bar in bars if bar.session < sessions[0]]
            for symbol, bars in universe.items()
        }
        pending: dict[str, Decimal] | None = None
        benchmark_shares = self.starting_cash / benchmark[0].open
        benchmark_cash = Decimal(0)
        points: list[MultiAssetPoint] = []
        transactions: list[Transaction] = []
        for index, session in enumerate(sessions):
            members = set(membership.members_on(session))
            if index:
                for symbol, quantity in shares.items():
                    bar = indexed.get(symbol, {}).get(session)
                    if quantity and bar is None:
                        raise ValueError(f"held security {symbol} has no exit price on {session}")
                    if bar is not None:
                        shares[symbol] = quantity * effective_split_coefficient(bar)
                        dividend = shares[symbol] * bar.dividend
                        cash += dividend
                        if dividend:
                            transactions.append(Transaction(
                                session, symbol, "DIVIDEND", shares[symbol], bar.dividend, dividend,
                            ))
                benchmark_bar = benchmark_index[session]
                benchmark_shares *= effective_split_coefficient(benchmark_bar)
                benchmark_cash += benchmark_shares * benchmark_bar.dividend
            opens = {
                symbol: indexed[symbol][session].open for symbol in shares
                if session in indexed[symbol]
            }
            turnover = cost = Decimal(0)
            if pending is not None:
                cash, shares, turnover, cost, trades = self._rebalance(
                    session, cash, shares, opens, pending,
                )
                transactions.extend(trades)
            closes = {
                symbol: indexed[symbol][session].close for symbol, quantity in shares.items()
                if quantity and session in indexed[symbol]
            }
            equity = cash + sum((shares[symbol] * price for symbol, price in closes.items()), Decimal(0))
            weights = {
                symbol: shares[symbol] * price / equity for symbol, price in closes.items() if equity
            }
            points.append(MultiAssetPoint(
                session, equity,
                benchmark_cash + benchmark_shares * benchmark_index[session].close,
                cash, weights, turnover, cost,
            ))
            for symbol, values in histories.items():
                bar = indexed[symbol].get(session)
                if bar is not None:
                    values.append(bar)
            if index % self.rebalance_every == 0:
                eligible_history = {
                    symbol: tuple(histories[symbol]) for symbol in sorted(members)
                    if histories.get(symbol)
                }
                pending = strategy.targets(eligible_history)
            else:
                pending = None
        metrics = performance_metrics(tuple(
            BacktestPoint(point.session, point.equity, point.benchmark_equity, Decimal(0), point.cost)
            for point in points
        ))
        years = max((sessions[-1] - sessions[0]).days / 365.25, 1 / 252)
        metrics["annual_turnover"] = sum(float(point.turnover) for point in points) / years
        return MultiAssetResult(strategy.name, tuple(points), metrics, tuple(transactions))

    def _rebalance(self, session, cash, shares, opens, targets):
        unknown = set(targets) - set(opens)
        if unknown:
            raise ValueError(f"targets lack execution prices: {sorted(unknown)}")
        held_without_price = {symbol for symbol, quantity in shares.items() if quantity and symbol not in opens}
        if held_without_price:
            raise ValueError(f"holdings lack execution prices: {sorted(held_without_price)}")
        if any(weight < 0 for weight in targets.values()) or sum(targets.values()) > Decimal(1):
            raise ValueError("targets must be long-only and fully funded")
        equity = cash + sum((shares[symbol] * opens[symbol] for symbol in opens), Decimal(0))
        desired = {
            symbol: equity * targets.get(symbol, Decimal(0)) / opens[symbol]
            for symbol in opens
        }
        # The universe contains securities that list later in the experiment. They have a
        # zero-share ledger entry but no opening price yet and must not enter trade arithmetic.
        deltas = {symbol: desired[symbol] - shares[symbol] for symbol in opens}
        traded = sum((abs(delta) * opens[symbol] for symbol, delta in deltas.items()), Decimal(0))
        cost = traded * self.cost_bps / Decimal(10_000)
        executed = {}
        for symbol, delta in deltas.items():
            if delta < 0:
                cash -= delta * opens[symbol]
                shares[symbol] += delta
                executed[symbol] = delta
        cash -= cost
        buys = sum((delta * opens[symbol] for symbol, delta in deltas.items() if delta > 0), Decimal(0))
        scale = min(Decimal(1), max(Decimal(0), cash / buys)) if buys else Decimal(1)
        for symbol, delta in deltas.items():
            if delta > 0:
                filled = delta * scale
                cash -= filled * opens[symbol]
                shares[symbol] += filled
                executed[symbol] = filled
        actual = sum((abs(qty) * opens[symbol] for symbol, qty in executed.items()), Decimal(0))
        transactions = tuple(Transaction(
            session=session, symbol=symbol, kind="BUY" if quantity > 0 else "SELL",
            quantity=abs(quantity), price=opens[symbol], cash_amount=abs(quantity) * opens[symbol],
            cost=cost * abs(quantity) * opens[symbol] / actual if actual else Decimal(0),
        ) for symbol, quantity in sorted(executed.items()) if quantity)
        return cash, shares, traded / equity if equity else Decimal(0), cost, transactions
