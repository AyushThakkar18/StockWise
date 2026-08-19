from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Protocol

from .backtest import BacktestPoint, performance_metrics
from .market_data import DailyBar
from .multi_backtest import CrossSectionalStrategy, MultiAssetPoint, MultiAssetResult
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
            return {}
        weight = Decimal(1) / Decimal(len(selected))
        return {symbol: weight for symbol in selected}


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
    ) -> MultiAssetResult:
        sessions = tuple(bar.session for bar in benchmark)
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
        histories: dict[str, list[DailyBar]] = {symbol: [] for symbol in universe}
        pending: dict[str, Decimal] | None = None
        benchmark_shares = self.starting_cash / benchmark[0].open
        benchmark_cash = Decimal(0)
        points: list[MultiAssetPoint] = []
        for index, session in enumerate(sessions):
            members = set(membership.members_on(session))
            if index:
                for symbol, quantity in shares.items():
                    bar = indexed.get(symbol, {}).get(session)
                    if quantity and bar is None:
                        raise ValueError(f"held security {symbol} has no exit price on {session}")
                    if bar is not None:
                        shares[symbol] = quantity * bar.split_coefficient
                        cash += shares[symbol] * bar.dividend
                benchmark_bar = benchmark_index[session]
                benchmark_shares *= benchmark_bar.split_coefficient
                benchmark_cash += benchmark_shares * benchmark_bar.dividend
            opens = {
                symbol: indexed[symbol][session].open for symbol in shares
                if session in indexed[symbol]
            }
            turnover = cost = Decimal(0)
            if pending is not None:
                cash, shares, turnover, cost = self._rebalance(cash, shares, opens, pending)
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
        return MultiAssetResult(strategy.name, tuple(points), metrics)

    def _rebalance(self, cash, shares, opens, targets):
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
        deltas = {symbol: desired.get(symbol, Decimal(0)) - shares[symbol] for symbol in shares}
        traded = sum((abs(delta) * opens[symbol] for symbol, delta in deltas.items()), Decimal(0))
        cost = traded * self.cost_bps / Decimal(10_000)
        for symbol, delta in deltas.items():
            if delta < 0:
                cash -= delta * opens[symbol]
                shares[symbol] += delta
        cash -= cost
        buys = sum((delta * opens[symbol] for symbol, delta in deltas.items() if delta > 0), Decimal(0))
        scale = min(Decimal(1), max(Decimal(0), cash / buys)) if buys else Decimal(1)
        for symbol, delta in deltas.items():
            if delta > 0:
                executed = delta * scale
                cash -= executed * opens[symbol]
                shares[symbol] += executed
        return cash, shares, traded / equity if equity else Decimal(0), cost
