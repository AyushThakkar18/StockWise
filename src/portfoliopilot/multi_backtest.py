from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Protocol

from .backtest import BacktestPoint, performance_metrics, total_return_ratio
from .market_data import DailyBar, effective_split_coefficient


class CrossSectionalStrategy(Protocol):
    name: str

    def targets(self, history: dict[str, tuple[DailyBar, ...]]) -> dict[str, Decimal]: ...


@dataclass(frozen=True)
class EqualWeight:
    name: str = "equal_weight"

    def targets(self, history: dict[str, tuple[DailyBar, ...]]) -> dict[str, Decimal]:
        eligible = sorted(symbol for symbol, bars in history.items() if bars)
        if not eligible:
            return {}
        weight = Decimal(1) / Decimal(len(eligible))
        return {symbol: weight for symbol in eligible}


@dataclass(frozen=True)
class RankedMomentum:
    lookback: int = 63
    top_n: int = 5
    name: str = "ranked_momentum"

    def targets(self, history: dict[str, tuple[DailyBar, ...]]) -> dict[str, Decimal]:
        ranked = []
        for symbol, bars in history.items():
            if len(bars) <= self.lookback:
                continue
            momentum = total_return_ratio(bars[-1 - self.lookback:]) - 1
            if momentum > 0:
                ranked.append((momentum, symbol))
        selected = sorted(ranked, key=lambda item: (-item[0], item[1]))[:self.top_n]
        if not selected:
            return {}
        weight = Decimal(1) / Decimal(len(selected))
        return {symbol: weight for _, symbol in selected}


@dataclass(frozen=True)
class MultiAssetPoint:
    session: date
    equity: Decimal
    benchmark_equity: Decimal
    cash: Decimal
    weights: dict[str, Decimal]
    turnover: Decimal
    cost: Decimal


@dataclass(frozen=True)
class MultiAssetResult:
    strategy: str
    points: tuple[MultiAssetPoint, ...]
    metrics: dict[str, float]
    transactions: tuple[Transaction, ...] = ()


@dataclass(frozen=True)
class Transaction:
    session: date
    symbol: str
    kind: str
    quantity: Decimal
    price: Decimal
    cash_amount: Decimal
    cost: Decimal = Decimal(0)


class MultiAssetBacktester:
    def __init__(
        self, starting_cash: Decimal = Decimal(100_000), cost_bps: Decimal = Decimal(5),
        rebalance_every: int = 21,
    ):
        if starting_cash <= 0 or cost_bps < 0 or rebalance_every <= 0:
            raise ValueError("invalid multi-asset backtest assumptions")
        self.starting_cash = starting_cash
        self.cost_bps = cost_bps
        self.rebalance_every = rebalance_every

    def run(
        self, universe: dict[str, tuple[DailyBar, ...]], benchmark: tuple[DailyBar, ...],
        strategy: CrossSectionalStrategy,
    ) -> MultiAssetResult:
        sessions, indexed, benchmark_index = self._align(universe, benchmark)
        if len(sessions) < 2:
            raise ValueError("at least two common sessions are required")
        cash = self.starting_cash
        shares = {symbol: Decimal(0) for symbol in sorted(universe)}
        pending_targets: dict[str, Decimal] | None = None
        histories: dict[str, list[DailyBar]] = {symbol: [] for symbol in sorted(universe)}
        benchmark_shares = self.starting_cash / benchmark_index[sessions[0]].open
        benchmark_cash = Decimal(0)
        points, transactions = [], []
        for index, session in enumerate(sessions):
            if index:
                for symbol, quantity in shares.items():
                    current = indexed[symbol][session]
                    shares[symbol] = quantity * effective_split_coefficient(current)
                    dividend = shares[symbol] * current.dividend
                    cash += dividend
                    if dividend:
                        transactions.append(Transaction(
                            session, symbol, "DIVIDEND", shares[symbol], current.dividend, dividend,
                        ))
                current_benchmark = benchmark_index[session]
                benchmark_shares *= effective_split_coefficient(current_benchmark)
                benchmark_cash += benchmark_shares * current_benchmark.dividend
            opens = {symbol: indexed[symbol][session].open for symbol in shares}
            turnover, cost = Decimal(0), Decimal(0)
            if pending_targets is not None:
                cash, shares, turnover, cost, trades = self._rebalance(
                    session, cash, shares, opens, pending_targets,
                )
                transactions.extend(trades)
            closes = {symbol: indexed[symbol][session].close for symbol in shares}
            equity = cash + sum((shares[symbol] * closes[symbol] for symbol in shares), Decimal(0))
            weights = {
                symbol: shares[symbol] * closes[symbol] / equity
                for symbol in shares if shares[symbol] and equity
            }
            points.append(MultiAssetPoint(
                session, equity, benchmark_cash + benchmark_shares * benchmark_index[session].close,
                cash, weights, turnover, cost,
            ))
            for symbol, values in histories.items():
                values.append(indexed[symbol][session])
            # Close-derived targets execute at the following session open.
            pending_targets = (
                strategy.targets({symbol: tuple(values) for symbol, values in histories.items()})
                if index % self.rebalance_every == 0 else None
            )
        metric_points = tuple(
            BacktestPoint(point.session, point.equity, point.benchmark_equity, Decimal(0), point.cost)
            for point in points
        )
        metrics = performance_metrics(metric_points)
        years = max((sessions[-1] - sessions[0]).days / 365.25, 1 / 252)
        metrics["annual_turnover"] = sum(float(point.turnover) for point in points) / years
        return MultiAssetResult(strategy.name, tuple(points), metrics, tuple(transactions))

    def run_window(
        self, universe: dict[str, tuple[DailyBar, ...]], benchmark: tuple[DailyBar, ...],
        strategy: CrossSectionalStrategy, evaluation_start: date,
    ) -> MultiAssetResult:
        """Start capital at evaluation_start while retaining earlier bars as feature warm-up."""
        if not universe:
            raise ValueError("universe cannot be empty")
        indexed = {symbol: {bar.session: bar for bar in bars} for symbol, bars in universe.items()}
        benchmark_index = {bar.session: bar for bar in benchmark}
        evaluation_sessions = tuple(sorted(
            session for session in benchmark_index if session >= evaluation_start
        ))
        if len(evaluation_sessions) < 2:
            raise ValueError("evaluation window requires at least two common sessions")
        incomplete = {
            symbol for symbol, values in indexed.items()
            if any(session not in values for session in evaluation_sessions)
        }
        if incomplete:
            raise ValueError(f"incomplete evaluation histories: {sorted(incomplete)}")
        histories = {
            symbol: [bar for bar in bars if bar.session < evaluation_start]
            for symbol in sorted(universe)
            for bars in (universe[symbol],)
        }
        cash = self.starting_cash
        shares = {symbol: Decimal(0) for symbol in sorted(universe)}
        pending_targets: dict[str, Decimal] | None = None
        first = evaluation_sessions[0]
        benchmark_shares = self.starting_cash / benchmark_index[first].open
        benchmark_cash = Decimal(0)
        points, transactions = [], []
        for index, session in enumerate(evaluation_sessions):
            if index:
                for symbol, quantity in shares.items():
                    current = indexed[symbol][session]
                    shares[symbol] = quantity * effective_split_coefficient(current)
                    dividend = shares[symbol] * current.dividend
                    cash += dividend
                    if dividend:
                        transactions.append(Transaction(
                            session, symbol, "DIVIDEND", shares[symbol], current.dividend, dividend,
                        ))
                current_benchmark = benchmark_index[session]
                benchmark_shares *= effective_split_coefficient(current_benchmark)
                benchmark_cash += benchmark_shares * current_benchmark.dividend
            opens = {symbol: indexed[symbol][session].open for symbol in shares}
            turnover, cost = Decimal(0), Decimal(0)
            if pending_targets is not None:
                cash, shares, turnover, cost, trades = self._rebalance(
                    session, cash, shares, opens, pending_targets,
                )
                transactions.extend(trades)
            closes = {symbol: indexed[symbol][session].close for symbol in shares}
            equity = cash + sum((shares[symbol] * closes[symbol] for symbol in shares), Decimal(0))
            weights = {
                symbol: shares[symbol] * closes[symbol] / equity
                for symbol in shares if shares[symbol] and equity
            }
            points.append(MultiAssetPoint(
                session, equity, benchmark_cash + benchmark_shares * benchmark_index[session].close,
                cash, weights, turnover, cost,
            ))
            for symbol, values in histories.items():
                values.append(indexed[symbol][session])
            pending_targets = (
                strategy.targets({symbol: tuple(values) for symbol, values in histories.items()})
                if index % self.rebalance_every == 0 else None
            )
        metric_points = tuple(
            BacktestPoint(point.session, point.equity, point.benchmark_equity, Decimal(0), point.cost)
            for point in points
        )
        metrics = performance_metrics(metric_points)
        years = max((evaluation_sessions[-1] - evaluation_sessions[0]).days / 365.25, 1 / 252)
        metrics["annual_turnover"] = sum(float(point.turnover) for point in points) / years
        return MultiAssetResult(strategy.name, tuple(points), metrics, tuple(transactions))

    def _rebalance(
        self, session: date, cash: Decimal, shares: dict[str, Decimal], opens: dict[str, Decimal],
        targets: dict[str, Decimal],
    ) -> tuple[Decimal, dict[str, Decimal], Decimal, Decimal, tuple[Transaction, ...]]:
        if any(weight < 0 for weight in targets.values()) or sum(targets.values()) > Decimal(1) + Decimal("1e-12"):
            raise ValueError("targets must be long-only and fully funded")
        equity = cash + sum((shares[symbol] * opens[symbol] for symbol in shares), Decimal(0))
        desired = {symbol: equity * targets.get(symbol, Decimal(0)) / opens[symbol] for symbol in shares}
        deltas = {symbol: desired[symbol] - shares[symbol] for symbol in shares}
        traded = sum((abs(delta) * opens[symbol] for symbol, delta in deltas.items()), Decimal(0))
        cost = traded * self.cost_bps / Decimal(10_000)
        # Execute sells first, then scale purchases to preserve non-negative cash after costs.
        executed: dict[str, Decimal] = {}
        for symbol, delta in deltas.items():
            if delta < 0:
                cash -= delta * opens[symbol]
                shares[symbol] += delta
                executed[symbol] = delta
        cash -= cost
        requested_buys = sum(
            (delta * opens[symbol] for symbol, delta in deltas.items() if delta > 0), Decimal(0)
        )
        scale = min(Decimal(1), max(Decimal(0), cash / requested_buys)) if requested_buys else Decimal(1)
        for symbol, delta in deltas.items():
            if delta > 0:
                filled = delta * scale
                cash -= filled * opens[symbol]
                shares[symbol] += filled
                executed[symbol] = filled
        actual_notional = sum((abs(qty) * opens[symbol] for symbol, qty in executed.items()), Decimal(0))
        transactions = tuple(Transaction(
            session=session, symbol=symbol, kind="BUY" if quantity > 0 else "SELL",
            quantity=abs(quantity), price=opens[symbol], cash_amount=abs(quantity) * opens[symbol],
            cost=cost * abs(quantity) * opens[symbol] / actual_notional if actual_notional else Decimal(0),
        ) for symbol, quantity in sorted(executed.items()) if quantity)
        return cash, shares, traded / equity if equity else Decimal(0), cost, transactions

    @staticmethod
    def _align(
        universe: dict[str, tuple[DailyBar, ...]], benchmark: tuple[DailyBar, ...]
    ) -> tuple[tuple[date, ...], dict[str, dict[date, DailyBar]], dict[date, DailyBar]]:
        if not universe:
            raise ValueError("universe cannot be empty")
        indexed = {symbol: {bar.session: bar for bar in bars} for symbol, bars in universe.items()}
        benchmark_index = {bar.session: bar for bar in benchmark}
        common = set(benchmark_index)
        for symbol_index in indexed.values():
            common &= set(symbol_index)
        return tuple(sorted(common)), indexed, benchmark_index
