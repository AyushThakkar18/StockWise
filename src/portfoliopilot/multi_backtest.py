from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Protocol

from .backtest import BacktestPoint, performance_metrics, total_return_ratio
from .market_data import DailyBar


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
        points = []
        for index, session in enumerate(sessions):
            if index:
                for symbol, quantity in shares.items():
                    current = indexed[symbol][session]
                    shares[symbol] = quantity * current.split_coefficient
                    cash += shares[symbol] * current.dividend
                current_benchmark = benchmark_index[session]
                benchmark_shares *= current_benchmark.split_coefficient
                benchmark_cash += benchmark_shares * current_benchmark.dividend
            opens = {symbol: indexed[symbol][session].open for symbol in shares}
            turnover, cost = Decimal(0), Decimal(0)
            if pending_targets is not None:
                cash, shares, turnover, cost = self._rebalance(cash, shares, opens, pending_targets)
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
        return MultiAssetResult(strategy.name, tuple(points), metrics)

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
        points = []
        for index, session in enumerate(evaluation_sessions):
            if index:
                for symbol, quantity in shares.items():
                    current = indexed[symbol][session]
                    shares[symbol] = quantity * current.split_coefficient
                    cash += shares[symbol] * current.dividend
                current_benchmark = benchmark_index[session]
                benchmark_shares *= current_benchmark.split_coefficient
                benchmark_cash += benchmark_shares * current_benchmark.dividend
            opens = {symbol: indexed[symbol][session].open for symbol in shares}
            turnover, cost = Decimal(0), Decimal(0)
            if pending_targets is not None:
                cash, shares, turnover, cost = self._rebalance(
                    cash, shares, opens, pending_targets,
                )
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
        return MultiAssetResult(strategy.name, tuple(points), metrics)

    def _rebalance(
        self, cash: Decimal, shares: dict[str, Decimal], opens: dict[str, Decimal],
        targets: dict[str, Decimal],
    ) -> tuple[Decimal, dict[str, Decimal], Decimal, Decimal]:
        if any(weight < 0 for weight in targets.values()) or sum(targets.values()) > Decimal(1) + Decimal("1e-12"):
            raise ValueError("targets must be long-only and fully funded")
        equity = cash + sum((shares[symbol] * opens[symbol] for symbol in shares), Decimal(0))
        desired = {symbol: equity * targets.get(symbol, Decimal(0)) / opens[symbol] for symbol in shares}
        deltas = {symbol: desired[symbol] - shares[symbol] for symbol in shares}
        traded = sum((abs(delta) * opens[symbol] for symbol, delta in deltas.items()), Decimal(0))
        cost = traded * self.cost_bps / Decimal(10_000)
        # Execute sells first, then scale purchases to preserve non-negative cash after costs.
        for symbol, delta in deltas.items():
            if delta < 0:
                cash -= delta * opens[symbol]
                shares[symbol] += delta
        cash -= cost
        requested_buys = sum(
            (delta * opens[symbol] for symbol, delta in deltas.items() if delta > 0), Decimal(0)
        )
        scale = min(Decimal(1), max(Decimal(0), cash / requested_buys)) if requested_buys else Decimal(1)
        for symbol, delta in deltas.items():
            if delta > 0:
                executed = delta * scale
                cash -= executed * opens[symbol]
                shares[symbol] += executed
        return cash, shares, traded / equity if equity else Decimal(0), cost

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
