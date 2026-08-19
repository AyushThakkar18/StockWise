from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from itertools import pairwise
from statistics import fmean, stdev
from typing import Protocol

from .market_data import DailyBar


class Strategy(Protocol):
    name: str

    def target(self, history: tuple[DailyBar, ...]) -> Decimal: ...


@dataclass(frozen=True)
class BuyAndHold:
    name: str = "buy_and_hold"

    def target(self, history: tuple[DailyBar, ...]) -> Decimal:
        return Decimal(1) if history else Decimal(0)


@dataclass(frozen=True)
class Cash:
    name: str = "cash"

    def target(self, history: tuple[DailyBar, ...]) -> Decimal:
        return Decimal(0)


@dataclass(frozen=True)
class Momentum:
    lookback: int = 126
    name: str = "momentum"

    def target(self, history: tuple[DailyBar, ...]) -> Decimal:
        if len(history) <= self.lookback:
            return Decimal(0)
        window = history[-1 - self.lookback:]
        return Decimal(1) if total_return_ratio(window) > 1 else Decimal(0)


@dataclass(frozen=True)
class BacktestPoint:
    session: date
    equity: Decimal
    benchmark_equity: Decimal
    target: Decimal
    cost: Decimal


@dataclass(frozen=True)
class BacktestResult:
    strategy: str
    points: tuple[BacktestPoint, ...]
    metrics: dict[str, float]


class Backtester:
    """Close-to-next-open event loop for one liquid instrument and its benchmark."""

    def __init__(self, starting_cash: Decimal = Decimal(100_000), cost_bps: Decimal = Decimal(5)):
        if starting_cash <= 0 or cost_bps < 0:
            raise ValueError("invalid backtest assumptions")
        self.starting_cash = starting_cash
        self.cost_bps = cost_bps

    def run(self, bars: tuple[DailyBar, ...], benchmark: tuple[DailyBar, ...], strategy: Strategy) -> BacktestResult:
        aligned = self._align(bars, benchmark)
        if len(aligned) < 2:
            raise ValueError("at least two aligned sessions are required")
        cash, shares, target = self.starting_cash, Decimal(0), Decimal(0)
        benchmark_shares = self.starting_cash / aligned[0][1].open
        benchmark_cash = Decimal(0)
        points: list[BacktestPoint] = []
        history: list[DailyBar] = []
        for index, (bar, bench) in enumerate(aligned):
            cost = Decimal(0)
            if index:
                shares *= bar.split_coefficient
                cash += shares * bar.dividend
                benchmark_shares *= bench.split_coefficient
                benchmark_cash += benchmark_shares * bench.dividend
            if index:
                equity_at_open = cash + shares * bar.open
                desired_shares = equity_at_open * target / bar.open
                trade_value = abs(desired_shares - shares) * bar.open
                cost = trade_value * self.cost_bps / Decimal(10_000)
                if desired_shares > shares:
                    affordable = max(Decimal(0), (cash - cost) / bar.open)
                    desired_shares = min(desired_shares, shares + affordable)
                cash -= (desired_shares - shares) * bar.open + cost
                shares = desired_shares
            equity = cash + shares * bar.close
            points.append(BacktestPoint(
                bar.session, equity, benchmark_cash + benchmark_shares * bench.close, target, cost
            ))
            history.append(bar)
            # Signal is computed after this close and can only trade at the next open.
            target = strategy.target(tuple(history))
        return BacktestResult(strategy.name, tuple(points), performance_metrics(tuple(points)))

    @staticmethod
    def _align(bars: tuple[DailyBar, ...], benchmark: tuple[DailyBar, ...]) -> list[tuple[DailyBar, DailyBar]]:
        left = {bar.session: bar for bar in bars}
        right = {bar.session: bar for bar in benchmark}
        return [(left[day], right[day]) for day in sorted(left.keys() & right.keys())]


def performance_metrics(points: tuple[BacktestPoint, ...]) -> dict[str, float]:
    equity = [float(point.equity) for point in points]
    benchmark = [float(point.benchmark_equity) for point in points]
    returns = [equity[i] / equity[i - 1] - 1 for i in range(1, len(equity))]
    bench_returns = [benchmark[i] / benchmark[i - 1] - 1 for i in range(1, len(benchmark))]
    years = max((points[-1].session - points[0].session).days / 365.25, 1 / 252)
    total = equity[-1] / equity[0] - 1
    benchmark_total = benchmark[-1] / benchmark[0] - 1
    annualized = (equity[-1] / equity[0]) ** (1 / years) - 1
    volatility = stdev(returns) * math.sqrt(252) if len(returns) > 1 else 0.0
    sharpe = fmean(returns) / stdev(returns) * math.sqrt(252) if len(returns) > 1 and stdev(returns) else 0.0
    downside = [min(value, 0) for value in returns]
    downside_deviation = math.sqrt(fmean([value * value for value in downside])) * math.sqrt(252) if downside else 0.0
    sortino = fmean(returns) * 252 / downside_deviation if downside_deviation else 0.0
    peak, max_drawdown = equity[0], 0.0
    for value in equity:
        peak = max(peak, value)
        max_drawdown = min(max_drawdown, value / peak - 1)
    active = [left - right for left, right in zip(returns, bench_returns, strict=True)]
    up_pairs = [(left, right) for left, right in zip(returns, bench_returns, strict=True) if right > 0]
    down_pairs = [(left, right) for left, right in zip(returns, bench_returns, strict=True) if right < 0]
    tracking_error = stdev(active) * math.sqrt(252) if len(active) > 1 else 0.0
    information_ratio = fmean(active) / stdev(active) * math.sqrt(252) if len(active) > 1 and stdev(active) else 0.0
    beta = _beta(returns, bench_returns)
    alpha = (fmean(returns) - beta * fmean(bench_returns)) * 252 if returns else 0.0
    ordered_returns = sorted(returns)
    tail_count = max(1, math.ceil(len(ordered_returns) * 0.05)) if ordered_returns else 0
    value_at_risk = -ordered_returns[tail_count - 1] if tail_count else 0.0
    expected_shortfall = -fmean(ordered_returns[:tail_count]) if tail_count else 0.0
    costs = sum(float(point.cost) for point in points)
    return {
        "total_return": total, "annualized_return": annualized,
        "benchmark_total_return": benchmark_total, "excess_return": total - benchmark_total,
        "volatility": volatility, "sharpe": sharpe, "sortino": sortino,
        "max_drawdown": max_drawdown,
        "calmar": annualized / abs(max_drawdown) if max_drawdown else 0.0,
        "tracking_error": tracking_error, "information_ratio": information_ratio,
        "alpha": alpha, "beta": beta, "value_at_risk_95": value_at_risk,
        "expected_shortfall_95": expected_shortfall,
        "daily_benchmark_win_rate": sum(value > 0 for value in active) / len(active) if active else 0.0,
        "upside_capture": _capture(up_pairs), "downside_capture": _capture(down_pairs),
        "trading_costs": costs, "decision_dates": float(len(points) - 1),
        **period_hit_metrics(points),
    }


def period_hit_metrics(
    points: tuple[BacktestPoint, ...], horizon: int = 21,
) -> dict[str, float]:
    """Non-overlapping holding-period outcomes; not per-stock forecast accuracy."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    periods = [
        (points[start], points[min(start + horizon, len(points) - 1)])
        for start in range(0, len(points) - 1, horizon)
        if min(start + horizon, len(points) - 1) > start
    ]
    strategy_returns = [float(end.equity / start.equity - 1) for start, end in periods]
    benchmark_returns = [
        float(end.benchmark_equity / start.benchmark_equity - 1) for start, end in periods
    ]
    return {
        f"positive_{horizon}_session_rate": (
            sum(value > 0 for value in strategy_returns) / len(periods) if periods else 0.0
        ),
        f"benchmark_win_{horizon}_session_rate": (
            sum(left > right for left, right in zip(strategy_returns, benchmark_returns, strict=True))
            / len(periods) if periods else 0.0
        ),
        f"evaluated_{horizon}_session_periods": float(len(periods)),
        f"positive_{horizon}_session_periods": float(sum(value > 0 for value in strategy_returns)),
        f"benchmark_wins_{horizon}_session_periods": float(sum(
            left > right for left, right in zip(strategy_returns, benchmark_returns, strict=True)
        )),
    }


def _capture(pairs: list[tuple[float, float]]) -> float:
    benchmark_mean = fmean(right for _, right in pairs) if pairs else 0.0
    return fmean(left for left, _ in pairs) / benchmark_mean if benchmark_mean else 0.0


def _beta(returns: list[float], benchmark_returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0
    benchmark_mean = fmean(benchmark_returns)
    denominator = sum((value - benchmark_mean) ** 2 for value in benchmark_returns)
    if denominator == 0:
        return 0.0
    return sum(
        (value - fmean(returns)) * (benchmark - benchmark_mean)
        for value, benchmark in zip(returns, benchmark_returns, strict=True)
    ) / denominator


def total_return_ratio(bars: tuple[DailyBar, ...]) -> Decimal:
    if len(bars) < 2:
        return Decimal(1)
    growth = Decimal(1)
    for previous, current in pairwise(bars):
        growth *= current.split_coefficient * (current.close + current.dividend) / previous.close
    return growth
