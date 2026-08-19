from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from statistics import fmean

from .features import FeatureSnapshot
from .market_data import DailyBar


@dataclass(frozen=True)
class TrainingRow:
    symbol: str
    decision_date: date
    feature_version: str
    input_hash: str
    features: dict[str, float]
    forward_return: float
    benchmark_forward_return: float
    excess_return: float
    label_end_date: date


def build_training_rows(
    snapshots: tuple[FeatureSnapshot, ...],
    prices: dict[str, tuple[DailyBar, ...]],
    benchmark: tuple[DailyBar, ...],
    horizon_sessions: int,
    label_cutoff: date,
) -> tuple[TrainingRow, ...]:
    """Create labels for training only; label values never enter feature dictionaries."""
    if horizon_sessions <= 0:
        raise ValueError("horizon must be positive")
    benchmark_index = _index(benchmark)
    rows = []
    for snapshot in snapshots:
        symbol_index = _index(prices.get(snapshot.symbol, ()))
        if snapshot.as_of not in symbol_index or snapshot.as_of not in benchmark_index:
            continue
        symbol_dates, symbol_position = symbol_index[snapshot.as_of]
        benchmark_dates, benchmark_position = benchmark_index[snapshot.as_of]
        symbol_end = symbol_position + horizon_sessions
        benchmark_end = benchmark_position + horizon_sessions
        if symbol_end >= len(symbol_dates) or benchmark_end >= len(benchmark_dates):
            continue
        if symbol_dates[symbol_end] != benchmark_dates[benchmark_end]:
            continue
        label_end = symbol_dates[symbol_end]
        if label_end > label_cutoff:
            continue
        numeric = {key: value for key, value in snapshot.values.items() if value is not None}
        if len(numeric) != len(snapshot.values):
            continue
        symbol_return = _return(prices[snapshot.symbol], snapshot.as_of, symbol_dates[symbol_end])
        benchmark_return = _return(benchmark, snapshot.as_of, benchmark_dates[benchmark_end])
        rows.append(TrainingRow(
            symbol=snapshot.symbol, decision_date=snapshot.as_of,
            feature_version=snapshot.version, input_hash=snapshot.input_hash,
            features=numeric, forward_return=symbol_return,
            benchmark_forward_return=benchmark_return,
            excess_return=symbol_return - benchmark_return, label_end_date=label_end,
        ))
    return tuple(sorted(rows, key=lambda row: (row.decision_date, row.symbol)))


def date_demean_labels(rows: tuple[TrainingRow, ...]) -> tuple[float, ...]:
    """Cross-sectional labels are centered per decision date, not per security row."""
    means: dict[date, float] = {}
    for decision_date in {row.decision_date for row in rows}:
        means[decision_date] = fmean(
            row.excess_return for row in rows if row.decision_date == decision_date
        )
    return tuple(row.excess_return - means[row.decision_date] for row in rows)


def _index(bars: tuple[DailyBar, ...]) -> dict[date, tuple[tuple[date, ...], int]]:
    dates = tuple(bar.session for bar in bars)
    return {session: (dates, index) for index, session in enumerate(dates)}


def _return(bars: tuple[DailyBar, ...], start: date, end: date) -> float:
    by_date = {bar.session: bar.close for bar in bars}
    return float(by_date[end] / by_date[start] - 1)
