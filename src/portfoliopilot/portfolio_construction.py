from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from sklearn.covariance import LedoitWolf

from .market_data import DailyBar


@dataclass(frozen=True)
class CorrelationEstimate:
    """Point-in-time return correlations used only for portfolio-fit tie-breaking."""

    correlations: dict[tuple[str, str], float]
    observations: int
    method: str = "LEDOIT_WOLF_SHRUNK_CORRELATION"

    def between(self, left: str, right: str) -> float | None:
        if left == right:
            return 1.0
        return self.correlations.get(tuple(sorted((left, right))))


def estimate_shrunk_correlations(
    histories: dict[str, tuple[DailyBar, ...]],
    symbols: tuple[str, ...],
    *,
    sessions: int = 126,
    minimum_observations: int = 63,
) -> CorrelationEstimate | None:
    """Estimate a stable correlation matrix from period-aligned daily returns.

    The estimator is intentionally excluded from candidate quality scores. It is supplied to the
    council only to order candidates whose rounded research scores are equal.
    """
    ordered = tuple(dict.fromkeys(symbols))
    if len(ordered) < 2 or sessions < minimum_observations:
        return None
    returns = {symbol: _returns_by_session(histories.get(symbol, ())) for symbol in ordered}
    if any(not values for values in returns.values()):
        return None
    common = set.intersection(*(set(values) for values in returns.values()))
    dates = sorted(common)[-sessions:]
    if len(dates) < minimum_observations:
        return None
    matrix = np.asarray(
        [[returns[symbol][session] for symbol in ordered] for session in dates],
        dtype=float,
    )
    covariance = LedoitWolf().fit(matrix).covariance_
    standard_deviations = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    output: dict[tuple[str, str], float] = {}
    for left_index, left in enumerate(ordered):
        for right_index in range(left_index + 1, len(ordered)):
            right = ordered[right_index]
            denominator = standard_deviations[left_index] * standard_deviations[right_index]
            value = covariance[left_index, right_index] / denominator if denominator else 0.0
            output[(left, right) if left < right else (right, left)] = max(
                -1.0, min(1.0, float(value)),
            )
    return CorrelationEstimate(output, len(dates))


def pairwise_summary(
    symbols: tuple[str, ...], estimate: CorrelationEstimate | None,
) -> tuple[float | None, float | None]:
    if estimate is None or len(symbols) < 2:
        return None, None
    values = [
        estimate.between(left, right)
        for index, left in enumerate(symbols)
        for right in symbols[index + 1:]
    ]
    present = [value for value in values if value is not None and math.isfinite(value)]
    if not present:
        return None, None
    return sum(present) / len(present), max(present)


def _returns_by_session(bars: tuple[DailyBar, ...]) -> dict[object, float]:
    ordered = sorted(bars, key=lambda item: item.session)
    return {
        current.session: float(current.adjusted_close / previous.adjusted_close - 1)
        for previous, current in pairwise(ordered)
        if previous.adjusted_close > 0
    }
