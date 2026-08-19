from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import date, datetime
from itertools import pairwise
from statistics import fmean, stdev

from .contracts import Quality
from .market_data import DailyBar

FEATURE_VERSION = "price-features-v1"


@dataclass(frozen=True)
class FeatureSnapshot:
    symbol: str
    as_of: date
    created_at: datetime
    version: str
    input_hash: str
    values: dict[str, float | None]


def build_price_features(
    bars: tuple[DailyBar, ...],
    benchmark: tuple[DailyBar, ...],
    decision_at: datetime,
) -> FeatureSnapshot:
    """Build features solely from records available by the decision timestamp."""
    eligible = tuple(
        bar for bar in bars
        if bar.available_to_strategy_at <= decision_at and bar.quality == Quality.PASS
    )
    bench_by_date = {
        bar.session: bar for bar in benchmark
        if bar.available_to_strategy_at <= decision_at and bar.quality == Quality.PASS
    }
    if len(eligible) < 2:
        raise ValueError("insufficient point-in-time price history")
    if any(left.session >= right.session for left, right in pairwise(eligible)):
        raise ValueError("bars must be strictly ordered and unique")
    returns = _returns(eligible)
    aligned_pairs = [
        (eligible[index - 1], eligible[index], bench_by_date.get(eligible[index - 1].session), bench_by_date.get(eligible[index].session))
        for index in range(1, len(eligible))
    ]
    paired_returns = [
        (float(current.close / previous.close - 1), float(bench_current.close / bench_previous.close - 1))
        for previous, current, bench_previous, bench_current in aligned_pairs
        if bench_previous is not None and bench_current is not None
    ]
    values: dict[str, float | None] = {
        "momentum_21d": _momentum(eligible, 21),
        "momentum_63d": _momentum(eligible, 63),
        "momentum_126d": _momentum(eligible, 126),
        "momentum_252d": _momentum(eligible, 252),
        "realized_volatility_21d": _annualized_volatility(returns[-21:]),
        "realized_volatility_63d": _annualized_volatility(returns[-63:]),
        "maximum_drawdown": _maximum_drawdown(eligible),
        "beta_63d": _beta(paired_returns[-63:]),
        "relative_strength_63d": _relative_strength(eligible, bench_by_date, 63),
        "average_dollar_volume_21d": fmean(
            float(bar.close * bar.volume) for bar in eligible[-21:]
        ),
    }
    canonical_inputs = [
        {
            "symbol": bar.symbol, "session": bar.session.isoformat(), "close": str(bar.close),
            "volume": bar.volume, "available": bar.available_to_strategy_at.isoformat(),
            "vintage": bar.vintage,
        }
        for bar in eligible
    ]
    digest = hashlib.sha256(
        json.dumps(canonical_inputs, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return FeatureSnapshot(
        symbol=eligible[-1].symbol, as_of=eligible[-1].session, created_at=decision_at,
        version=FEATURE_VERSION, input_hash=digest, values=values,
    )


def _returns(bars: tuple[DailyBar, ...]) -> list[float]:
    return [float(bars[index].close / bars[index - 1].close - 1) for index in range(1, len(bars))]


def _momentum(bars: tuple[DailyBar, ...], lookback: int) -> float | None:
    if len(bars) <= lookback:
        return None
    return float(bars[-1].close / bars[-1 - lookback].close - 1)


def _annualized_volatility(returns: list[float]) -> float | None:
    return stdev(returns) * math.sqrt(252) if len(returns) > 1 else None


def _maximum_drawdown(bars: tuple[DailyBar, ...]) -> float:
    peak = float(bars[0].close)
    drawdown = 0.0
    for bar in bars:
        value = float(bar.close)
        peak = max(peak, value)
        drawdown = min(drawdown, value / peak - 1)
    return drawdown


def _beta(pairs: list[tuple[float, float]]) -> float | None:
    if len(pairs) < 2:
        return None
    stocks, market = zip(*pairs, strict=True)
    market_mean = fmean(market)
    denominator = sum((value - market_mean) ** 2 for value in market)
    if denominator == 0:
        return None
    stock_mean = fmean(stocks)
    return sum(
        (stock - stock_mean) * (benchmark - market_mean)
        for stock, benchmark in pairs
    ) / denominator


def _relative_strength(
    bars: tuple[DailyBar, ...], benchmark_by_date: dict[date, DailyBar], lookback: int
) -> float | None:
    if len(bars) <= lookback:
        return None
    old, new = bars[-1 - lookback], bars[-1]
    bench_old, bench_new = benchmark_by_date.get(old.session), benchmark_by_date.get(new.session)
    if not bench_old or not bench_new:
        return None
    return float(new.close / old.close - bench_new.close / bench_old.close)
