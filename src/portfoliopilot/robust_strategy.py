from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

from .fundamental_features import build_fundamental_features
from .market_data import DailyBar
from .reliable_strategy import _mean, _percentile_ranks, _volatility
from .sec_edgar import SECEdgarCache

DEFAULT_WEIGHTS = {
    "momentum_12_1": .20,
    "momentum_6_1": .15,
    "relative_strength": .10,
    "trend": .10,
    "low_volatility": .10,
    "downside_risk": .05,
    "growth": .12,
    "quality": .18,
}

SIMPLE_MTG_WEIGHTS = {
    "momentum_12_1": .35,
    "momentum_6_1": .25,
    "trend": .25,
    "growth": .15,
}


@dataclass
class RobustMultifactorStrategy:
    """Point-in-time multifactor selection with risk-budgeted portfolio construction."""

    benchmark: tuple[DailyBar, ...]
    sec: SECEdgarCache
    cik_by_symbol: dict[str, int]
    top_n: int = 20
    buffer_rank: int = 30
    maximum_position_weight: Decimal = Decimal("0.08")
    maximum_sector_weight: Decimal = Decimal("0.25")
    target_volatility: float = .15
    minimum_dollar_volume: float = 5_000_000
    fundamental_universe_size: int = 150
    correlation_penalty: float = .08
    retention_bonus: float = 0.0
    weighting_method: str = "inverse_volatility"
    use_fundamentals: bool = True
    risk_aware: bool = True
    factor_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    name: str = "robust_point_in_time_multifactor"
    previous: tuple[str, ...] = ()
    audits: dict[date, dict[str, object]] = field(default_factory=dict)

    def targets(self, history: dict[str, tuple[DailyBar, ...]]) -> dict[str, Decimal]:
        decision_on = max(bars[-1].session for bars in history.values() if bars)
        market = tuple(bar for bar in self.benchmark if bar.session <= decision_on)
        raw = self._raw_features(history, market, decision_on)
        if not raw:
            return {}
        price_names = tuple(name for name in DEFAULT_WEIGHTS if name not in {"growth", "quality"})
        preliminary_percentiles = {
            factor: _percentile_ranks(raw, factor) for factor in price_names
        }
        preliminary = sorted(raw, key=lambda symbol: (-sum(
            preliminary_percentiles[factor][symbol] for factor in price_names
        ), symbol))
        retained = set(preliminary[:self.fundamental_universe_size]) | set(self.previous)
        raw = {symbol: raw[symbol] for symbol in retained if symbol in raw}
        self._add_fundamentals(raw, decision_on)
        active_factors = tuple(
            factor for factor, weight in self.factor_weights.items()
            if weight and (self.use_fundamentals or factor not in {"growth", "quality"})
        )
        percentiles = {factor: _percentile_ranks(raw, factor) for factor in active_factors}
        total_weight = sum(self.factor_weights[factor] for factor in active_factors)
        scores = {
            symbol: sum(
                self.factor_weights[factor] * percentiles[factor][symbol]
                for factor in active_factors
            ) / total_weight * (.80 + .20 * float(raw[symbol]["completeness"]))
            for symbol in raw
        }
        for symbol, score in scores.items():
            raw[symbol]["composite_score"] = score
        ranked = sorted(scores, key=lambda symbol: (-scores[symbol], symbol))
        rank = {symbol: index for index, symbol in enumerate(ranked, 1)}
        buffered = [symbol for symbol in self.previous if rank.get(symbol, math.inf) <= self.buffer_rank]
        pool = buffered + [symbol for symbol in ranked if symbol not in buffered]
        selected = self._diversified_selection(pool, scores, history)
        exposure, regime = self._exposure(market)
        targets = self._risk_weights(selected, raw, exposure)
        self.previous = tuple(targets)
        audit_symbols = tuple(dict.fromkeys((*ranked[:self.buffer_rank], *targets)))
        self.audits[decision_on] = {
            "market_regime": regime,
            "invested_target": float(sum(targets.values(), Decimal(0))),
            "selected": list(targets),
            "targets": {symbol: float(weight) for symbol, weight in targets.items()},
            "ranked": [{
                "rank": rank[symbol], "symbol": symbol, "score": scores[symbol],
                "sector": raw[symbol]["sector"],
                "fundamental_completeness": raw[symbol]["completeness"],
                "volatility": raw[symbol]["volatility"],
                **{factor: raw[symbol].get(factor) for factor in DEFAULT_WEIGHTS},
            } for symbol in audit_symbols],
        }
        return targets

    def _raw_features(self, history, market, decision_on):
        output = {}
        market_12_1 = _skip_month_return(market, 252)
        for symbol, bars in history.items():
            if len(bars) < 253:
                continue
            cik = self.cik_by_symbol.get(symbol.replace(".", "-"))
            if cik is None:
                continue
            returns = _returns(bars[-253:])
            dollar_volume = sum(float(bar.close) * bar.volume for bar in bars[-21:]) / 21
            if dollar_volume < self.minimum_dollar_volume:
                continue
            negative = [min(value, 0.0) for value in returns[-63:]]
            output[symbol] = {
                "momentum_12_1": _skip_month_return(bars, 252),
                "momentum_6_1": _skip_month_return(bars, 126),
                "relative_strength": _skip_month_return(bars, 252) - market_12_1,
                "trend": float(bars[-1].adjusted_close) / _average(bars[-200:]) - 1,
                "low_volatility": -_volatility(returns[-63:]),
                "downside_risk": -math.sqrt(sum(value * value for value in negative) / 63 * 252),
                "volatility": _volatility(returns[-63:]),
                "dollar_volume": dollar_volume,
                "cik": cik,
            }
        return output

    def _add_fundamentals(self, raw, decision_on):
        for symbol, values in raw.items():
            fundamentals = build_fundamental_features(
                self.sec.company_facts(symbol, int(values["cik"])), decision_on,
            )
            values.update({
                "growth": fundamentals.revenue_growth,
                "quality": _mean([
                    fundamentals.net_margin, fundamentals.operating_margin,
                    fundamentals.return_on_assets,
                    -fundamentals.liabilities_to_assets
                    if fundamentals.liabilities_to_assets is not None else None,
                    fundamentals.cash_to_assets,
                ]),
                "completeness": fundamentals.completeness if self.use_fundamentals else 1.0,
                "sector": self.sec.sector(symbol, int(values["cik"])),
            })

    def _diversified_selection(self, pool, scores, history):
        selected: list[str] = []
        candidates = pool[:max(self.buffer_rank, self.top_n)]
        if self.correlation_penalty == 0:
            return sorted(
                candidates,
                key=lambda symbol: (
                    -(scores[symbol] + (self.retention_bonus if symbol in self.previous else 0.0)),
                    symbol,
                ),
            )[:self.top_n]
        while candidates and len(selected) < self.top_n:
            def adjusted(symbol):
                correlation = max(
                    (_correlation(history[symbol], history[held]) for held in selected),
                    default=0.0,
                )
                retention = self.retention_bonus if symbol in self.previous else 0.0
                return scores[symbol] + retention - self.correlation_penalty * max(0.0, correlation)
            winner = max(candidates, key=lambda symbol: (adjusted(symbol), scores[symbol], symbol))
            selected.append(winner)
            candidates.remove(winner)
        return selected

    def _exposure(self, market):
        returns = _returns(market[-253:])
        volatility = _volatility(returns[-63:])
        above_trend = float(market[-1].adjusted_close) > _average(market[-200:])
        drawdown = float(market[-1].adjusted_close) / max(
            float(bar.adjusted_close) for bar in market[-126:]
        ) - 1
        regime_limit = 1.0 if above_trend and volatility < .25 else .75
        if not above_trend and (volatility >= .25 or drawdown <= -.10):
            regime_limit = .50
        volatility_limit = min(1.0, self.target_volatility / max(volatility, .01))
        exposure = min(regime_limit, volatility_limit) if self.risk_aware else 1.0
        regime = "RISK_ON" if exposure >= .95 else "CAUTIOUS" if exposure > .50 else "RISK_OFF"
        return Decimal(str(round(exposure, 8))), regime

    def _risk_weights(self, selected, raw, exposure):
        if not selected:
            return {}
        if self.weighting_method not in {
            "equal", "score", "mild_volatility", "inverse_volatility",
        }:
            raise ValueError(f"unknown weighting method {self.weighting_method}")
        minimum_score = min(float(raw[symbol]["composite_score"]) for symbol in selected)
        budgets = {}
        for symbol in selected:
            volatility = max(float(raw[symbol]["volatility"]), .05)
            if self.weighting_method == "score":
                budget = max(.05, float(raw[symbol]["composite_score"]) - minimum_score + .05)
            elif self.weighting_method == "mild_volatility":
                budget = 1 / math.sqrt(volatility)
            elif self.weighting_method == "inverse_volatility":
                budget = 1 / volatility
            else:
                budget = 1.0
            budgets[symbol] = budget
        targets: dict[str, Decimal] = {}
        sectors: dict[str, Decimal] = defaultdict(Decimal)
        remaining = set(selected)
        for _ in range(len(selected) + 1):
            room = exposure - sum(targets.values(), Decimal(0))
            if room <= Decimal("0.000001") or not remaining:
                break
            budget_total = sum(budgets[symbol] for symbol in remaining)
            progressed = False
            for symbol in tuple(sorted(remaining)):
                sector = str(raw[symbol]["sector"])
                proposed = room * Decimal(str(budgets[symbol] / budget_total))
                capacity = min(
                    self.maximum_position_weight - targets.get(symbol, Decimal(0)),
                    self.maximum_sector_weight - sectors[sector],
                )
                addition = max(Decimal(0), min(proposed, capacity))
                if addition:
                    targets[symbol] = targets.get(symbol, Decimal(0)) + addition
                    sectors[sector] += addition
                    progressed = True
                if capacity <= proposed + Decimal("0.000001"):
                    remaining.remove(symbol)
            if not progressed:
                break
        ordered = {symbol: targets[symbol] for symbol in selected if targets.get(symbol, 0) > 0}
        total = sum(ordered.values(), Decimal(0))
        if total > exposure:
            scale = exposure / total
            ordered = {symbol: weight * scale for symbol, weight in ordered.items()}
        final_total = sum(ordered.values(), Decimal(0))
        if final_total > exposure:
            last = next(reversed(ordered))
            ordered[last] -= final_total - exposure
        return ordered


class SimpleMomentumTrendGrowthStrategy(RobustMultifactorStrategy):
    """Small, interpretable baseline; no LLM scoring or inverse-volatility tilts."""

    def __init__(self, benchmark, sec, cik_by_symbol, **overrides):
        parameters = {
            "factor_weights": dict(SIMPLE_MTG_WEIGHTS),
            "risk_aware": False,
            "correlation_penalty": 0.0,
            "weighting_method": "equal",
            "name": "simple_momentum_trend_growth",
        }
        parameters.update(overrides)
        super().__init__(benchmark, sec, cik_by_symbol, **parameters)


def _skip_month_return(bars, lookback: int) -> float:
    return float(bars[-22].adjusted_close / bars[-1 - lookback].adjusted_close - 1)


def _returns(bars) -> list[float]:
    return [
        float(bars[index].adjusted_close / bars[index - 1].adjusted_close - 1)
        for index in range(1, len(bars))
    ]


def _average(bars) -> float:
    return sum(float(bar.adjusted_close) for bar in bars) / len(bars)


def _correlation(left, right, sessions: int = 63) -> float:
    left_by_date = {bar.session: bar for bar in left[-sessions - 10:]}
    right_by_date = {bar.session: bar for bar in right[-sessions - 10:]}
    common = sorted(set(left_by_date) & set(right_by_date))[-sessions - 1:]
    if len(common) < 21:
        return 0.0
    a = [float(left_by_date[day].adjusted_close) for day in common]
    b = [float(right_by_date[day].adjusted_close) for day in common]
    ar = [a[index] / a[index - 1] - 1 for index in range(1, len(a))]
    br = [b[index] / b[index - 1] - 1 for index in range(1, len(b))]
    am, bm = sum(ar) / len(ar), sum(br) / len(br)
    covariance = sum((x - am) * (y - bm) for x, y in zip(ar, br, strict=True))
    denominator = math.sqrt(sum((x - am) ** 2 for x in ar) * sum((y - bm) ** 2 for y in br))
    return covariance / denominator if denominator else 0.0
