from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

from .fundamental_features import build_fundamental_features
from .market_data import DailyBar
from .sec_edgar import SECEdgarCache


@dataclass
class ReliableCompositeStrategy:
    benchmark: tuple[DailyBar, ...]
    sec: SECEdgarCache
    cik_by_symbol: dict[str, int]
    top_n: int = 20
    buffer_rank: int = 30
    maximum_position_weight: Decimal = Decimal("0.10")
    maximum_sector_weight: Decimal = Decimal("0.25")
    risk_off_invested_weight: Decimal = Decimal("0.60")
    fundamental_universe_size: int = 150
    minimum_dollar_volume: float = 5_000_000
    name: str = "reliable_growth_quality_composite"
    previous: tuple[str, ...] = ()
    audits: dict[date, dict[str, object]] = field(default_factory=dict)

    def targets(self, history: dict[str, tuple[DailyBar, ...]]) -> dict[str, Decimal]:
        decision_on = max(bars[-1].session for bars in history.values() if bars)
        raw = {}
        for symbol, bars in history.items():
            if len(bars) < 253:
                continue
            cik = self.cik_by_symbol.get(symbol.replace(".", "-"))
            if cik is None:
                continue
            returns = [float(bars[index].adjusted_close / bars[index - 1].adjusted_close - 1)
                       for index in range(len(bars) - 63, len(bars))]
            dollar_volume = sum(float(bar.close) * bar.volume for bar in bars[-21:]) / 21
            if dollar_volume < self.minimum_dollar_volume:
                continue
            raw[symbol] = {
                "momentum": _mean([
                    _return(bars, 63), _return(bars, 126), _return(bars, 252),
                ]),
                "relative_strength": _return(bars, 126) - self._benchmark_return(decision_on, 126),
                "low_volatility": -_volatility(returns),
                "cik": cik, "dollar_volume": dollar_volume,
            }
        price_factors = ("momentum", "relative_strength", "low_volatility")
        price_percentiles = {factor: _percentile_ranks(raw, factor) for factor in price_factors}
        preliminary = sorted(raw, key=lambda symbol: (-sum(
            price_percentiles[factor][symbol] for factor in price_factors
        ), symbol))
        retained = set(preliminary[:self.fundamental_universe_size]) | set(self.previous)
        raw = {symbol: raw[symbol] for symbol in retained if symbol in raw}
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
                "completeness": fundamentals.completeness,
                "sector": self.sec.sector(symbol, int(values["cik"])),
            })
        factors = (*price_factors, "growth", "quality")
        percentiles = {factor: _percentile_ranks(raw, factor) for factor in factors}
        scores = {
            symbol: (
                .30 * percentiles["momentum"][symbol]
                + .15 * percentiles["relative_strength"][symbol]
                + .10 * percentiles["low_volatility"][symbol]
                + .20 * percentiles["growth"][symbol]
                + .25 * percentiles["quality"][symbol]
            ) * (0.75 + 0.25 * raw[symbol]["completeness"])
            for symbol in raw
        }
        ranked = sorted(scores, key=lambda symbol: (-scores[symbol], symbol))
        rank = {symbol: index for index, symbol in enumerate(ranked, 1)}
        candidates = [
            symbol for symbol in self.previous if rank.get(symbol, math.inf) <= self.buffer_rank
        ]
        candidates.extend(symbol for symbol in ranked if symbol not in candidates)
        invested_limit = (
            Decimal(1) if self._benchmark_return(decision_on, 200) > 0
            else self.risk_off_invested_weight
        )
        equal_weight = invested_limit / Decimal(self.top_n)
        targets, sectors = {}, {}
        for symbol in candidates:
            sector = str(raw[symbol]["sector"])
            remaining = invested_limit - sum(targets.values(), Decimal(0))
            sector_room = self.maximum_sector_weight - sectors.get(sector, Decimal(0))
            weight = min(equal_weight, self.maximum_position_weight, sector_room, remaining)
            if weight > 0:
                targets[symbol] = weight
                sectors[sector] = sectors.get(sector, Decimal(0)) + weight
            if len(targets) >= self.top_n or remaining <= 0:
                break
        self.previous = tuple(targets)
        audit_symbols = tuple(dict.fromkeys((*ranked[:self.buffer_rank], *targets)))
        self.audits[decision_on] = {
            "market_regime": "RISK_ON" if invested_limit == 1 else "RISK_OFF",
            "invested_target": float(sum(targets.values(), Decimal(0))),
            "ranked": [{"rank": rank[symbol], "symbol": symbol, "score": scores[symbol],
                        "sector": raw[symbol]["sector"],
                        "fundamental_completeness": raw[symbol]["completeness"],
                        "dollar_volume": raw[symbol]["dollar_volume"], **{
                            factor: raw[symbol][factor] for factor in factors
                        }} for symbol in audit_symbols],
            "selected": list(targets),
        }
        return targets

    def _benchmark_return(self, decision_on: date, lookback: int) -> float:
        bars = tuple(bar for bar in self.benchmark if bar.session <= decision_on)
        return _return(bars, lookback) if len(bars) > lookback else 0.0


def _return(bars: tuple[DailyBar, ...], lookback: int) -> float:
    return float(bars[-1].adjusted_close / bars[-1 - lookback].adjusted_close - 1)


def _volatility(values: list[float]) -> float:
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / max(1, len(values) - 1)) * math.sqrt(252)


def _mean(values: list[float | None]) -> float | None:
    usable = [value for value in values if value is not None]
    return sum(usable) / len(usable) if usable else None


def _percentile_ranks(raw: dict[str, dict[str, object]], factor: str) -> dict[str, float]:
    usable = sorted(
        (float(values[factor]), symbol) for symbol, values in raw.items()
        if values[factor] is not None and math.isfinite(float(values[factor]))
    )
    ranks = {
        symbol: index / max(1, len(usable) - 1)
        for index, (_, symbol) in enumerate(usable)
    }
    return {symbol: ranks.get(symbol, 0.5) for symbol in raw}
