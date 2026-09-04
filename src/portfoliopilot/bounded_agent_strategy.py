from __future__ import annotations

import math
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

from .bynara_agents import AgentFeature

ROLE_WEIGHTS = {"TECHNICAL": .40, "QUALITY": .25, "RISK": .35}


def bounded_agent_adjustment(features: dict[str, AgentFeature], limit: float = .08) -> float:
    if set(features) != set(ROLE_WEIGHTS):
        raise ValueError("all bounded agent roles are required")
    consensus = sum(
        ROLE_WEIGHTS[role] * feature.signal / 2 * feature.confidence
        for role, feature in features.items()
    )
    return max(-limit, min(limit, limit * consensus))


def feature_packet(symbol: str, bars, benchmark, factor: dict[str, object]) -> dict[str, object]:
    decision_on = bars[-1].session
    market = tuple(bar for bar in benchmark if bar.session <= decision_on)
    returns = [float(bars[index].adjusted_close / bars[index - 1].adjusted_close - 1)
               for index in range(1, len(bars))]
    market_returns = [float(market[index].adjusted_close / market[index - 1].adjusted_close - 1)
                      for index in range(1, len(market))]
    aligned = min(126, len(returns), len(market_returns))
    beta = _covariance(returns[-aligned:], market_returns[-aligned:]) / max(
        1e-12, _variance(market_returns[-aligned:]),
    )
    return {
        "identity": "anonymous_candidate",
        "as_of": decision_on.isoformat(),
        "horizon_sessions": 21,
        "deterministic_rank": factor["rank"],
        "deterministic_score": factor["score"],
        "factor_percentiles": {
            name: factor.get(name) for name in (
                "momentum", "relative_strength", "low_volatility", "growth", "quality",
            )
        },
        "fundamental_completeness": factor.get("fundamental_completeness"),
        "sector": factor["sector"],
        "price_features": {
            "return_21d": _return(bars, 21), "return_63d": _return(bars, 63),
            "return_126d": _return(bars, 126), "return_252d": _return(bars, 252),
            "relative_return_21d": _return(bars, 21) - _return(market, 21),
            "relative_return_126d": _return(bars, 126) - _return(market, 126),
            "annualized_volatility_63d": _volatility(returns[-63:]),
            "annualized_volatility_252d": _volatility(returns[-252:]),
            "maximum_drawdown_126d": _maximum_drawdown(bars[-127:]),
            "beta_126d": beta,
            "above_50d_average": float(bars[-1].adjusted_close) / _average_close(bars[-50:]) - 1,
            "above_200d_average": float(bars[-1].adjusted_close) / _average_close(bars[-200:]) - 1,
            "average_dollar_volume_21d": factor.get("dollar_volume"),
        },
        "market_regime": {
            "spy_return_21d": _return(market, 21), "spy_return_200d": _return(market, 200),
            "spy_above_200d_average": float(market[-1].adjusted_close) / _average_close(market[-200:]) - 1,
            "spy_volatility_63d": _volatility(market_returns[-63:]),
        },
        "instructions": "Assess only supplied, period-aligned deterministic measurements.",
    }


@dataclass
class BoundedMultiAgentStrategy:
    candidate_strategy: object
    agent: object
    benchmark: tuple
    candidate_count: int = 30
    position_count: int = 20
    locked_count: int = 10
    maximum_per_sector: int = 4
    adjustment_limit: float = .08
    workers: int = 2
    name: str = "bounded_multi_agent_features"
    target_history: dict[date, dict[str, Decimal]] = field(default_factory=dict)
    audits: dict[date, dict[str, object]] = field(default_factory=dict)

    def targets(self, history) -> dict[str, Decimal]:
        decision_on = max(bars[-1].session for bars in history.values() if bars)
        base_targets = self.candidate_strategy.targets(history)
        candidates = tuple(base_targets)[:self.candidate_count]
        if not 0 <= self.locked_count < self.position_count <= len(candidates):
            raise ValueError("invalid locked, position, or candidate count")
        locked = candidates[:self.locked_count]
        review_candidates = candidates[self.locked_count:]
        factors = {
            item["symbol"]: item for item in self.candidate_strategy.audits[decision_on]["ranked"]
        }
        packets = {
            symbol: feature_packet(symbol, history[symbol], self.benchmark, factors[symbol])
            for symbol in review_candidates
        }
        print(
            f"[{decision_on}] deterministic lock {len(locked)}; three bounded agents review "
            f"{len(review_candidates)} cutoff candidates", flush=True,
        )
        features = {symbol: {} for symbol in review_candidates}
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {
                pool.submit(self.agent.evaluate_cross_section, role, packets): role
                for role in ROLE_WEIGHTS
            }
            for future in as_completed(futures):
                role = futures[future]
                for symbol, value in future.result().items():
                    features[symbol][role] = value
                print(f"[{decision_on}] {role} cross-sectional agent complete", flush=True)
        reviewed = sorted((
            symbol,
            float(factors[symbol]["score"])
            + bounded_agent_adjustment(features[symbol], self.adjustment_limit),
        ) for symbol in review_candidates)
        reviewed.sort(key=lambda item: (-item[1], int(factors[item[0]]["rank"]), item[0]))
        ranked = [(symbol, float(factors[symbol]["score"])) for symbol in locked] + reviewed
        selected, sectors = [], Counter()
        for symbol, _ in ranked:
            sector = str(factors[symbol]["sector"])
            if sectors[sector] >= self.maximum_per_sector:
                continue
            selected.append(symbol)
            sectors[sector] += 1
            if len(selected) == self.position_count:
                break
        sector_cap_relaxed = False
        if len(selected) < self.position_count:
            sector_cap_relaxed = True
            selected.extend(
                symbol for symbol, _ in ranked
                if symbol not in selected
            )
            selected = selected[:self.position_count]
        if len(selected) != self.position_count:
            raise ValueError("unable to construct diversified Top 20")
        market = tuple(bar for bar in self.benchmark if bar.session <= decision_on)
        if hasattr(self.candidate_strategy, "_exposure") and hasattr(
            self.candidate_strategy, "_risk_weights",
        ):
            invested, regime = self.candidate_strategy._exposure(market)
            targets = self.candidate_strategy._risk_weights(selected, factors, invested)
        else:
            invested = Decimal(1) if _return(market, 200) > 0 else Decimal("0.60")
            regime = "RISK_ON" if invested == 1 else "RISK_OFF"
            targets = {symbol: invested / Decimal(self.position_count) for symbol in selected}
        self.target_history[decision_on] = targets
        self.audits[decision_on] = {
            "selected": selected, "invested_weight": float(invested),
            "market_regime": regime,
            "sector_cap_relaxed": sector_cap_relaxed,
            "ranked": [{
                "rank": index, "symbol": symbol, "final_score": score,
                "base_rank": factors[symbol]["rank"], "base_score": factors[symbol]["score"],
                "agent_adjustment": (
                    bounded_agent_adjustment(features[symbol], self.adjustment_limit)
                    if symbol in features else 0.0
                ),
                "decision_source": "DETERMINISTIC_LOCK" if symbol in locked else "AGENT_REVIEW",
                "sector": factors[symbol]["sector"],
                "agents": {role: value.model_dump(mode="json")
                           for role, value in features.get(symbol, {}).items()},
            } for index, (symbol, score) in enumerate(ranked, 1)],
        }
        print(f"[{decision_on}] bounded-agent Top 20: {', '.join(selected)}", flush=True)
        return targets


def _return(bars, lookback: int) -> float:
    return float(bars[-1].adjusted_close / bars[-1 - lookback].adjusted_close - 1)


def _average_close(bars) -> float:
    return sum(float(bar.adjusted_close) for bar in bars) / len(bars)


def _variance(values: list[float]) -> float:
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / max(1, len(values) - 1)


def _covariance(left: list[float], right: list[float]) -> float:
    left_mean, right_mean = sum(left) / len(left), sum(right) / len(right)
    return sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right, strict=True)) / max(
        1, len(left) - 1,
    )


def _volatility(values: list[float]) -> float:
    return math.sqrt(_variance(values) * 252)


def _maximum_drawdown(bars) -> float:
    peak, drawdown = 0.0, 0.0
    for bar in bars:
        close = float(bar.adjusted_close)
        peak = max(peak, close)
        drawdown = min(drawdown, close / peak - 1)
    return drawdown
