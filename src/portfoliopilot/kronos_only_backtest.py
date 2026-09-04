from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from pathlib import Path

from .config import Settings
from .council_backtest import position_attribution
from .historical_council_backtest import prepare_universe
from .kronos_forecast import KronosForecast, KronosForecaster
from .point_in_time import PointInTimeBacktester
from .point_in_time_download import required_symbols
from .price_cache import PriceCache
from .reliable_strategy import ReliableCompositeStrategy
from .sec_edgar import SECEdgarCache
from .universe import load_membership_history


def forecast_score(candidate: KronosForecast, benchmark: KronosForecast) -> float:
    """Frozen, interpretable score for the next 21 sessions; higher is better."""
    excess = candidate.median_return - benchmark.median_return
    return excess - 0.25 * candidate.forecast_dispersion + 0.10 * candidate.predicted_max_drawdown


@dataclass
class KronosOnlyStrategy:
    candidate_strategy: object
    forecaster: object
    benchmark: tuple
    candidate_count: int = 100
    position_count: int = 20
    maximum_per_sector: int = 5
    name: str = "kronos_base_top100_to_top20"
    target_history: dict[date, dict[str, Decimal]] = field(default_factory=dict)
    audits: dict[date, dict[str, object]] = field(default_factory=dict)

    def targets(self, history) -> dict[str, Decimal]:
        decision_on = max(bars[-1].session for bars in history.values() if bars)
        candidates = tuple(self.candidate_strategy.targets(history))[:self.candidate_count]
        factors = {
            item["symbol"]: item for item in self.candidate_strategy.audits[decision_on]["ranked"]
        }
        benchmark_history = tuple(bar for bar in self.benchmark if bar.session <= decision_on)
        benchmark_forecast = self.forecaster(benchmark_history)
        ranked = []
        print(f"[{decision_on}] Kronos-only forecasts 0/{len(candidates)}", flush=True)
        for index, symbol in enumerate(candidates, 1):
            forecast = self.forecaster(history[symbol])
            ranked.append((symbol, forecast_score(forecast, benchmark_forecast), forecast))
            if index % 10 == 0:
                print(f"[{decision_on}] Kronos-only forecasts {index}/{len(candidates)}", flush=True)
        ranked.sort(key=lambda item: (-item[1], item[0]))
        selected, sectors = [], Counter()
        for symbol, _, _ in ranked:
            sector = str(factors[symbol]["sector"])
            if sectors[sector] >= self.maximum_per_sector:
                continue
            selected.append(symbol)
            sectors[sector] += 1
            if len(selected) == self.position_count:
                break
        if len(selected) != self.position_count:
            raise ValueError("sector constraints did not permit exactly 20 positions")
        targets = {symbol: Decimal(1) / Decimal(self.position_count) for symbol in selected}
        self.target_history[decision_on] = targets
        self.audits[decision_on] = {
            "selected": selected,
            "benchmark_forecast": benchmark_forecast.model_dump(mode="json"),
            "ranked": [{
                "rank": index, "symbol": symbol, "score": score,
                "sector": factors[symbol]["sector"],
                "selected": symbol in targets, "forecast": forecast.model_dump(mode="json"),
            } for index, (symbol, score, forecast) in enumerate(ranked, 1)],
        }
        print(f"[{decision_on}] Kronos-only Top 20: {', '.join(selected)}", flush=True)
        return targets


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Kronos-only 2025 baseline")
    parser.add_argument("--start", type=date.fromisoformat, default=date(2024, 1, 1))
    parser.add_argument("--evaluation-start", type=date.fromisoformat, default=date(2025, 1, 2))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2025, 12, 31))
    parser.add_argument("--membership", type=Path, default=Path("private_data/universe/sp500-components-updated.csv"))
    parser.add_argument("--prices", type=Path, default=Path("private_data/prices-yahoo"))
    parser.add_argument("--kronos-repository", type=Path, default=Path("private_data/Kronos"))
    parser.add_argument("--kronos-cache", type=Path, default=Path("private_data/kronos-base"))
    parser.add_argument("--kronos-device", default="cuda")
    parser.add_argument("--output", type=Path, default=Path("private_data/results/kronos-only-2025.json"))
    arguments = parser.parse_args()

    history = load_membership_history(arguments.membership)
    histories, benchmark, filtered, exclusions, coverage = prepare_universe(
        history, PriceCache(arguments.prices), arguments.start, arguments.end,
        arguments.evaluation_start,
    )
    settings = Settings.from_env()
    if not settings.sec_user_agent:
        parser.error("SEC_USER_AGENT is required")
    sec = SECEdgarCache(settings.sec_user_agent, Path("private_data/sec-edgar"))
    candidate_strategy = ReliableCompositeStrategy(
        benchmark, sec, sec.ticker_map(), top_n=100, buffer_rank=120,
        maximum_position_weight=Decimal("0.01"), maximum_sector_weight=Decimal(1),
        risk_off_invested_weight=Decimal(1),
    )
    strategy = KronosOnlyStrategy(
        candidate_strategy,
        KronosForecaster(
            arguments.kronos_repository, arguments.kronos_cache,
            model_id="NeoQuasar/Kronos-base", device=arguments.kronos_device,
        ),
        benchmark,
    )
    result = PointInTimeBacktester(cost_bps=Decimal(5), rebalance_every=21).run(
        histories, benchmark, filtered, strategy, arguments.evaluation_start,
    )
    payload = {
        "experiment": "kronos-base-only-top100-to-top20",
        "evaluation_start": arguments.evaluation_start.isoformat(),
        "evaluation_end": result.points[-1].session.isoformat(),
        "selection_rule": (
            "median candidate return minus SPY median return, minus 0.25x forecast dispersion, "
            "plus 0.10x predicted maximum drawdown"
        ),
        "universe_method": "point-in-time S&P 500 members; deterministic Top 100",
        "required_symbols": len(required_symbols(history, arguments.evaluation_start, arguments.end)),
        "excluded_symbols": exclusions, "price_coverage": coverage.coverage,
        "ending_equity": float(result.points[-1].equity), "metrics": result.metrics,
        "position_attribution": position_attribution(result),
        "monthly_rankings": {str(day): audit for day, audit in strategy.audits.items()},
        "limitations": [
            "Unavailable historical securities leave residual survivorship bias.",
            "Kronos checkpoint training cutoff is not documented precisely enough for a causal backtest claim.",
            "This diagnostic must not be interpreted as expected future profitability.",
        ],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "ending_equity": payload["ending_equity"], "metrics": payload["metrics"],
        "output": str(arguments.output),
    }, indent=2))


if __name__ == "__main__":
    main()
