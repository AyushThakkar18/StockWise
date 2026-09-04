from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter
from datetime import date
from decimal import Decimal
from pathlib import Path

from scipy.stats import spearmanr

from .config import Settings
from .council_backtest import CachedCouncilGate, position_attribution
from .historical_council_backtest import prepare_universe
from .kronos_forecast import KronosForecaster
from .point_in_time import PointInTimeBacktester
from .price_cache import PriceCache
from .reliable_strategy import ReliableCompositeStrategy
from .sec_edgar import SECEdgarCache
from .stock_scoring import PROMPT_VERSION, PerStockCouncilStrategy, StockAssessment
from .universe import load_membership_history


class CacheOnlyScorer:
    def __init__(self, model: str, directory: Path):
        self.model, self.directory = model, directory

    def __call__(self, packet: dict[str, object]) -> StockAssessment:
        encoded = json.dumps(packet, sort_keys=True, separators=(",", ":"))
        fingerprint = hashlib.sha256(
            f"{self.model}:{PROMPT_VERSION}:{encoded}".encode(),
        ).hexdigest()
        path = self.directory / f"{fingerprint}.json"
        if not path.exists():
            raise FileNotFoundError(f"missing cached stock score {fingerprint}")
        return StockAssessment.model_validate_json(path.read_text(encoding="utf-8"))


class CacheOnlyCouncil:
    def evaluate(self, *args, **kwargs):
        raise RuntimeError("diagnostic attempted an uncached council call")


def realized_return(bars, decision_on: date) -> float | None:
    following = [bar for bar in bars if bar.session > decision_on]
    if len(following) <= 21:
        return None
    return float(following[21].open / following[0].open - 1)


def main() -> None:
    start, evaluation_start, end = date(2024, 1, 1), date(2025, 1, 2), date(2025, 12, 3)
    history = load_membership_history(Path("private_data/universe/sp500-components-updated.csv"))
    histories, benchmark, filtered, _, _ = prepare_universe(
        history, PriceCache(Path("private_data/prices-yahoo")), start, end, evaluation_start,
    )
    settings = Settings.from_env()
    sec = SECEdgarCache(settings.sec_user_agent or "StockWise diagnostic", Path("private_data/sec-edgar"))
    base = ReliableCompositeStrategy(
        benchmark, sec, sec.ticker_map(), top_n=100, buffer_rank=120,
        maximum_position_weight=Decimal("0.01"), maximum_sector_weight=Decimal(1),
        risk_off_invested_weight=Decimal(1),
    )
    gate = CachedCouncilGate(
        CacheOnlyCouncil(), histories, sec, Path("private_data/council-pit-2021-2025"), 4,
        factor_audits=base.audits, factor_version="top100-comparative-factors-v1",
    )
    scoring_model = settings.openai_scoring_model or settings.openai_selection_model or settings.openai_model
    strategy = PerStockCouncilStrategy(
        base, gate, CacheOnlyScorer(scoring_model, Path("private_data/stock-scores-v1")),
        forecaster=KronosForecaster(
            Path("private_data/Kronos"), Path("private_data/kronos-base"),
            model_id="NeoQuasar/Kronos-base", device="cuda",
        ),
        benchmark=benchmark, workers=4,
    )
    result = PointInTimeBacktester(cost_bps=Decimal(5), rebalance_every=21).run(
        histories, benchmark, filtered, strategy, evaluation_start,
    )

    records = []
    route_counts, blocker_counts, selected_routes = Counter(), Counter(), Counter()
    for decision_on, audit in strategy.selection_audits.items():
        benchmark_forecast = strategy.forecaster(
            tuple(bar for bar in benchmark if bar.session <= decision_on)
        )
        for item in audit["ranked"]:
            symbol = item["symbol"]
            council = gate.decisions[f"{decision_on}:{symbol}"]
            route_counts[council.route] += 1
            blocker_counts.update(council.council.audit.blocker_codes)
            selected = symbol in strategy.target_history[decision_on]
            if selected:
                selected_routes[council.route] += 1
            forecast = item["kronos"]
            records.append({
                "date": decision_on.isoformat(), "symbol": symbol,
                "rating": item["rating"], "selected": selected,
                "route": council.route, "deterministic_rank": item["deterministic_rank"],
                "kronos_score": (
                    float(forecast["median_return"]) - benchmark_forecast.median_return
                    - .25 * float(forecast["forecast_dispersion"])
                    + .10 * float(forecast["predicted_max_drawdown"])
                ),
                "realized_return": realized_return(histories[symbol], decision_on),
            })
    usable = [item for item in records if item["realized_return"] is not None]
    rating_correlation = spearmanr(
        [item["rating"] for item in usable], [item["realized_return"] for item in usable],
    ).statistic
    kronos_correlation = spearmanr(
        [item["kronos_score"] for item in usable], [item["realized_return"] for item in usable],
    ).statistic
    deterministic_correlation = spearmanr(
        [-item["deterministic_rank"] for item in usable],
        [item["realized_return"] for item in usable],
    ).statistic
    selected = [item for item in usable if item["selected"]]
    rejected = [item for item in usable if not item["selected"]]
    monthly_spreads = []
    for day in sorted({item["date"] for item in usable}):
        month = [item for item in usable if item["date"] == day]
        chosen = [item["realized_return"] for item in month if item["selected"]]
        others = [item["realized_return"] for item in month if not item["selected"]]
        monthly_spreads.append(sum(chosen) / len(chosen) - sum(others) / len(others))
    payload = {
        "window": [evaluation_start.isoformat(), end.isoformat()],
        "candidate_observations": len(usable),
        "llm_rating_spearman_to_next_period_return": rating_correlation,
        "kronos_score_spearman_to_next_period_return": kronos_correlation,
        "deterministic_rank_spearman_to_next_period_return": deterministic_correlation,
        "llm_rating_distribution": {
            "minimum": min(item["rating"] for item in usable),
            "maximum": max(item["rating"] for item in usable),
            "mean": statistics.mean(item["rating"] for item in usable),
            "standard_deviation": statistics.pstdev(item["rating"] for item in usable),
            "selected_mean": statistics.mean(item["rating"] for item in selected),
            "unselected_mean": statistics.mean(item["rating"] for item in rejected),
        },
        "selected_average_next_period_return": sum(item["realized_return"] for item in selected) / len(selected),
        "unselected_average_next_period_return": sum(item["realized_return"] for item in rejected) / len(rejected),
        "average_monthly_selection_spread": sum(monthly_spreads) / len(monthly_spreads),
        "positive_monthly_selection_spreads": sum(value > 0 for value in monthly_spreads),
        "monthly_selection_spreads": monthly_spreads,
        "all_candidate_routes": route_counts,
        "selected_routes": selected_routes,
        "audit_blocker_codes": blocker_counts,
        "portfolio_metrics": result.metrics,
        "position_attribution": position_attribution(result),
    }
    output = Path("private_data/results/council-diagnostic-2025.json")
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
