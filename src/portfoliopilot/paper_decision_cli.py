from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

from .config import Settings
from .core_satellite import CoreSatelliteStrategy
from .paper_store import PaperTradingStore, StrategyVersion, canonical_hash
from .price_cache import PriceCache
from .robust_strategy import SIMPLE_MTG_WEIGHTS, SimpleMomentumTrendGrowthStrategy
from .sec_edgar import SECEdgarCache
from .universe import load_membership_history


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze a prospective monthly paper decision")
    parser.add_argument("--as-of", type=date.fromisoformat, required=True)
    parser.add_argument("--execution-date", type=date.fromisoformat, required=True)
    parser.add_argument("--core-weight", type=Decimal, default=Decimal("0.70"))
    parser.add_argument("--membership", type=Path, default=Path("private_data/universe/sp500-components-updated.csv"))
    parser.add_argument("--prices", type=Path, default=Path("private_data/prices-yahoo"))
    parser.add_argument("--database", type=Path, default=Path("data/paper-trading.db"))
    arguments = parser.parse_args()
    validate_prospective_date(arguments.as_of, datetime.now(ZoneInfo("America/Chicago")).date())
    settings = Settings.from_env()
    if not settings.sec_user_agent:
        parser.error("SEC_USER_AGENT is required")
    history = load_membership_history(arguments.membership)
    cache = PriceCache(arguments.prices)
    start = arguments.as_of - timedelta(days=550)
    benchmark = _cached_bars(cache, "SPY", start, arguments.as_of)
    decision_on = benchmark[-1].session
    if decision_on != arguments.as_of:
        parser.error(f"as-of date is not a cached SPY session; latest is {decision_on}")
    members = history.members_on(decision_on)
    histories, excluded = {}, []
    for symbol in members:
        try:
            bars = _cached_bars(cache, symbol, start, decision_on)
        except ValueError:
            excluded.append(symbol)
            continue
        if len(bars) >= 253 and bars[-1].session == decision_on:
            histories[symbol] = bars
        else:
            excluded.append(symbol)
    sec = SECEdgarCache(settings.sec_user_agent, Path("private_data/sec-edgar"))
    active = SimpleMomentumTrendGrowthStrategy(benchmark, sec, sec.ticker_map())
    strategy = CoreSatelliteStrategy(active, core_weight=arguments.core_weight)
    targets = strategy.targets(histories)
    definition = StrategyVersion(
        name=strategy.name,
        parameters={
            "core_symbol": "SPY", "core_weight": str(arguments.core_weight),
            "active_signals": SIMPLE_MTG_WEIGHTS, "active_positions": active.top_n,
            "rebalance_sessions": 21, "execution": "next_session_open",
        },
        source_hash=_source_hash(),
    )
    store = PaperTradingStore(arguments.database)
    version_hash = store.register_strategy(definition)
    active_audit = active.audits[decision_on]
    evidence_hash = canonical_hash({
        "membership_version": history.version,
        "bars": {
            symbol: [bars[-1].vintage, bars[-1].retrieved_at.isoformat()]
            for symbol, bars in sorted(histories.items())
        },
    })
    payload = {
        "strategy_version": version_hash,
        "decision_date": decision_on.isoformat(),
        "earliest_execution_date": arguments.execution_date.isoformat(),
        "targets": {symbol: str(weight) for symbol, weight in targets.items()},
        "selected_active": active_audit["selected"],
        "market_regime": active_audit["market_regime"],
        "evidence_hash": evidence_hash,
        "membership_version": history.version,
        "eligible_symbols": len(histories), "excluded_symbols": sorted(excluded),
        "llm_used": False,
    }
    decision_hash = store.record_decision(
        version_hash, decision_on, arguments.execution_date, payload, datetime.now(UTC),
    )
    print(json.dumps({
        "strategy_version": version_hash, "decision_hash": decision_hash,
        "decision_date": decision_on.isoformat(), "targets": payload["targets"],
        "database": str(arguments.database),
    }, indent=2))


def _cached_bars(cache: PriceCache, symbol: str, start: date, end: date):
    if cache.covering_path(symbol, start, end) is None:
        raise ValueError(f"no cached coverage for {symbol} from {start} through {end}")
    return cache.daily(None, symbol, start, end)  # type: ignore[arg-type]


def _source_hash() -> str:
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for name in ("core_satellite.py", "robust_strategy.py", "paper_decision_cli.py"):
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def validate_prospective_date(as_of: date, today: date) -> None:
    if as_of != today:
        raise ValueError(
            f"paper decisions cannot be backfilled or predated: as-of {as_of}, today {today}"
        )


if __name__ == "__main__":
    main()
