from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

from .council_backtest import position_attribution
from .historical_council_backtest import prepare_universe
from .point_in_time import PointInTimeBacktester
from .price_cache import PriceCache
from .universe import load_membership_history

SELECTION = re.compile(r"\[(\d{4}-\d{2}-\d{2})\] selected Top 20: (.+)")
RESEARCH = re.compile(r"\[(\d{4}-\d{2}-\d{2})\] researching 100 candidates")


@dataclass
class LoggedSelections:
    selections: dict[date, tuple[str, ...]]
    name: str = "logged_partial_council"

    def targets(self, history) -> dict[str, Decimal]:
        decision_on = max(bars[-1].session for bars in history.values() if bars)
        selected = self.selections.get(decision_on)
        if selected is None:
            raise ValueError(f"no completed selection for {decision_on}")
        return {symbol: Decimal("0.05") for symbol in selected}


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct equity from completed council months")
    parser.add_argument("--log", type=Path, default=Path("private_data/logs/per-stock-kronos-2025.log"))
    parser.add_argument("--membership", type=Path, default=Path("private_data/universe/sp500-components-updated.csv"))
    parser.add_argument("--prices", type=Path, default=Path("private_data/prices-yahoo"))
    arguments = parser.parse_args()

    text = arguments.log.read_text(encoding="utf-8")
    selections = {
        date.fromisoformat(day): tuple(symbol.strip() for symbol in symbols.split(","))
        for day, symbols in SELECTION.findall(text)
    }
    if not selections:
        raise ValueError("no completed monthly selections found")
    latest = max(selections)
    later_research = sorted(
        date.fromisoformat(day) for day in RESEARCH.findall(text)
        if date.fromisoformat(day) > latest
    )
    if not later_research:
        raise ValueError("next rebalance has not started; partial cutoff is ambiguous")
    cutoff = later_research[0] - timedelta(days=1)
    evaluation_start = min(selections)
    history = load_membership_history(arguments.membership)
    histories, benchmark, filtered, _, _ = prepare_universe(
        history, PriceCache(arguments.prices), date(2024, 1, 1), cutoff, evaluation_start,
    )
    result = PointInTimeBacktester(cost_bps=Decimal(5), rebalance_every=21).run(
        histories, benchmark, filtered, LoggedSelections(selections), evaluation_start,
    )
    print(json.dumps({
        "through": result.points[-1].session.isoformat(),
        "completed_rebalances": len(selections),
        "ending_equity": float(result.points[-1].equity),
        "metrics": result.metrics,
        "position_attribution": position_attribution(result),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
