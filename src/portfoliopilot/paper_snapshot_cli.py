from __future__ import annotations

import argparse
import json
from datetime import date
from decimal import Decimal
from pathlib import Path

from .paper_store import PaperTradingStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Append an immutable paper-portfolio snapshot")
    parser.add_argument("--strategy-version", required=True)
    parser.add_argument("--portfolio", choices=("spy", "active", "core_satellite"), required=True)
    parser.add_argument("--session", type=date.fromisoformat, required=True)
    parser.add_argument("--equity", type=Decimal, required=True)
    parser.add_argument("--cash", type=Decimal, required=True)
    parser.add_argument("--positions", type=Path, required=True)
    parser.add_argument("--source-hash", required=True)
    parser.add_argument("--database", type=Path, default=Path("data/paper-trading.db"))
    arguments = parser.parse_args()
    raw = json.loads(arguments.positions.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        parser.error("positions file must contain a JSON object of symbol to quantity")
    positions = {str(symbol): Decimal(str(quantity)) for symbol, quantity in raw.items()}
    if arguments.equity < 0 or arguments.cash < 0 or any(value < 0 for value in positions.values()):
        parser.error("equity, cash, and long-only quantities cannot be negative")
    store = PaperTradingStore(arguments.database)
    store.record_snapshot(
        arguments.strategy_version, arguments.portfolio, arguments.session,
        arguments.equity, arguments.cash, positions, arguments.source_hash,
    )
    print(json.dumps({
        "recorded": True, "strategy_version": arguments.strategy_version,
        "portfolio": arguments.portfolio, "session": arguments.session.isoformat(),
        "database": str(arguments.database),
    }, indent=2))


if __name__ == "__main__":
    main()
