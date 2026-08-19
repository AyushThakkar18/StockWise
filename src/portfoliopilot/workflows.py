from __future__ import annotations

import sqlite3
from collections.abc import Callable
from datetime import date, datetime
from typing import Any, Protocol

from .market_data import DailyBar
from .snapshots import ResearchStore
from .store import EventStore


class DailyPriceClient(Protocol):
    def daily(
        self, symbol: str, retrieved_at: datetime | None = None
    ) -> tuple[DailyBar, ...]: ...


class MarketSnapshotHandler:
    def __init__(
        self, client_factory: Callable[[], DailyPriceClient], snapshots: ResearchStore,
        events: EventStore, clock: Callable[[], datetime],
    ):
        self.client_factory = client_factory
        self.snapshots = snapshots
        self.events = events
        self.clock = clock

    def __call__(self, payload: dict[str, Any]) -> None:
        session = date.fromisoformat(payload["session"])
        decision_at = datetime.fromisoformat(payload["decision_at"])
        symbols = tuple(payload["symbols"])
        if not symbols:
            raise ValueError("snapshot universe cannot be empty")
        retrieved_at = self.clock()
        if retrieved_at < decision_at:
            raise ValueError("snapshot job ran before the decision cutoff")
        client = self.client_factory()
        collected = []
        missing = []
        for symbol in symbols:
            eligible = tuple(
                bar for bar in client.daily(symbol, retrieved_at)
                if bar.session <= session and bar.available_to_strategy_at <= decision_at
            )
            if not eligible or eligible[-1].session != session:
                missing.append(symbol)
            else:
                collected.extend(eligible)
        if missing:
            raise RuntimeError(f"incomplete universe snapshot: {','.join(sorted(missing))}")
        snapshot_id = f"{payload['universe_id']}:{session.isoformat()}"
        digest = self.snapshots.save_bars(snapshot_id, decision_at, tuple(collected))
        event_id = f"snapshot:{snapshot_id}:created"
        event_payload = {
            "snapshot_id": snapshot_id, "snapshot_hash": digest,
            "decision_at": decision_at.isoformat(), "symbols": list(symbols),
            "calendar_version": payload["calendar_version"],
        }
        try:
            self.events.append(event_id, "MARKET_SNAPSHOT_CREATED", snapshot_id, event_payload)
        except sqlite3.IntegrityError:
            existing = [event for event in self.events.events(snapshot_id) if event["event_id"] == event_id]
            if not existing:
                raise
