from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class EventStore:
    """Small durable append-only boundary; PostgreSQL replaces it in deployment."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL UNIQUE,
                occurred_at TEXT NOT NULL,
                event_type TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                payload TEXT NOT NULL,
                payload_hash TEXT NOT NULL
            )
        """)
        self.connection.commit()

    def append(self, event_id: str, event_type: str, entity_id: str, payload: dict[str, Any]) -> int:
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        digest = hashlib.sha256(canonical.encode()).hexdigest()
        cursor = self.connection.execute(
            "INSERT INTO events(event_id, occurred_at, event_type, entity_id, payload, payload_hash) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (event_id, datetime.now(UTC).isoformat(), event_type, entity_id, canonical, digest),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def events(self, entity_id: str | None = None) -> tuple[dict[str, Any], ...]:
        query = "SELECT * FROM events"
        parameters: tuple[str, ...] = ()
        if entity_id:
            query += " WHERE entity_id = ?"
            parameters = (entity_id,)
        query += " ORDER BY sequence"
        return tuple(dict(row) for row in self.connection.execute(query, parameters))

