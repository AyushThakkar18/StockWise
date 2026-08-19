from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path

from .market_data import DailyBar
from .research_contracts import CouncilResult


class ResearchStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS data_snapshots (
                snapshot_id TEXT PRIMARY KEY,
                as_of TEXT NOT NULL,
                payload_hash TEXT NOT NULL UNIQUE,
                payload_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS council_results (
                result_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                as_of TEXT NOT NULL,
                payload_hash TEXT NOT NULL UNIQUE,
                payload_json TEXT NOT NULL
            );
            CREATE TRIGGER IF NOT EXISTS no_snapshot_update BEFORE UPDATE ON data_snapshots
            BEGIN SELECT RAISE(ABORT, 'data snapshot is immutable'); END;
            CREATE TRIGGER IF NOT EXISTS no_council_update BEFORE UPDATE ON council_results
            BEGIN SELECT RAISE(ABORT, 'council result is immutable'); END;
        """)
        self.connection.commit()

    def save_bars(self, snapshot_id: str, as_of: datetime, bars: tuple[DailyBar, ...]) -> str:
        if any(bar.available_to_strategy_at > as_of for bar in bars):
            raise ValueError("snapshot contains data unavailable at its as-of time")
        payload = json.dumps(
            [bar.model_dump(mode="json") for bar in sorted(bars, key=lambda item: (item.symbol, item.session))],
            sort_keys=True, separators=(",", ":"),
        )
        digest = hashlib.sha256(payload.encode()).hexdigest()
        try:
            self.connection.execute(
                "INSERT INTO data_snapshots VALUES (?, ?, ?, ?)",
                (snapshot_id, as_of.isoformat(), digest, payload),
            )
            self.connection.commit()
        except sqlite3.IntegrityError:
            row = self.connection.execute(
                "SELECT as_of, payload_hash FROM data_snapshots WHERE snapshot_id = ?", (snapshot_id,)
            ).fetchone()
            if row is None or row["as_of"] != as_of.isoformat() or row["payload_hash"] != digest:
                raise ValueError("snapshot ID reused with different inputs") from None
        return digest

    def save_council(self, result_id: str, result: CouncilResult) -> str:
        payload = result.model_dump_json()
        digest = hashlib.sha256(payload.encode()).hexdigest()
        try:
            self.connection.execute(
                "INSERT INTO council_results VALUES (?, ?, ?, ?, ?)",
                (result_id, result.symbol, result.as_of.isoformat(), digest, payload),
            )
            self.connection.commit()
        except sqlite3.IntegrityError:
            row = self.connection.execute(
                "SELECT payload_hash FROM council_results WHERE result_id = ?", (result_id,)
            ).fetchone()
            if row is None or row["payload_hash"] != digest:
                raise ValueError("result ID reused with different inputs") from None
        return digest

    def load_council(self, result_id: str) -> CouncilResult:
        row = self.connection.execute(
            "SELECT payload_json, payload_hash FROM council_results WHERE result_id = ?", (result_id,)
        ).fetchone()
        if row is None:
            raise KeyError(result_id)
        if hashlib.sha256(row["payload_json"].encode()).hexdigest() != row["payload_hash"]:
            raise ValueError("stored council result hash mismatch")
        return CouncilResult.model_validate_json(row["payload_json"])
