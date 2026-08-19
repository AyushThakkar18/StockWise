from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .validation import ProtocolDates


@dataclass(frozen=True)
class ExperimentProtocol:
    experiment_id: str
    protocol_version: str
    model_version: str
    universe: tuple[str, ...]
    dates: ProtocolDates
    embargo_days: int
    transaction_cost_bps: float
    seed: int

    @property
    def protocol_hash(self) -> str:
        payload = asdict(self)
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()


class ExperimentRegistry:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY,
                protocol_hash TEXT NOT NULL UNIQUE,
                protocol_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                holdout_opened_at TEXT,
                holdout_result_json TEXT,
                eligible_for_paper INTEGER NOT NULL DEFAULT 0
            );
            CREATE TRIGGER IF NOT EXISTS no_protocol_update
            BEFORE UPDATE OF protocol_hash, protocol_json ON experiments
            BEGIN SELECT RAISE(ABORT, 'protocol is immutable'); END;
            CREATE TRIGGER IF NOT EXISTS no_holdout_rewrite
            BEFORE UPDATE OF holdout_result_json ON experiments
            WHEN OLD.holdout_result_json IS NOT NULL
            BEGIN SELECT RAISE(ABORT, 'holdout result is immutable'); END;
        """)
        self.connection.commit()

    def register(self, protocol: ExperimentProtocol) -> str:
        protocol.dates.validate(protocol.embargo_days)
        payload = json.dumps(asdict(protocol), sort_keys=True, default=str)
        self.connection.execute(
            "INSERT INTO experiments(experiment_id, protocol_hash, protocol_json, created_at) "
            "VALUES (?, ?, ?, ?)",
            (protocol.experiment_id, protocol.protocol_hash, payload, datetime.now(UTC).isoformat()),
        )
        self.connection.commit()
        return protocol.protocol_hash

    def open_holdout(self, experiment_id: str, result: dict[str, Any], eligible: bool) -> None:
        cursor = self.connection.execute(
            "UPDATE experiments SET holdout_opened_at = ?, holdout_result_json = ?, "
            "eligible_for_paper = ? WHERE experiment_id = ? AND holdout_opened_at IS NULL",
            (datetime.now(UTC).isoformat(), json.dumps(result, sort_keys=True), int(eligible), experiment_id),
        )
        if cursor.rowcount != 1:
            self.connection.rollback()
            raise ValueError("unknown experiment or holdout already opened")
        self.connection.commit()

    def get(self, experiment_id: str) -> dict[str, Any]:
        row = self.connection.execute(
            "SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,)
        ).fetchone()
        if row is None:
            raise KeyError(experiment_id)
        return dict(row)

