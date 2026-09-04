from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()


@dataclass(frozen=True)
class StrategyVersion:
    name: str
    parameters: dict[str, Any]
    source_hash: str

    @property
    def version_hash(self) -> str:
        return canonical_hash(asdict(self))


class PaperTradingStore:
    """Append-only strategy decisions and portfolio snapshots for prospective evaluation."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.execute("PRAGMA busy_timeout = 5000")
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS strategy_versions (
                version_hash TEXT PRIMARY KEY,
                definition_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS paper_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version_hash TEXT NOT NULL REFERENCES strategy_versions(version_hash),
                decision_date TEXT NOT NULL,
                decided_at TEXT NOT NULL,
                earliest_execution_date TEXT NOT NULL,
                payload_hash TEXT NOT NULL UNIQUE,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(version_hash, decision_date)
            );
            CREATE TABLE IF NOT EXISTS paper_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version_hash TEXT NOT NULL REFERENCES strategy_versions(version_hash),
                portfolio_name TEXT NOT NULL,
                session TEXT NOT NULL,
                equity TEXT NOT NULL,
                cash TEXT NOT NULL,
                positions_json TEXT NOT NULL,
                source_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(version_hash, portfolio_name, session)
            );
            CREATE TRIGGER IF NOT EXISTS no_strategy_version_update
            BEFORE UPDATE ON strategy_versions BEGIN SELECT RAISE(ABORT, 'strategy version is immutable'); END;
            CREATE TRIGGER IF NOT EXISTS no_strategy_version_delete
            BEFORE DELETE ON strategy_versions BEGIN SELECT RAISE(ABORT, 'strategy version is immutable'); END;
            CREATE TRIGGER IF NOT EXISTS no_paper_decision_update
            BEFORE UPDATE ON paper_decisions BEGIN SELECT RAISE(ABORT, 'paper decision is immutable'); END;
            CREATE TRIGGER IF NOT EXISTS no_paper_decision_delete
            BEFORE DELETE ON paper_decisions BEGIN SELECT RAISE(ABORT, 'paper decision is immutable'); END;
            CREATE TRIGGER IF NOT EXISTS no_paper_snapshot_update
            BEFORE UPDATE ON paper_snapshots BEGIN SELECT RAISE(ABORT, 'paper snapshot is immutable'); END;
            CREATE TRIGGER IF NOT EXISTS no_paper_snapshot_delete
            BEFORE DELETE ON paper_snapshots BEGIN SELECT RAISE(ABORT, 'paper snapshot is immutable'); END;
        """)
        self.connection.commit()

    def register_strategy(self, version: StrategyVersion) -> str:
        payload = json.dumps(asdict(version), sort_keys=True, default=str)
        self.connection.execute(
            "INSERT OR IGNORE INTO strategy_versions VALUES (?, ?, ?)",
            (version.version_hash, payload, datetime.now(UTC).isoformat()),
        )
        row = self.connection.execute(
            "SELECT definition_json FROM strategy_versions WHERE version_hash = ?",
            (version.version_hash,),
        ).fetchone()
        if row is None or row["definition_json"] != payload:
            raise ValueError("strategy hash collision")
        self.connection.commit()
        return version.version_hash

    def record_decision(
        self, version_hash: str, decision_date: date, earliest_execution_date: date,
        payload: dict[str, Any], decided_at: datetime | None = None,
    ) -> str:
        if earliest_execution_date <= decision_date:
            raise ValueError("execution must occur after the decision date")
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        payload_hash = hashlib.sha256(canonical.encode()).hexdigest()
        existing = self.connection.execute(
            "SELECT payload_hash FROM paper_decisions WHERE version_hash = ? AND decision_date = ?",
            (version_hash, decision_date.isoformat()),
        ).fetchone()
        if existing:
            if existing["payload_hash"] != payload_hash:
                raise ValueError("decision already frozen with different content")
            return payload_hash
        now = datetime.now(UTC)
        self.connection.execute(
            "INSERT INTO paper_decisions(version_hash, decision_date, decided_at, "
            "earliest_execution_date, payload_hash, payload_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (version_hash, decision_date.isoformat(), (decided_at or now).isoformat(),
             earliest_execution_date.isoformat(), payload_hash, canonical, now.isoformat()),
        )
        self.connection.commit()
        return payload_hash

    def record_snapshot(
        self, version_hash: str, portfolio_name: str, session: date,
        equity: Decimal, cash: Decimal, positions: dict[str, Decimal], source_hash: str,
    ) -> None:
        canonical_positions = json.dumps(
            {symbol: str(value) for symbol, value in sorted(positions.items())}, sort_keys=True,
        )
        try:
            self.connection.execute(
                "INSERT INTO paper_snapshots(version_hash, portfolio_name, session, equity, cash, "
                "positions_json, source_hash, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (version_hash, portfolio_name, session.isoformat(), str(equity), str(cash),
                 canonical_positions, source_hash, datetime.now(UTC).isoformat()),
            )
            self.connection.commit()
        except sqlite3.IntegrityError:
            row = self.connection.execute(
                "SELECT equity, cash, positions_json, source_hash FROM paper_snapshots "
                "WHERE version_hash = ? AND portfolio_name = ? AND session = ?",
                (version_hash, portfolio_name, session.isoformat()),
            ).fetchone()
            expected = (str(equity), str(cash), canonical_positions, source_hash)
            if row is None or tuple(row) != expected:
                raise ValueError("snapshot already frozen with different content") from None

    def decisions(self, version_hash: str) -> tuple[dict[str, Any], ...]:
        rows = self.connection.execute(
            "SELECT * FROM paper_decisions WHERE version_hash = ? ORDER BY decision_date",
            (version_hash,),
        )
        return tuple(dict(row) for row in rows)

    def decision_payloads(self, version_hash: str) -> tuple[dict[str, Any], ...]:
        """Return immutable decisions in a form suitable for API and dashboard read models."""
        return tuple(
            {
                "decision_date": row["decision_date"],
                "decided_at": row["decided_at"],
                "earliest_execution_date": row["earliest_execution_date"],
                "payload_hash": row["payload_hash"],
                "payload": json.loads(row["payload_json"]),
            }
            for row in self.decisions(version_hash)
        )

    def all_decision_payloads(self) -> tuple[dict[str, Any], ...]:
        versions = self.connection.execute(
            "SELECT version_hash FROM strategy_versions ORDER BY created_at",
        )
        return tuple(
            {"version_hash": row["version_hash"], **decision}
            for row in versions for decision in self.decision_payloads(row["version_hash"])
        )

    def snapshots(self) -> tuple[dict[str, Any], ...]:
        rows = self.connection.execute("SELECT * FROM paper_snapshots ORDER BY session, id")
        return tuple({
            **dict(row), "positions": json.loads(row["positions_json"]),
        } for row in rows)
