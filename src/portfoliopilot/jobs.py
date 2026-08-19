from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any


class JobStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    DEAD = "DEAD"


@dataclass(frozen=True)
class Job:
    id: int
    idempotency_key: str
    job_type: str
    payload: dict[str, Any]
    status: JobStatus
    attempts: int
    maximum_attempts: int
    available_at: datetime
    lease_until: datetime | None
    last_error: str | None


class JobQueue:
    def __init__(self, path: Path, clock: Callable[[], datetime] | None = None):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.clock = clock or (lambda: datetime.now(UTC))
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                idempotency_key TEXT NOT NULL UNIQUE,
                job_type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('PENDING','RUNNING','SUCCEEDED','DEAD')),
                attempts INTEGER NOT NULL DEFAULT 0,
                maximum_attempts INTEGER NOT NULL,
                available_at TEXT NOT NULL,
                lease_until TEXT,
                last_error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS jobs_ready ON jobs(status, available_at, id);
        """)
        self.connection.commit()

    def enqueue(
        self, idempotency_key: str, job_type: str, payload: dict[str, Any],
        maximum_attempts: int = 3, available_at: datetime | None = None,
    ) -> int:
        if not idempotency_key or not job_type or maximum_attempts <= 0:
            raise ValueError("invalid job definition")
        now = self.clock()
        try:
            cursor = self.connection.execute(
                "INSERT INTO jobs(idempotency_key, job_type, payload_json, status, maximum_attempts, "
                "available_at, created_at, updated_at) VALUES (?, ?, ?, 'PENDING', ?, ?, ?, ?)",
                (idempotency_key, job_type, json.dumps(payload, sort_keys=True), maximum_attempts,
                 (available_at or now).isoformat(), now.isoformat(), now.isoformat()),
            )
            self.connection.commit()
            return int(cursor.lastrowid)
        except sqlite3.IntegrityError:
            row = self.connection.execute(
                "SELECT id, job_type, payload_json FROM jobs WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
            canonical = json.dumps(payload, sort_keys=True)
            if row is None or row["job_type"] != job_type or row["payload_json"] != canonical:
                raise ValueError("idempotency key reused with different job") from None
            return int(row["id"])

    def claim(self, lease: timedelta = timedelta(minutes=5)) -> Job | None:
        if lease <= timedelta(0):
            raise ValueError("lease must be positive")
        now = self.clock()
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            self.connection.execute(
                "UPDATE jobs SET status = 'PENDING', lease_until = NULL, updated_at = ? "
                "WHERE status = 'RUNNING' AND lease_until <= ? AND attempts < maximum_attempts",
                (now.isoformat(), now.isoformat()),
            )
            self.connection.execute(
                "UPDATE jobs SET status = 'DEAD', lease_until = NULL, "
                "last_error = COALESCE(last_error, 'lease expired after final attempt'), updated_at = ? "
                "WHERE status = 'RUNNING' AND lease_until <= ? AND attempts >= maximum_attempts",
                (now.isoformat(), now.isoformat()),
            )
            row = self.connection.execute(
                "SELECT id FROM jobs WHERE status = 'PENDING' AND available_at <= ? "
                "ORDER BY available_at, id LIMIT 1", (now.isoformat(),),
            ).fetchone()
            if row is None:
                self.connection.commit()
                return None
            lease_until = now + lease
            self.connection.execute(
                "UPDATE jobs SET status = 'RUNNING', attempts = attempts + 1, lease_until = ?, "
                "updated_at = ? WHERE id = ?",
                (lease_until.isoformat(), now.isoformat(), row["id"]),
            )
            self.connection.commit()
            return self.get(int(row["id"]))
        except Exception:
            self.connection.rollback()
            raise

    def succeed(self, job_id: int) -> None:
        self._finish(job_id, JobStatus.SUCCEEDED, None)

    def fail(self, job_id: int, error: str, retry_delay: timedelta = timedelta(minutes=1)) -> None:
        if retry_delay < timedelta(0):
            raise ValueError("retry delay cannot be negative")
        job = self.get(job_id)
        if job.status != JobStatus.RUNNING:
            raise ValueError("only a running job can fail")
        now = self.clock()
        terminal = job.attempts >= job.maximum_attempts
        self.connection.execute(
            "UPDATE jobs SET status = ?, available_at = ?, lease_until = NULL, last_error = ?, "
            "updated_at = ? WHERE id = ? AND status = 'RUNNING'",
            (JobStatus.DEAD if terminal else JobStatus.PENDING,
             (now + retry_delay).isoformat(), error[:1000], now.isoformat(), job_id),
        )
        self.connection.commit()

    def get(self, job_id: int) -> Job:
        row = self.connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return _job(row)

    def summary(self) -> dict[str, int]:
        counts = {status.value: 0 for status in JobStatus}
        for row in self.connection.execute("SELECT status, COUNT(*) AS count FROM jobs GROUP BY status"):
            counts[row["status"]] = int(row["count"])
        return counts

    def _finish(self, job_id: int, status: JobStatus, error: str | None) -> None:
        if status != JobStatus.SUCCEEDED:
            raise ValueError("invalid terminal transition")
        cursor = self.connection.execute(
            "UPDATE jobs SET status = ?, lease_until = NULL, last_error = ?, updated_at = ? "
            "WHERE id = ? AND status = 'RUNNING'",
            (status, error, self.clock().isoformat(), job_id),
        )
        if cursor.rowcount != 1:
            self.connection.rollback()
            raise ValueError("only a running job can succeed")
        self.connection.commit()


class JobWorker:
    def __init__(self, queue: JobQueue, handlers: dict[str, Callable[[dict[str, Any]], None]]):
        self.queue, self.handlers = queue, handlers

    def run_one(self) -> bool:
        job = self.queue.claim()
        if job is None:
            return False
        handler = self.handlers.get(job.job_type)
        if handler is None:
            self.queue.fail(job.id, f"unknown job type: {job.job_type}", timedelta(0))
            return True
        try:
            handler(job.payload)
        except Exception as exc:  # noqa: BLE001 - durable worker records all handler failures
            delay = timedelta(seconds=min(3600, 2 ** max(0, job.attempts - 1) * 30))
            self.queue.fail(job.id, f"{type(exc).__name__}: {exc}", delay)
        else:
            self.queue.succeed(job.id)
        return True


def _job(row: sqlite3.Row) -> Job:
    return Job(
        id=int(row["id"]), idempotency_key=row["idempotency_key"], job_type=row["job_type"],
        payload=json.loads(row["payload_json"]), status=JobStatus(row["status"]),
        attempts=int(row["attempts"]), maximum_attempts=int(row["maximum_attempts"]),
        available_at=datetime.fromisoformat(row["available_at"]),
        lease_until=datetime.fromisoformat(row["lease_until"]) if row["lease_until"] else None,
        last_error=row["last_error"],
    )

