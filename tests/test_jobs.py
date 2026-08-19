from datetime import UTC, datetime, timedelta

import pytest

from portfoliopilot.jobs import JobQueue, JobStatus, JobWorker


class Clock:
    def __init__(self):
        self.now = datetime(2025, 1, 1, tzinfo=UTC)

    def __call__(self):
        return self.now


def test_enqueue_is_idempotent_but_rejects_key_reuse(tmp_path) -> None:
    queue = JobQueue(tmp_path / "jobs.db", Clock())
    assert queue.enqueue("daily:2025-01-01", "snapshot", {"date": "2025-01-01"}) == 1
    assert queue.enqueue("daily:2025-01-01", "snapshot", {"date": "2025-01-01"}) == 1
    with pytest.raises(ValueError, match="different"):
        queue.enqueue("daily:2025-01-01", "snapshot", {"date": "changed"})


def test_expired_lease_is_recovered_after_restart(tmp_path) -> None:
    clock = Clock()
    path = tmp_path / "jobs.db"
    queue = JobQueue(path, clock)
    job_id = queue.enqueue("j1", "snapshot", {})
    assert queue.claim(timedelta(minutes=1)).id == job_id
    clock.now += timedelta(minutes=2)
    restarted = JobQueue(path, clock)
    recovered = restarted.claim()
    assert recovered.id == job_id
    assert recovered.attempts == 2


def test_worker_retries_then_dead_letters(tmp_path) -> None:
    clock = Clock()
    queue = JobQueue(tmp_path / "jobs.db", clock)
    job_id = queue.enqueue("j1", "broken", {}, maximum_attempts=2)

    def broken(_):
        raise RuntimeError("failure")

    worker = JobWorker(queue, {"broken": broken})
    assert worker.run_one()
    assert queue.get(job_id).status == JobStatus.PENDING
    clock.now += timedelta(seconds=31)
    assert worker.run_one()
    assert queue.get(job_id).status == JobStatus.DEAD
    assert "RuntimeError" in queue.get(job_id).last_error


def test_successful_job_is_not_claimed_twice(tmp_path) -> None:
    queue = JobQueue(tmp_path / "jobs.db", Clock())
    calls = []
    job_id = queue.enqueue("j1", "ok", {"value": 1})
    worker = JobWorker(queue, {"ok": lambda payload: calls.append(payload["value"])})
    assert worker.run_one()
    assert not worker.run_one()
    assert calls == [1]
    assert queue.get(job_id).status == JobStatus.SUCCEEDED

