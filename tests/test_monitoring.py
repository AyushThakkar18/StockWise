from datetime import UTC, datetime, timedelta

from portfoliopilot.jobs import JobQueue
from portfoliopilot.monitoring import operational_health


def test_health_requires_fresh_snapshot_and_no_dead_letters(tmp_path) -> None:
    now = datetime(2025, 1, 3, tzinfo=UTC)
    queue = JobQueue(tmp_path / "jobs.db", lambda: now)
    assert operational_health(queue, now, now - timedelta(hours=1)).healthy
    stale = operational_health(queue, now, now - timedelta(days=3))
    assert not stale.healthy
    assert "stale" in stale.reasons[0]

