from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from .jobs import JobQueue


@dataclass(frozen=True)
class OperationalHealth:
    healthy: bool
    queue_counts: dict[str, int]
    reasons: tuple[str, ...]


def operational_health(
    queue: JobQueue, now: datetime, last_successful_snapshot_at: datetime | None,
    maximum_snapshot_age: timedelta = timedelta(days=2),
) -> OperationalHealth:
    counts = queue.summary()
    reasons = []
    if counts["DEAD"]:
        reasons.append("dead-letter jobs require review")
    if last_successful_snapshot_at is None:
        reasons.append("no successful data snapshot")
    elif now - last_successful_snapshot_at > maximum_snapshot_age:
        reasons.append("latest data snapshot is stale")
    return OperationalHealth(not reasons, counts, tuple(reasons))

