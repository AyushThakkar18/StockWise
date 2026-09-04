from datetime import UTC, date, datetime

import pytest

from portfoliopilot.jobs import JobQueue
from portfoliopilot.scheduling import (
    AfterCloseScheduler,
    ExchangeSession,
    MonthlyCouncilScheduler,
    TradingCalendar,
)


def calendar() -> TradingCalendar:
    return TradingCalendar((
        ExchangeSession(
            date(2025, 1, 2), datetime(2025, 1, 2, 14, 30, tzinfo=UTC),
            datetime(2025, 1, 2, 21, tzinfo=UTC),
        ),
        ExchangeSession(
            date(2025, 1, 3), datetime(2025, 1, 3, 14, 30, tzinfo=UTC),
            datetime(2025, 1, 3, 21, tzinfo=UTC),
        ),
    ), "fixture-v1")


def test_after_close_schedule_is_idempotent_and_uses_next_open(tmp_path) -> None:
    now = datetime(2025, 1, 2, 22, tzinfo=UTC)
    queue = JobQueue(tmp_path / "jobs.db", lambda: now)
    scheduler = AfterCloseScheduler(queue, calendar())
    first = scheduler.schedule_latest(now, "core", ("spy", "abc"))
    second = scheduler.schedule_latest(now, "core", ("spy", "abc"))
    assert first == second
    job = queue.get(first)
    assert job.payload["symbols"] == ["ABC", "SPY"]
    assert job.payload["next_execution_at"] == "2025-01-03T14:30:00+00:00"


def test_scheduler_refuses_unknown_next_session(tmp_path) -> None:
    now = datetime(2025, 1, 3, 22, tzinfo=UTC)
    scheduler = AfterCloseScheduler(JobQueue(tmp_path / "jobs.db", lambda: now), calendar())
    with pytest.raises(ValueError, match="coverage"):
        scheduler.schedule_latest(now, "core", ("SPY",))


def test_monthly_council_only_schedules_at_month_boundary(tmp_path) -> None:
    sessions = (
        ExchangeSession(
            date(2025, 1, 30), datetime(2025, 1, 30, 14, 30, tzinfo=UTC),
            datetime(2025, 1, 30, 21, tzinfo=UTC),
        ),
        ExchangeSession(
            date(2025, 1, 31), datetime(2025, 1, 31, 14, 30, tzinfo=UTC),
            datetime(2025, 1, 31, 21, tzinfo=UTC),
        ),
        ExchangeSession(
            date(2025, 2, 3), datetime(2025, 2, 3, 14, 30, tzinfo=UTC),
            datetime(2025, 2, 3, 21, tzinfo=UTC),
        ),
    )
    now = datetime(2025, 1, 31, 22, tzinfo=UTC)
    queue = JobQueue(tmp_path / "jobs.db", lambda: now)
    scheduler = MonthlyCouncilScheduler(queue, TradingCalendar(sessions, "fixture-v2"))
    job_id = scheduler.schedule_if_due(now, "sp500")
    assert job_id is not None
    assert scheduler.schedule_if_due(now, "sp500") == job_id
    assert queue.get(job_id).payload["execution_session"] == "2025-02-03"
