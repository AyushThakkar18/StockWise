from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

from .jobs import JobQueue


@dataclass(frozen=True)
class ExchangeSession:
    session: date
    open_at: datetime
    close_at: datetime

    def __post_init__(self) -> None:
        if self.open_at.tzinfo is None or self.close_at.tzinfo is None:
            raise ValueError("session timestamps must be timezone-aware")
        if self.open_at >= self.close_at or self.open_at.date() != self.session:
            raise ValueError("invalid exchange session bounds")


class TradingCalendar:
    """Explicit sessions avoid guessing holidays, half-days, or daylight-saving changes."""

    def __init__(self, sessions: tuple[ExchangeSession, ...], calendar_version: str):
        if not calendar_version or not sessions:
            raise ValueError("versioned exchange sessions are required")
        if tuple(sorted(sessions, key=lambda item: item.session)) != sessions:
            raise ValueError("sessions must be sorted")
        if len({item.session for item in sessions}) != len(sessions):
            raise ValueError("sessions must be unique")
        self.sessions = sessions
        self.calendar_version = calendar_version
        self.by_date = {item.session: item for item in sessions}

    def latest_closed(self, now: datetime) -> ExchangeSession:
        eligible = [session for session in self.sessions if session.close_at <= now]
        if not eligible:
            raise ValueError("no known closed exchange session")
        return eligible[-1]

    def next_session(self, session_date: date) -> ExchangeSession:
        for index, session in enumerate(self.sessions):
            if session.session == session_date and index + 1 < len(self.sessions):
                return self.sessions[index + 1]
        raise ValueError("next exchange session is outside calendar coverage")


class AfterCloseScheduler:
    def __init__(self, queue: JobQueue, calendar: TradingCalendar):
        self.queue, self.calendar = queue, calendar

    def schedule_latest(self, now: datetime, universe_id: str, symbols: tuple[str, ...]) -> int:
        if not universe_id or not symbols or len(set(symbols)) != len(symbols):
            raise ValueError("a named universe with unique symbols is required")
        session = self.calendar.latest_closed(now)
        following = self.calendar.next_session(session.session)
        payload = {
            "session": session.session.isoformat(), "decision_at": session.close_at.isoformat(),
            "next_execution_at": following.open_at.isoformat(), "universe_id": universe_id,
            "symbols": sorted(symbol.upper() for symbol in symbols),
            "calendar_version": self.calendar.calendar_version,
        }
        return self.queue.enqueue(
            f"market-snapshot:{universe_id}:{session.session.isoformat()}",
            "MARKET_SNAPSHOT", payload, maximum_attempts=4,
        )

