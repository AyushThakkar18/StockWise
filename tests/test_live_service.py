from datetime import UTC, date, datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from portfoliopilot.config import Settings
from portfoliopilot.contracts import Quality
from portfoliopilot.kronos_forecast import KronosForecast
from portfoliopilot.live_council import LiveCouncilDecision
from portfoliopilot.live_service import (
    LivePaperService,
    add_calendar_month,
    kronos_signal,
    next_market_open,
    next_weekday_open,
    select_reviewed_candidates,
)
from portfoliopilot.market_data import DailyBar


def bar(symbol: str, session: date, price: str = "100") -> DailyBar:
    moment = datetime(session.year, session.month, session.day, 21, tzinfo=UTC)
    value = Decimal(price)
    return DailyBar(
        symbol=symbol, session=session, open=value, high=value, low=value, close=value,
        adjusted_close=value, volume=1_000_000, dividend=Decimal(0),
        split_coefficient=Decimal(1), source="test", observed_at=moment,
        published_at=moment, available_to_strategy_at=moment, retrieved_at=moment,
        vintage="test", quality=Quality.PASS,
    )


class Prices:
    def histories(self, symbols, end):
        return {symbol: (bar(symbol, end),) for symbol in symbols}

    def marks(self, symbols):
        return {symbol: Decimal("100") for symbol in symbols}


def test_next_market_open_uses_same_weekday_before_open() -> None:
    eastern = ZoneInfo("America/New_York")
    before_open = datetime(2026, 10, 5, 2, 30, tzinfo=eastern)
    after_close = datetime(2026, 10, 5, 17, tzinfo=eastern)

    assert next_market_open(before_open).astimezone(eastern) == datetime(
        2026, 10, 5, 9, 30, tzinfo=eastern,
    )
    assert next_market_open(after_close).astimezone(eastern) == datetime(
        2026, 10, 6, 9, 30, tzinfo=eastern,
    )


class Runner:
    def __init__(self):
        self.calls = 0
        self.prices = Prices()

    def run(self, decision_at):
        self.calls += 1
        decision = LiveCouncilDecision(
            decision_id=f"live-{decision_at.date()}", decision_at=decision_at,
            model="gpt-4o-mini", prompt_version="live-council-v1",
            maximum_selections=20, minimum_score=55, candidates=(),
            selected_symbols=tuple(f"S{index}" for index in range(10)),
        )
        return decision, {}, (bar("SPY", decision_at.date()),)


def settings(tmp_path) -> Settings:
    return Settings(None, None, "key", "gpt-4o-mini", tmp_path / "paper.db")


def test_service_waits_for_close_then_decides_once_and_executes_next_session(tmp_path) -> None:
    runner = Runner()
    service = LivePaperService(settings(tmp_path), runner)
    before_close = datetime(2026, 8, 28, 19, tzinfo=UTC)
    assert service.cycle(before_close)["decision_created"] is None
    after_close = datetime(2026, 8, 28, 21, tzinfo=UTC)
    created = service.cycle(after_close)
    assert created["decision_created"] == "live-2026-08-28"
    assert runner.calls == 1
    assert service.cycle(after_close)["decision_created"] is None
    assert runner.calls == 1

    following = next_weekday_open(after_close.date())
    executed = service.cycle(following)
    assert executed["executed_sessions"] == ["live-2026-08-28"]
    assert service.engine.paper.snapshots()
    assert service.cycle(following)["executed_sessions"] == []


def test_existing_decision_waits_one_calendar_month_before_research(tmp_path) -> None:
    runner = Runner()
    service = LivePaperService(settings(tmp_path), runner)
    service.cycle(datetime(2026, 8, 28, 21, tzinfo=UTC))
    early_september = service.cycle(datetime(2026, 9, 2, 22, tzinfo=UTC))
    assert early_september["decision_created"] is None
    assert early_september["next_research_date"] == "2026-09-30"
    assert early_september["next_action"] == "WAIT_FOR_REBALANCE_DATE"
    assert runner.calls == 1


def test_calendar_month_clamps_month_end() -> None:
    assert add_calendar_month(date(2026, 1, 31)) == date(2026, 2, 28)


def test_one_time_initialization_uses_latest_completed_close(tmp_path) -> None:
    runner = Runner()
    service = LivePaperService(settings(tmp_path), runner)
    before_open = datetime(2026, 8, 28, 12, tzinfo=UTC)
    result = service.initialize_from_latest_close(before_open)
    assert result["decision_cutoff"] == "2026-08-27T20:15:00+00:00"
    assert result["execution_at"] == "2026-08-28T13:30:00+00:00"
    with pytest.raises(ValueError, match="already exists"):
        service.initialize_from_latest_close(before_open)


def test_shadow_rerun_does_not_freeze_or_replace_decision(tmp_path) -> None:
    runner = Runner()
    service = LivePaperService(settings(tmp_path), runner)
    decision = service.shadow_from_latest_close(datetime(2026, 8, 28, 12, tzinfo=UTC))
    assert decision.decision_id == "shadow-live-2026-08-27-v5"
    assert service.engine.paper.all_decision_payloads() == ()


def test_extreme_kronos_forecast_is_reduced_to_relative_signal() -> None:
    def forecast(median: float, dispersion: float = .04) -> KronosForecast:
        return KronosForecast(
            model="test", as_of=date(2026, 8, 27), horizon=21, paths=10,
            median_return=median, mean_return=median, probability_positive=.4,
            bear_return=median - .05, bull_return=median + .05,
            predicted_volatility=.2, predicted_max_drawdown=-.1,
            forecast_dispersion=dispersion,
        )

    signal = kronos_signal(forecast(-.19), forecast(-.12), 1, 100)
    assert signal["status"] == "UNRELIABLE"
    assert signal["cross_sectional_percentile"] == 1
    assert "candidate_forecast" not in signal


def test_unreliable_benchmark_falls_back_to_deterministic_candidates() -> None:
    benchmark = KronosForecast(
        model="test", as_of=date(2026, 8, 27), horizon=21, paths=10,
        median_return=-.12, mean_return=-.12, probability_positive=0,
        bear_return=-.2, bull_return=-.05, predicted_volatility=.2,
        predicted_max_drawdown=-.2, forecast_dispersion=.04,
    )
    ranked = [("C", 3.0, object()), ("B", 2.0, object()), ("A", 1.0, object())]
    reviewed, method = select_reviewed_candidates(("A", "B", "C"), ranked, benchmark, 2)
    assert [item[0] for item in reviewed] == ["A", "B"]
    assert method == "DETERMINISTIC_TOP_2_KRONOS_BENCHMARK_REJECTED"


def test_unreliable_candidate_forecast_cannot_control_kronos_admission() -> None:
    def forecast(median: float, dispersion: float = .04) -> KronosForecast:
        return KronosForecast(
            model="test", as_of=date(2026, 8, 27), horizon=21, paths=10,
            median_return=median, mean_return=median, probability_positive=.6,
            bear_return=median - .05, bull_return=median + .05,
            predicted_volatility=.2, predicted_max_drawdown=-.1,
            forecast_dispersion=dispersion,
        )

    benchmark = forecast(.01)
    ranked = [
        ("C", 3.0, forecast(.30)),
        ("B", 2.0, forecast(.04)),
        ("A", 1.0, forecast(.03)),
    ]

    reviewed, method = select_reviewed_candidates(("A", "B", "C"), ranked, benchmark, 2)

    assert [item[0] for item in reviewed] == ["B", "A"]
    assert method == "KRONOS_RELIABLE_CROSS_SECTIONAL_TOP_2"
