from datetime import UTC, datetime, timedelta
from decimal import Decimal

from portfoliopilot.forward_paper import ForwardPaperEngine
from portfoliopilot.live_council import LiveCouncilDecision


def test_frozen_decision_executes_once_at_next_open_and_records_costs(tmp_path) -> None:
    decision_at = datetime(2026, 8, 28, 20, tzinfo=UTC)
    decision = LiveCouncilDecision(
        decision_id="monthly-2026-08", decision_at=decision_at, model="gpt-4o-mini",
        prompt_version="live-council-v1", maximum_selections=20, minimum_score=55,
        candidates=(), selected_symbols=tuple(f"S{index}" for index in range(10)),
    )
    engine = ForwardPaperEngine(tmp_path / "paper.db")
    execution_at = decision_at + timedelta(days=3)
    prices = {f"S{index}": Decimal("100") for index in range(10)}
    engine.freeze(decision, execution_at, prices)
    frozen = engine.paper.all_decision_payloads()[0]["payload"]["target_weights"]
    assert "SPY" not in frozen
    assert sum(Decimal(weight) for weight in frozen.values()) == Decimal("0.998")
    results = engine.execute_next_open("monthly-2026-08", execution_at, prices)

    assert len(results) == 10
    assert {result["fill"]["symbol"] for result in results} == set(prices)
    assert Decimal(results[0]["fill"]["fee"]) > 0
    assert engine.execute_next_open(
        "monthly-2026-08", execution_at, prices,
    ) == ()
    snapshot = engine.paper.snapshots()[0]
    assert Decimal(snapshot["equity"]) < Decimal("1000")


def test_fewer_than_ten_safe_stocks_leave_unused_slots_in_cash() -> None:
    targets = ForwardPaperEngine._targets(("A", "B", "C", "D", "E"))

    assert "SPY" not in targets
    assert set(targets.values()) == {Decimal("0.0998")}
    assert sum(targets.values()) == Decimal("0.4990")
    assert ForwardPaperEngine._targets(()) == {}


def test_initial_cash_is_persisted_per_isolated_portfolio(tmp_path) -> None:
    database = tmp_path / "comparison.db"
    first = ForwardPaperEngine(database, initial_cash=Decimal("1027.24"))
    resumed = ForwardPaperEngine(database)

    assert first.initial_cash == Decimal("1027.24")
    assert resumed.initial_cash == Decimal("1027.24")
    assert resumed._recover()[0].cash == Decimal("1027.24")


def test_historical_liquidation_is_append_only_disclosed_and_idempotent(tmp_path) -> None:
    decision_at = datetime(2026, 8, 27, 20, tzinfo=UTC)
    symbols = tuple(f"S{index}" for index in range(10))
    decision = LiveCouncilDecision(
        decision_id="monthly-2026-08", decision_at=decision_at, model="gpt-4o-mini",
        prompt_version="test", maximum_selections=20, minimum_score=55,
        candidates=(), selected_symbols=symbols,
    )
    engine = ForwardPaperEngine(tmp_path / "paper.db")
    entry = {symbol: Decimal("100") for symbol in symbols}
    engine.freeze(decision, decision_at + timedelta(days=1), entry)
    engine.execute_next_open("monthly-2026-08", decision_at + timedelta(days=1), entry)

    close_at = datetime(2026, 9, 28, 20, tzinfo=UTC)
    result = engine.reconstruct_liquidation_at_close(
        "missed-rebalance-liquidation-2026-09-28", close_at,
        {symbol: Decimal("110") for symbol in symbols},
    )

    assert result["status"] == "RECONSTRUCTED"
    ledger, _ = engine._recover()
    assert not any(position.quantity for position in ledger.positions.values())
    assert ledger.cash > Decimal("1000")
    assert any(
        event["event_type"] == "MISSED_REBALANCE_RECONSTRUCTED"
        for event in engine.events.events()
    )
    repeated = engine.reconstruct_liquidation_at_close(
        "missed-rebalance-liquidation-2026-09-28", close_at,
        {symbol: Decimal("110") for symbol in symbols},
    )
    assert repeated["status"] == "ALREADY_RECONSTRUCTED"


def test_late_evening_utc_timestamp_records_the_eastern_market_date(tmp_path) -> None:
    decision_at = datetime(2026, 10, 2, 2, 0, tzinfo=UTC)
    symbols = tuple(f"S{index}" for index in range(10))
    decision = LiveCouncilDecision(
        decision_id="live-2026-10-01", decision_at=decision_at, model="gpt-4o-mini",
        prompt_version="test", maximum_selections=20, minimum_score=70,
        candidates=(), selected_symbols=symbols,
    )
    engine = ForwardPaperEngine(tmp_path / "paper.db")
    execution_at = datetime(2026, 10, 2, 13, 30, tzinfo=UTC)
    engine.freeze(decision, execution_at, {symbol: Decimal("100") for symbol in symbols})

    stored = engine.paper.all_decision_payloads()[0]
    assert stored["decision_date"] == "2026-10-01"
    assert stored["earliest_execution_date"] == "2026-10-02"


def test_mixed_price_targets_all_execute_without_false_weight_rejections(tmp_path) -> None:
    decision_at = datetime(2026, 10, 1, 20, 15, tzinfo=UTC)
    symbols = tuple(f"S{index}" for index in range(10))
    decision = LiveCouncilDecision(
        decision_id="mixed-price", decision_at=decision_at, model="gpt-4o-mini",
        prompt_version="test", maximum_selections=20, minimum_score=70,
        candidates=(), selected_symbols=symbols,
    )
    engine = ForwardPaperEngine(tmp_path / "paper.db")
    prices = {symbol: Decimal(10 + index * 100) for index, symbol in enumerate(symbols)}
    execution_at = decision_at + timedelta(days=1)
    engine.freeze(decision, execution_at, prices)
    results = engine.execute_next_open("mixed-price", execution_at, prices)

    assert len(results) == 10
    assert {result["status"] for result in results} == {"FILLED"}
