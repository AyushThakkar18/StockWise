from datetime import UTC, datetime, timedelta
from decimal import Decimal

from portfoliopilot.forward_paper import ForwardPaperEngine
from portfoliopilot.live_council import LiveCouncilDecision


def test_frozen_decision_executes_once_at_next_open_and_records_costs(tmp_path) -> None:
    decision_at = datetime(2026, 8, 28, 20, tzinfo=UTC)
    decision = LiveCouncilDecision(
        decision_id="monthly-2026-08", decision_at=decision_at, model="gpt-4o-mini",
        prompt_version="live-council-v1", maximum_selections=20, minimum_score=55,
        candidates=(), selected_symbols=(),
    )
    engine = ForwardPaperEngine(tmp_path / "paper.db")
    execution_at = decision_at + timedelta(days=3)
    engine.freeze(decision, execution_at, {})
    results = engine.execute_next_open("monthly-2026-08", execution_at, {"SPY": Decimal("100")})

    assert len(results) == 1
    assert results[0]["fill"]["symbol"] == "SPY"
    assert Decimal(results[0]["fill"]["fee"]) > 0
    assert engine.execute_next_open(
        "monthly-2026-08", execution_at, {"SPY": Decimal("100")},
    ) == ()
    snapshot = engine.paper.snapshots()[0]
    assert Decimal(snapshot["equity"]) < Decimal("1000")
