from datetime import datetime, timedelta, timezone

from portfoliopilot.contracts import Evidence, Quality
from portfoliopilot.integrity import all_pass, check_evidence


def evidence(available_at: datetime) -> Evidence:
    return Evidence(
        id="e1", symbol="ABC", claim="reported revenue", source="issuer filing",
        observed_at=available_at, published_at=available_at,
        available_to_strategy_at=available_at, retrieved_at=available_at,
        vintage="v1", quality=Quality.PASS,
    )


def test_future_evidence_is_blocked() -> None:
    now = datetime(2025, 1, 2, tzinfo=timezone.utc)
    assert not all_pass(check_evidence((evidence(now + timedelta(minutes=1)),), now, timedelta(days=7)))


def test_current_evidence_passes() -> None:
    now = datetime(2025, 1, 2, tzinfo=timezone.utc)
    assert all_pass(check_evidence((evidence(now - timedelta(days=1)),), now, timedelta(days=7)))

