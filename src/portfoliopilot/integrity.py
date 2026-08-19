from datetime import datetime, timedelta

from .contracts import Evidence, PolicyCheck, Quality


def check_evidence(
    evidence: tuple[Evidence, ...], decision_at: datetime, max_age: timedelta
) -> tuple[PolicyCheck, ...]:
    if not evidence:
        return (PolicyCheck(name="evidence_present", passed=False, detail="no evidence"),)
    return tuple(
        PolicyCheck(
            name=f"evidence:{item.id}",
            passed=(
                item.quality == Quality.PASS
                and item.published_at <= item.available_to_strategy_at <= decision_at
                and item.observed_at <= decision_at
                and decision_at - item.available_to_strategy_at <= max_age
            ),
            detail="point-in-time and freshness policy",
        )
        for item in evidence
    )


def all_pass(checks: tuple[PolicyCheck, ...]) -> bool:
    return bool(checks) and all(check.passed for check in checks)

