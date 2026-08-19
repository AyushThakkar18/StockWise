from __future__ import annotations

from datetime import datetime, timedelta

from .contracts import Evidence, Quality


def curate_evidence(
    evidence: tuple[Evidence, ...], symbol: str, decision_at: datetime, max_age: timedelta,
) -> tuple[Evidence, ...]:
    """Return deduplicated, usable evidence; never silently retain an invalid record."""
    usable = []
    seen = set()
    for item in sorted(evidence, key=lambda value: (value.available_to_strategy_at, value.id)):
        if item.symbol != symbol or item.quality != Quality.PASS:
            continue
        if not (
            item.published_at <= item.available_to_strategy_at <= decision_at
            and item.observed_at <= decision_at
            and item.retrieved_at >= item.published_at
            and decision_at - item.available_to_strategy_at <= max_age
        ):
            continue
        identity = (
            item.source, item.claim.strip().casefold(), item.published_at,
            item.available_to_strategy_at, item.vintage,
        )
        if identity in seen:
            continue
        seen.add(identity)
        usable.append(item)
    return tuple(usable)

