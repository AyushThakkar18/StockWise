import json
import sqlite3

import pytest

from portfoliopilot.store import EventStore


def test_event_store_is_append_only_and_idempotency_key_is_unique(tmp_path) -> None:
    store = EventStore(tmp_path / "events.db")
    assert store.append("event-1", "DECISION", "d1", {"b": 2, "a": 1}) == 1
    event = store.events("d1")[0]
    assert json.loads(event["payload"]) == {"a": 1, "b": 2}
    with pytest.raises(sqlite3.IntegrityError):
        store.append("event-1", "DECISION", "d1", {"a": 1})
