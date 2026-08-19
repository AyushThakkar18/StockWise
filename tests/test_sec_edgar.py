from datetime import UTC, date, datetime

from portfoliopilot.sec_edgar import SECEdgarCache


def test_sec_evidence_never_uses_future_filing(tmp_path) -> None:
    cache = SECEdgarCache("StockWise test@example.com", tmp_path)
    payload = {"facts": {"us-gaap": {"Revenues": {"units": {"USD": [
        {"val": 10, "form": "10-Q", "filed": "2025-01-02", "end": "2024-12-31"},
        {"val": 99, "form": "10-Q", "filed": "2025-04-01", "end": "2025-03-31"},
    ]}}}}}
    (tmp_path / "ABC-companyfacts.json").write_text(__import__("json").dumps(payload))
    item = cache.evidence_on("ABC", 1, date(2025, 2, 1), datetime(2026, 1, 1, tzinfo=UTC))
    assert item is not None
    assert "Revenues=10" in item.claim
    assert "99" not in item.claim
