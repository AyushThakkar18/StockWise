from datetime import UTC, datetime

from portfoliopilot.live_evidence import LiveNewsEvidence, YahooNewsEvidence


class NewsClient:
    def news(self, symbols, start, end, limit=50):
        return {"news": [
            {
                "id": 1, "created_at": "2026-08-28T18:00:00Z", "symbols": ["AAPL"],
                "headline": "Company reports results", "content": "<p>Revenue increased.</p>",
                "source": "wire", "url": "https://example.com/article",
            },
            {
                "id": 2, "created_at": "2026-08-29T18:00:00Z", "symbols": ["AAPL"],
                "headline": "Future article", "content": "Must not be visible.",
            },
        ]}


def test_news_is_symbol_bound_sanitized_and_cut_off() -> None:
    decision = datetime(2026, 8, 28, 20, tzinfo=UTC)
    result = LiveNewsEvidence(NewsClient()).collect(
        "AAPL", datetime(2026, 8, 21, tzinfo=UTC), decision, decision,
    )
    assert len(result) == 1
    assert result[0].id == "alpaca-news:1"
    assert "<p>" not in result[0].claim


def test_yahoo_headlines_are_broker_free_symbol_bound_and_cut_off() -> None:
    decision = datetime(2026, 8, 28, 20, tzinfo=UTC)
    timestamp = int(datetime(2026, 8, 28, 18, tzinfo=UTC).timestamp())
    result = YahooNewsEvidence(lambda symbol: [{
        "uuid": "n1", "title": "Company raises guidance", "publisher": "wire",
        "link": "https://finance.yahoo.com/article", "providerPublishTime": timestamp,
        "relatedTickers": ["AAPL"],
    }]).collect("AAPL", decision, decision)
    assert result[0].id == "yahoo-news:n1"
    assert result[0].published_at < decision
