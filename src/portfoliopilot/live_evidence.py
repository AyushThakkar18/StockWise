from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from html import unescape

from .alpaca_paper import AlpacaPaperClient
from .contracts import Evidence, Quality


def _text(value: str, limit: int = 4000) -> str:
    plain = re.sub(r"<[^>]+>", " ", unescape(value or ""))
    return re.sub(r"\s+", " ", plain).strip()[:limit]


class LiveNewsEvidence:
    def __init__(self, client: AlpacaPaperClient) -> None:
        self.client = client

    def collect(
        self, symbol: str, start: datetime, decision_at: datetime,
        retrieved_at: datetime | None = None,
    ) -> tuple[Evidence, ...]:
        if start >= decision_at or decision_at.tzinfo is None:
            raise ValueError("news window and decision timestamp are invalid")
        retrieved = retrieved_at or datetime.now(UTC)
        payload = self.client.news(
            (symbol.upper(),), start.isoformat(), decision_at.isoformat(), limit=50,
        )
        evidence = []
        for item in payload.get("news", []):
            published = datetime.fromisoformat(item["created_at"])
            if published > decision_at or symbol.upper() not in {
                value.upper() for value in item.get("symbols", [])
            }:
                continue
            content = _text(item.get("content") or item.get("summary") or "")
            headline = _text(item.get("headline") or "", 500)
            if not headline:
                continue
            evidence.append(Evidence(
                id=f"alpaca-news:{item['id']}", symbol=symbol.upper(),
                claim=f"Headline: {headline}. Article: {content}",
                source=f"Alpaca News / {item.get('source', 'unknown')}",
                source_url=item.get("url") or None, observed_at=published,
                published_at=published, available_to_strategy_at=published,
                retrieved_at=retrieved, vintage=f"retrieved:{retrieved.isoformat()}",
                quality=Quality.PASS,
            ))
        return tuple(sorted(evidence, key=lambda item: (item.published_at, item.id)))


class YahooNewsEvidence:
    """Broker-free, timestamped headline evidence for live forward decisions."""

    def __init__(self, fetch=None) -> None:
        self.fetch = fetch or self._fetch

    @staticmethod
    def _fetch(symbol: str) -> list[dict]:
        import yfinance as yf
        return list(yf.Search(symbol, news_count=8, max_results=1).news)

    def collect(
        self, symbol: str, decision_at: datetime, retrieved_at: datetime | None = None,
    ) -> tuple[Evidence, ...]:
        retrieved = retrieved_at or datetime.now(UTC)
        earliest = decision_at - timedelta(days=30)
        output = []
        for item in self.fetch(symbol.upper()):
            published = datetime.fromtimestamp(int(item["providerPublishTime"]), tz=UTC)
            related = {value.upper() for value in item.get("relatedTickers", [])}
            if not earliest <= published <= decision_at or symbol.upper() not in related:
                continue
            headline = _text(item.get("title") or "", 500)
            if not headline:
                continue
            output.append(Evidence(
                id=f"yahoo-news:{item['uuid']}", symbol=symbol.upper(),
                claim=f"Headline: {headline}",
                source=f"Yahoo Finance / {item.get('publisher', 'unknown')}",
                source_url=item.get("link") or None, observed_at=published,
                published_at=published, available_to_strategy_at=published,
                retrieved_at=retrieved, vintage=f"retrieved:{retrieved.isoformat()}",
                quality=Quality.PASS,
            ))
        return tuple(sorted(output, key=lambda item: (item.published_at, item.id)))
