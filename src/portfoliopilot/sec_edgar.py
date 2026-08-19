from __future__ import annotations

import json
from datetime import UTC, date, datetime, time
from pathlib import Path
from urllib.request import Request, urlopen

from .contracts import Evidence, Quality

FACTS = (
    "Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "NetIncomeLoss",
    "OperatingIncomeLoss", "Assets", "Liabilities",
    "CashAndCashEquivalentsAtCarryingValue",
)


class SECEdgarCache:
    def __init__(self, user_agent: str, directory: Path):
        if "@" not in user_agent:
            raise ValueError("SEC_USER_AGENT must identify the application and contact email")
        self.user_agent = user_agent
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)

    def ticker_map(self) -> dict[str, int]:
        payload = self._cached("company_tickers.json", "https://www.sec.gov/files/company_tickers.json")
        return {item["ticker"].upper().replace(".", "-"): int(item["cik_str"])
                for item in payload.values()}

    def company_facts(self, symbol: str, cik: int) -> dict:
        return self._cached(
            f"{symbol.upper()}-companyfacts.json",
            f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json",
        )

    def evidence_on(
        self, symbol: str, cik: int, decision_on: date, retrieved_at: datetime,
    ) -> Evidence | None:
        payload = self.company_facts(symbol, cik)
        selected = []
        for concept in FACTS:
            fact = payload.get("facts", {}).get("us-gaap", {}).get(concept)
            if not fact:
                continue
            candidates = []
            for unit_values in fact.get("units", {}).values():
                candidates.extend(item for item in unit_values
                                  if item.get("form") in {"10-Q", "10-K"}
                                  and item.get("filed") and date.fromisoformat(item["filed"]) <= decision_on)
            if candidates:
                latest = max(candidates, key=lambda item: (item["filed"], item.get("end", "")))
                selected.append((concept, latest))
        if not selected:
            return None
        filed = max(date.fromisoformat(item["filed"]) for _, item in selected)
        claims = "; ".join(
            f"{concept}={item.get('val')} ({item.get('form')}, period ended {item.get('end')})"
            for concept, item in selected
        )
        available = datetime.combine(filed, time(12), tzinfo=UTC)
        return Evidence(
            id=f"sec:{symbol}:{decision_on}", symbol=symbol, claim=claims,
            source="SEC EDGAR company facts", observed_at=available, published_at=available,
            available_to_strategy_at=available, retrieved_at=retrieved_at,
            vintage=f"retrieved:{retrieved_at.isoformat()}", quality=Quality.PASS,
        )

    def _cached(self, name: str, url: str) -> dict:
        path = self.directory / name
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        request = Request(url, headers={"User-Agent": self.user_agent, "Accept": "application/json"})
        with urlopen(request, timeout=60) as response:
            payload = json.loads(response.read())
        path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        return payload
