from __future__ import annotations

import csv
import hashlib
import io
import re
from bisect import bisect_right
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from urllib.request import urlopen

HISTORICAL_COMPONENTS_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes%20%28Updated%29.csv"
)
DATASET_VERSION = "fja05680-sp500-updated-components"
_DATED_SUFFIX = re.compile(r"-\d{6}$")


@dataclass(frozen=True)
class MarketCapitalization:
    symbol: str
    observed_date: date
    available_to_strategy_at: datetime
    market_cap: Decimal
    source: str
    vintage: str


class MembershipHistory:
    def __init__(self, records: dict[date, tuple[str, ...]], version: str):
        if not records:
            raise ValueError("membership history cannot be empty")
        self.records = records
        self.dates = tuple(sorted(records))
        self.version = version

    def members_on(self, as_of: date) -> tuple[str, ...]:
        index = bisect_right(self.dates, as_of) - 1
        if index < 0:
            raise ValueError("date precedes membership coverage")
        return self.records[self.dates[index]]


def load_membership_history(
    cache_path: Path, fetch: Callable[[str], bytes] | None = None,
) -> MembershipHistory:
    if cache_path.exists():
        content = cache_path.read_bytes()
    else:
        content = (fetch or _fetch)(HISTORICAL_COMPONENTS_URL)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(content)
    reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig")))
    if reader.fieldnames != ["date", "tickers"]:
        raise ValueError("unexpected historical-membership schema")
    records = {}
    for row in reader:
        session = date.fromisoformat(row["date"])
        records[session] = tuple(sorted({_provider_symbol(item) for item in row["tickers"].split(",")}))
    digest = hashlib.sha256(content).hexdigest()[:16]
    return MembershipHistory(records, f"{DATASET_VERSION}-sha256-{digest}")


def select_top_market_cap(
    as_of: datetime, members: tuple[str, ...],
    observations: dict[str, tuple[MarketCapitalization, ...]], count: int = 100,
    maximum_age: timedelta = timedelta(days=10),
) -> tuple[str, ...]:
    if count <= 0:
        raise ValueError("selection count must be positive")
    ranked = []
    for symbol in members:
        eligible = [
            item for item in observations.get(symbol, ())
            if item.available_to_strategy_at <= as_of
            and timedelta(0) <= as_of - item.available_to_strategy_at <= maximum_age
        ]
        if eligible:
            latest = max(eligible, key=lambda item: item.available_to_strategy_at)
            ranked.append((latest.market_cap, symbol))
    if len(ranked) < count:
        raise ValueError(f"only {len(ranked)} members have fresh point-in-time market caps")
    return tuple(symbol for _, symbol in sorted(ranked, key=lambda item: (-item[0], item[1]))[:count])


def _provider_symbol(symbol: str) -> str:
    return _DATED_SUFFIX.sub("", symbol.strip()).replace(".", "-")


def _fetch(url: str) -> bytes:
    with urlopen(url, timeout=60) as response:
        return response.read()
