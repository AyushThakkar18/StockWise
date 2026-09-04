from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

PAPER_BASE_URL = "https://paper-api.alpaca.markets"
DATA_BASE_URL = "https://data.alpaca.markets"
Transport = Callable[[Request], tuple[int, bytes]]


class AlpacaPaperClient:
    """Minimal Alpaca adapter intentionally incapable of reaching the live trading domain."""

    def __init__(self, key_id: str, secret_key: str, transport: Transport | None = None) -> None:
        if not key_id or not secret_key:
            raise ValueError("Alpaca paper credentials are required")
        self.key_id, self.secret_key = key_id, secret_key
        self.transport = transport or self._transport

    def account(self) -> dict[str, Any]:
        return self._request("GET", f"{PAPER_BASE_URL}/v2/account")

    def positions(self) -> list[dict[str, Any]]:
        return self._request("GET", f"{PAPER_BASE_URL}/v2/positions")

    def orders(self, status: str = "all") -> list[dict[str, Any]]:
        query = urlencode({"status": status, "direction": "asc", "nested": "true"})
        return self._request("GET", f"{PAPER_BASE_URL}/v2/orders?{query}")

    def submit_notional_order(
        self, *, symbol: str, notional: str, side: str, client_order_id: str,
    ) -> dict[str, Any]:
        if side not in {"buy", "sell"}:
            raise ValueError("side must be buy or sell")
        if not client_order_id:
            raise ValueError("an idempotent client_order_id is required")
        payload = {
            "symbol": symbol.upper(), "notional": notional, "side": side,
            "type": "market", "time_in_force": "day", "client_order_id": client_order_id,
        }
        return self._request("POST", f"{PAPER_BASE_URL}/v2/orders", payload)

    def cancel_order(self, order_id: str) -> None:
        self._request("DELETE", f"{PAPER_BASE_URL}/v2/orders/{order_id}", expect_json=False)

    def news(self, symbols: tuple[str, ...], start: str, end: str, limit: int = 50) -> dict[str, Any]:
        query = urlencode({
            "symbols": ",".join(symbols), "start": start, "end": end,
            "limit": min(limit, 50), "sort": "desc",
        })
        return self._request("GET", f"{DATA_BASE_URL}/v1beta1/news?{query}")

    def _request(
        self, method: str, url: str, payload: dict[str, Any] | None = None,
        *, expect_json: bool = True,
    ) -> Any:
        if not url.startswith((PAPER_BASE_URL + "/", DATA_BASE_URL + "/")):
            raise ValueError("refusing non-paper Alpaca endpoint")
        body = json.dumps(payload).encode() if payload is not None else None
        request = Request(url, data=body, method=method, headers={
            "APCA-API-KEY-ID": self.key_id,
            "APCA-API-SECRET-KEY": self.secret_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        })
        status, raw = self.transport(request)
        if status >= 400:
            raise RuntimeError(f"Alpaca paper API returned HTTP {status}: {raw[:500].decode(errors='replace')}")
        return json.loads(raw) if expect_json and raw else None

    @staticmethod
    def _transport(request: Request) -> tuple[int, bytes]:
        try:
            with urlopen(request, timeout=30) as response:
                return response.status, response.read()
        except HTTPError as exc:
            return exc.code, exc.read()
