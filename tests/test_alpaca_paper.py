import json

import pytest

from portfoliopilot.alpaca_execution import AlpacaPaperExecutor
from portfoliopilot.alpaca_paper import PAPER_BASE_URL, AlpacaPaperClient
from portfoliopilot.store import EventStore


def test_orders_can_only_reach_paper_domain() -> None:
    requests = []

    def transport(request):
        requests.append(request)
        return 200, json.dumps({"id": "paper-order-1"}).encode()

    result = AlpacaPaperClient("paper-key", "paper-secret", transport).submit_notional_order(
        symbol="aapl", notional="50", side="buy", client_order_id="decision-1:AAPL",
    )
    assert result["id"] == "paper-order-1"
    assert requests[0].full_url == f"{PAPER_BASE_URL}/v2/orders"
    assert json.loads(requests[0].data)["client_order_id"] == "decision-1:AAPL"


def test_client_rejects_non_paper_trading_endpoint() -> None:
    client = AlpacaPaperClient("paper-key", "paper-secret", lambda request: (200, b"{}"))
    with pytest.raises(ValueError, match="non-paper"):
        client._request("GET", "https://api.alpaca.markets/v2/account")


def test_submit_requires_idempotency_key_and_valid_side() -> None:
    client = AlpacaPaperClient("paper-key", "paper-secret", lambda request: (200, b"{}"))
    with pytest.raises(ValueError, match="side"):
        client.submit_notional_order(symbol="A", notional="50", side="hold", client_order_id="x")
    with pytest.raises(ValueError, match="idempotent"):
        client.submit_notional_order(symbol="A", notional="50", side="buy", client_order_id="")


def test_executor_is_disabled_by_default_and_idempotent(tmp_path) -> None:
    calls = []

    def transport(request):
        calls.append(json.loads(request.data))
        return 200, json.dumps({"id": f"order-{len(calls)}"}).encode()

    client = AlpacaPaperClient("paper-key", "paper-secret", transport)
    store = EventStore(tmp_path / "events.db")
    with pytest.raises(ValueError, match="disabled"):
        AlpacaPaperExecutor(client, store).rebalance("d1", ("AAPL",), 1000, {})
    executor = AlpacaPaperExecutor(client, store, enabled=True)
    executor.rebalance("d1", ("AAPL",), 1000, {})
    executor.rebalance("d1", ("AAPL",), 1000, {})
    assert len(calls) == 2  # AAPL 5% and SPY 95%; second invocation submits nothing.
