from fastapi.testclient import TestClient

from portfoliopilot.api import app


def test_dashboard_page_exposes_portfolio_decisions_and_trades() -> None:
    response = TestClient(app).get("/dashboard")
    assert response.status_code == 200
    assert "StockWise" in response.text
    assert "Current positions" in response.text
    assert "Latest council" in response.text
    assert "Execution history" in response.text
    assert "Council ranking" in response.text
    assert "Evidence and source links" in response.text
    assert "/dashboard/overview" in response.text
    assert "/dashboard/refresh-prices" in response.text


def test_dashboard_price_refresh_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(
        "portfoliopilot.api.LivePaperService.price_cycle",
        lambda self: {"checked_at": "2026-09-04T12:00:00Z", "valuation_updated": True},
    )
    response = TestClient(app).post("/dashboard/refresh-prices")
    assert response.status_code == 200
    assert response.json()["status"] == "UPDATED"


def test_root_redirects_to_dashboard_without_favicon_noise() -> None:
    client = TestClient(app)
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/dashboard"
    assert client.get("/favicon.ico").status_code == 204
