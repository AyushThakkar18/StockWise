from fastapi.testclient import TestClient

from portfoliopilot.api import app


def test_capabilities_make_safety_boundary_explicit() -> None:
    response = TestClient(app).get("/capabilities")
    assert response.status_code == 200
    assert response.json()["live_broker"] is False
    assert response.json()["llm_order_authority"] is False
