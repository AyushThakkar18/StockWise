from fastapi import FastAPI
from fastapi.testclient import TestClient

from portfoliopilot.security import BearerTokenMiddleware


def protected_app(token: str | None) -> FastAPI:
    app = FastAPI()
    app.add_middleware(BearerTokenMiddleware, token=token)

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/private")
    def private():
        return {"private": True}

    return app


def test_token_protects_private_routes_but_not_health() -> None:
    client = TestClient(protected_app("x" * 32))
    assert client.get("/health").status_code == 200
    assert client.get("/private").status_code == 401
    assert client.get("/private", headers={"Authorization": f"Bearer {'x' * 32}"}).status_code == 200


def test_missing_token_keeps_local_development_usable() -> None:
    assert TestClient(protected_app(None)).get("/private").status_code == 200

