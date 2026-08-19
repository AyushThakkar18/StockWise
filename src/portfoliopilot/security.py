from __future__ import annotations

import secrets
from collections.abc import Awaitable, Callable

from fastapi import Request
from starlette.responses import JSONResponse, Response


class BearerTokenMiddleware:
    def __init__(self, app, token: str | None, public_paths: tuple[str, ...] = ("/health",)):
        self.app, self.token, self.public_paths = app, token, public_paths

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http" or not self.token or scope.get("path") in self.public_paths:
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers", ()))
        supplied = headers.get(b"authorization", b"").decode("utf-8", errors="ignore")
        expected = f"Bearer {self.token}"
        if not secrets.compare_digest(supplied, expected):
            response = JSONResponse({"detail": "unauthorized"}, status_code=401)
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


RequestHandler = Callable[[Request], Awaitable[Response]]

