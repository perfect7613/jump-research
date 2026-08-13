"""Authenticated HTTP boundary for the general workbench coordinator."""

import hmac
import os
from typing import Any, Awaitable, Callable

MAX_BODY_BYTES = 4096
MAX_REQUESTS_PER_CONTAINER = 32


def build_general_gateway(
    run_action: Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]],
    *,
    health: dict[str, Any],
):
    from fastapi import FastAPI, HTTPException, Request

    app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)
    request_count = {"value": 0}

    def authorize(request: Request) -> None:
        expected = os.environ.get("JUMP_MODAL_TOKEN", "")
        supplied = request.headers.get("authorization", "")
        if not expected or not supplied.startswith("Bearer ") or not hmac.compare_digest(supplied[7:], expected):
            raise HTTPException(status_code=401, detail="unauthorized")

    async def read_body(request: Request) -> dict[str, Any]:
        length = request.headers.get("content-length")
        if length is not None and (not length.isdigit() or int(length) > MAX_BODY_BYTES):
            raise HTTPException(status_code=413, detail="request body exceeds cap")
        if request_count["value"] >= MAX_REQUESTS_PER_CONTAINER:
            raise HTTPException(status_code=429, detail="gateway request guard reached")
        try:
            body = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail="request body must be JSON") from exc
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="request body must be a JSON object")
        request_count["value"] += 1
        return body

    @app.get("/health")
    async def get_health(request: Request) -> dict[str, Any]:
        authorize(request)
        return dict(health)

    @app.post("/v1/experiments/plan")
    async def plan(request: Request) -> dict[str, Any]:
        authorize(request)
        body = await read_body(request)
        try:
            return await run_action("plan", body)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/experiments/confirm")
    async def confirm(request: Request) -> dict[str, Any]:
        authorize(request)
        body = await read_body(request)
        try:
            return await run_action("confirm", body)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v2/thought-experiments/spec")
    async def visual_spec(request: Request) -> dict[str, Any]:
        authorize(request)
        body = await read_body(request)
        try:
            return await run_action("visual_spec", body)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v2/thought-experiments/confirm")
    async def visual_confirm(request: Request) -> dict[str, Any]:
        authorize(request)
        body = await read_body(request)
        try:
            return await run_action("visual_confirm", body)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


__all__ = ["MAX_BODY_BYTES", "MAX_REQUESTS_PER_CONTAINER", "build_general_gateway"]
