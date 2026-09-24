"""Optional FastAPI wrapper: POST /v1/systemone with the request and response JSON of docs/API_SPEC.md.

Run with `uv run --extra serve python scripts/serve.py --port 8000`. The model
loads once at startup (the systemone default model: configs/base.yaml, plus the
run directory named by JEVMARK_CHECKPOINT if set). Errors: ValueError -> 400 and
RuntimeError -> 500, with the message as detail. The body is parsed with a
duplicate-key check, because json.loads would otherwise keep only the last of two
equal keys and duplicate option labels would go unnoticed.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi import Request as HttpRequest
from starlette.concurrency import run_in_threadpool

from jevmark.model import JevMark
from jevmark.systemone import default_model, systemone_batch


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    obj: dict[str, Any] = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"request: duplicate key {key!r}")
        obj[key] = value
    return obj


def parse_body(raw: bytes) -> Any:
    """JSON body to Python objects; raises ValueError on invalid JSON or any duplicate key."""
    try:
        return json.loads(raw, object_pairs_hook=_reject_duplicates)
    except json.JSONDecodeError as err:
        raise ValueError(f"request: invalid JSON ({err})") from None


def create_app(model: JevMark | None = None) -> FastAPI:
    """The app; with model=None the default model is loaded once, at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.model = model if model is not None else default_model()
        yield

    app = FastAPI(title="jevmark", lifespan=lifespan)

    @app.post("/v1/systemone")
    async def post_systemone(http_request: HttpRequest) -> dict[str, Any]:
        try:
            body = parse_body(await http_request.body())
            responses = await run_in_threadpool(systemone_batch, [body], model=app.state.model)
        except ValueError as err:
            raise HTTPException(status_code=400, detail=str(err)) from None
        except RuntimeError as err:
            raise HTTPException(status_code=500, detail=str(err)) from None
        return responses[0]

    return app


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
