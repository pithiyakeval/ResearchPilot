"""Server-level error handling, tested through uvicorn's real HTTP protocol code.

Each request is fed as raw bytes to uvicorn's ``H11Protocol`` over an in-memory transport,
so what uvicorn itself logs is captured too. No socket is opened, no server or lifespan is
started, and no network, model or MCP server is contacted.
"""

import asyncio
import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass

import pytest
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ValidationError
from uvicorn.config import Config
from uvicorn.protocols.http.h11_impl import H11Protocol
from uvicorn.server import ServerState

from researchpilot.api import create_app, problems
from researchpilot.api.problems import PROBLEM_MEDIA_TYPE
from researchpilot.api.settings import ApiSettings
from researchpilot.config import Settings

SECRET = "SECRET-SOURCE-TEXT"
UVICORN_TRACEBACK = "Exception in ASGI application"
LIMIT = 1_048_576
TIMEOUT = 5.0


class MemoryTransport(asyncio.Transport):
    """Socket-free transport that records everything uvicorn writes."""

    def __init__(self) -> None:
        super().__init__()
        self.written = bytearray()
        self.closed = False

    def get_extra_info(self, name: str, default=None):
        addresses = {"sockname": ("127.0.0.1", 8000), "peername": ("127.0.0.1", 50000)}
        return addresses.get(name, default)

    def write(self, data: bytes) -> None:
        self.written += data

    def is_closing(self) -> bool:
        return self.closed

    def close(self) -> None:
        self.closed = True

    def pause_reading(self) -> None:
        pass

    def resume_reading(self) -> None:
        pass


@dataclass
class RawResponse:
    status: int
    headers: dict[str, str]
    raw: bytes

    def json(self) -> dict:
        return json.loads(self.raw.partition(b"\r\n\r\n")[2])


def serve(app: FastAPI, request: bytes) -> RawResponse:
    """Run one raw HTTP/1.1 request through uvicorn's protocol and return its response."""

    async def exchange() -> bytes:
        config = Config(app=app, lifespan="off", log_config=None)
        state = ServerState()
        protocol = H11Protocol(config=config, server_state=state, app_state={})
        transport = MemoryTransport()
        protocol.connection_made(transport)
        protocol.data_received(request)
        await asyncio.wait_for(asyncio.gather(*list(state.tasks)), TIMEOUT)
        protocol.connection_lost(None)
        return bytes(transport.written)

    raw = asyncio.run(exchange())
    head = raw.partition(b"\r\n\r\n")[0].decode("latin-1").split("\r\n")
    headers = dict(line.split(": ", 1) for line in head[1:])
    return RawResponse(int(head[0].split()[1]), {k.lower(): v for k, v in headers.items()}, raw)


def get(path: str) -> bytes:
    return f"GET {path} HTTP/1.1\r\nhost: test\r\nconnection: close\r\n\r\n".encode()


def post(path: str, body: bytes) -> bytes:
    head = f"POST {path} HTTP/1.1\r\nhost: test\r\nconnection: close\r\n"
    head += f"content-type: application/json\r\ncontent-length: {len(body)}\r\n\r\n"
    return head.encode() + body


def post_chunked(path: str, chunks: list[bytes]) -> bytes:
    head = f"POST {path} HTTP/1.1\r\nhost: test\r\nconnection: close\r\n"
    head += "content-type: application/json\r\ntransfer-encoding: chunked\r\n\r\n"
    framed = b"".join(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n" for chunk in chunks)
    return head.encode() + framed + b"0\r\n\r\n"


class Count(BaseModel):
    count: int


class Payload(BaseModel):
    question: str
    count: int


def add_test_routes(app: FastAPI, calls: list[str]) -> FastAPI:
    """Routes that fail in each way a real route can, plus one that records its calls."""

    @app.get("/test/crash")
    def crash() -> dict:
        raise RuntimeError(f"crash {SECRET}")

    @app.get("/test/invalid-response", response_model=Count)
    def invalid_response() -> dict:
        return {"count": f"not a number {SECRET}"}

    @app.get("/test/stream")
    def stream() -> StreamingResponse:
        def chunks() -> Iterator[bytes]:
            yield b"partial "
            raise RuntimeError(f"mid-stream {SECRET}")

        return StreamingResponse(chunks(), media_type="text/plain")

    @app.post("/test/echo")
    def echo(payload: Payload) -> dict:
        calls.append(payload.question[:8])
        return {"ok": True}

    return app


def phase2_app() -> FastAPI:
    """The Phase 2 setup: an ``Exception`` handler alone, without containment middleware."""
    app = FastAPI()
    app.add_exception_handler(Exception, problems._unexpected_error)
    return add_test_routes(app, [])


def current_app(calls: list[str] | None = None, **api_overrides) -> FastAPI:
    app = create_app(
        settings=Settings(_env_file=None),  # type: ignore[call-arg]
        api_settings=ApiSettings(_env_file=None, **api_overrides),  # type: ignore[call-arg]
        ollama_probe=lambda: True,
    )
    return add_test_routes(app, calls if calls is not None else [])


@pytest.fixture
def logs(caplog):
    """Everything logged during the test, by any logger, uvicorn's included."""
    caplog.set_level(logging.DEBUG)
    return caplog


# --- The harness reproduces the Phase 2 leak ---


@pytest.mark.parametrize("path", ["/test/crash", "/test/invalid-response", "/test/stream"])
def test_harness_shows_uvicorn_logging_the_message_with_a_handler_alone(logs, path) -> None:
    serve(phase2_app(), get(path))

    assert UVICORN_TRACEBACK in logs.text
    assert SECRET in logs.text


# --- Containment ---


@pytest.mark.parametrize(
    ("path", "error_type"),
    [("/test/crash", "RuntimeError"), ("/test/invalid-response", "ResponseValidationError")],
)
def test_unexpected_errors_are_contained_before_the_response_starts(logs, path, error_type) -> None:
    response = serve(current_app(), get(path))

    assert response.status == 500
    assert response.headers["content-type"] == PROBLEM_MEDIA_TYPE
    body = response.json()
    assert (body["code"], body["detail"]) == (
        "internal_error",
        "The server could not complete the request.",
    )
    assert f"Unhandled {error_type} while handling GET {path}" in logs.text
    assert UVICORN_TRACEBACK not in logs.text
    assert SECRET not in logs.text
    assert SECRET.encode() not in response.raw


def test_errors_after_the_response_starts_are_contained_without_details(logs) -> None:
    response = serve(current_app(), get("/test/stream"))

    assert response.status == 200
    assert b"partial" in response.raw
    assert "while handling GET /test/stream" in logs.text
    assert UVICORN_TRACEBACK not in logs.text
    assert SECRET not in logs.text
    assert SECRET.encode() not in response.raw


def test_expected_errors_keep_their_problem_responses(logs) -> None:
    app = current_app()

    assert serve(app, get("/api/v1/nope")).json()["code"] == "not_found"
    invalid = serve(app, post("/test/echo", json.dumps({"question": SECRET}).encode()))
    assert invalid.status == 422
    assert invalid.json()["code"] == "validation_failed"
    assert SECRET.encode() not in invalid.raw
    assert UVICORN_TRACEBACK not in logs.text


# --- Request body limit ---


def payload_of_size(size: int) -> bytes:
    """A valid ``Payload`` JSON body of exactly ``size`` bytes."""
    empty = json.dumps({"question": "", "count": 1}).encode()
    return json.dumps({"question": "x" * (size - len(empty)), "count": 1}).encode()


def assert_too_large(response: RawResponse) -> None:
    assert response.status == 413
    assert response.headers["content-type"] == PROBLEM_MEDIA_TYPE
    body = response.json()
    assert (body["code"], body["title"], body["status"]) == (
        "content_too_large",
        "Content too large",
        413,
    )
    assert SECRET.encode() not in response.raw


def test_a_declared_oversized_body_is_refused_before_the_route_runs(logs) -> None:
    calls: list[str] = []
    body = SECRET.encode() + b"x" * (LIMIT + 1 - len(SECRET))

    response = serve(current_app(calls), post("/test/echo", body))

    assert_too_large(response)
    assert calls == []
    assert SECRET not in logs.text
    assert UVICORN_TRACEBACK not in logs.text


def test_a_streamed_oversized_body_is_refused_before_the_endpoint_runs(logs) -> None:
    calls: list[str] = []
    chunks = [SECRET.encode()] + [b"x" * 65_536] * 16  # just over 1 MiB, no Content-Length

    response = serve(current_app(calls), post_chunked("/test/echo", chunks))

    assert_too_large(response)
    assert calls == []
    assert SECRET not in logs.text
    assert UVICORN_TRACEBACK not in logs.text


@pytest.mark.parametrize("chunked", [False, True])
def test_a_body_of_exactly_the_limit_is_accepted(chunked) -> None:
    calls: list[str] = []
    body = payload_of_size(LIMIT)
    assert len(body) == LIMIT
    request = (
        post_chunked("/test/echo", [body[:500_000], body[500_000:]])
        if chunked
        else post("/test/echo", body)
    )

    response = serve(current_app(calls), request)

    assert response.status == 200
    assert calls == ["xxxxxxxx"]


def test_the_limit_comes_from_the_api_settings(monkeypatch) -> None:
    calls: list[str] = []
    small = current_app(calls, max_request_bytes=2_048)

    assert_too_large(serve(small, post("/test/echo", payload_of_size(2_049))))
    assert serve(small, post("/test/echo", payload_of_size(2_048))).status == 200
    assert calls == ["xxxxxxxx"]

    assert ApiSettings(_env_file=None).max_request_bytes == LIMIT  # type: ignore[call-arg]
    monkeypatch.setenv("RESEARCHPILOT_API_MAX_REQUEST_BYTES", "4096")
    assert ApiSettings(_env_file=None).max_request_bytes == 4_096  # type: ignore[call-arg]
    for bad in (1_023, 16_777_217):
        with pytest.raises(ValidationError):
            ApiSettings(_env_file=None, max_request_bytes=bad)  # type: ignore[call-arg]
