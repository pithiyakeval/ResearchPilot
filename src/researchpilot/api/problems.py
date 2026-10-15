"""RFC 9457 ``application/problem+json`` errors for every API failure.

Responses never contain request bodies, source text, settings or raw exception messages:
validation errors list only the field location, a message and an error type, and
unexpected errors get a fixed message while only the exception's type is logged.

Unexpected errors are contained by ``UnexpectedErrorMiddleware`` rather than left to a 500
handler alone: Starlette's ``ServerErrorMiddleware`` re-raises after calling that handler so
the server can log the error, and uvicorn would then log the full traceback, message included.
"""

import logging
from collections.abc import Iterable, Mapping
from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger(__name__)

PROBLEM_MEDIA_TYPE = "application/problem+json"
_HTTP_CODES = {404: "not_found", 405: "method_not_allowed", 413: "content_too_large"}


class FieldError(BaseModel):
    field: str
    message: str
    type: str


class Problem(BaseModel):
    """Error body; ``code`` is the stable, machine-readable identifier."""

    type: str
    title: str
    status: int
    detail: str
    code: str
    errors: list[FieldError] = Field(default_factory=list)


class ProblemError(Exception):
    """Raise from a route to return a problem response."""

    def __init__(
        self,
        status: int,
        code: str,
        title: str,
        detail: str,
        headers: Mapping[str, str] | None = None,
        errors: list[FieldError] | None = None,
    ) -> None:
        super().__init__(detail)
        self.status, self.code, self.title, self.detail = status, code, title, detail
        self.headers = dict(headers or {})
        self.errors = list(errors or [])


def field_errors(
    errors: Iterable[Mapping[str, Any]], prefix: tuple[str, ...] = ()
) -> list[FieldError]:
    """Pydantic-style errors as field, message and type only; the input is never included."""
    return [
        FieldError(
            field=".".join(str(part) for part in (*prefix, *error.get("loc", ()))),
            message=str(error.get("msg", "Invalid value.")),
            type=str(error.get("type", "value_error")),
        )
        for error in errors
    ]


def validation_problem(errors: list[FieldError]) -> ProblemError:
    return ProblemError(
        422,
        "validation_failed",
        "Request validation failed",
        "One or more request fields are invalid.",
        errors=errors,
    )


def problem_response(
    status: int,
    code: str,
    title: str,
    detail: str,
    *,
    errors: list[FieldError] | None = None,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    body = Problem(
        type=f"urn:researchpilot:problem:{code}",
        title=title,
        status=status,
        detail=detail,
        code=code,
        errors=errors or [],
    )
    return JSONResponse(
        body.model_dump(),
        status_code=status,
        media_type=PROBLEM_MEDIA_TYPE,
        headers=dict(headers or {}),
    )


async def _problem_error(_request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, ProblemError)
    return problem_response(
        exc.status, exc.code, exc.title, exc.detail, errors=exc.errors, headers=exc.headers
    )


async def _validation_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    # Only location, message and type: the submitted value ("input") is never echoed back.
    return await _problem_error(request, validation_problem(field_errors(exc.errors())))


async def _http_error(_request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)
    status = HTTPStatus(exc.status_code)
    return problem_response(
        status.value,
        _HTTP_CODES.get(status.value, "http_error"),
        status.phrase,
        f"{status.description}.",
        headers=exc.headers,
    )


class RequestBodyTooLarge(StarletteHTTPException):
    """Raised while a route reads a request body larger than the limit."""

    def __init__(self) -> None:
        super().__init__(status_code=413)


def _content_too_large() -> JSONResponse:
    # A fixed title: Python versions disagree on the standard phrase for 413.
    return problem_response(
        413,
        "content_too_large",
        "Content too large",
        "The request body exceeds the maximum allowed size.",
    )


async def _body_too_large(_request: Request, _exc: Exception) -> JSONResponse:
    return _content_too_large()


def _internal_error() -> JSONResponse:
    return problem_response(
        500,
        "internal_error",
        "Internal server error",
        "The server could not complete the request.",
    )


def _log_unexpected(exc: BaseException, method: str, path: str) -> None:
    # The exception message may contain request or model content, so only its type is logged.
    logger.error("Unhandled %s while handling %s %s", type(exc).__name__, method, path)


async def _unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    # Fallback for anything raised outside UnexpectedErrorMiddleware.
    _log_unexpected(exc, request.method, request.url.path)
    return _internal_error()


class UnexpectedErrorMiddleware:
    """Answer any unhandled ``Exception`` with the fixed 500 problem and never re-raise it.

    If the response has already started, the client's response is cut short instead; either
    way only the exception's type, the method and the path are logged.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        response_started = False

        async def tracking_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, receive, tracking_send)
        except Exception as exc:  # noqa: BLE001 - re-raising would let the server log its message
            _log_unexpected(exc, scope["method"], scope["path"])
            if response_started:
                return
            try:
                await _internal_error()(scope, receive, send)
            except Exception as send_exc:  # noqa: BLE001 - e.g. the client has disconnected
                logger.warning("Could not send the error response: %s", type(send_exc).__name__)


def _declared_length(scope: Scope) -> int | None:
    for name, value in scope["headers"]:
        if name == b"content-length":
            try:
                return int(value)
            except ValueError:
                return None
    return None


class RequestBodyLimitMiddleware:
    """Refuse request bodies larger than ``max_bytes`` with a 413 problem.

    A larger ``Content-Length`` is refused before any route runs; a body sent without one
    (chunked) is counted as it is read, and the route's body parsing fails with
    ``RequestBodyTooLarge`` before the endpoint is called.
    """

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        declared = _declared_length(scope)
        if declared is not None and declared > self.max_bytes:
            await _content_too_large()(scope, receive, send)
            return
        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise RequestBodyTooLarge
            return message

        await self.app(scope, limited_receive, send)


def install_problem_handlers(app: FastAPI, *, max_request_bytes: int) -> None:
    """Make every error response, including FastAPI's own, a problem+json body."""
    app.add_exception_handler(ProblemError, _problem_error)
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(RequestBodyTooLarge, _body_too_large)
    app.add_exception_handler(StarletteHTTPException, _http_error)
    app.add_exception_handler(Exception, _unexpected_error)
    # Each added middleware wraps the ones added before it, so the containment middleware
    # is outermost and also catches anything the body limit raises.
    app.add_middleware(RequestBodyLimitMiddleware, max_bytes=max_request_bytes)
    app.add_middleware(UnexpectedErrorMiddleware)
