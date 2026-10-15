"""Offline tests for the API skeleton: app factory, health endpoints and problem responses.

The Ollama readiness check is always injected or patched; no network call is made.
"""

import logging

import httpx
import pytest
from fakes import FakeHistory
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel, ValidationError

from researchpilot.api import __main__ as api_main
from researchpilot.api import create_app
from researchpilot.api import dependencies as api_dependencies
from researchpilot.api.dependencies import AppDependencies, check_ollama
from researchpilot.api.problems import PROBLEM_MEDIA_TYPE
from researchpilot.api.settings import ApiSettings
from researchpilot.config import Settings

SECRET_BASE_URL = "http://user:hunter2@ollama.internal:11434"


class Probe:
    """Readiness probe double that records calls."""

    def __init__(self, result: bool | Exception) -> None:
        self.result = result
        self.calls = 0

    def __call__(self) -> bool:
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def make_app(probe: Probe | None = None, **settings_overrides) -> FastAPI:
    settings = Settings(_env_file=None, **settings_overrides)  # type: ignore[call-arg]
    api_settings = ApiSettings(_env_file=None)  # type: ignore[call-arg]
    return create_app(
        settings=settings,
        api_settings=api_settings,
        ollama_probe=probe or Probe(True),
        history=FakeHistory(),
    )


def assert_problem(response: httpx.Response, status: int, code: str) -> dict:
    assert response.status_code == status
    assert response.headers["content-type"] == PROBLEM_MEDIA_TYPE
    body = response.json()
    assert body["status"] == status
    assert body["code"] == code
    assert body["type"] == f"urn:researchpilot:problem:{code}"
    assert set(body) == {"type", "title", "status", "detail", "code", "errors"}
    return body


# --- Application factory ---


def test_create_app_uses_injected_dependencies_and_contacts_nothing() -> None:
    probe = Probe(True)
    settings = Settings(_env_file=None, ollama_model="custom:1b")  # type: ignore[call-arg]
    api_settings = ApiSettings(_env_file=None, ready_timeout_seconds=0.5)  # type: ignore[call-arg]

    app = create_app(settings=settings, api_settings=api_settings, ollama_probe=probe)

    dependencies = app.state.dependencies
    assert isinstance(dependencies, AppDependencies)
    assert dependencies.settings is settings
    assert dependencies.api_settings is api_settings
    assert dependencies.ollama_probe is probe
    assert probe.calls == 0
    paths = set(app.openapi()["paths"])
    assert paths == {
        "/api/v1/health",
        "/api/v1/health/ready",
        "/api/v1/jobs",
        "/api/v1/jobs/{job_id}",
        "/api/v1/jobs/{job_id}/result",
        "/api/v1/jobs/{job_id}/cancel",
    }


def test_default_probe_uses_the_configured_url_and_timeout(monkeypatch) -> None:
    calls: list[tuple[str, float]] = []

    def fake_get(url: str, *, timeout: float) -> httpx.Response:
        calls.append((url, timeout))
        return httpx.Response(200, json={"version": "test"})

    monkeypatch.setattr(api_dependencies.httpx, "get", fake_get)
    settings = Settings(_env_file=None, ollama_base_url="http://127.0.0.1:11555")  # type: ignore[call-arg]
    api_settings = ApiSettings(_env_file=None, ready_timeout_seconds=0.75)  # type: ignore[call-arg]
    app = create_app(settings=settings, api_settings=api_settings)

    assert app.state.dependencies.ollama_probe() is True
    assert calls == [("http://127.0.0.1:11555/api/version", 0.75)]


def test_api_settings_default_to_localhost_and_validate(monkeypatch) -> None:
    defaults = ApiSettings(_env_file=None)  # type: ignore[call-arg]
    assert (defaults.host, defaults.port, defaults.ready_timeout_seconds) == (
        "127.0.0.1",
        8000,
        2.0,
    )

    monkeypatch.setenv("RESEARCHPILOT_API_PORT", "8123")
    assert ApiSettings(_env_file=None).port == 8123  # type: ignore[call-arg]
    for bad in ({"port": 0}, {"ready_timeout_seconds": 0}, {"ready_timeout_seconds": 30}):
        with pytest.raises(ValidationError):
            ApiSettings(_env_file=None, **bad)  # type: ignore[call-arg]


# --- Liveness ---


def test_liveness_is_ok_without_contacting_ollama() -> None:
    probe = Probe(ConnectionError("Ollama is down"))

    response = TestClient(make_app(probe)).get("/api/v1/health")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    assert response.json() == {
        "status": "ok",
        "service": "researchpilot",
        "version": response.json()["version"],
    }
    assert response.json()["version"]
    assert probe.calls == 0


# --- Readiness ---


def test_readiness_is_200_when_ollama_answers() -> None:
    probe = Probe(True)

    response = TestClient(make_app(probe)).get("/api/v1/health/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "checks": {"ollama": "ok", "jobs": "ok", "history": "ok"},
    }
    assert probe.calls == 1


@pytest.mark.parametrize(
    "result", [False, ConnectionError("boom: secret detail"), RuntimeError("x")]
)
def test_readiness_is_503_problem_when_ollama_is_unavailable(result) -> None:
    response = TestClient(make_app(Probe(result), ollama_base_url=SECRET_BASE_URL)).get(
        "/api/v1/health/ready"
    )

    body = assert_problem(response, 503, "ollama_unavailable")
    assert response.headers["retry-after"] == "5"
    assert body["errors"] == []
    for leaked in ("hunter2", "ollama.internal", "11434", "secret detail", "boom"):
        assert leaked not in response.text


@pytest.mark.parametrize(
    "failure",
    [
        httpx.ConnectError("connection refused"),
        httpx.ReadTimeout("timed out"),
        httpx.Response(500),
    ],
)
def test_default_probe_reports_not_ready_on_errors_timeouts_and_bad_status(
    monkeypatch, failure
) -> None:
    def fake_get(_url: str, *, timeout: float) -> httpx.Response:
        assert timeout == 2.0
        if isinstance(failure, Exception):
            raise failure
        return failure

    monkeypatch.setattr(api_dependencies.httpx, "get", fake_get)

    assert check_ollama("http://127.0.0.1:11434/", 2.0) is False


# --- Problem responses ---


def test_unknown_route_and_wrong_method_are_problems() -> None:
    client = TestClient(make_app())

    assert_problem(client.get("/api/v1/nope"), 404, "not_found")
    not_allowed = client.post("/api/v1/health")
    assert_problem(not_allowed, 405, "method_not_allowed")
    assert not_allowed.headers["allow"] == "GET"


class Payload(BaseModel):
    question: str
    count: int


def test_validation_errors_list_fields_without_echoing_input() -> None:
    app = make_app()

    @app.post("/test/validate")  # test-only route; Phase 2 has no endpoints with a body
    def validate(payload: Payload) -> dict:
        return {"ok": True}

    response = TestClient(app).post(
        "/test/validate", json={"question": "PRIVATE-SOURCE-TEXT", "count": "not-a-number"}
    )

    body = assert_problem(response, 422, "validation_failed")
    assert body["errors"] == [
        {
            "field": "body.count",
            "message": "Input should be a valid integer, unable to parse string as an integer",
            "type": "int_parsing",
        }
    ]
    assert "PRIVATE-SOURCE-TEXT" not in response.text
    assert "not-a-number" not in response.text


def test_malformed_json_is_a_validation_problem() -> None:
    app = make_app()

    @app.post("/test/validate")
    def validate(payload: Payload) -> dict:
        return {"ok": True}

    response = TestClient(app).post(
        "/test/validate",
        content=b'{"question": "PRIVATE',
        headers={"content-type": "application/json"},
    )

    assert_problem(response, 422, "validation_failed")
    assert "PRIVATE" not in response.text


def test_unexpected_errors_are_500_problems_without_details(caplog) -> None:
    app = make_app()

    @app.get("/test/crash")  # test-only route
    def crash() -> dict:
        raise RuntimeError("secret source text inside an exception")

    with caplog.at_level(logging.ERROR, logger="researchpilot.api.problems"):
        response = TestClient(app, raise_server_exceptions=False).get("/test/crash")

    body = assert_problem(response, 500, "internal_error")
    assert body["detail"] == "The server could not complete the request."
    assert "secret source text" not in response.text
    assert "RuntimeError" in caplog.text
    assert "secret source text" not in caplog.text


# --- Entry point ---


def test_entry_point_serves_on_the_configured_host_and_port(monkeypatch) -> None:
    served: dict = {}

    def fake_run(app, *, host: str, port: int) -> None:
        served.update(app=app, host=host, port=port)

    monkeypatch.setenv("RESEARCHPILOT_API_PORT", "8124")
    monkeypatch.setattr(api_main.uvicorn, "run", fake_run)
    monkeypatch.setattr(
        api_main, "create_app", lambda *, api_settings: FastAPI(title=f"stub {api_settings.port}")
    )

    api_main.main()

    assert served["host"] == "127.0.0.1"
    assert served["port"] == 8124
    assert served["app"].title == "stub 8124"
