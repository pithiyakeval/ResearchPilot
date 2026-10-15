"""Offline tests for research history in the app lifecycle.

Covers degraded startup, starting the worker once history is ready, readiness, lock loss and
shutdown order. History is a FakeHistory, or a real HistoryRuntime whose connections always
fail; no database, model, network or MCP server is contacted.
"""

import logging
import os
import threading

import httpx
import psycopg
import pytest
from fakes import FakeHistory, Probe
from fastapi.testclient import TestClient

from researchpilot.api import create_app
from researchpilot.api.history import DRIVER_LOGGERS, HistoryRuntime, HistoryState, HistoryStatus
from researchpilot.api.jobs import JobInput, JobManager
from researchpilot.api.problems import PROBLEM_MEDIA_TYPE
from researchpilot.api.settings import ApiSettings
from researchpilot.config import Settings
from researchpilot.history.database import create_history_engine
from researchpilot.history.settings import DatabaseSettings
from researchpilot.pipeline import prepare_request

TIMEOUT = 5.0
JOBS = "/api/v1/jobs"
READINESS = "/api/v1/health/ready"
SUBMISSION = {
    "question": "Compare System Alpha and System Beta",
    "source_material": "# Source A: Tasks\n\nSystem Alpha completed 85 of 100 tasks in a test.",
}
SECRET = "hunter2-SECRET"
READY = HistoryStatus(HistoryState.READY)


def unavailable(reason: str) -> HistoryStatus:
    return HistoryStatus(HistoryState.UNAVAILABLE, reason)  # type: ignore[arg-type]


class Research:
    """Research double; with ``block``, each call waits until ``release`` is set."""

    def __init__(self, *, block: bool = False) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        if not block:
            self.release.set()
        self.inputs: list[JobInput] = []

    @property
    def calls(self) -> int:
        return len(self.inputs)

    def __call__(self, job_input: JobInput) -> object:
        self.inputs.append(job_input)
        self.started.set()
        assert self.release.wait(TIMEOUT), "test never released the job"
        return object()


@pytest.fixture(autouse=True)
def no_database_environment(monkeypatch):
    for name in list(os.environ):
        if name.startswith("RESEARCHPILOT_DATABASE_"):
            monkeypatch.delenv(name)


@pytest.fixture(autouse=True)
def restore_driver_loggers():
    levels = {name: logging.getLogger(name).level for name in DRIVER_LOGGERS}
    yield
    for name, level in levels.items():
        logging.getLogger(name).setLevel(level)


@pytest.fixture
def make():
    managers: list[JobManager] = []

    def make(history, research: Research | None = None):
        manager = JobManager(research or Research(), lambda _outcome: object())  # type: ignore[arg-type]
        managers.append(manager)
        probe = Probe()
        app = create_app(
            settings=Settings(_env_file=None),  # type: ignore[call-arg]
            api_settings=ApiSettings(_env_file=None),  # type: ignore[call-arg]
            ollama_probe=probe,
            job_manager=manager,
            history=history,
        )
        return app, manager, probe

    yield make
    for manager in managers:
        manager.shutdown(TIMEOUT)


def assert_problem(response: httpx.Response, code: str, retry_after: str | None) -> None:
    assert response.status_code == 503
    assert response.headers["content-type"] == PROBLEM_MEDIA_TYPE
    assert response.json()["code"] == code
    assert response.headers.get("retry-after") == retry_after


# --- Degraded startup ---


def test_unavailable_history_at_startup_leaves_the_app_alive_and_degraded(make) -> None:
    history = FakeHistory(unavailable("database_unavailable"))
    app, manager, probe = make(history)

    with TestClient(app) as client:
        assert client.get("/api/v1/health").status_code == 200
        assert manager.started is False
        assert_problem(client.get(READINESS), "history_unavailable", "5")
        assert_problem(client.post(JOBS, json=SUBMISSION), "history_unavailable", "5")
        assert manager.started is False

    assert probe.calls == 0  # history is reported without asking Ollama
    assert history.events[-1] == "close"


def test_a_real_runtime_that_cannot_connect_starts_degraded_without_leaking_details(
    make, caplog
) -> None:
    caplog.set_level(logging.DEBUG)

    def refuse(*_args, **_kwargs):
        raise psycopg.OperationalError(f'connection to "db.invalid" failed: password {SECRET}')

    history = HistoryRuntime(
        DatabaseSettings(_env_file=None, url=f"postgresql+psycopg://rp:{SECRET}@db.invalid/rp"),  # type: ignore[call-arg]
        engine_factory=lambda settings: create_history_engine(settings, creator=refuse),
    )
    app, manager, _probe = make(history)

    with TestClient(app) as client:
        readiness = client.get(READINESS)
        submission = client.post(JOBS, json=SUBMISSION)

    assert_problem(readiness, "history_unavailable", "5")
    assert_problem(submission, "history_unavailable", "5")
    assert manager.started is False
    assert history.status == unavailable("closed")
    for text in (readiness.text, submission.text, caplog.text):
        assert SECRET not in text and "db.invalid" not in text


def test_by_default_history_is_built_from_settings_and_unconfigured_without_a_url(make) -> None:
    app = create_app(
        settings=Settings(_env_file=None),  # type: ignore[call-arg]
        api_settings=ApiSettings(_env_file=None),  # type: ignore[call-arg]
        ollama_probe=Probe(),
    )
    history = app.state.dependencies.history

    assert isinstance(history, HistoryRuntime)
    assert history.status.state is HistoryState.UNCONFIGURED
    with TestClient(app) as client:
        assert_problem(client.get(READINESS), "history_unconfigured", None)
    assert app.state.dependencies.jobs.started is False


# --- Starting the worker ---


def test_the_worker_starts_once_history_becomes_ready_and_only_once(make) -> None:
    history = FakeHistory(unavailable("database_unavailable"))
    app, manager, _probe = make(history)

    with TestClient(app) as client:
        assert manager.started is False
        history.current = READY

        first = client.get(READINESS)
        worker = manager._worker
        for _ in range(3):
            assert client.get(READINESS).status_code == 200

        job = client.post(JOBS, json=SUBMISSION).json()
        assert manager.wait(job["id"], TIMEOUT).state == "succeeded"

    assert first.json() == {
        "status": "ready",
        "checks": {"ollama": "ok", "jobs": "ok", "history": "ok"},
    }
    assert worker is not None and manager._worker is worker  # never a second worker


def test_a_submission_starts_the_worker_once_history_becomes_ready(make) -> None:
    history = FakeHistory(unavailable("database_unavailable"))
    app, manager, _probe = make(history)

    with TestClient(app) as client:
        history.current = READY
        response = client.post(JOBS, json=SUBMISSION)
        assert manager.wait(response.json()["id"], TIMEOUT).state == "succeeded"

    assert response.status_code == 202
    assert manager.started is True


def test_ready_history_never_starts_a_worker_without_the_lifespan(make) -> None:
    app, manager, _probe = make(FakeHistory())

    client = TestClient(app)  # no "with": the app is not serving
    assert client.get(READINESS).status_code == 200
    assert_problem(client.post(JOBS, json=SUBMISSION), "jobs_unavailable", None)
    assert manager.started is False


# --- Readiness ---


@pytest.mark.parametrize(
    ("status", "code", "retry_after"),
    [
        (HistoryStatus(HistoryState.UNCONFIGURED, "not_configured"), "history_unconfigured", None),
        (unavailable("schema_outdated"), "history_schema_outdated", None),
        (unavailable("encoding_unsupported"), "history_schema_outdated", None),
        (unavailable("lock_held_elsewhere"), "history_unavailable", "5"),
        (unavailable("recovery_failed"), "history_unavailable", "5"),
    ],
)
def test_readiness_reports_the_history_state(make, status, code, retry_after) -> None:
    app, manager, probe = make(FakeHistory(status))

    with TestClient(app) as client:
        response = client.get(READINESS)

    assert_problem(response, code, retry_after)
    assert probe.calls == 0
    assert manager.started is False


# --- Lock loss ---


def test_losing_the_lock_stops_the_worker_before_the_next_job(make) -> None:
    history = FakeHistory()
    research = Research(block=True)
    app, manager, _probe = make(history, research)

    with TestClient(app) as client:
        first = client.post(JOBS, json=SUBMISSION).json()["id"]
        assert research.started.wait(TIMEOUT)
        second = client.post(JOBS, json=SUBMISSION).json()["id"]

        history.current = unavailable("lock_lost")  # e.g. the lock's connection was dropped
        research.release.set()
        stranded = manager.wait(second, TIMEOUT)
        readiness = client.get(READINESS)

    # The running job was not interrupted; the next one never started.
    assert manager.get(first).state == "succeeded"
    assert (stranded.state, stranded.cancel_reason) == ("cancelled", "worker_stopped")
    assert research.calls == 1
    assert manager.closed
    assert_problem(readiness, "jobs_unavailable", None)


def test_a_run_gate_refusal_or_error_stops_a_manager_without_running_the_job() -> None:
    for gate in (lambda: False, lambda: 1 / 0):
        research = Research()
        manager = JobManager(research, lambda _outcome: object())  # type: ignore[arg-type]
        manager.set_run_gate(gate)
        job = manager.submit(JobInput(request=_request()))
        manager.start()

        stranded = manager.wait(job.id, TIMEOUT)

        assert (stranded.state, stranded.cancel_reason) == ("cancelled", "worker_stopped")
        assert research.calls == 0
        assert manager.shutdown(TIMEOUT) is True


def _request():
    return prepare_request(SUBMISSION["question"], SUBMISSION["source_material"])


# --- Shutdown ---


@pytest.mark.parametrize("initial", [READY, unavailable("database_unavailable")])
def test_shutdown_stops_the_manager_before_closing_history(make, initial) -> None:
    history = FakeHistory(initial)
    app, manager, _probe = make(history)
    order: list[str] = []
    shutdown = manager.shutdown

    def recording_shutdown(timeout: float) -> bool:
        order.append("jobs.shutdown")
        return shutdown(timeout)

    manager.shutdown = recording_shutdown  # type: ignore[method-assign]
    close = history.close

    def recording_close() -> None:
        order.append("history.close")
        close()

    history.close = recording_close  # type: ignore[method-assign]

    with TestClient(app):
        pass

    assert order == ["jobs.shutdown", "history.close"]
    assert manager.closed


def test_startup_quiets_the_database_driver_loggers(make) -> None:
    for name in DRIVER_LOGGERS:
        logging.getLogger(name).setLevel(logging.DEBUG)
    app, _manager, _probe = make(FakeHistory())

    with TestClient(app):
        levels = [logging.getLogger(name).level for name in DRIVER_LOGGERS]

    assert levels == [logging.CRITICAL] * len(DRIVER_LOGGERS)
