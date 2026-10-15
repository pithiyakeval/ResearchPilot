"""Offline tests for history-backed job reads (Phase 4B.5).

Status and result reads use the in-memory job when there is one, and otherwise the job's
record in research history. History is a FakeHistory whose store can be seeded with records
or made to fail; no database, model, network or MCP server is contacted.
"""

import threading
import uuid
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from fakes import FakeHistory, ScriptedModel, complete
from fastapi.testclient import TestClient

from researchpilot.api import create_app
from researchpilot.api.history import HistoryState, HistoryStatus
from researchpilot.api.jobs import FAILURE_MESSAGES, UNKNOWN_FAILURE_MESSAGE, JobInput, JobManager
from researchpilot.api.problems import PROBLEM_MEDIA_TYPE
from researchpilot.api.settings import ApiSettings
from researchpilot.config import Settings
from researchpilot.history.records import HistoryRecord
from researchpilot.history.store import (
    INTERRUPTED_BY_RESTART,
    HistorySchemaError,
    HistoryUnavailableError,
)
from researchpilot.pipeline import prepare_request, run_research
from researchpilot.results import ResearchResult, build_result

TIMEOUT = 5.0
JOBS = "/api/v1/jobs"
QUESTION = "Compare System Alpha and System Beta"
SOURCES = "# Source A: Tasks\n\nSystem Alpha completed 85 of 100 tasks in a fictional test."
SUBMISSION = {"question": QUESTION, "source_material": SOURCES}
CREATED = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
STATUS_FIELDS = {
    "id",
    "state",
    "mode",
    "created_at",
    "started_at",
    "finished_at",
    "queue_position",
    "cancel_reason",
    "failure",
}


@pytest.fixture(scope="module")
def result() -> ResearchResult:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    model = ScriptedModel(["notes", complete(), "Report [S1]."])
    outcome = run_research(
        prepare_request(QUESTION, SOURCES),
        settings=settings,
        model_factory=lambda _settings, **_kwargs: model,
    )
    return build_result(outcome, settings)


class Research:
    def __init__(self) -> None:
        self.calls = 0
        self.lock = threading.Lock()

    def __call__(self, job_input: JobInput) -> object:
        with self.lock:
            self.calls += 1
        return object()


def stored(
    state: str,
    *,
    result: ResearchResult | None = None,
    failure_code: str | None = None,
    cancel_reason: str | None = None,
    started: bool = True,
    finished: bool = True,
) -> HistoryRecord:
    return HistoryRecord(
        id=uuid.uuid4().hex,
        label="Stored research",
        mode="web",
        state=state,  # type: ignore[arg-type]
        created_at=CREATED,
        started_at=CREATED + timedelta(seconds=1) if started else None,
        finished_at=CREATED + timedelta(minutes=1) if finished else None,
        failure_code=failure_code,
        cancel_reason=cancel_reason,  # type: ignore[arg-type]
        expires_at=CREATED + timedelta(days=14) if finished else None,
        result=result,
    )


@pytest.fixture
def app_parts(result):
    """A served app whose FakeHistory store can be seeded; shut down afterwards."""
    history = FakeHistory()
    research = Research()
    manager = JobManager(research, lambda _outcome: result)
    app = create_app(
        settings=Settings(_env_file=None),  # type: ignore[call-arg]
        api_settings=ApiSettings(_env_file=None),  # type: ignore[call-arg]
        ollama_probe=lambda: True,
        job_manager=manager,
        history=history,
    )
    with TestClient(app) as client:
        yield client, manager, history, research


def seed(history: FakeHistory, record: HistoryRecord) -> str:
    history.backing_store.stored[record.id] = record
    return record.id


def assert_problem(response: httpx.Response, status: int, code: str) -> dict:
    assert response.status_code == status
    assert response.headers["content-type"] == PROBLEM_MEDIA_TYPE
    body = response.json()
    assert body["code"] == code
    return body


# --- In memory first ---


def test_in_memory_jobs_are_served_without_reading_history(app_parts, result) -> None:
    client, manager, history, _research = app_parts
    job_id = client.post(JOBS, json=SUBMISSION).json()["id"]
    manager.wait(job_id, TIMEOUT)

    status = client.get(f"{JOBS}/{job_id}")
    fetched = client.get(f"{JOBS}/{job_id}/result")

    assert status.status_code == 200 and status.json()["state"] == "succeeded"
    assert fetched.json() == result.model_dump(mode="json")
    assert history.backing_store.gets == []


def test_an_unrecorded_in_memory_job_is_503_and_never_read_from_history(app_parts) -> None:
    client, manager, history, _research = app_parts
    history.backing_store.failing.add("record_finished")
    job_id = client.post(JOBS, json=SUBMISSION).json()["id"]
    assert manager.wait(job_id, TIMEOUT).unrecorded
    history.backing_store.stored[job_id] = stored("succeeded")  # stale data must be ignored

    for path in (f"{JOBS}/{job_id}", f"{JOBS}/{job_id}/result"):
        assert_problem(client.get(path), 503, "history_unavailable")
    assert history.backing_store.gets == []


# --- From history ---


def test_a_stored_succeeded_job_and_its_result_are_served(app_parts, result) -> None:
    client, _manager, history, research = app_parts
    record = stored("succeeded", result=result)
    job_id = seed(history, record)

    status = client.get(f"{JOBS}/{job_id}")
    fetched = client.get(f"{JOBS}/{job_id}/result")

    assert status.status_code == 200
    assert set(status.json()) == STATUS_FIELDS  # exactly the existing contract
    assert status.json() == {
        "id": job_id,
        "state": "succeeded",
        "mode": "web",
        "created_at": "2026-01-01T12:00:00Z",
        "started_at": "2026-01-01T12:00:01Z",
        "finished_at": "2026-01-01T12:01:00Z",
        "queue_position": None,
        "cancel_reason": None,
        "failure": None,
    }
    assert fetched.status_code == 200 and fetched.json() == result.model_dump(mode="json")
    assert research.calls == 0  # nothing is re-run


def test_a_stored_failed_job_is_honest_and_has_no_result(app_parts) -> None:
    client, _manager, history, _research = app_parts
    job_id = seed(history, stored("failed", failure_code="model_failed"))

    status = client.get(f"{JOBS}/{job_id}").json()

    assert status["state"] == "failed"
    assert status["failure"] == {
        "code": "model_failed",
        "message": FAILURE_MESSAGES["model_failed"],
    }
    assert_problem(client.get(f"{JOBS}/{job_id}/result"), 409, "job_failed")


def test_a_stored_cancelled_job_is_honest_and_has_no_result(app_parts) -> None:
    client, _manager, history, _research = app_parts
    job_id = seed(history, stored("cancelled", cancel_reason="client", started=False))

    status = client.get(f"{JOBS}/{job_id}").json()

    assert (status["state"], status["cancel_reason"], status["failure"]) == (
        "cancelled",
        "client",
        None,
    )
    assert_problem(client.get(f"{JOBS}/{job_id}/result"), 409, "job_cancelled")


@pytest.mark.parametrize("started", [True, False])
def test_a_job_interrupted_by_a_restart_is_reported_as_failed(app_parts, started) -> None:
    client, _manager, history, research = app_parts
    job_id = seed(history, stored("failed", failure_code=INTERRUPTED_BY_RESTART, started=started))

    status = client.get(f"{JOBS}/{job_id}").json()
    no_result = client.get(f"{JOBS}/{job_id}/result")

    assert status["state"] == "failed"
    assert status["failure"] == {
        "code": "interrupted_by_restart",
        "message": FAILURE_MESSAGES["interrupted_by_restart"],
    }
    assert_problem(no_result, 409, "job_failed")
    assert research.calls == 0


def test_an_unknown_stored_failure_code_gets_a_generic_message(app_parts) -> None:
    client, _manager, history, _research = app_parts
    job_id = seed(history, stored("failed", failure_code="future_failure"))

    failure = client.get(f"{JOBS}/{job_id}").json()["failure"]

    assert failure == {"code": "future_failure", "message": UNKNOWN_FAILURE_MESSAGE}


@pytest.mark.parametrize("state", ["queued", "running"])
def test_a_stored_unfinished_job_is_not_finished(app_parts, state) -> None:
    client, _manager, history, _research = app_parts
    job_id = seed(history, stored(state, started=state == "running", finished=False))

    status = client.get(f"{JOBS}/{job_id}").json()

    assert (status["state"], status["queue_position"]) == (state, None)
    assert_problem(client.get(f"{JOBS}/{job_id}/result"), 409, "job_not_finished")


def test_a_stored_success_without_a_result_is_never_served_as_complete(app_parts) -> None:
    client, _manager, history, _research = app_parts
    job_id = seed(history, stored("succeeded", result=None))

    assert_problem(client.get(f"{JOBS}/{job_id}/result"), 503, "history_unavailable")


# --- Not found versus unavailable ---


def test_unknown_and_hidden_jobs_are_404(app_parts) -> None:
    client, _manager, history, _research = app_parts
    unknown = uuid.uuid4().hex  # also how the store reports an expired record

    for path in (f"{JOBS}/{unknown}", f"{JOBS}/{unknown}/result"):
        assert_problem(client.get(path), 404, "job_not_found")
    assert history.backing_store.gets == [unknown, unknown]


def test_an_id_no_stored_job_can_have_is_404_without_reading_history(app_parts) -> None:
    client, _manager, history, _research = app_parts

    assert_problem(client.get(f"{JOBS}/not-a-job"), 404, "job_not_found")
    assert history.backing_store.gets == []


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (HistoryUnavailableError(), "history_unavailable"),
        # Also what the store raises for a stored result that no longer validates.
        (HistorySchemaError(), "history_schema_outdated"),
    ],
)
def test_history_read_failures_are_503_never_404(app_parts, error, code) -> None:
    client, _manager, history, _research = app_parts
    history.backing_store.get_error = error
    job_id = uuid.uuid4().hex

    for path in (f"{JOBS}/{job_id}", f"{JOBS}/{job_id}/result"):
        body = assert_problem(client.get(path), 503, code)
        assert "Traceback" not in body["detail"]


@pytest.mark.parametrize(
    ("status", "code"),
    [
        (HistoryStatus(HistoryState.UNAVAILABLE, "database_unavailable"), "history_unavailable"),
        (HistoryStatus(HistoryState.UNAVAILABLE, "schema_outdated"), "history_schema_outdated"),
        (HistoryStatus(HistoryState.UNCONFIGURED, "not_configured"), "history_unconfigured"),
    ],
)
def test_history_that_is_not_ready_is_503_without_reading(app_parts, status, code) -> None:
    client, _manager, history, _research = app_parts
    job_id = seed(history, stored("succeeded"))
    history.current = status

    for path in (f"{JOBS}/{job_id}", f"{JOBS}/{job_id}/result"):
        assert_problem(client.get(path), 503, code)
    assert history.backing_store.gets == []


def test_cancelling_a_job_known_only_to_history_is_unchanged(app_parts) -> None:
    client, _manager, history, _research = app_parts
    job_id = seed(history, stored("succeeded"))

    # Cancellation is unchanged in this stage: only jobs in memory can be cancelled.
    assert_problem(client.post(f"{JOBS}/{job_id}/cancel"), 404, "job_not_found")
