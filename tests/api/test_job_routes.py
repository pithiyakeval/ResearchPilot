"""Offline tests for the research job HTTP API.

The job manager is always injected: its research is a double synchronised with
``threading.Event``, or the real pipeline driven by ``ScriptedModel`` and a fake web
gatherer. No real server is started and no model, network or MCP server is contacted.
"""

import logging
import threading
import time

import httpx
import pytest
from fakes import FakeHistory, Probe, Research, ScriptedModel, complete
from fastapi.testclient import TestClient

from researchpilot.api import create_app
from researchpilot.api.jobs import (
    FAILURE_MESSAGES,
    JobInput,
    JobManager,
    JobState,
    ManagerClosedError,
)
from researchpilot.api.problems import PROBLEM_MEDIA_TYPE
from researchpilot.api.settings import ApiSettings
from researchpilot.config import Settings
from researchpilot.errors import InputTooLargeError
from researchpilot.pipeline import check_input_fits, prepare_request, run_research
from researchpilot.results import ResearchResult, build_result
from researchpilot.schemas import WebSource
from researchpilot.web_research import WebEvidence

# Upper bound for every wait, so a broken API fails the test instead of hanging it.
TIMEOUT = 5.0
SECRET = "SECRET-SOURCE-TEXT"
QUESTION = "Compare System Alpha and System Beta"
SOURCES = (
    f"# Source A: Task completion\n\nSystem Alpha completed 85 of 100 tasks. {SECRET}\n\n"
    "# Source B: Configuration\n\nSystem Beta required less initial configuration."
)
# Within the 80,000-character limit, but too large for the default 16,384-token context.
BIG_SOURCES = f"# Source A: Long\n\n{SECRET} " + "word " * 15_900
SUBMISSION = {"question": QUESTION, "source_material": SOURCES}
JOBS = "/api/v1/jobs"
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


def settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


def api_settings(**overrides) -> ApiSettings:
    return ApiSettings(_env_file=None, **overrides)  # type: ignore[call-arg]


@pytest.fixture(scope="module")
def result() -> ResearchResult:
    model = ScriptedModel(["notes", complete(), "Report [S1]."])
    outcome = run_research(
        prepare_request(QUESTION, SOURCES),
        settings=settings(),
        model_factory=lambda _settings, **_kwargs: model,
    )
    return build_result(outcome, settings())


@pytest.fixture
def make(result):
    """Builds an app around an injected manager; shuts every manager down afterwards."""
    managers: list[JobManager] = []

    def make(research=None, *, probe=None, max_queued=4, **api_overrides):
        manager = JobManager(research or Research(), lambda _outcome: result, max_queued=max_queued)
        managers.append(manager)
        app = create_app(
            settings=settings(),
            api_settings=api_settings(**api_overrides),
            ollama_probe=probe or Probe(),
            job_manager=manager,
            history=FakeHistory(),
        )
        return app, manager

    yield make
    for manager in managers:
        manager.shutdown(TIMEOUT)


@pytest.fixture
def logs(caplog):
    """Everything logged during the test, by any logger."""
    caplog.set_level(logging.DEBUG)
    return caplog


def assert_problem(response: httpx.Response, status: int, code: str) -> dict:
    assert response.status_code == status
    assert response.headers["content-type"] == PROBLEM_MEDIA_TYPE
    body = response.json()
    assert (body["status"], body["code"]) == (status, code)
    assert body["type"] == f"urn:researchpilot:problem:{code}"
    assert set(body) == {"type", "title", "status", "detail", "code", "errors"}
    return body


def assert_no_secret(logs, *responses: httpx.Response) -> None:
    for response in responses:
        assert SECRET not in response.text
    assert SECRET not in logs.text


# --- Submission and the job lifecycle ---


def test_a_job_is_queued_then_its_status_and_result_are_served(make, result, logs) -> None:
    research = Research(block=True)
    app, manager = make(research)

    with TestClient(app) as client:
        submitted = client.post(JOBS, json=SUBMISSION)
        job = submitted.json()
        assert research.started.wait(TIMEOUT)
        running = client.get(submitted.headers["location"])
        research.release.set()
        manager.wait(job["id"], TIMEOUT)
        done = client.get(f"{JOBS}/{job['id']}")
        fetched = client.get(f"{JOBS}/{job['id']}/result")

    assert submitted.status_code == 202
    assert submitted.headers["location"] == f"{JOBS}/{job['id']}"
    assert set(job) == STATUS_FIELDS
    assert (job["state"], job["mode"], job["queue_position"]) == ("queued", "local", 1)
    assert job["started_at"] is None and job["failure"] is None
    assert (running.json()["state"], running.json()["queue_position"]) == ("running", None)
    assert done.json()["state"] == "succeeded"
    assert done.json()["started_at"] and done.json()["finished_at"]
    assert fetched.status_code == 200
    assert fetched.json() == result.model_dump(mode="json")
    # The research ran on the validated request, exactly as the pipeline prepares it.
    assert research.inputs == [JobInput(request=prepare_request(QUESTION, SOURCES), web=False)]
    assert_no_secret(logs, submitted, running, done, fetched)


@pytest.mark.parametrize("web", [False, True])
def test_local_and_web_jobs_run_the_real_pipeline_offline(logs, web) -> None:
    model = ScriptedModel(["notes", complete(), "Report [S1]."])
    gathered: list[str] = []
    evidence = WebEvidence(
        sources=(
            WebSource(
                kind="search_excerpt",
                title="Benchmark page",
                url="https://docs.example.org/benchmarks",
                text=f"Excerpt about benchmark tasks. {SECRET}-WEB",
            ),
        ),
        warnings=(),
        search_calls=1,
        fetch_calls=0,
    )

    def gatherer(question: str) -> WebEvidence:
        gathered.append(question)
        return evidence

    manager = JobManager.from_settings(
        settings(),
        api_settings(),
        model_factory=lambda _settings, **_kwargs: model,
        web_gatherer=gatherer,
    )
    app = create_app(
        settings=settings(),
        api_settings=api_settings(),
        ollama_probe=Probe(),
        job_manager=manager,
        history=FakeHistory(),
    )

    with TestClient(app) as client:
        job = client.post(JOBS, json={**SUBMISSION, "web": web}).json()
        manager.wait(job["id"], TIMEOUT)
        status = client.get(f"{JOBS}/{job['id']}").json()
        fetched = client.get(f"{JOBS}/{job['id']}/result")

    assert status["state"] == "succeeded", status["failure"]
    assert status["mode"] == fetched.json()["mode"] == ("web" if web else "local")
    assert [source["id"] for source in fetched.json()["sources"]] == (
        ["S1", "S2", "S3"] if web else ["S1", "S2"]
    )
    assert gathered == ([QUESTION] if web else [])
    assert model.call_count == 3
    assert_no_secret(logs, fetched)


def test_create_app_builds_the_real_manager_without_starting_it() -> None:
    app = create_app(settings=settings(), api_settings=api_settings(), ollama_probe=Probe())

    jobs = app.state.dependencies.jobs
    assert isinstance(jobs, JobManager)
    assert (jobs.started, jobs.closed) == (False, False)


# --- Validation ---


@pytest.mark.parametrize(
    ("payload", "field", "error_type"),
    [
        ({"question": QUESTION}, "body.source_material", "missing"),
        ({**SUBMISSION, "extra": SECRET}, "body.extra", "extra_forbidden"),
        ({**SUBMISSION, "web": "yes"}, "body.web", "bool_type"),
        ({**SUBMISSION, "web": 1}, "body.web", "bool_type"),
        ({**SUBMISSION, "question": 12_345_678}, "body.question", "string_type"),
        # From the research engine's own request validation:
        ({**SUBMISSION, "question": " " * 20}, "body.question", "value_error"),
        ({**SUBMISSION, "question": "x" * 2_001 + SECRET}, "body.question", "string_too_long"),
        ({**SUBMISSION, "source_material": SECRET}, "body.source_material", "string_too_short"),
    ],
)
def test_invalid_submissions_are_422_without_echoing_input(
    make, logs, payload, field, error_type
) -> None:
    research = Research()
    app, _manager = make(research)

    with TestClient(app) as client:
        response = client.post(JOBS, json=payload)

    body = assert_problem(response, 422, "validation_failed")
    assert [(error["field"], error["type"]) for error in body["errors"]] == [(field, error_type)]
    assert research.calls == 0
    assert_no_secret(logs, response)


def test_oversized_research_input_is_rejected_before_it_is_queued(make, logs) -> None:
    research = Research()
    app, manager = make(research)

    with TestClient(app) as client:
        rejected = client.post(JOBS, json={**SUBMISSION, "source_material": BIG_SOURCES})
        accepted = client.post(JOBS, json=SUBMISSION).json()
        manager.wait(accepted["id"], TIMEOUT)

    body = assert_problem(rejected, 422, "input_too_large")
    assert body["errors"] == []
    assert accepted["queue_position"] == 1
    assert research.inputs == [JobInput(request=prepare_request(QUESTION, SOURCES))]
    assert_no_secret(logs, rejected)


def test_check_input_fits_uses_the_configured_context_budget() -> None:
    request = prepare_request(QUESTION, BIG_SOURCES)

    with pytest.raises(InputTooLargeError):
        check_input_fits(request, settings())
    check_input_fits(request, settings(ollama_num_ctx=131_072))
    check_input_fits(prepare_request(QUESTION, SOURCES), settings())


def test_a_body_over_the_size_limit_is_413_and_never_queued(make, logs) -> None:
    research = Research()
    app, _manager = make(research, max_request_bytes=4_096)

    with TestClient(app) as client:
        response = client.post(JOBS, json={**SUBMISSION, "source_material": SECRET * 400})

    assert_problem(response, 413, "content_too_large")
    assert research.calls == 0
    assert_no_secret(logs, response)


# --- Availability ---


def test_submissions_are_503_unless_the_worker_is_running(make) -> None:
    app, _manager = make()
    no_lifespan = TestClient(app)  # without "with", the lifespan and so the worker never start

    before = no_lifespan.post(JOBS, json=SUBMISSION)
    with TestClient(app) as client:
        during = client.post(JOBS, json=SUBMISSION)
    after = no_lifespan.post(JOBS, json=SUBMISSION)

    assert during.status_code == 202
    for response in (before, after):
        assert_problem(response, 503, "jobs_unavailable")
        assert "retry-after" not in response.headers


def test_a_full_queue_is_503_with_retry_after(make) -> None:
    research = Research(block=True)
    app, _manager = make(research, max_queued=1)

    with TestClient(app) as client:
        assert client.post(JOBS, json=SUBMISSION).status_code == 202
        assert research.started.wait(TIMEOUT)
        assert client.post(JOBS, json=SUBMISSION).status_code == 202
        full = client.post(JOBS, json=SUBMISSION)
        research.release.set()

    assert_problem(full, 503, "queue_full")
    assert full.headers["retry-after"] == "30"


def test_a_fatal_worker_failure_makes_jobs_and_readiness_unavailable(
    make, monkeypatch, logs
) -> None:
    class Stop(BaseException):
        pass

    thread_errors: list[type[BaseException]] = []
    monkeypatch.setattr(threading, "excepthook", lambda args: thread_errors.append(args.exc_type))
    research = Research(error=Stop(f"fatal {SECRET}"), block=True)
    probe = Probe()
    app, manager = make(research, probe=probe)

    with TestClient(app) as client:
        ready_before = client.get("/api/v1/health/ready")
        first = client.post(JOBS, json=SUBMISSION).json()["id"]
        assert research.started.wait(TIMEOUT)
        second = client.post(JOBS, json=SUBMISSION).json()["id"]
        research.release.set()
        manager.wait(first, TIMEOUT)
        failed = client.get(f"{JOBS}/{first}")
        stranded = client.get(f"{JOBS}/{second}")
        no_result = client.get(f"{JOBS}/{first}/result")
        submit = client.post(JOBS, json=SUBMISSION)
        ready_after = client.get("/api/v1/health/ready")

    assert ready_before.status_code == 200
    assert failed.json()["state"] == "failed"
    assert failed.json()["failure"] == {
        "code": "internal_error",
        "message": FAILURE_MESSAGES["internal_error"],
    }
    assert (stranded.json()["state"], stranded.json()["cancel_reason"]) == (
        "cancelled",
        "worker_stopped",
    )
    assert_problem(no_result, 409, "job_failed")
    assert_problem(submit, 503, "jobs_unavailable")
    assert_problem(ready_after, 503, "jobs_unavailable")
    assert probe.calls == 1  # the closed manager is reported without asking Ollama
    assert thread_errors == []
    assert "Stop" in logs.text
    assert_no_secret(logs, failed, stranded, no_result, submit, ready_after)


def test_readiness_reports_jobs_and_skips_ollama_once_closed(make) -> None:
    probe = Probe()
    app, _manager = make(probe=probe)

    before = TestClient(app).get("/api/v1/health/ready")  # not started yet: not a failure
    with TestClient(app) as client:
        during = client.get("/api/v1/health/ready")
    after = TestClient(app).get("/api/v1/health/ready")

    for response in (before, during):
        assert response.json() == {
            "status": "ready",
            "checks": {"ollama": "ok", "jobs": "ok", "history": "ok"},
        }
    assert_problem(after, 503, "jobs_unavailable")
    assert "retry-after" not in after.headers
    assert probe.calls == 2


# --- Status, results and cancellation ---


@pytest.mark.parametrize(
    ("method", "suffix"), [("get", ""), ("get", "/result"), ("post", "/cancel")]
)
def test_unknown_jobs_are_404_without_echoing_the_id(make, method, suffix) -> None:
    app, _manager = make()

    response = getattr(TestClient(app), method)(f"{JOBS}/missing-{SECRET}{suffix}")

    assert_problem(response, 404, "job_not_found")
    assert SECRET not in response.text


def test_results_are_409_with_a_state_specific_code_until_the_job_succeeds(make) -> None:
    research = Research(block=True)
    app, manager = make(research)

    with TestClient(app) as client:
        running = client.post(JOBS, json=SUBMISSION).json()["id"]
        assert research.started.wait(TIMEOUT)
        queued = client.post(JOBS, json=SUBMISSION).json()["id"]
        cancelled = client.post(JOBS, json=SUBMISSION).json()["id"]
        assert client.post(f"{JOBS}/{cancelled}/cancel").status_code == 200
        early = {
            job_id: client.get(f"{JOBS}/{job_id}/result") for job_id in (running, queued, cancelled)
        }
        research.release.set()
        manager.wait(queued, TIMEOUT)
        succeeded = client.get(f"{JOBS}/{running}/result")

    assert_problem(early[running], 409, "job_not_finished")
    assert_problem(early[queued], 409, "job_not_finished")
    assert_problem(early[cancelled], 409, "job_cancelled")
    assert succeeded.status_code == 200


def test_a_failed_job_shows_only_its_failure_code(make, logs) -> None:
    app, manager = make(Research(error=RuntimeError(f"boom {SECRET}")))

    with TestClient(app) as client:
        job = client.post(JOBS, json=SUBMISSION).json()["id"]
        manager.wait(job, TIMEOUT)
        status = client.get(f"{JOBS}/{job}")
        no_result = client.get(f"{JOBS}/{job}/result")

    assert status.json()["state"] == "failed"
    assert status.json()["failure"] == {
        "code": "internal_error",
        "message": FAILURE_MESSAGES["internal_error"],
    }
    assert_problem(no_result, 409, "job_failed")
    assert "RuntimeError" in logs.text
    assert_no_secret(logs, status, no_result)


def test_a_model_failure_in_the_real_pipeline_is_model_failed(logs) -> None:
    class FailingModel:
        def invoke(self, *_args, **_kwargs):
            raise ConnectionError(f"Ollama refused {SECRET}")

    manager = JobManager.from_settings(
        settings(), api_settings(), model_factory=lambda _settings, **_kwargs: FailingModel()
    )
    app = create_app(
        settings=settings(),
        api_settings=api_settings(),
        ollama_probe=Probe(),
        job_manager=manager,
        history=FakeHistory(),
    )

    with TestClient(app) as client:
        job = client.post(JOBS, json=SUBMISSION).json()["id"]
        manager.wait(job, TIMEOUT)
        status = client.get(f"{JOBS}/{job}")

    assert status.json()["failure"] == {
        "code": "model_failed",
        "message": FAILURE_MESSAGES["model_failed"],
    }
    assert "ModelRunError" in logs.text
    assert_no_secret(logs, status)


def test_queue_positions_follow_cancellations(make) -> None:
    research = Research(block=True)
    app, _manager = make(research)

    with TestClient(app) as client:
        client.post(JOBS, json=SUBMISSION)
        assert research.started.wait(TIMEOUT)
        ids = [client.post(JOBS, json=SUBMISSION).json()["id"] for _ in range(3)]
        positions = [client.get(f"{JOBS}/{job_id}").json()["queue_position"] for job_id in ids]
        cancelled = client.post(f"{JOBS}/{ids[0]}/cancel")
        again = client.post(f"{JOBS}/{ids[0]}/cancel")
        later = [client.get(f"{JOBS}/{job_id}").json()["queue_position"] for job_id in ids[1:]]
        research.release.set()

    assert positions == [1, 2, 3]
    assert cancelled.status_code == again.status_code == 200
    assert set(cancelled.json()) == STATUS_FIELDS
    body = cancelled.json()
    assert (body["state"], body["cancel_reason"], body["queue_position"]) == (
        "cancelled",
        "client",
        None,
    )
    assert again.json() == body
    assert later == [1, 2]


def test_running_and_finished_jobs_cannot_be_cancelled(make) -> None:
    research = Research(block=True)
    app, manager = make(research)

    with TestClient(app) as client:
        job = client.post(JOBS, json=SUBMISSION).json()["id"]
        assert research.started.wait(TIMEOUT)
        running = client.post(f"{JOBS}/{job}/cancel")
        research.release.set()
        manager.wait(job, TIMEOUT)
        finished = client.post(f"{JOBS}/{job}/cancel")
        state = client.get(f"{JOBS}/{job}").json()["state"]

    assert "running" in assert_problem(running, 409, "job_not_cancellable")["detail"]
    assert "succeeded" in assert_problem(finished, 409, "job_not_cancellable")["detail"]
    assert state == "succeeded"


# --- Lifespan ---


def test_shutdown_cancels_queued_jobs_and_is_bounded_while_one_runs(make) -> None:
    research = Research(block=True)
    app, manager = make(research, shutdown_timeout_seconds=0.1)
    assert (manager.started, manager.closed) == (False, False)

    with TestClient(app) as client:
        assert (manager.started, manager.closed) == (True, False)
        running = client.post(JOBS, json=SUBMISSION).json()["id"]
        assert research.started.wait(TIMEOUT)
        queued = client.post(JOBS, json=SUBMISSION).json()["id"]
        exiting = time.monotonic()
    shutdown_seconds = time.monotonic() - exiting

    assert manager.closed
    assert shutdown_seconds < TIMEOUT
    after = TestClient(app)
    cancelled = after.get(f"{JOBS}/{queued}").json()
    assert (cancelled["state"], cancelled["cancel_reason"]) == ("cancelled", "shutdown")
    assert after.get(f"{JOBS}/{running}").json()["state"] == "running"

    # The abandoned job still finishes once released, but history is closed by then, so its
    # outcome cannot be recorded and is never presented as durably finished.
    research.release.set()
    assert manager.shutdown(TIMEOUT) is True
    assert manager.get(running).state is JobState.SUCCEEDED
    late = after.get(f"{JOBS}/{running}")
    assert late.status_code == 503 and late.json()["code"] == "history_unavailable"


def test_a_second_lifespan_on_the_same_app_fails_at_startup(make) -> None:
    app, _manager = make()
    with TestClient(app):
        pass

    with pytest.raises(ManagerClosedError), TestClient(app):
        pass


def test_openapi_documents_the_job_models(make) -> None:
    app, _manager = make()

    schema = app.openapi()

    components = schema["components"]["schemas"]
    assert {"JobSubmission", "JobStatus", "ResearchResult", "Problem"} <= set(components)
    assert components["JobSubmission"]["additionalProperties"] is False
    assert set(schema["paths"][JOBS]["post"]["responses"]) == {"202", "413", "422", "503"}
