"""Structured job lifecycle log events: safe fields only, and never fatal to a job."""

import logging
import threading

import pytest
from fakes import ScriptedModel, complete

from researchpilot.api.jobs import JobInput, JobManager, JobState
from researchpilot.config import Settings
from researchpilot.errors import ModelRunError
from researchpilot.pipeline import prepare_request, run_research
from researchpilot.results import build_result

TIMEOUT = 5.0
QUESTION_SECRET = "QUESTION-SENTINEL how did System Alpha do"
SOURCE_SECRET = "SOURCE-SENTINEL"
SOURCES = f"# Source A: Tasks\n\nSystem Alpha completed 85 of 100 tasks. {SOURCE_SECRET}"
SETTINGS = Settings(_env_file=None)  # type: ignore[call-arg]


def job() -> JobInput:
    return JobInput(request=prepare_request(QUESTION_SECRET, SOURCES))


def events(caplog) -> list[dict]:
    return [r.job_event for r in caplog.records if hasattr(r, "job_event")]


@pytest.fixture
def managers():
    made: list[JobManager] = []
    yield made
    for manager in made:
        manager.shutdown(TIMEOUT)


def test_a_succeeded_job_logs_queued_started_and_finished_with_run_metadata(
    managers, caplog
) -> None:
    caplog.set_level(logging.INFO, logger="researchpilot.api.jobs")
    model = ScriptedModel(["notes", complete(), "Alpha completed 85 tasks [S1]."])

    def research(job_input: JobInput):
        return run_research(
            job_input.request, settings=SETTINGS, model_factory=lambda _s, **_k: model
        )

    manager = JobManager(research, lambda o: build_result(o, SETTINGS), model="gemma3:4b")
    managers.append(manager)
    manager.start()
    snapshot = manager.wait(manager.submit(job()).id, TIMEOUT)

    assert snapshot.state is JobState.SUCCEEDED
    logged = events(caplog)
    assert [e["event"] for e in logged] == ["job_queued", "job_started", "job_finished"]
    assert {e["job_id"] for e in logged} == {snapshot.id}
    finished = logged[-1]
    assert finished["state"] == "succeeded"
    assert (finished["mode"], finished["model"]) == ("local", "gemma3:4b")
    assert finished["stop_reason"] == "evaluation_complete"
    assert (finished["analysis_passes"], finished["model_calls"]) == (1, 3)
    assert finished["invalid_citations"] == 0
    assert finished["failure_code"] is None
    assert finished["total_seconds"] >= finished["run_seconds"] >= 0
    for text in (QUESTION_SECRET, SOURCE_SECRET, "Alpha completed 85 tasks"):
        assert text not in caplog.text


def test_a_failed_job_logs_its_failure_code_but_not_the_exception_message(managers, caplog) -> None:
    caplog.set_level(logging.INFO, logger="researchpilot.api.jobs")

    def research(_job_input: JobInput):
        raise ModelRunError(f"connection refused while sending {SOURCE_SECRET}")

    manager = JobManager(research, lambda o: o)
    managers.append(manager)
    manager.start()
    manager.wait(manager.submit(job()).id, TIMEOUT)

    finished = events(caplog)[-1]
    assert (finished["event"], finished["state"]) == ("job_finished", "failed")
    assert finished["failure_code"] == "model_failed"
    assert finished["stop_reason"] is None
    assert SOURCE_SECRET not in caplog.text
    assert "connection refused" not in caplog.text


def test_a_cancelled_job_logs_its_cancel_reason_and_no_run_time(managers, caplog) -> None:
    caplog.set_level(logging.INFO, logger="researchpilot.api.jobs")
    manager = JobManager(lambda _job: None, lambda o: o)  # type: ignore[arg-type,return-value]
    managers.append(manager)
    queued = manager.submit(job())  # never started, so it stays queued

    manager.cancel(queued.id)

    finished = events(caplog)[-1]
    assert (finished["state"], finished["cancel_reason"]) == ("cancelled", "client")
    assert finished["run_seconds"] is None and finished["queue_seconds"] is None
    assert finished["total_seconds"] >= 0


def test_logging_never_breaks_a_job_whose_result_has_an_unexpected_shape(managers, caplog) -> None:
    caplog.set_level(logging.INFO, logger="researchpilot.api.jobs")
    hook_calls: list[object] = []
    previous_hook = threading.excepthook
    threading.excepthook = hook_calls.append
    try:
        manager = JobManager(lambda _job: object(), lambda _outcome: object())  # type: ignore[arg-type,return-value]
        managers.append(manager)
        manager.start()
        first = manager.wait(manager.submit(job()).id, TIMEOUT)
        second = manager.wait(manager.submit(job()).id, TIMEOUT)
    finally:
        threading.excepthook = previous_hook

    assert (first.state, second.state) == (JobState.SUCCEEDED, JobState.SUCCEEDED)
    assert hook_calls == []
    assert events(caplog)[-1]["stop_reason"] is None
