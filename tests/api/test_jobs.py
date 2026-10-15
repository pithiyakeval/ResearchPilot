"""Offline tests for the research job manager.

Research and result building are injected doubles synchronised with ``threading.Event``,
or the real pipeline driven by ``ScriptedModel``; no model, network or MCP server is used.
"""

import logging
import threading
import time
from datetime import UTC, datetime, timedelta

import pytest
from fakes import ScriptedModel, complete
from pydantic import ValidationError

from researchpilot.api.jobs import (
    FAILURE_MESSAGES,
    JobInput,
    JobManager,
    JobNotCancellableError,
    JobNotFoundError,
    JobState,
    ManagerClosedError,
    QueueFullError,
)
from researchpilot.api.settings import ApiSettings
from researchpilot.config import Settings
from researchpilot.errors import (
    InputError,
    InputTooLargeError,
    ModelRunError,
    ResearchPilotError,
    RetrievalError,
)
from researchpilot.pipeline import prepare_request, run_research
from researchpilot.results import ResearchResult, build_result
from researchpilot.schemas import WebSource
from researchpilot.web_research import WebEvidence

# Upper bound for every wait, so a broken manager fails the test instead of hanging it.
TIMEOUT = 5.0
SECRET = "SECRET-SOURCE-TEXT"
QUESTION = f"Compare System Alpha and System Beta {SECRET}-QUESTION"
SOURCES = (
    f"# Source A: Task completion\n\nSystem Alpha completed 85 of 100 tasks. {SECRET}\n\n"
    "# Source B: Configuration\n\nSystem Beta required less initial configuration."
)
JOB = JobInput(request=prepare_request(QUESTION, SOURCES))
OUTCOME = object()  # stands in for a ResearchOutcome when ``build`` is faked


def settings() -> Settings:
    return Settings(_env_file=None)  # type: ignore[call-arg]


@pytest.fixture(scope="module")
def result() -> ResearchResult:
    model = ScriptedModel(["notes", complete(), "Report [S1]."])
    outcome = run_research(
        JOB.request, settings=settings(), model_factory=lambda _settings, **_kwargs: model
    )
    return build_result(outcome, settings())


class FakeClock:
    """Returns one second later on every call."""

    def __init__(self) -> None:
        self.ticks = 0

    def __call__(self) -> datetime:
        self.ticks += 1
        return datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=self.ticks)


class Research:
    """Research double; with ``block``, each call waits until ``release`` is set."""

    def __init__(self, *, error: BaseException | None = None, block: bool = False) -> None:
        self.error = error
        self.started = threading.Event()
        self.release = threading.Event()
        if not block:
            self.release.set()
        self.inputs: list[JobInput] = []
        self._lock = threading.Lock()

    @property
    def calls(self) -> int:
        with self._lock:
            return len(self.inputs)

    def __call__(self, job_input: JobInput) -> object:
        with self._lock:
            self.inputs.append(job_input)
        self.started.set()
        assert self.release.wait(TIMEOUT), "test never released the job"
        if self.error is not None:
            raise self.error
        return OUTCOME


def manager_for(research: Research, result: ResearchResult, **options) -> JobManager:
    def build(outcome: object) -> ResearchResult:
        assert outcome is OUTCOME
        return result

    return JobManager(research, build, **options)  # type: ignore[arg-type]


@pytest.fixture
def managers():
    """Shuts down every manager a test creates, so no worker outlives its test."""
    created: list[JobManager] = []
    yield created
    for manager in created:
        manager.shutdown(TIMEOUT)


@pytest.fixture
def make(managers, result):
    def make(research: Research | None = None, **options) -> JobManager:
        manager = manager_for(research or Research(), result, **options)
        managers.append(manager)
        return manager

    return make


def failed_with(cause: Exception, error_type: type[ResearchPilotError]) -> ResearchPilotError:
    """A typed error built as the pipeline builds it: original message, original cause."""
    error = error_type(str(cause))
    error.__cause__ = cause
    return error


# --- Settings ---


def test_job_settings_have_the_approved_defaults_and_validate(monkeypatch) -> None:
    defaults = ApiSettings(_env_file=None)  # type: ignore[call-arg]
    assert (
        defaults.max_queued_jobs,
        defaults.max_retained_jobs,
        defaults.shutdown_timeout_seconds,
    ) == (4, 50, 5.0)

    monkeypatch.setenv("RESEARCHPILOT_API_MAX_QUEUED_JOBS", "7")
    assert ApiSettings(_env_file=None).max_queued_jobs == 7  # type: ignore[call-arg]
    for bad in (
        {"max_queued_jobs": 0},
        {"max_retained_jobs": 0},
        {"shutdown_timeout_seconds": -1},
        {"shutdown_timeout_seconds": 61},
    ):
        with pytest.raises(ValidationError):
            ApiSettings(_env_file=None, **bad)  # type: ignore[call-arg]


def test_manager_rejects_invalid_limits(result) -> None:
    for limits in ({"max_queued": 0}, {"max_retained": 0}):
        with pytest.raises(ValueError):
            manager_for(Research(), result, **limits)


# --- Success ---


def test_job_runs_to_success_with_ordered_timestamps(make, result) -> None:
    research = Research(block=True)
    manager = make(research, clock=FakeClock(), id_factory=lambda: "job-1")

    queued = manager.submit(JOB)
    assert (queued.id, queued.state, queued.mode, queued.queue_position) == (
        "job-1",
        JobState.QUEUED,
        "local",
        1,
    )
    assert queued.started_at is None and queued.result is None

    manager.start()
    assert research.started.wait(TIMEOUT)
    running = manager.get("job-1")
    assert running.state is JobState.RUNNING
    assert running.queue_position is None and running.finished_at is None

    research.release.set()
    done = manager.wait("job-1", TIMEOUT)

    assert done.state is JobState.SUCCEEDED
    assert done.result is result
    assert done.failure is None and done.cancel_reason is None
    assert done.started_at is not None and done.finished_at is not None
    assert done.created_at < done.started_at < done.finished_at
    assert research.inputs == [JOB]
    # The finished job no longer holds the request or its source material.
    assert manager._jobs["job-1"].input is None


@pytest.mark.parametrize("web", [False, True])
def test_real_pipeline_job_succeeds_without_source_text_in_the_result(managers, web) -> None:
    model = ScriptedModel(["notes", complete(), "Report [S1]."])
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
    manager = JobManager.from_settings(
        settings(),
        ApiSettings(_env_file=None),  # type: ignore[call-arg]
        model_factory=lambda _settings, **_kwargs: model,
        web_gatherer=lambda _question: evidence,
    )
    managers.append(manager)
    manager.start()

    job = manager.submit(JobInput(request=JOB.request, web=web))
    done = manager.wait(job.id, TIMEOUT)

    assert done.state is JobState.SUCCEEDED, done.failure
    assert done.mode == ("web" if web else "local")
    assert done.result is not None
    assert done.result.mode == done.mode
    assert [source.id for source in done.result.sources] == (
        ["S1", "S2", "S3"] if web else ["S1", "S2"]
    )
    assert "System Alpha completed 85" not in done.result.model_dump_json()
    assert model.call_count == 3


# --- Failures ---


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (failed_with(ValueError(f"bad {SECRET}"), InputError), "validation_failed"),
        (failed_with(ValueError(f"too big {SECRET}"), InputTooLargeError), "input_too_large"),
        (failed_with(OSError(f"search {SECRET}"), RetrievalError), "web_retrieval_failed"),
        (failed_with(ConnectionError(f"ollama {SECRET}"), ModelRunError), "model_failed"),
        (ResearchPilotError(f"base {SECRET}"), "internal_error"),
        (RuntimeError(f"unexpected {SECRET}"), "internal_error"),
    ],
)
def test_failures_store_a_code_and_fixed_message_only(make, caplog, error, code) -> None:
    manager = make(Research(error=error))
    manager.start()

    with caplog.at_level(logging.INFO, logger="researchpilot.api.jobs"):
        job = manager.submit(JOB)
        done = manager.wait(job.id, TIMEOUT)

    assert done.state is JobState.FAILED
    assert done.failure is not None
    assert (done.failure.code, done.failure.message) == (code, FAILURE_MESSAGES[code])
    assert done.result is None
    assert type(error).__name__ in caplog.text
    assert SECRET not in caplog.text
    assert SECRET not in repr(done)


def test_build_result_failure_is_result_failed(managers, caplog) -> None:
    def broken_build(_outcome: object) -> ResearchResult:
        raise KeyError(f"missing {SECRET}")

    manager = JobManager(Research(), broken_build)  # type: ignore[arg-type]
    managers.append(manager)
    manager.start()

    with caplog.at_level(logging.WARNING, logger="researchpilot.api.jobs"):
        done = manager.wait(manager.submit(JOB).id, TIMEOUT)

    assert done.state is JobState.FAILED
    assert done.failure is not None and done.failure.code == "result_failed"
    assert "KeyError" in caplog.text
    assert SECRET not in caplog.text


def test_worker_keeps_running_after_a_failed_job(make) -> None:
    outcomes = iter([RuntimeError("first job breaks"), None])

    class FlakyResearch(Research):
        def __call__(self, job_input: JobInput) -> object:
            self.error = next(outcomes)
            return super().__call__(job_input)

    manager = make(FlakyResearch())
    first, second = manager.submit(JOB), manager.submit(JOB)
    manager.start()

    assert manager.wait(first.id, TIMEOUT).state is JobState.FAILED
    assert manager.wait(second.id, TIMEOUT).state is JobState.SUCCEEDED


def test_a_base_exception_fails_the_job_and_closes_the_manager(make, monkeypatch, caplog) -> None:
    class Stop(BaseException):
        pass

    thread_errors: list[type[BaseException]] = []
    monkeypatch.setattr(threading, "excepthook", lambda args: thread_errors.append(args.exc_type))
    research = Research(error=Stop(f"fatal {SECRET}"), block=True)
    manager = make(research)
    first = manager.submit(JOB)

    with caplog.at_level(logging.INFO, logger="researchpilot.api.jobs"):
        manager.start()
        assert research.started.wait(TIMEOUT)
        second = manager.submit(JOB)
        research.release.set()
        failed = manager.wait(first.id, TIMEOUT)
        stranded = manager.get(second.id)

        with pytest.raises(ManagerClosedError):
            manager.submit(JOB)
        with pytest.raises(ManagerClosedError):
            manager.start()
        assert manager.shutdown(TIMEOUT) is True

    assert failed.state is JobState.FAILED
    assert failed.failure is not None
    assert (failed.failure.code, failed.failure.message) == (
        "internal_error",
        FAILURE_MESSAGES["internal_error"],
    )
    assert (stranded.state, stranded.cancel_reason) == (JobState.CANCELLED, "worker_stopped")
    assert research.calls == 1
    assert thread_errors == []
    assert "Stop" in caplog.text
    assert SECRET not in caplog.text
    assert SECRET not in repr(failed) + repr(stranded)


# --- Queue limits and concurrency ---


def test_queue_refuses_jobs_beyond_its_limit_until_one_leaves(make) -> None:
    manager = make(max_queued=2)  # never started, so every job stays queued
    first = manager.submit(JOB)
    second = manager.submit(JOB)

    with pytest.raises(QueueFullError):
        manager.submit(JOB)

    manager.cancel(first.id)
    third = manager.submit(JOB)
    assert manager.get(second.id).queue_position == 1
    assert third.queue_position == 2


def test_the_running_job_does_not_count_against_the_queue(make) -> None:
    research = Research(block=True)
    manager = make(research, max_queued=1)
    running = manager.submit(JOB)
    manager.start()
    assert research.started.wait(TIMEOUT)

    queued = manager.submit(JOB)
    with pytest.raises(QueueFullError):
        manager.submit(JOB)

    research.release.set()
    assert manager.wait(running.id, TIMEOUT).state is JobState.SUCCEEDED
    assert manager.wait(queued.id, TIMEOUT).state is JobState.SUCCEEDED


def test_concurrent_submissions_never_overfill_the_queue(make) -> None:
    manager = make(max_queued=5)
    barrier = threading.Barrier(20)
    accepted: list[str] = []
    refused: list[QueueFullError] = []

    def submit() -> None:
        barrier.wait(TIMEOUT)
        try:
            accepted.append(manager.submit(JOB).id)
        except QueueFullError as exc:
            refused.append(exc)

    threads = [threading.Thread(target=submit) for _ in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(TIMEOUT)

    assert len(accepted) == 5 and len(set(accepted)) == 5
    assert len(refused) == 15
    positions = sorted(manager.get(job_id).queue_position for job_id in accepted)
    assert positions == [1, 2, 3, 4, 5]


def test_cancel_racing_the_worker_never_loses_or_repeats_a_transition(make) -> None:
    research = Research()
    manager = make(research, max_queued=1)
    manager.start()
    cancelled = 0

    for _ in range(50):
        job = manager.submit(JOB)
        try:
            manager.cancel(job.id)
        except JobNotCancellableError as exc:
            assert exc.state in {JobState.RUNNING, JobState.SUCCEEDED}
            expected = JobState.SUCCEEDED
        else:
            cancelled += 1
            expected = JobState.CANCELLED
        assert manager.wait(job.id, TIMEOUT).state is expected

    assert research.calls == 50 - cancelled


# --- Cancellation ---


def test_cancelling_a_queued_job_is_final_and_idempotent(make) -> None:
    manager = make(clock=FakeClock())
    job = manager.submit(JOB)

    cancelled = manager.cancel(job.id)
    again = manager.cancel(job.id)

    assert cancelled.state is JobState.CANCELLED
    assert cancelled.cancel_reason == "client"
    assert cancelled.queue_position is None and cancelled.started_at is None
    assert cancelled.finished_at is not None
    assert again == cancelled
    assert manager._jobs[job.id].input is None


def test_cancelled_jobs_are_never_run(make) -> None:
    research = Research()
    manager = make(research)
    skipped = manager.submit(JOB)
    kept = manager.submit(JOB)
    manager.cancel(skipped.id)

    manager.start()

    assert manager.wait(kept.id, TIMEOUT).state is JobState.SUCCEEDED
    assert research.calls == 1


def test_a_running_job_cannot_be_cancelled_and_still_finishes(make) -> None:
    research = Research(block=True)
    manager = make(research)
    job = manager.submit(JOB)
    manager.start()
    assert research.started.wait(TIMEOUT)

    with pytest.raises(JobNotCancellableError) as raised:
        manager.cancel(job.id)
    assert raised.value.state is JobState.RUNNING

    research.release.set()
    assert manager.wait(job.id, TIMEOUT).state is JobState.SUCCEEDED
    with pytest.raises(JobNotCancellableError):
        manager.cancel(job.id)


def test_unknown_job_ids_are_not_found(make) -> None:
    manager = make()
    for call in (manager.get, manager.cancel, lambda job_id: manager.wait(job_id, 0)):
        with pytest.raises(JobNotFoundError):
            call("missing")


def test_wait_returns_the_current_state_after_its_timeout(make) -> None:
    manager = make()  # never started
    job = manager.submit(JOB)

    start = time.monotonic()
    snapshot = manager.wait(job.id, 0.05)

    assert snapshot.state is JobState.QUEUED
    assert time.monotonic() - start < TIMEOUT


# --- Retention ---


def test_retention_forgets_the_oldest_finished_jobs_first(make) -> None:
    manager = make(max_retained=2)
    manager.start()
    ids = [manager.wait(manager.submit(JOB).id, TIMEOUT).id for _ in range(4)]

    for forgotten in ids[:2]:
        with pytest.raises(JobNotFoundError):
            manager.get(forgotten)
    assert [manager.get(job_id).state for job_id in ids[2:]] == [JobState.SUCCEEDED] * 2


def test_retention_never_forgets_queued_jobs(make) -> None:
    manager = make(max_retained=1, max_queued=3)
    first, second, waiting = (manager.submit(JOB) for _ in range(3))

    manager.cancel(first.id)
    manager.cancel(second.id)

    with pytest.raises(JobNotFoundError):
        manager.get(first.id)
    assert manager.get(second.id).state is JobState.CANCELLED
    assert manager.get(waiting.id).state is JobState.QUEUED


# --- Start and shutdown ---


def test_started_and_closed_report_the_lifecycle(make) -> None:
    manager = make()
    assert (manager.started, manager.closed) == (False, False)

    manager.start()
    assert (manager.started, manager.closed) == (True, False)

    assert manager.shutdown(TIMEOUT) is True
    assert (manager.started, manager.closed) == (True, True)

    unstarted = make()
    unstarted.shutdown(TIMEOUT)
    assert (unstarted.started, unstarted.closed) == (False, True)


def test_start_is_idempotent(make) -> None:
    manager = make()
    manager.start()
    worker = manager._worker

    manager.start()

    assert manager._worker is worker


def test_shutdown_cancels_queued_jobs_and_does_not_wait_for_a_running_one(make) -> None:
    research = Research(block=True)
    manager = make(research)
    running = manager.submit(JOB)
    manager.start()
    assert research.started.wait(TIMEOUT)
    queued = [manager.submit(JOB), manager.submit(JOB)]

    start = time.monotonic()
    stopped = manager.shutdown(0.1)

    assert stopped is False
    assert time.monotonic() - start < TIMEOUT
    for job in queued:
        snapshot = manager.get(job.id)
        assert (snapshot.state, snapshot.cancel_reason) == (JobState.CANCELLED, "shutdown")
    assert manager.get(running.id).state is JobState.RUNNING
    with pytest.raises(ManagerClosedError):
        manager.submit(JOB)
    with pytest.raises(ManagerClosedError):
        manager.start()

    # The abandoned job still finishes, after which the worker exits.
    research.release.set()
    assert manager.shutdown(TIMEOUT) is True
    assert manager.get(running.id).state is JobState.SUCCEEDED
    assert research.calls == 1


def test_shutdown_of_an_idle_or_unstarted_manager_is_immediate(make) -> None:
    idle = make()
    idle.start()
    unstarted = make()
    job = unstarted.submit(JOB)

    assert idle.shutdown(TIMEOUT) is True
    assert idle.shutdown(TIMEOUT) is True
    assert unstarted.shutdown(TIMEOUT) is True
    assert unstarted.get(job.id).cancel_reason == "shutdown"
