"""Offline tests for recording jobs in research history (Phase 4B.4).

Jobs are recorded through HistoryRecorder into a FakeHistoryStore, so the real adapters and
record types are used without a database. Synchronisation uses events and conditions with
bounded waits; nothing sleeps.
"""

import threading

import httpx
import pytest
from fakes import FakeHistory, Research, ScriptedModel, complete
from fastapi.testclient import TestClient

from researchpilot.api import create_app
from researchpilot.api.history import HistoryRecorder, HistoryState, HistoryStatus
from researchpilot.api.jobs import (
    FAILURE_MESSAGES,
    JobInput,
    JobManager,
    JobNotCancellableError,
    JobNotFoundError,
    JobRecordingError,
    JobState,
    ManagerClosedError,
    QueueFullError,
)
from researchpilot.api.problems import PROBLEM_MEDIA_TYPE
from researchpilot.api.settings import ApiSettings
from researchpilot.config import Settings
from researchpilot.history.records import FinishedJob, QueuedJob, StartedJob
from researchpilot.pipeline import prepare_request, run_research
from researchpilot.results import ResearchResult, build_result

TIMEOUT = 5.0
JOBS = "/api/v1/jobs"
QUESTION = "Compare System Alpha and System Beta"
SOURCES = "# Source A: Tasks\n\nSystem Alpha completed 85 of 100 tasks in a fictional test."
SUBMISSION = {"question": QUESTION, "source_material": SOURCES}
JOB = JobInput(request=prepare_request(QUESTION, SOURCES))
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
LABEL_SECRET = "LABEL-SECRET"


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


@pytest.fixture
def managed(result):
    """Builds managers that record through a FakeHistory; shuts them all down afterwards."""
    managers: list[JobManager] = []

    def make(research: Research | None = None, **options):
        history = FakeHistory()
        manager = JobManager(research or Research(), lambda _outcome: result, **options)
        manager.set_recorder(HistoryRecorder(history))
        managers.append(manager)
        return manager, history.backing_store, history

    yield make
    for manager in managers:
        manager.shutdown(TIMEOUT)


def app_for(manager: JobManager, history: FakeHistory):
    return create_app(
        settings=Settings(_env_file=None),  # type: ignore[call-arg]
        api_settings=ApiSettings(_env_file=None),  # type: ignore[call-arg]
        ollama_probe=lambda: True,
        job_manager=manager,
        history=history,
    )


def assert_history_unavailable(response: httpx.Response) -> None:
    assert response.status_code == 503
    assert response.headers["content-type"] == PROBLEM_MEDIA_TYPE
    assert response.json()["code"] == "history_unavailable"
    assert response.headers["retry-after"] == "5"


# --- Submission ---


def test_the_queued_record_is_written_before_the_job_can_run(managed) -> None:
    research = Research()
    manager, store, _history = managed(research)
    manager.start()
    seen: list[str] = []

    def before(operation, record):
        if operation == "record_queued":
            with pytest.raises(JobNotFoundError):
                manager.get(record.id)  # not visible, so not runnable, while being written
            assert research.calls == 0
            seen.append(record.id)

    store.before = before
    job = manager.submit(JOB, label="My research")
    done = manager.wait(job.id, TIMEOUT)

    assert seen == [job.id]
    assert done.state is JobState.SUCCEEDED and not done.unrecorded
    assert [operation for operation, _ in store.writes] == [
        "record_queued",
        "record_started",
        "record_finished",
    ]
    queued, started, finished = store.records[job.id]
    assert queued == QueuedJob(job.id, "local", job.created_at, label="My research")
    assert started == StartedJob(job.id, done.started_at)
    assert isinstance(finished, FinishedJob) and finished.state == "succeeded"
    assert finished.result is done.result


def test_a_failed_queued_write_refuses_the_job_and_frees_its_slot(managed) -> None:
    manager, store, _history = managed(max_queued=1)
    store.failing.add("record_queued")

    with pytest.raises(JobRecordingError):
        manager.submit(JOB)

    assert manager._jobs == {} and manager._reserved == set()
    store.failing.clear()
    accepted = manager.submit(JOB)  # the only slot is free again
    assert accepted.queue_position == 1
    with pytest.raises(QueueFullError):
        manager.submit(JOB)


def test_concurrent_submissions_never_overfill_the_queue_while_writing(managed) -> None:
    manager, store, _history = managed(max_queued=3)
    state = threading.Condition()
    writing: list[str] = []
    refused: list[Exception] = []
    accepted: list[str] = []
    go = threading.Event()

    def before(operation, record):
        if operation == "record_queued":
            with state:
                writing.append(record.id)
                state.notify_all()
            assert go.wait(TIMEOUT)

    store.before = before

    def submit() -> None:
        try:
            accepted.append(manager.submit(JOB).id)
        except QueueFullError as exc:
            with state:
                refused.append(exc)
                state.notify_all()

    threads = [threading.Thread(target=submit) for _ in range(10)]
    for thread in threads:
        thread.start()
    with state:
        # Three reservations are held mid-write; every other submission is already refused.
        assert state.wait_for(lambda: len(writing) == 3 and len(refused) == 7, TIMEOUT)
    go.set()
    for thread in threads:
        thread.join(TIMEOUT)

    assert len(accepted) == 3 and len(set(accepted)) == 3
    assert sorted(manager.get(job_id).queue_position for job_id in accepted) == [1, 2, 3]
    assert manager._reserved == set()


@pytest.mark.parametrize("discard_fails", [False, True])
def test_a_shutdown_during_the_queued_write_never_leaves_a_runnable_job(
    managed, discard_fails
) -> None:
    research = Research()
    manager, store, _history = managed(research)
    manager.start()
    if discard_fails:
        store.failing.add("delete_queued")

    def before(operation, _record):
        if operation == "record_queued":
            manager.shutdown(TIMEOUT)  # shutdown wins the race while the record is written

    store.before = before

    with pytest.raises(ManagerClosedError):
        manager.submit(JOB)

    assert manager._jobs == {} and manager._reserved == set()
    assert research.calls == 0
    (job_id,) = {job_id for _, job_id in store.writes}
    # Removed when possible; otherwise left queued, for restart recovery to interrupt.
    assert (job_id in store.rows) is discard_fails


# --- Starting ---


def test_a_failed_running_write_fails_the_job_without_running_research(managed) -> None:
    research = Research()
    manager, store, _history = managed(research)
    store.failing.add("record_started")
    manager.start()

    failed = manager.wait(manager.submit(JOB).id, TIMEOUT)

    assert failed.state is JobState.FAILED
    assert failed.failure is not None
    assert (failed.failure.code, failed.failure.message) == (
        "history_unavailable",
        FAILURE_MESSAGES["history_unavailable"],
    )
    assert research.calls == 0
    assert store.rows[failed.id] == "failed"  # its failure is recorded from the queued row

    store.failing.clear()  # the worker keeps going; the failed job is never retried
    succeeded = manager.wait(manager.submit(JOB).id, TIMEOUT)
    assert succeeded.state is JobState.SUCCEEDED
    assert research.calls == 1


# --- Finishing ---


def test_every_terminal_outcome_is_recorded(managed) -> None:
    research = Research(error=RuntimeError("model broke"), block=True)
    manager, store, _history = managed(research)
    failed = manager.submit(JOB)
    manager.start()
    assert research.started.wait(TIMEOUT)
    cancelled = manager.submit(JOB)
    manager.cancel(cancelled.id)
    shut_down = manager.submit(JOB)
    manager.shutdown(0)  # cancels the queued job while the first is still running
    research.release.set()
    manager.wait(failed.id, TIMEOUT)
    manager.shutdown(TIMEOUT)

    records = {job_id: store.records[job_id][-1] for job_id in store.records}
    assert records[failed.id].state == "failed"
    assert records[failed.id].failure_code == "internal_error"
    assert (records[cancelled.id].state, records[cancelled.id].cancel_reason) == (
        "cancelled",
        "client",
    )
    assert (records[shut_down.id].state, records[shut_down.id].cancel_reason) == (
        "cancelled",
        "shutdown",
    )
    assert not any(manager.get(job_id).unrecorded for job_id in (failed.id, cancelled.id))


def test_a_failed_finished_write_marks_the_job_unrecorded_without_rerunning_it(managed) -> None:
    research = Research()
    manager, store, _history = managed(research)
    store.failing.add("record_finished")
    manager.start()

    done = manager.wait(manager.submit(JOB).id, TIMEOUT)

    assert done.state is JobState.SUCCEEDED  # the real outcome is kept for diagnosis
    assert done.unrecorded
    assert store.rows[done.id] == "running"  # no failure record is fabricated
    assert research.calls == 1


# --- HTTP: labels, durability and contracts ---


@pytest.fixture
def served(managed):
    clients: list[TestClient] = []

    def serve(research: Research | None = None, **options):
        manager, store, history = managed(research, **options)
        client = TestClient(app_for(manager, history))
        client.__enter__()
        clients.append(client)
        return client, manager, store

    yield serve
    for client in clients:
        client.__exit__(None, None, None)


@pytest.mark.parametrize(
    ("label", "stored"),
    [
        (None, None),
        ("  My report  ", "My report"),
        ("   ", None),
        ("\tTabbed\n", "Tabbed"),
        ("x" * 120, "x" * 120),
    ],
)
def test_labels_are_normalized_before_recording(served, label, stored) -> None:
    client, manager, store = served()

    response = client.post(JOBS, json={**SUBMISSION, "label": label})
    job_id = response.json()["id"]
    manager.wait(job_id, TIMEOUT)

    assert response.status_code == 202
    assert set(response.json()) == STATUS_FIELDS  # no label or other new field in JobStatus
    assert store.records[job_id][0].label == stored


@pytest.mark.parametrize(
    ("label", "error_type"),
    [
        (f"{LABEL_SECRET}\tinside", "value_error"),
        (f"{LABEL_SECRET}\x85inside", "value_error"),
        (LABEL_SECRET * 11, "value_error"),  # 132 characters
        (123, "string_type"),
    ],
)
def test_invalid_labels_are_422_at_body_label_without_echoing(served, label, error_type) -> None:
    client, _manager, store = served()

    response = client.post(JOBS, json={**SUBMISSION, "label": label})

    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "validation_failed"
    assert [(error["field"], error["type"]) for error in body["errors"]] == [
        ("body.label", error_type)
    ]
    assert LABEL_SECRET not in response.text
    assert store.writes == []


def test_a_failed_queued_write_is_a_503_and_accepts_nothing(served) -> None:
    client, manager, store = served()
    store.failing.add("record_queued")

    response = client.post(JOBS, json=SUBMISSION)

    assert_history_unavailable(response)
    assert manager._jobs == {}


def test_reads_of_an_unrecorded_outcome_are_503(served) -> None:
    client, manager, store = served()
    store.failing.add("record_finished")
    job_id = client.post(JOBS, json=SUBMISSION).json()["id"]
    assert manager.wait(job_id, TIMEOUT).unrecorded

    status = client.get(f"{JOBS}/{job_id}")
    result = client.get(f"{JOBS}/{job_id}/result")
    cancel = client.post(f"{JOBS}/{job_id}/cancel")

    for response in (status, result, cancel):
        assert_history_unavailable(response)
        assert "succeeded" not in response.text


def test_an_unrecorded_cancellation_is_503(served) -> None:
    research = Research(block=True)
    client, manager, store = served(research)
    client.post(JOBS, json=SUBMISSION)
    assert research.started.wait(TIMEOUT)
    queued = client.post(JOBS, json=SUBMISSION).json()["id"]
    store.failing.add("record_finished")

    cancel = client.post(f"{JOBS}/{queued}/cancel")
    status = client.get(f"{JOBS}/{queued}")
    research.release.set()

    assert_history_unavailable(cancel)
    assert_history_unavailable(status)
    assert manager.get(queued).state is JobState.CANCELLED  # it will never run


def test_recorded_jobs_keep_the_existing_status_contract(served, result) -> None:
    client, manager, _store = served()
    job_id = client.post(JOBS, json=SUBMISSION).json()["id"]
    manager.wait(job_id, TIMEOUT)

    status = client.get(f"{JOBS}/{job_id}")
    fetched = client.get(f"{JOBS}/{job_id}/result")

    assert status.status_code == 200 and set(status.json()) == STATUS_FIELDS
    assert status.json()["state"] == "succeeded"
    assert fetched.status_code == 200 and fetched.json() == result.model_dump(mode="json")
    schema = client.app.openapi()["components"]["schemas"]
    assert set(schema["JobStatus"]["properties"]) == STATUS_FIELDS
    assert "label" in schema["JobSubmission"]["properties"]


def test_submissions_while_history_is_not_ready_record_nothing(managed) -> None:
    manager, store, history = managed()
    history.current = HistoryStatus(HistoryState.UNAVAILABLE, "database_unavailable")

    with TestClient(app_for(manager, history)) as client:
        response = client.post(JOBS, json=SUBMISSION)

    assert_history_unavailable(response)
    assert store.writes == [] and manager._jobs == {}


# --- Audit regressions: an unrecorded outcome stays unrecorded and is never served ---


def test_an_unrecorded_outcome_stays_unrecorded_after_later_events(managed) -> None:
    research = Research()
    manager, store, _history = managed(research)
    store.failing.add("record_finished")
    manager.start()
    first = manager.wait(manager.submit(JOB).id, TIMEOUT)

    store.failing.clear()
    second = manager.wait(manager.submit(JOB).id, TIMEOUT)  # a later write succeeds
    with pytest.raises(JobNotCancellableError):
        manager.cancel(first.id)
    manager.shutdown(TIMEOUT)

    kept = manager.get(first.id)
    assert (kept.state, kept.unrecorded) == (JobState.SUCCEEDED, True)
    assert not manager.get(second.id).unrecorded
    assert store.rows[first.id] == "running"  # the database never claims it finished
    assert research.calls == 2  # nothing was re-run to repair it


def test_retention_keeps_unrecorded_jobs_so_reads_never_use_stale_history(served) -> None:
    client, manager, store = served(max_retained=1)
    store.failing.add("record_finished")
    unrecorded = client.post(JOBS, json=SUBMISSION).json()["id"]
    assert manager.wait(unrecorded, TIMEOUT).unrecorded
    store.failing.clear()
    recorded = client.post(JOBS, json=SUBMISSION).json()["id"]
    with pytest.raises(JobNotFoundError):
        manager.wait(recorded, TIMEOUT)  # recorded, then forgotten by retention at once

    stale = client.get(f"{JOBS}/{unrecorded}")
    forgotten = client.get(f"{JOBS}/{recorded}")

    # The unrecorded job stays in memory and is refused; its "running" row is never read.
    assert manager.get(unrecorded).unrecorded
    assert stale.status_code == 503 and stale.json()["code"] == "history_unavailable"
    assert unrecorded not in store.gets
    # The recorded job was forgotten by retention instead, and is read from history.
    with pytest.raises(JobNotFoundError):
        manager.get(recorded)
    assert forgotten.status_code == 200 and forgotten.json()["state"] == "succeeded"
    assert store.gets == [recorded]


def test_a_record_the_history_layer_rejects_counts_as_a_failed_write(managed) -> None:
    manager, store, _history = managed(id_factory=lambda: "job-1")  # not a history job ID

    with pytest.raises(JobRecordingError):
        manager.submit(JOB)

    assert manager._jobs == {} and manager._reserved == set()
    assert store.writes == []


def test_a_fatal_worker_failure_with_failed_writes_leaves_both_outcomes_unrecorded(
    managed, monkeypatch
) -> None:
    class Stop(BaseException):
        pass

    monkeypatch.setattr(threading, "excepthook", lambda _args: None)
    research = Research(error=Stop(), block=True)
    manager, store, _history = managed(research)
    first = manager.submit(JOB)
    manager.start()
    assert research.started.wait(TIMEOUT)
    second = manager.submit(JOB)
    store.failing.add("record_finished")

    research.release.set()
    failed = manager.wait(first.id, TIMEOUT)
    cancelled = manager.wait(second.id, TIMEOUT)

    assert (failed.state, failed.unrecorded) == (JobState.FAILED, True)
    assert (cancelled.state, cancelled.cancel_reason, cancelled.unrecorded) == (
        JobState.CANCELLED,
        "worker_stopped",
        True,
    )
    assert (store.rows[first.id], store.rows[second.id]) == ("running", "queued")
    assert research.calls == 1
