"""Offline tests for the API's history adapter: labels, record conversion and error mapping.

No database, model, network or MCP server is used.
"""

from datetime import UTC, datetime, timedelta

import pytest
from fakes import ScriptedModel, complete

from researchpilot.api.history import (
    HISTORY_RETRY_AFTER_SECONDS,
    InvalidLabelError,
    finished_record,
    history_problem,
    normalize_label,
    queued_record,
    started_record,
)
from researchpilot.api.jobs import JobFailure, JobSnapshot, JobState
from researchpilot.config import Settings
from researchpilot.history.records import FinishedJob, QueuedJob, StartedJob
from researchpilot.history.store import (
    HistoryConfigError,
    HistoryConflictError,
    HistoryEncodingError,
    HistoryError,
    HistoryLockError,
    HistorySchemaError,
    HistoryUnavailableError,
)
from researchpilot.pipeline import prepare_request, run_research
from researchpilot.results import ResearchResult, build_result

JOB_ID = "0123456789abcdef0123456789abcdef"
CREATED = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
STARTED = CREATED + timedelta(seconds=1)
FINISHED = CREATED + timedelta(minutes=1)
SECRET = "LABEL-SECRET"


@pytest.fixture(scope="module")
def result() -> ResearchResult:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    model = ScriptedModel(["notes", complete(), "Report [S1]."])
    outcome = run_research(
        prepare_request(
            "Compare System Alpha and System Beta",
            "# Source A: Tasks\n\nSystem Alpha completed 85 of 100 tasks in a fictional test.",
        ),
        settings=settings,
        model_factory=lambda _settings, **_kwargs: model,
    )
    return build_result(outcome, settings)


def snapshot(state: JobState, **fields) -> JobSnapshot:
    values = {
        "id": JOB_ID,
        "state": state,
        "mode": "web",
        "created_at": CREATED,
        "started_at": STARTED if state is not JobState.QUEUED else None,
        "finished_at": None,
        "queue_position": 1 if state is JobState.QUEUED else None,
        "failure": None,
        "cancel_reason": None,
        "result": None,
    }
    values.update(fields)
    return JobSnapshot(**values)


# --- Labels ---


@pytest.mark.parametrize("raw", [None, "", "   ", "\t\n", "  　"])
def test_blank_labels_are_absent(raw) -> None:
    assert normalize_label(raw) is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Alpha vs Beta", "Alpha vs Beta"),
        ("  padded  ", "padded"),
        (" no-break space edges ", "no-break space edges"),
        ("\tTabbed report\n", "Tabbed report"),
        ("inner separator", "inner separator"),
        ("x" * 120, "x" * 120),
        (" " + "x" * 120 + " ", "x" * 120),
    ],
)
def test_labels_are_stripped_then_kept(raw, expected) -> None:
    assert normalize_label(raw) == expected


@pytest.mark.parametrize(
    "raw",
    ["x" * 121, f"{SECRET}\tinside", f"{SECRET}\ninside", f"{SECRET}\x00", f"{SECRET}\x85x"],
)
def test_invalid_labels_raise_without_echoing_the_label(raw) -> None:
    with pytest.raises(InvalidLabelError) as raised:
        normalize_label(raw)

    assert isinstance(raised.value, ValueError)
    assert SECRET not in str(raised.value)
    assert "x" * 121 not in str(raised.value)


# --- Snapshots to records ---


def test_a_queued_snapshot_becomes_a_queued_record() -> None:
    record = queued_record(snapshot(JobState.QUEUED), "My research")

    assert record == QueuedJob(JOB_ID, "web", CREATED, label="My research")
    assert queued_record(snapshot(JobState.QUEUED), None).label is None


def test_a_running_snapshot_becomes_a_started_record() -> None:
    assert started_record(snapshot(JobState.RUNNING)) == StartedJob(JOB_ID, STARTED)


def test_terminal_snapshots_carry_only_their_state_specific_field(result) -> None:
    failure = JobFailure(code="model_failed", message="The research model failed.")
    succeeded = snapshot(JobState.SUCCEEDED, finished_at=FINISHED, result=result)
    failed = snapshot(JobState.FAILED, finished_at=FINISHED, failure=failure)
    cancelled = snapshot(
        JobState.CANCELLED, started_at=None, finished_at=FINISHED, cancel_reason="shutdown"
    )

    assert finished_record(succeeded) == FinishedJob(JOB_ID, "succeeded", FINISHED, result=result)
    assert finished_record(failed) == FinishedJob(
        JOB_ID, "failed", FINISHED, failure_code="model_failed"
    )
    assert finished_record(cancelled) == FinishedJob(
        JOB_ID, "cancelled", FINISHED, cancel_reason="shutdown"
    )


@pytest.mark.parametrize(
    "build",
    [
        lambda: queued_record(snapshot(JobState.RUNNING), None),
        lambda: started_record(snapshot(JobState.QUEUED)),
        lambda: started_record(snapshot(JobState.RUNNING, started_at=None)),
        lambda: finished_record(snapshot(JobState.RUNNING)),
        lambda: finished_record(snapshot(JobState.SUCCEEDED, finished_at=None)),
        # The history records' own checks still apply: a success needs its result.
        lambda: finished_record(snapshot(JobState.SUCCEEDED, finished_at=FINISHED)),
        # Only history-compatible job IDs can be recorded.
        lambda: queued_record(snapshot(JobState.QUEUED, id="job-1"), None),
    ],
)
def test_snapshots_in_the_wrong_state_are_rejected(build) -> None:
    with pytest.raises(ValueError):
        build()


# --- History errors to problems ---


@pytest.mark.parametrize(
    ("error", "code", "retry_after"),
    [
        (HistoryConfigError(), "history_unconfigured", None),
        (HistorySchemaError(), "history_schema_outdated", None),
        (HistoryEncodingError(), "history_schema_outdated", None),
        (HistoryUnavailableError(), "history_unavailable", str(HISTORY_RETRY_AFTER_SECONDS)),
        (HistoryLockError(), "history_unavailable", str(HISTORY_RETRY_AFTER_SECONDS)),
        (HistoryConflictError(), "history_unavailable", str(HISTORY_RETRY_AFTER_SECONDS)),
        (HistoryError(), "history_unavailable", str(HISTORY_RETRY_AFTER_SECONDS)),
    ],
)
def test_history_errors_map_to_fixed_503_problems(error, code, retry_after) -> None:
    problem = history_problem(error)

    assert (problem.status, problem.code) == (503, code)
    assert problem.headers.get("Retry-After") == retry_after
    assert problem.errors == []
    assert problem.detail == FIXED_DETAILS[code]


FIXED_DETAILS = {
    "history_unconfigured": "The research history database has not been configured.",
    "history_schema_outdated": (
        "The research history database is not at the expected schema; run the migrations."
    ),
    "history_unavailable": "The research history database is unavailable.",
}
