"""Version 1 research job routes.

Status and result reads look in memory first. A job that is no longer there (after a
restart, or once forgotten by in-memory retention) is read from research history; a job
whose outcome is not durably recorded is always kept in memory, so it is never read from a
stale history record.

A submission is validated, context budget included, on the request thread before it is
queued; the research itself runs only on the job manager's worker thread. Responses never
contain the submitted source material, internal request objects or exception messages, and
these routes log nothing themselves.
"""

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response
from pydantic import BaseModel, ConfigDict, StrictBool, ValidationError

from researchpilot.api.dependencies import (
    AppDependencies,
    ensure_history_and_worker,
    get_dependencies,
)
from researchpilot.api.history import InvalidLabelError, history_problem, normalize_label
from researchpilot.api.jobs import (
    CancelReason,
    JobInput,
    JobManager,
    JobMode,
    JobNotCancellableError,
    JobNotFoundError,
    JobRecordingError,
    JobSnapshot,
    JobState,
    ManagerClosedError,
    QueueFullError,
    failure_message,
)
from researchpilot.api.problems import (
    PROBLEM_MEDIA_TYPE,
    FieldError,
    Problem,
    ProblemError,
    field_errors,
    validation_problem,
)
from researchpilot.api.routes import jobs_unavailable
from researchpilot.config import Settings
from researchpilot.errors import InputError, InputTooLargeError
from researchpilot.history.records import HistoryRecord, is_job_id
from researchpilot.history.store import HistoryError, HistoryUnavailableError
from researchpilot.pipeline import check_input_fits, prepare_request
from researchpilot.results import ResearchResult
from researchpilot.schemas import ResearchRequest

router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])
Dependencies = Annotated[AppDependencies, Depends(get_dependencies)]
# Seconds a client should wait before submitting again when the queue is full.
QUEUE_FULL_RETRY_AFTER_SECONDS = 30


def _problems(*statuses: int) -> dict[int | str, dict]:
    return {status: {"model": Problem, "content": {PROBLEM_MEDIA_TYPE: {}}} for status in statuses}


class JobSubmission(BaseModel):
    """A research request. Length limits are those of the research engine's own request."""

    model_config = ConfigDict(extra="forbid")

    question: str
    source_material: str
    # Retrieve web sources with Parallel Search before analysis.
    web: StrictBool = False
    # Optional name for finding the job in research history; surrounding whitespace is
    # removed, and a blank label counts as none.
    label: str | None = None


class JobFailureBody(BaseModel):
    code: str
    message: str


class JobStatus(BaseModel):
    """A job's lifecycle state; the result is served separately."""

    id: str
    state: JobState
    mode: JobMode
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    # 1 for the next job to run; null unless queued.
    queue_position: int | None
    cancel_reason: CancelReason | None
    failure: JobFailureBody | None

    @classmethod
    def from_snapshot(cls, snapshot: JobSnapshot) -> "JobStatus":
        # Field by field, so internal snapshot fields such as the result never leak in.
        failure = snapshot.failure
        return cls(
            id=snapshot.id,
            state=snapshot.state,
            mode=snapshot.mode,
            created_at=snapshot.created_at,
            started_at=snapshot.started_at,
            finished_at=snapshot.finished_at,
            queue_position=snapshot.queue_position,
            cancel_reason=snapshot.cancel_reason,
            failure=None
            if failure is None
            else JobFailureBody(code=failure.code, message=failure.message),
        )

    @classmethod
    def from_history(cls, record: HistoryRecord) -> "JobStatus":
        """The same fields from a stored job; a stored job is never waiting in the queue."""
        return cls(
            id=record.id,
            state=JobState(record.state),
            mode=record.mode,
            created_at=record.created_at,
            started_at=record.started_at,
            finished_at=record.finished_at,
            queue_position=None,
            cancel_reason=record.cancel_reason,
            failure=None
            if record.failure_code is None
            else JobFailureBody(
                code=record.failure_code, message=failure_message(record.failure_code)
            ),
        )


def _job_not_found() -> ProblemError:
    # The requested ID is never echoed back.
    return ProblemError(
        404,
        "job_not_found",
        "Job not found",
        "No job has this ID; finished jobs are kept only for a limited time.",
    )


_NO_RESULT = {
    JobState.QUEUED: ("job_not_finished", "Job not finished", "The job has not finished yet."),
    JobState.RUNNING: ("job_not_finished", "Job not finished", "The job has not finished yet."),
    JobState.FAILED: (
        "job_failed",
        "Job failed",
        "The job failed and has no result; its status shows the failure code.",
    ),
    JobState.CANCELLED: (
        "job_cancelled",
        "Job cancelled",
        "The job was cancelled before it ran and has no result.",
    ),
}


def _validated_request(submission: JobSubmission, settings: Settings) -> ResearchRequest:
    """Run the research engine's own validation and budget check; nothing is queued yet."""
    try:
        request = prepare_request(submission.question, submission.source_material)
        check_input_fits(request, settings)
    except InputError as exc:
        # The error's message quotes the input, so only field, message and type are kept.
        cause = exc.__cause__
        errors = (
            cause.errors(include_url=False, include_context=False, include_input=False)
            if isinstance(cause, ValidationError)
            else []
        )
        raise validation_problem(field_errors(errors, prefix=("body",))) from None
    except InputTooLargeError:
        raise ProblemError(
            422,
            "input_too_large",
            "Input too large",
            "The question and source material would not fit the model's context window.",
        ) from None
    return request


def _normalized_label(submission: JobSubmission) -> str | None:
    try:
        return normalize_label(submission.label)
    except InvalidLabelError as exc:
        # The message is fixed and never quotes the label.
        error = FieldError(field="body.label", message=str(exc), type="value_error")
        raise validation_problem([error]) from None


def _history_unavailable() -> ProblemError:
    return history_problem(HistoryUnavailableError())


def _durable(snapshot: JobSnapshot) -> JobSnapshot:
    """Refuse to present a finished job whose outcome is not durably recorded."""
    if snapshot.unrecorded:
        raise _history_unavailable()
    return snapshot


def _snapshot(jobs: JobManager, job_id: str) -> JobSnapshot:
    try:
        return _durable(jobs.get(job_id))
    except JobNotFoundError:
        raise _job_not_found() from None


def _job(dependencies: AppDependencies, job_id: str) -> JobSnapshot | HistoryRecord:
    """The job from memory if it is there, else from research history.

    A missing or expired record is 404; history that cannot be read is a 503, never a 404.
    """
    try:
        return _durable(dependencies.jobs.get(job_id))
    except JobNotFoundError:
        pass
    if not is_job_id(job_id):
        raise _job_not_found()  # no stored job can have this ID; history is not consulted
    status = ensure_history_and_worker(dependencies)
    store = dependencies.history.store
    if not status.ready:
        raise status.problem()
    if store is None:
        raise _history_unavailable()
    try:
        record = store.get(job_id)
    except HistoryError as error:
        raise history_problem(error) from None
    if record is None:
        raise _job_not_found()
    return record


@router.post(
    "",
    status_code=202,
    response_model=JobStatus,
    summary="Submit a research job",
    responses=_problems(413, 422, 503),
)
def submit_job(
    submission: JobSubmission, request: Request, response: Response, dependencies: Dependencies
) -> JobStatus:
    """Queue a validated research job; poll its status at the ``Location`` URL."""
    research_request = _validated_request(submission, dependencies.settings)
    label = _normalized_label(submission)
    # No job is accepted unless research history is ready (this may also start the worker).
    history = ensure_history_and_worker(dependencies)
    if not history.ready:
        raise history.problem()
    jobs = dependencies.jobs
    if not jobs.started:
        raise jobs_unavailable()
    try:
        snapshot = jobs.submit(JobInput(request=research_request, web=submission.web), label=label)
    except JobRecordingError:
        raise _history_unavailable() from None
    except QueueFullError:
        raise ProblemError(
            503,
            "queue_full",
            "The job queue is full",
            "Too many research jobs are waiting; try again later.",
            headers={"Retry-After": str(QUEUE_FULL_RETRY_AFTER_SECONDS)},
        ) from None
    except ManagerClosedError:
        raise jobs_unavailable() from None
    response.headers["Location"] = str(request.app.url_path_for("get_job", job_id=snapshot.id))
    return JobStatus.from_snapshot(snapshot)


@router.get(
    "/{job_id}",
    response_model=JobStatus,
    summary="Get a research job's status",
    responses=_problems(404, 503),
)
def get_job(job_id: str, dependencies: Dependencies) -> JobStatus:
    job = _job(dependencies, job_id)
    if isinstance(job, HistoryRecord):
        return JobStatus.from_history(job)
    return JobStatus.from_snapshot(job)


@router.get(
    "/{job_id}/result",
    response_model=ResearchResult,
    summary="Get a succeeded research job's result",
    responses=_problems(404, 409, 503),
)
def get_job_result(job_id: str, dependencies: Dependencies) -> ResearchResult:
    """The result of a succeeded job; 409 with a state-specific code otherwise."""
    job = _job(dependencies, job_id)
    state = JobState(job.state)
    if state is JobState.SUCCEEDED:
        if job.result is None:
            # A success without a stored result is never served as complete.
            raise _history_unavailable()
        return job.result
    code, title, detail = _NO_RESULT[state]
    raise ProblemError(409, code, title, detail)


@router.post(
    "/{job_id}/cancel",
    response_model=JobStatus,
    summary="Cancel a queued research job",
    responses=_problems(404, 409, 503),
)
def cancel_job(job_id: str, dependencies: Dependencies) -> JobStatus:
    """Cancel a queued job; cancelling it again returns the same job. Running jobs finish."""
    try:
        snapshot = dependencies.jobs.cancel(job_id)
    except JobNotFoundError:
        raise _job_not_found() from None
    except JobNotCancellableError as exc:
        # The detail names the job's state, so an unrecorded outcome is refused first.
        _snapshot(dependencies.jobs, job_id)
        raise ProblemError(
            409,
            "job_not_cancellable",
            "Job cannot be cancelled",
            f"The job is {exc.state} and can no longer be cancelled.",
        ) from None
    return JobStatus.from_snapshot(_durable(snapshot))
