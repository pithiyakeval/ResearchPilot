"""Research jobs run one at a time by a background worker thread.

``JobManager`` keeps a bounded queue of waiting jobs and runs each through the injected
``research`` and ``build`` functions (by default ``run_research`` and ``build_result``) on
one dedicated thread, so no caller waits for a run and the event loop is never blocked.

Failures are stored as a stable code with a fixed message; exception messages, questions
and source text are never stored on a job or logged. A finished job keeps only its
``ResearchResult``; its request, including the source material, is dropped.

An ``Exception`` fails only its job. A ``BaseException`` (such as ``SystemExit`` or
``asyncio.CancelledError``) also fails its job, is not re-raised, so the thread's exception
hook never prints its message, and stops the worker: in the same locked step the manager
closes and cancels every queued job with reason ``worker_stopped``, so no job is accepted
or left waiting without a worker.

With a ``JobRecorder`` (research history), every job is recorded durably: its queued row
is written before the job can run, its running row before research starts, and its finished
row after it ends. A job whose finished row could not be written is marked ``unrecorded``,
so callers never present it as a durably finished job. Without a recorder, jobs live only
in this process's memory.
"""

import logging
import threading
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from functools import partial
from typing import Literal, Protocol

from researchpilot.api.settings import ApiSettings
from researchpilot.config import Settings
from researchpilot.errors import ResearchPilotError
from researchpilot.pipeline import ModelFactory, ResearchOutcome, WebGatherer, run_research
from researchpilot.results import ResearchResult, build_result
from researchpilot.schemas import ResearchRequest

logger = logging.getLogger(__name__)


class JobState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_STATES = frozenset({JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED})
_TRANSITIONS = {
    JobState.QUEUED: frozenset({JobState.RUNNING, JobState.CANCELLED}),
    JobState.RUNNING: frozenset({JobState.SUCCEEDED, JobState.FAILED}),
}
# Fixed, content-free messages for each failure code.
FAILURE_MESSAGES = {
    "validation_failed": "The research request failed validation.",
    "input_too_large": "A prompt would not fit the model's context window.",
    "web_retrieval_failed": "Web retrieval failed or returned no usable sources.",
    "model_failed": "The research model failed or could not be reached.",
    "result_failed": "The research finished, but its result could not be built.",
    "internal_error": "The job failed unexpectedly.",
    "history_unavailable": "The job could not be recorded in research history.",
    # Recorded by restart recovery for a job left queued or running by an earlier process.
    "interrupted_by_restart": "The job was interrupted by a server restart and did not finish.",
}
# For a failure code read from history that this version does not know.
UNKNOWN_FAILURE_MESSAGE = "The job did not finish."


def failure_message(code: str) -> str:
    """The fixed message for a failure code, including codes read back from history."""
    return FAILURE_MESSAGES.get(code, UNKNOWN_FAILURE_MESSAGE)


JobMode = Literal["local", "web"]
CancelReason = Literal["client", "shutdown", "worker_stopped"]


@dataclass(frozen=True)
class JobInput:
    """A validated research request and whether to retrieve web sources first."""

    request: ResearchRequest
    web: bool = False


@dataclass(frozen=True)
class JobFailure:
    code: str
    message: str


@dataclass(frozen=True)
class JobSnapshot:
    """A job's state at one moment; safe to share between threads."""

    id: str
    state: JobState
    mode: JobMode
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    # 1 for the next job to run; None unless queued.
    queue_position: int | None
    failure: JobFailure | None
    cancel_reason: CancelReason | None
    result: ResearchResult | None
    # True while a finished job's outcome is not durably recorded: still being written, or
    # the write failed. Such a job must not be presented as durably finished.
    unrecorded: bool = False


class JobRecorder(Protocol):
    """Durable recording of a job's lifecycle; any exception means the write failed."""

    def queued(self, snapshot: JobSnapshot, label: str | None) -> None: ...

    def started(self, snapshot: JobSnapshot) -> None: ...

    def finished(self, snapshot: JobSnapshot) -> None: ...

    def discard(self, job_id: str) -> None:
        """Remove the queued record of a job that was never made runnable."""


@dataclass
class _JobRecord:
    """Mutable job state, changed only while the manager's lock is held."""

    id: str
    input: JobInput | None
    mode: JobMode
    created_at: datetime
    state: JobState = JobState.QUEUED
    started_at: datetime | None = None
    finished_at: datetime | None = None
    failure: JobFailure | None = None
    cancel_reason: CancelReason | None = None
    result: ResearchResult | None = None
    # Finished but its record is still being written, or the write failed.
    finish_pending: bool = False
    finish_failed: bool = False


class JobError(Exception):
    """Base class for job manager errors."""


class JobNotFoundError(JobError):
    """No job has this ID, or it was forgotten under the retention limit."""

    def __init__(self, job_id: str) -> None:
        super().__init__("Job not found.")
        self.job_id = job_id


class QueueFullError(JobError):
    """The waiting queue is at its limit."""

    def __init__(self) -> None:
        super().__init__("The job queue is full.")


class JobNotCancellableError(JobError):
    """Only queued jobs can be cancelled; a running model call is never interrupted."""

    def __init__(self, state: JobState) -> None:
        super().__init__(f"A {state} job cannot be cancelled.")
        self.state = state


class JobRecordingError(JobError):
    """The job could not be recorded durably, so it was not accepted."""

    def __init__(self) -> None:
        super().__init__("The job could not be recorded.")


class ManagerClosedError(JobError):
    """The manager has shut down and accepts no more jobs."""

    def __init__(self) -> None:
        super().__init__("The job manager has shut down.")


# Runs one job; raises the typed errors in ``researchpilot.errors`` on failure.
Research = Callable[[JobInput], ResearchOutcome]
BuildResult = Callable[[ResearchOutcome], ResearchResult]


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _new_id() -> str:
    return uuid.uuid4().hex


def _gate_allows(gate: Callable[[], bool]) -> bool:
    try:
        return bool(gate())
    except Exception as exc:  # noqa: BLE001 - a broken gate must refuse, not crash the worker
        logger.warning("Job run gate raised %s", type(exc).__name__)
        return False


def _failure_code(exc: Exception) -> str:
    if isinstance(exc, ResearchPilotError) and exc.code in FAILURE_MESSAGES:
        return exc.code
    return "internal_error"


class JobManager:
    """Thread-safe job lifecycle with one worker thread and a bounded waiting queue.

    Jobs can be submitted before ``start``; they wait until the worker runs them.
    """

    def __init__(
        self,
        research: Research,
        build: BuildResult,
        *,
        max_queued: int = 4,
        max_retained: int = 50,
        clock: Callable[[], datetime] = _utc_now,
        id_factory: Callable[[], str] = _new_id,
    ) -> None:
        if max_queued < 1 or max_retained < 1:
            raise ValueError("max_queued and max_retained must be at least 1.")
        self._research = research
        self._build = build
        self._max_queued = max_queued
        self._max_retained = max_retained
        self._clock = clock
        self._id_factory = id_factory
        self._cond = threading.Condition()
        self._jobs: dict[str, _JobRecord] = {}
        self._waiting: deque[str] = deque()
        # Finished job IDs, oldest first, for the retention limit.
        self._finished: deque[str] = deque()
        self._closed = False
        self._worker: threading.Thread | None = None
        self._run_gate: Callable[[], bool] | None = None
        self._recorder: JobRecorder | None = None
        # Queue slots held by submissions whose queued record is being written.
        self._reserved: set[str] = set()
        # Finished jobs whose records still have to be written, oldest first.
        self._to_record: deque[JobSnapshot] = deque()

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        api_settings: ApiSettings,
        *,
        model_factory: ModelFactory | None = None,
        web_gatherer: WebGatherer | None = None,
    ) -> "JobManager":
        """A manager running the real pipeline, with limits from the API settings."""
        options = {} if model_factory is None else {"model_factory": model_factory}

        def research(job_input: JobInput) -> ResearchOutcome:
            return run_research(
                job_input.request,
                settings=settings,
                web=job_input.web,
                web_gatherer=web_gatherer,
                **options,
            )

        return cls(
            research,
            partial(build_result, settings=settings),
            max_queued=api_settings.max_queued_jobs,
            max_retained=api_settings.max_retained_jobs,
        )

    # --- Public API ---

    @property
    def started(self) -> bool:
        """True once ``start`` has created the worker, even if it has since stopped."""
        with self._cond:
            return self._worker is not None

    @property
    def closed(self) -> bool:
        """True after shutdown, or after a fatal worker failure; no jobs are accepted."""
        with self._cond:
            return self._closed

    def set_run_gate(self, gate: Callable[[], bool] | None) -> None:
        """Ask ``gate`` before each job; ``False`` or an error stops the worker for good.

        A refused job is not run: the manager closes as after a fatal worker failure, and
        queued jobs are cancelled with ``worker_stopped``. A job already running finishes.
        """
        with self._cond:
            self._run_gate = gate

    def set_recorder(self, recorder: JobRecorder | None) -> None:
        """Record every job durably; set before the first submission."""
        with self._cond:
            self._recorder = recorder

    def start(self) -> None:
        """Start the worker thread; calling it again has no effect."""
        with self._cond:
            if self._closed:
                raise ManagerClosedError
            if self._worker is None:
                # A daemon, so a job still running at exit cannot keep the process alive.
                self._worker = threading.Thread(
                    target=self._work, name="researchpilot-job-worker", daemon=True
                )
                self._worker.start()

    def submit(self, job_input: JobInput, *, label: str | None = None) -> JobSnapshot:
        """Queue a job, or raise ``QueueFullError``, ``ManagerClosedError`` or
        ``JobRecordingError``.

        With a recorder, a queue slot is reserved, the queued record is written without the
        lock held, and only then does the job become runnable. A failed write releases the
        slot; a shutdown during the write discards the record and refuses the job.
        """
        with self._cond:
            if self._closed:
                raise ManagerClosedError
            if len(self._waiting) + len(self._reserved) >= self._max_queued:
                raise QueueFullError
            job_id = self._id_factory()
            if job_id in self._jobs or job_id in self._reserved:
                raise RuntimeError("Job ID factory returned a duplicate ID.")
            record = _JobRecord(
                id=job_id,
                input=job_input,
                mode="web" if job_input.web else "local",
                created_at=self._clock(),
            )
            recorder = self._recorder
            if recorder is None:
                return self._enqueue(record)
            self._reserved.add(job_id)
            pending = self._snapshot(record, queued=False)
        try:
            recorder.queued(pending, label)
        except Exception as exc:  # noqa: BLE001 - any recording failure refuses the job
            logger.warning("Job %s could not be recorded: %s", job_id, type(exc).__name__)
            with self._cond:
                self._reserved.discard(job_id)
                self._cond.notify_all()
            raise JobRecordingError from None
        with self._cond:
            self._reserved.discard(job_id)
            if not self._closed:
                return self._enqueue(record)
        # Shut down while the record was written: the job must never become runnable.
        try:
            recorder.discard(job_id)
        except Exception as exc:  # noqa: BLE001 - a restart marks it interrupted instead
            logger.warning("Job %s record could not be discarded: %s", job_id, type(exc).__name__)
        raise ManagerClosedError

    def get(self, job_id: str) -> JobSnapshot:
        with self._cond:
            return self._snapshot(self._record(job_id))

    def cancel(self, job_id: str) -> JobSnapshot:
        """Cancel a queued job; cancelling it again returns the same cancelled job."""
        with self._cond:
            record = self._record(job_id)
            if record.state is JobState.CANCELLED:
                return self._snapshot(record)
            if record.state is not JobState.QUEUED:
                raise JobNotCancellableError(record.state)
            self._waiting.remove(job_id)
            self._finish(record, JobState.CANCELLED, cancel_reason="client")
            self._cond.notify_all()
        self._flush_records()
        with self._cond:
            return self._snapshot(record)

    def wait(self, job_id: str, timeout: float) -> JobSnapshot:
        """Wait up to ``timeout`` seconds for the job to finish, then return its state."""
        with self._cond:
            self._record(job_id)
            self._cond.wait_for(lambda: self._is_finished_or_gone(job_id), timeout)
            return self._snapshot(self._record(job_id))

    def shutdown(self, timeout: float) -> bool:
        """Refuse new jobs, cancel queued ones and wait up to ``timeout`` for the worker.

        A running job is not interrupted; returns False if it was still running.
        """
        with self._cond:
            self._close("shutdown")
            self._cond.notify_all()
            worker = self._worker
        self._flush_records()
        if worker is None or worker is threading.current_thread():
            return True
        worker.join(timeout)
        if worker.is_alive():
            logger.warning("Job manager shut down with a job still running")
            return False
        return True

    # --- Internals; the caller holds the lock unless noted ---

    def _close(self, reason: CancelReason) -> None:
        """Refuse new jobs and cancel every queued one; only the first call has an effect."""
        if self._closed:
            return
        self._closed = True
        while self._waiting:
            record = self._jobs[self._waiting.popleft()]
            self._finish(record, JobState.CANCELLED, cancel_reason=reason)

    def _record(self, job_id: str) -> _JobRecord:
        record = self._jobs.get(job_id)
        if record is None:
            raise JobNotFoundError(job_id)
        return record

    def _is_finished_or_gone(self, job_id: str) -> bool:
        # A finished job counts once its recording has settled, whatever the outcome.
        record = self._jobs.get(job_id)
        return record is None or (record.state in TERMINAL_STATES and not record.finish_pending)

    def _enqueue(self, record: _JobRecord) -> JobSnapshot:
        self._jobs[record.id] = record
        self._waiting.append(record.id)
        self._cond.notify_all()
        logger.info("Job %s queued mode=%s", record.id, record.mode)
        return self._snapshot(record)

    def _snapshot(self, record: _JobRecord, *, queued: bool = True) -> JobSnapshot:
        position = None
        if record.state is JobState.QUEUED and queued:
            position = self._waiting.index(record.id) + 1
        return JobSnapshot(
            id=record.id,
            state=record.state,
            mode=record.mode,
            created_at=record.created_at,
            started_at=record.started_at,
            finished_at=record.finished_at,
            queue_position=position,
            failure=record.failure,
            cancel_reason=record.cancel_reason,
            result=record.result,
            unrecorded=record.finish_pending or record.finish_failed,
        )

    def _transition(self, record: _JobRecord, state: JobState) -> None:
        if state not in _TRANSITIONS.get(record.state, frozenset()):
            raise RuntimeError(f"Invalid job transition: {record.state} -> {state}.")
        record.state = state

    def _finish(
        self,
        record: _JobRecord,
        state: JobState,
        *,
        failure: JobFailure | None = None,
        result: ResearchResult | None = None,
        cancel_reason: CancelReason | None = None,
    ) -> None:
        self._transition(record, state)
        record.finished_at = self._clock()
        record.failure, record.result, record.cancel_reason = failure, result, cancel_reason
        # The request holds the source material; a finished job no longer needs it.
        record.input = None
        if self._recorder is not None:
            # Written by _flush_records once the lock is released.
            record.finish_pending = True
            self._to_record.append(self._snapshot(record))
        self._finished.append(record.id)
        self._evict()
        logger.info("Job %s %s", record.id, state)

    def _evict(self) -> None:
        """Forget the oldest finished jobs beyond the retention limit.

        A job whose outcome is not durably recorded is kept: forgetting it would let reads
        fall back to its stale history record. Such jobs only arise from writes in progress
        when history fails, since no job is accepted while history is unavailable.
        """
        excess = len(self._finished) - self._max_retained
        if excess <= 0:
            return
        for job_id in list(self._finished):
            if excess == 0:
                break
            record = self._jobs[job_id]
            if record.finish_pending or record.finish_failed:
                continue
            self._finished.remove(job_id)
            del self._jobs[job_id]
            excess -= 1

    def _work(self) -> None:
        """Worker thread loop; runs without the lock except while picking or finishing a job."""
        while True:
            with self._cond:
                self._cond.wait_for(lambda: self._waiting or self._closed)
                if not self._waiting:
                    return
                gate = self._run_gate
            # Asked outside the lock: the gate may contact the history database.
            if gate is not None and not _gate_allows(gate):
                with self._cond:
                    logger.error("Job worker stopped: the run gate refused the next job")
                    self._close("worker_stopped")
                    self._cond.notify_all()
                self._flush_records()
                return
            with self._cond:
                if not self._waiting:
                    continue  # cancelled or shut down while the gate was asked
                record = self._jobs[self._waiting.popleft()]
                self._transition(record, JobState.RUNNING)
                record.started_at = self._clock()
                job_input = record.input
                running = self._snapshot(record)
                recorder = self._recorder
            assert job_input is not None
            if recorder is not None and not self._record_start(recorder, record, running):
                continue  # failed without running research
            logger.info("Job %s running", record.id)
            keep_working = self._run(record, job_input)
            job_input = None
            self._flush_records()
            if not keep_working:
                return

    def _record_start(
        self, recorder: JobRecorder, record: _JobRecord, running: JobSnapshot
    ) -> bool:
        """Write the running record; if that fails, fail the job without running research."""
        try:
            recorder.started(running)
        except Exception as exc:  # noqa: BLE001 - research must not run unrecorded
            failure = self._failure(record.id, "history_unavailable", exc)
            with self._cond:
                self._finish(record, JobState.FAILED, failure=failure)
                self._cond.notify_all()
            self._flush_records()
            return False
        return True

    def _flush_records(self) -> None:
        """Write pending finished records, without the lock held; a failure is remembered."""
        while True:
            with self._cond:
                recorder = self._recorder
                if recorder is None or not self._to_record:
                    return
                snapshot = self._to_record.popleft()
            try:
                recorder.finished(snapshot)
                recorded = True
            except Exception as exc:  # noqa: BLE001 - the job is marked unrecorded instead
                logger.warning(
                    "Job %s outcome could not be recorded: %s", snapshot.id, type(exc).__name__
                )
                recorded = False
            with self._cond:
                record = self._jobs.get(snapshot.id)
                if record is not None:
                    record.finish_pending = False
                    record.finish_failed = not recorded
                    self._evict()  # a recorded job may now be forgotten
                self._cond.notify_all()

    def _run(self, record: _JobRecord, job_input: JobInput) -> bool:
        """Run one job without the lock; it always ends succeeded or failed.

        Returns False after a ``BaseException``, once the manager has been closed.
        """
        state = JobState.FAILED
        failure: JobFailure | None = self._failure(record.id, "internal_error", None)
        result = None
        fatal = False
        try:
            outcome = self._research(job_input)
            try:
                result = self._build(outcome)
            except Exception as exc:  # noqa: BLE001 - any build failure fails only this job
                failure = self._failure(record.id, "result_failed", exc)
            else:
                state, failure = JobState.SUCCEEDED, None
        except Exception as exc:  # noqa: BLE001 - any job failure must not stop the worker
            failure = self._failure(record.id, _failure_code(exc), exc)
        except BaseException as exc:  # noqa: BLE001 - re-raising would print its message
            failure = self._failure(record.id, "internal_error", exc)
            fatal = True
        finally:
            with self._cond:
                self._finish(record, state, failure=failure, result=result)
                if fatal:
                    logger.error("Job worker stopped after job %s; manager closed", record.id)
                    self._close("worker_stopped")
                self._cond.notify_all()
        return not fatal

    @staticmethod
    def _failure(job_id: str, code: str, exc: BaseException | None) -> JobFailure:
        # Only type names are logged: messages can contain request or model content.
        if exc is not None:
            cause = exc.__cause__
            logger.warning(
                "Job %s failed code=%s error=%s cause=%s",
                job_id,
                code,
                type(exc).__name__,
                type(cause).__name__ if cause is not None else None,
            )
        return JobFailure(code=code, message=FAILURE_MESSAGES[code])
