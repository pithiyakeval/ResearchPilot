"""The API's bridge to persistent research history.

This module converts job-manager snapshots into ``researchpilot.history`` records, normalizes
user labels, maps history errors to problem responses, and provides ``HistoryRuntime``, which
owns the history engine, store and instance lock. ``JobManager`` stays free of database code
and the history package stays free of API types.
"""

import logging
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, Protocol

from sqlalchemy import Engine

from researchpilot.api.jobs import TERMINAL_STATES, JobSnapshot, JobState
from researchpilot.api.problems import ProblemError
from researchpilot.history.database import InstanceLock, create_history_engine, session_factory
from researchpilot.history.records import (
    MAX_LABEL_LENGTH,
    FinishedJob,
    QueuedJob,
    StartedJob,
    label_is_valid,
)
from researchpilot.history.settings import DatabaseSettings
from researchpilot.history.store import (
    HistoryConfigError,
    HistoryEncodingError,
    HistoryError,
    HistoryLockError,
    HistorySchemaError,
    HistoryStore,
    HistoryUnavailableError,
    describe_error,
)

logger = logging.getLogger(__name__)

# Seconds a client should wait before retrying while the history database is unavailable.
HISTORY_RETRY_AFTER_SECONDS = 5


class InvalidLabelError(ValueError):
    """A nonblank label that breaks the stored-label rule; the message never quotes it."""

    def __init__(self) -> None:
        super().__init__(
            f"A label must be at most {MAX_LABEL_LENGTH} characters and must not contain "
            "control characters."
        )


def normalize_label(raw: str | None) -> str | None:
    """Strip Unicode whitespace; a blank label is absent. Raises ``InvalidLabelError``."""
    if raw is None:
        return None
    label = raw.strip()
    if not label:
        return None
    if not label_is_valid(label):
        raise InvalidLabelError
    return label


# --- Snapshots to history records ---


def queued_record(snapshot: JobSnapshot, label: str | None) -> QueuedJob:
    """The record written when a job is accepted; ``label`` must already be normalized."""
    if snapshot.state is not JobState.QUEUED:
        raise ValueError("A queued record needs a queued job.")
    return QueuedJob(snapshot.id, snapshot.mode, snapshot.created_at, label=label)


def started_record(snapshot: JobSnapshot) -> StartedJob:
    if snapshot.state is not JobState.RUNNING or snapshot.started_at is None:
        raise ValueError("A started record needs a running job with a start time.")
    return StartedJob(snapshot.id, snapshot.started_at)


def finished_record(snapshot: JobSnapshot) -> FinishedJob:
    """The terminal record: the result, the failure code or the cancel reason by state."""
    if snapshot.state not in TERMINAL_STATES or snapshot.finished_at is None:
        raise ValueError("A finished record needs a finished job.")
    return FinishedJob(
        snapshot.id,
        snapshot.state.value,  # type: ignore[arg-type]  # a terminal JobState value
        snapshot.finished_at,
        result=snapshot.result if snapshot.state is JobState.SUCCEEDED else None,
        failure_code=snapshot.failure.code
        if snapshot.state is JobState.FAILED and snapshot.failure is not None
        else None,
        cancel_reason=snapshot.cancel_reason if snapshot.state is JobState.CANCELLED else None,
    )


# --- History errors to problem responses ---


def history_problem(error: HistoryError) -> ProblemError:
    """A 503 problem with a fixed message; connection details never reach the client."""
    if isinstance(error, HistoryConfigError):
        return ProblemError(
            503,
            "history_unconfigured",
            "Research history is not configured",
            "The research history database has not been configured.",
        )
    if isinstance(error, HistorySchemaError):
        return ProblemError(
            503,
            "history_schema_outdated",
            "Research history schema is not ready",
            "The research history database is not at the expected schema; run the migrations.",
        )
    return ProblemError(
        503,
        "history_unavailable",
        "Research history is unavailable",
        "The research history database is unavailable.",
        headers={"Retry-After": str(HISTORY_RETRY_AFTER_SECONDS)},
    )


# --- Runtime: the engine, store and instance lock, brought to a ready state ---


class HistoryState(StrEnum):
    UNCONFIGURED = "unconfigured"
    UNAVAILABLE = "unavailable"
    READY = "ready"


# Fixed reason codes; safe to log and expose, never derived from database messages.
HistoryReason = Literal[
    "not_configured",
    "not_checked",
    "database_unavailable",
    "lock_held_elsewhere",
    "lock_lost",
    "schema_outdated",
    "encoding_unsupported",
    "recovery_failed",
    "purge_failed",
    "closed",
]
_SCHEMA_REASONS = frozenset({"schema_outdated", "encoding_unsupported"})


@dataclass(frozen=True)
class HistoryStatus:
    """The runtime's state and, unless ready, a fixed reason code. Holds no secrets."""

    state: HistoryState
    reason: HistoryReason | None = None

    @property
    def ready(self) -> bool:
        return self.state is HistoryState.READY

    def problem(self) -> ProblemError:
        """The 503 problem for a status that is not ready."""
        if self.ready:
            raise ValueError("A ready history has no problem to report.")
        if self.state is HistoryState.UNCONFIGURED:
            return history_problem(HistoryConfigError())
        if self.reason in _SCHEMA_REASONS:
            return history_problem(HistorySchemaError())
        return history_problem(HistoryUnavailableError())


READY = HistoryStatus(HistoryState.READY)


class HistoryControl(Protocol):
    """What the app needs from a history runtime; tests inject a fake without a database."""

    @property
    def status(self) -> HistoryStatus: ...

    @property
    def store(self) -> HistoryStore | None:
        """The store, only while ready."""

    def writing(self) -> AbstractContextManager[HistoryStore]:
        """The store for one write; raises ``HistoryUnavailableError`` if writing is unsafe."""

    def ensure_ready(self) -> HistoryStatus: ...

    def close(self) -> None: ...


class HistoryRecorder:
    """Records each job's lifecycle through a history runtime's ``writing()`` guard.

    Implements ``JobRecorder``: any failure, including history not being ready, closed or no
    longer holding the instance lock, raises, and the job manager then refuses, fails or
    marks the job as described there.
    """

    def __init__(self, history: HistoryControl) -> None:
        self._history = history

    def queued(self, snapshot: JobSnapshot, label: str | None) -> None:
        record = queued_record(snapshot, label)
        with self._history.writing() as store:
            store.record_queued(record)

    def started(self, snapshot: JobSnapshot) -> None:
        record = started_record(snapshot)
        with self._history.writing() as store:
            store.record_started(record)

    def finished(self, snapshot: JobSnapshot) -> None:
        record = finished_record(snapshot)
        with self._history.writing() as store:
            store.record_finished(record)

    def discard(self, job_id: str) -> None:
        with self._history.writing() as store:
            store.delete_queued(job_id)


# Database driver loggers that can log tracebacks or connection descriptions (host, port,
# user, database) on their own; see researchpilot.history.database.
DRIVER_LOGGERS = ("sqlalchemy.pool", "psycopg")


def quiet_driver_loggers() -> None:
    """Silence driver loggers below CRITICAL; history errors are logged by type only."""
    for name in DRIVER_LOGGERS:
        logging.getLogger(name).setLevel(logging.CRITICAL)


StoreFactory = Callable[[Engine, str], HistoryStore]
LockFactory = Callable[[Engine], InstanceLock]


def _default_store(engine: Engine, instance_id: str) -> HistoryStore:
    return HistoryStore(session_factory(engine), instance_id=instance_id)


def _unavailable(reason: HistoryReason) -> HistoryStatus:
    return HistoryStatus(HistoryState.UNAVAILABLE, reason)


class HistoryRuntime:
    """Owns the history engine, store and instance lock, and brings them to a ready state.

    While not ready, ``ensure_ready`` makes at most one attempt per ``retry_seconds``:

    1. create the engine, store and lock (once; they are reused afterwards);
    2. acquire the instance lock, so this is the only instance recovering and running jobs;
    3. check connectivity, UTF8 encoding and the schema revision;
    4. mark jobs left queued or running as ``interrupted_by_restart``;
    5. purge expired jobs.

    Steps 4 and 5 run once per lock acquisition, only while the lock is held. A failed step
    keeps a held lock, so the next attempt resumes at that step and no other instance can
    recover jobs in between. Once ready, each call only asks the server whether the lock is
    still held; if it was lost, the runtime becomes unavailable, and the next acquisition
    repeats steps 3-5, since another instance may have run jobs meanwhile.

    Writes go through ``writing()``, which hands out the store only while ready, confirms
    with the server that this instance still holds the lock, and counts the write as in
    progress. ``close()`` refuses new writes and waits up to ``drain_seconds`` for those in
    progress before releasing the lock, so no write starts, and normally none commits, after
    the lock is released.

    There is no background thread: callers drive retries. A missing or unsupported database
    URL is permanent until restart. Statuses and logs carry only fixed reason codes.
    """

    def __init__(
        self,
        settings: DatabaseSettings,
        *,
        instance_id: str | None = None,
        retry_seconds: float = 5.0,
        monotonic: Callable[[], float] = time.monotonic,
        engine_factory: Callable[[DatabaseSettings], Engine] = create_history_engine,
        store_factory: StoreFactory = _default_store,
        lock_factory: LockFactory = InstanceLock,
        drain_seconds: float = 15.0,
    ) -> None:
        if retry_seconds <= 0:
            raise ValueError("retry_seconds must be positive.")
        if drain_seconds < 0:
            raise ValueError("drain_seconds must not be negative.")
        self._settings = settings
        self.instance_id = instance_id or uuid.uuid4().hex
        self._retry_seconds = retry_seconds
        self._monotonic = monotonic
        self._engine_factory = engine_factory
        self._store_factory = store_factory
        self._lock_factory = lock_factory
        # Above the longest bounded write: connect, pool and statement timeouts together.
        self._drain_seconds = drain_seconds
        self._guard = threading.Condition()
        self._active_writes = 0
        self._engine: Engine | None = None
        self._store: HistoryStore | None = None
        self._lock: InstanceLock | None = None
        self._recovered = False
        self._purged = False
        self._last_attempt: float | None = None
        self._closed = False
        self._status = (
            HistoryStatus(HistoryState.UNCONFIGURED, "not_configured")
            if settings.url is None
            else _unavailable("not_checked")
        )

    @property
    def status(self) -> HistoryStatus:
        """The last known status, without contacting the database."""
        with self._guard:
            return self._status

    @property
    def store(self) -> HistoryStore | None:
        """The store, only while ready."""
        with self._guard:
            return self._store if self._status.ready else None

    @contextmanager
    def writing(self) -> Iterator[HistoryStore]:
        """The store for one write, only while ready and holding the instance lock.

        Raises ``HistoryUnavailableError`` when closed or not ready, or when the server says
        the lock is no longer held, which also makes the runtime unavailable. The write
        counts as in progress until the block exits, and ``close`` waits for it.
        """
        with self._guard:
            if self._closed or not self._status.ready:
                raise HistoryUnavailableError
            assert self._lock is not None and self._store is not None
            if not self._lock.is_held():
                self._last_attempt = self._monotonic()
                self._set_status(_unavailable("lock_lost"))
                raise HistoryUnavailableError
            self._active_writes += 1
            store = self._store
        try:
            yield store
        finally:
            with self._guard:
                self._active_writes -= 1
                self._guard.notify_all()

    def ensure_ready(self) -> HistoryStatus:
        """Bring the runtime to ready if possible, attempting at most once per interval."""
        with self._guard:
            if self._closed or self._status.state is HistoryState.UNCONFIGURED:
                return self._status
            if self._status.ready:
                assert self._lock is not None
                if self._lock.is_held():
                    return self._status
                # The retry interval starts now: the lock was lost, not merely untried.
                self._last_attempt = self._monotonic()
                self._set_status(_unavailable("lock_lost"))
                return self._status
            now = self._monotonic()
            if self._last_attempt is not None and now - self._last_attempt < self._retry_seconds:
                return self._status
            self._last_attempt = now
            self._set_status(self._attempt())
            return self._status

    def close(self) -> None:
        """Refuse new writes, let those in progress finish, then release the lock and dispose
        of the engine; calling it again does nothing."""
        with self._guard:
            if self._closed:
                return
            self._closed = True
            if not self._guard.wait_for(lambda: self._active_writes == 0, self._drain_seconds):
                logger.warning("Closing research history with a write still in progress")
            lock, engine = self._lock, self._engine
            self._lock = self._store = self._engine = None
            if self._status.state is not HistoryState.UNCONFIGURED:
                self._set_status(_unavailable("closed"))
            if lock is not None:
                lock.release()
            if engine is not None:
                try:
                    engine.dispose()
                except Exception as exc:  # noqa: BLE001 - closing must not fail shutdown
                    logger.warning("Disposing the history engine failed: %s", describe_error(exc))

    # --- Internals; the caller holds the guard ---

    def _attempt(self) -> HistoryStatus:
        try:
            self._create_resources()
        except HistoryConfigError:
            return HistoryStatus(HistoryState.UNCONFIGURED, "not_configured")
        assert self._store is not None and self._lock is not None
        try:
            if not self._lock.is_held():
                # A fresh acquisition: recovery and purge must run again under this lock.
                self._recovered = self._purged = False
                if not self._lock.acquire():
                    return _unavailable("lock_held_elsewhere")
        except HistoryUnavailableError:
            return _unavailable("database_unavailable")
        try:
            self._store.check_schema()
        except HistoryEncodingError:
            return _unavailable("encoding_unsupported")
        except HistorySchemaError:
            return _unavailable("schema_outdated")
        except HistoryError:
            return _unavailable("database_unavailable")
        if not self._recovered:
            try:
                self._store.interrupt_unfinished(self._lock)
            except HistoryLockError:
                return _unavailable("lock_lost")
            except HistoryError:
                return _unavailable("recovery_failed")
            self._recovered = True
        if not self._purged:
            try:
                self._store.purge_expired()
            except HistoryError:
                return _unavailable("purge_failed")
            self._purged = True
        return READY

    def _create_resources(self) -> None:
        if self._engine is not None:
            return
        engine = self._engine_factory(self._settings)
        try:
            store = self._store_factory(engine, self.instance_id)
            lock = self._lock_factory(engine)
        except BaseException:
            engine.dispose()
            raise
        self._engine, self._store, self._lock = engine, store, lock

    def _set_status(self, status: HistoryStatus) -> None:
        if status == self._status:
            return
        self._status = status
        if status.ready:
            logger.info("Research history is ready")
        else:
            logger.warning("Research history is %s: %s", status.state, status.reason)
