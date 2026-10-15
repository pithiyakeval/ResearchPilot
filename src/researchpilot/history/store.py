"""Repository for persistent research history.

Each method runs in one short transaction on its own session, so one store can be shared by
request threads and the job worker. Errors carry fixed messages only:

* database failures become ``HistoryUnavailableError``;
* writes that conflict with the stored job become ``HistoryConflictError``.

The database's own exception is never chained, logged or included: its message can name
hosts, addresses and users, quote SQL, or quote row values. Only ``describe_error`` output
(type names and the SQLSTATE code) is logged.

Reads hide expired jobs; ``purge_expired`` deletes them.
"""

import logging
import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import fields
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import psycopg
from pydantic import ValidationError
from sqlalchemy import Row, delete, func, insert, literal, or_, select, text, tuple_, update
from sqlalchemy.exc import IntegrityError, ProgrammingError, SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from researchpilot.history import EXPECTED_SCHEMA_REVISION
from researchpilot.history.models import ResearchJobRow
from researchpilot.history.records import (
    STATE_FIELDS,
    FinishedJob,
    HistoryCursor,
    HistoryRecord,
    HistorySummary,
    QueuedJob,
    StartedJob,
    is_job_id,
    require_aware,
    require_job_id,
)
from researchpilot.results import ResearchResult

if TYPE_CHECKING:
    from researchpilot.history.database import InstanceLock

logger = logging.getLogger(__name__)

SUCCESS_RETENTION = timedelta(days=30)
FAILURE_RETENTION = timedelta(days=14)
INTERRUPTED_BY_RESTART = "interrupted_by_restart"
RESULT_VERSION = 1
REQUIRED_ENCODING = "UTF8"
MAX_LIST_LIMIT = 200
DATABASE_ERRORS = (SQLAlchemyError, psycopg.Error)
_UNFINISHED = ("queued", "running")
_SQLSTATE = re.compile(r"[0-9A-Z]{5}")
# The states a job must be in to reach each finished state, from whether it must have started.
_PREVIOUS_STATES = {True: ("running",), False: ("queued",), None: _UNFINISHED}
_SUMMARY_COLUMNS = [getattr(ResearchJobRow, field.name) for field in fields(HistorySummary)]


class HistoryError(Exception):
    """Base class for history errors; every message is fixed."""

    message = "The research history failed."

    def __init__(self) -> None:
        super().__init__(self.message)


class HistoryConfigError(HistoryError):
    message = (
        "The research history database is not configured; set RESEARCHPILOT_DATABASE_URL "
        "to a postgresql+psycopg:// URL."
    )


class HistoryUnavailableError(HistoryError):
    message = "The research history database is unavailable."


class HistorySchemaError(HistoryError):
    message = "The research history schema is not at the expected revision; run the migrations."


class HistoryEncodingError(HistorySchemaError):
    message = "The research history database must use UTF8 encoding."


class HistoryConflictError(HistoryError):
    message = "The history write conflicts with the stored job."


class HistoryLockError(HistoryError):
    message = "This instance does not hold the research history lock."


def _utc_now() -> datetime:
    return datetime.now(UTC)


class HistoryStore:
    def __init__(
        self,
        sessions: sessionmaker[Session],
        *,
        instance_id: str,
        clock: Callable[[], datetime] = _utc_now,
        success_retention: timedelta = SUCCESS_RETENTION,
        failure_retention: timedelta = FAILURE_RETENTION,
    ) -> None:
        require_job_id(instance_id)
        if success_retention <= timedelta(0) or failure_retention <= timedelta(0):
            raise ValueError("Retention periods must be positive.")
        self._sessions = sessions
        self._instance_id = instance_id
        self._clock = clock
        self._success_retention = success_retention
        self._failure_retention = failure_retention

    # --- Health ---

    def check_schema(self) -> None:
        """Raise unless the database answers, uses UTF8 and is at the expected revision.

        UTF8 is required so that PostgreSQL's ``char_length`` counts the same characters as
        Python's ``len`` when label lengths are checked.
        """
        with self._transaction("schema check") as session:
            encoding = session.execute(text("SHOW server_encoding")).scalar_one()
            if encoding != REQUIRED_ENCODING:
                logger.warning("History database encoding is not %s", REQUIRED_ENCODING)
                raise HistoryEncodingError
            try:
                versions = session.execute(text("SELECT version_num FROM alembic_version"))
                revisions = list(versions.scalars())
            except ProgrammingError:
                logger.warning("History schema check found no migration version table")
                raise HistorySchemaError from None
        if revisions != [EXPECTED_SCHEMA_REVISION]:
            logger.warning("History schema is at an unexpected revision")
            raise HistorySchemaError

    # --- Writes ---

    def record_queued(self, job: QueuedJob) -> None:
        values = {
            "id": job.id,
            "label": job.label,
            "mode": job.mode,
            "state": "queued",
            "created_at": job.created_at,
            "instance_id": self._instance_id,
            "updated_at": self._now(),
        }
        with self._transaction("record_queued") as session:
            session.execute(insert(ResearchJobRow).values(**values))

    def record_started(self, job: StartedJob) -> None:
        statement = (
            update(ResearchJobRow)
            .where(ResearchJobRow.id == job.id, ResearchJobRow.state == "queued")
            .values(state="running", started_at=job.started_at, updated_at=self._now())
        )
        self._update_one("record_started", job.id, statement)

    def record_finished(self, job: FinishedJob) -> None:
        retention = self._success_retention if job.state == "succeeded" else self._failure_retention
        previous = _PREVIOUS_STATES[STATE_FIELDS[job.state]["started_at"]]
        statement = (
            update(ResearchJobRow)
            .where(ResearchJobRow.id == job.id, ResearchJobRow.state.in_(previous))
            .values(
                state=job.state,
                finished_at=job.finished_at,
                result=None if job.result is None else job.result.model_dump(mode="json"),
                result_version=None if job.result is None else RESULT_VERSION,
                failure_code=job.failure_code,
                cancel_reason=job.cancel_reason,
                expires_at=job.finished_at + retention,
                updated_at=self._now(),
            )
        )
        self._update_one("record_finished", job.id, statement)

    def delete_queued(self, job_id: str) -> None:
        """Remove a job the queue refused after it was recorded."""
        require_job_id(job_id)
        statement = delete(ResearchJobRow).where(
            ResearchJobRow.id == job_id, ResearchJobRow.state == "queued"
        )
        self._update_one("delete_queued", job_id, statement)

    # --- Reads ---

    def get(self, job_id: str) -> HistoryRecord | None:
        if not is_job_id(job_id):
            return None
        now = self._now()
        statement = select(ResearchJobRow).where(ResearchJobRow.id == job_id, _not_expired(now))
        with self._transaction("get") as session:
            row = session.execute(statement).scalar_one_or_none()
            if row is None:
                return None
            summary = {field.name: getattr(row, field.name) for field in fields(HistorySummary)}
            return HistoryRecord(**summary, result=_stored_result(row))

    def list_recent(
        self, *, limit: int = 50, before: HistoryCursor | None = None
    ) -> list[HistorySummary]:
        """Unexpired jobs without results, newest first by ``(created_at, id)``.

        With ``before``, only jobs after that position in this order are returned, so passing
        the last summary's ``cursor`` fetches the next page. A short page is the last one.
        """
        if not 1 <= limit <= MAX_LIST_LIMIT:
            raise ValueError(f"limit must be between 1 and {MAX_LIST_LIMIT}.")
        if before is not None and not isinstance(before, HistoryCursor):
            raise ValueError("before must be a HistoryCursor.")
        statement = _recent_statement(self._now(), limit, before)
        with self._transaction("list_recent") as session:
            return [_summary(row) for row in session.execute(statement)]

    # --- Maintenance ---

    def interrupt_unfinished(self, lock: "InstanceLock") -> int:
        """Fail every queued or running job with ``interrupted_by_restart``.

        Only the instance holding the history lock may do this, before its worker starts:
        any unfinished job then belongs to a process that no longer runs.
        """
        if not lock.is_held():
            raise HistoryLockError
        now = self._now()
        # Never earlier than the job's own timestamps, even if clocks disagree.
        finished_at = func.greatest(
            literal(now, ResearchJobRow.finished_at.type),
            func.coalesce(ResearchJobRow.started_at, ResearchJobRow.created_at),
        )
        statement = (
            update(ResearchJobRow)
            .where(ResearchJobRow.state.in_(_UNFINISHED))
            .values(
                state="failed",
                failure_code=INTERRUPTED_BY_RESTART,
                finished_at=finished_at,
                expires_at=finished_at + self._failure_retention,
                updated_at=now,
            )
        )
        with self._transaction("interrupt_unfinished") as session:
            count = session.execute(statement).rowcount
        if count:
            logger.warning("Marked %d unfinished job(s) as interrupted by a restart", count)
        return count

    def purge_expired(self) -> int:
        statement = delete(ResearchJobRow).where(ResearchJobRow.expires_at <= self._now())
        with self._transaction("purge_expired") as session:
            count = session.execute(statement).rowcount
        if count:
            logger.info("Purged %d expired job(s) from history", count)
        return count

    # --- Internals ---

    def _now(self) -> datetime:
        now = self._clock()
        require_aware("The history clock's time", now)
        return now

    @contextmanager
    def _transaction(self, operation: str) -> Iterator[Session]:
        try:
            with self._sessions.begin() as session:
                yield session
        except IntegrityError as exc:
            logger.warning("History %s conflicted: %s", operation, describe_error(exc))
            raise HistoryConflictError from None
        except DATABASE_ERRORS as exc:
            logger.warning("History %s failed: %s", operation, describe_error(exc))
            raise HistoryUnavailableError from None

    def _update_one(self, operation: str, job_id: str, statement: Any) -> None:
        with self._transaction(operation) as session:
            if session.execute(statement).rowcount != 1:
                logger.warning(
                    "History %s found job %s missing or in another state", operation, job_id
                )
                raise HistoryConflictError


def describe_error(exc: BaseException) -> str:
    """Error type names and the SQLSTATE code only, safe to log.

    Database error messages are never logged or chained: they can name hosts, addresses and
    users, quote SQL, or (in PostgreSQL's details) quote row values.
    """
    original = getattr(exc, "orig", None) or exc
    parts = [type(exc).__name__]
    if original is not exc:
        parts.append(type(original).__name__)
    sqlstate = getattr(original, "sqlstate", None)
    if isinstance(sqlstate, str) and _SQLSTATE.fullmatch(sqlstate):
        parts.append(f"SQLSTATE {sqlstate}")
    return " / ".join(parts)


def _not_expired(now: datetime) -> Any:
    return or_(ResearchJobRow.expires_at.is_(None), ResearchJobRow.expires_at > now)


def _recent_statement(now: datetime, limit: int, before: HistoryCursor | None) -> Any:
    # The row comparison matches the (created_at DESC, id DESC) order and index exactly.
    statement = select(*_SUMMARY_COLUMNS).where(_not_expired(now))
    if before is not None:
        statement = statement.where(
            tuple_(ResearchJobRow.created_at, ResearchJobRow.id)
            < tuple_(
                literal(before.created_at, ResearchJobRow.created_at.type),
                literal(before.id, ResearchJobRow.id.type),
            )
        )
    return statement.order_by(ResearchJobRow.created_at.desc(), ResearchJobRow.id.desc()).limit(
        limit
    )


def _summary(row: Row[Any]) -> HistorySummary:
    return HistorySummary(**row._asdict())


def _stored_result(row: ResearchJobRow) -> ResearchResult | None:
    if row.result is None:
        return None
    if row.result_version != RESULT_VERSION:
        logger.warning("History job %s has an unsupported result version", row.id)
        raise HistorySchemaError
    try:
        return ResearchResult.model_validate(row.result)
    except ValidationError:
        # The validation error quotes the stored text, so it is dropped.
        logger.warning("History job %s has a stored result that does not validate", row.id)
        raise HistorySchemaError from None
