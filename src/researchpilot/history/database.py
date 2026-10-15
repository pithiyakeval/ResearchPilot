"""Engine, sessions and the single-instance lock for the research history database.

Creating an engine never connects. ``hide_parameters`` keeps bound values (results, labels)
out of SQLAlchemy's error messages, and the URL, which holds credentials, is never logged.

Known logging risk, to be handled by the API's logging setup (Stage 4B): the libraries log
on their own loggers outside this package's control. ``sqlalchemy.pool`` logs full
tracebacks when closing or resetting a connection fails, and some ``psycopg`` warnings
include the connection's description, which names the host, port, user and database (never
the password). Those loggers must be raised to CRITICAL or filtered wherever these engines
run; this package itself logs only ``describe_error`` output.
"""

import logging
import threading

from sqlalchemy import Connection, Engine, create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError
from sqlalchemy.orm import Session, sessionmaker

from researchpilot.history.settings import DatabaseSettings
from researchpilot.history.store import (
    DATABASE_ERRORS,
    HistoryConfigError,
    HistoryUnavailableError,
    describe_error,
)

logger = logging.getLogger(__name__)

DRIVER = "postgresql+psycopg"
# Advisory-lock key owned by the one ResearchPilot instance allowed to run jobs on a database:
# "RPJOBS" followed by 1, as a signed 64-bit integer.
INSTANCE_LOCK_KEY = 0x5250_4A4F_4253_0001


def create_history_engine(settings: DatabaseSettings, **engine_options: object) -> Engine:
    """A lazy, pooled engine; raises ``HistoryConfigError`` for a missing or unsupported URL.

    ``engine_options`` are passed to ``create_engine``, for example a test ``creator``.
    """
    if settings.url is None:
        raise HistoryConfigError
    try:
        url = make_url(settings.url.get_secret_value())
    except ArgumentError:
        # The parse error quotes the URL, which may hold a password.
        raise HistoryConfigError from None
    if url.drivername != DRIVER:
        raise HistoryConfigError
    timeouts = (
        f"-c statement_timeout={settings.statement_timeout_ms} "
        f"-c lock_timeout={settings.lock_timeout_ms}"
    )
    return create_engine(
        url,
        pool_pre_ping=True,
        pool_size=settings.pool_size,
        max_overflow=settings.max_overflow,
        pool_timeout=settings.pool_timeout_seconds,
        pool_recycle=1800,
        hide_parameters=True,
        connect_args={
            "connect_timeout": settings.connect_timeout_seconds,
            "application_name": "researchpilot",
            "options": timeouts,
        },
        **engine_options,
    )


def session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(engine, expire_on_commit=False)


class InstanceLock:
    """A session-level PostgreSQL advisory lock on one dedicated connection.

    Held for the application's lifetime, it ensures only one instance recovers unfinished
    jobs and runs the worker. PostgreSQL releases it when that connection ends, including
    when the process dies, so ``is_held`` checks the server rather than trusting local state.
    """

    def __init__(self, engine: Engine, *, key: int = INSTANCE_LOCK_KEY) -> None:
        if not -(2**63) <= key < 2**63:
            raise ValueError("An advisory-lock key must be a signed 64-bit integer.")
        self._engine = engine
        self._key = key
        self._connection: Connection | None = None
        self._guard = threading.Lock()

    def acquire(self) -> bool:
        """Take the lock without waiting; False if another instance holds it."""
        with self._guard:
            if self._connection is not None:
                return True
            connection = None
            try:
                connection = self._engine.connect()
                acquired = connection.execute(
                    text("SELECT pg_try_advisory_lock(:key)"), {"key": self._key}
                ).scalar_one()
                # Ends the transaction only; a session-level lock outlives it.
                connection.commit()
            except DATABASE_ERRORS as exc:
                logger.warning("History lock acquisition failed: %s", describe_error(exc))
                if connection is not None:
                    _discard(connection)
                raise HistoryUnavailableError from None
            if not acquired:
                connection.close()
                logger.warning("The research history lock is held by another instance")
                return False
            self._connection = connection
            return True

    def is_held(self) -> bool:
        """Ask the server whether this lock's own session still holds the lock."""
        with self._guard:
            connection = self._connection
            if connection is None:
                return False
            high, low = (self._key >> 32) & 0xFFFF_FFFF, self._key & 0xFFFF_FFFF
            try:
                held = connection.execute(
                    text(
                        "SELECT EXISTS (SELECT 1 FROM pg_locks WHERE locktype = 'advisory' "
                        "AND pid = pg_backend_pid() AND granted AND objsubid = 1 "
                        "AND classid::bigint = :high AND objid::bigint = :low)"
                    ),
                    {"high": high, "low": low},
                ).scalar_one()
                connection.commit()
            except DATABASE_ERRORS as exc:
                logger.warning("History lock check failed: %s", describe_error(exc))
                held = False
            if not held:
                logger.warning("The research history lock has been lost")
                self._connection = None
                _discard(connection)
            return bool(held)

    def release(self) -> None:
        """Release the lock and discard its connection; calling it again does nothing."""
        with self._guard:
            connection, self._connection = self._connection, None
        if connection is None:
            return
        try:
            connection.execute(text("SELECT pg_advisory_unlock(:key)"), {"key": self._key})
            connection.commit()
        except DATABASE_ERRORS as exc:
            logger.warning("History lock release failed: %s", describe_error(exc))
        finally:
            # Never return this connection to the pool: ending its session releases the lock
            # even if the unlock above failed.
            _discard(connection)


def _discard(connection: Connection) -> None:
    try:
        connection.invalidate()
        connection.close()
    except DATABASE_ERRORS as exc:
        logger.warning("Discarding a history connection failed: %s", describe_error(exc))
