"""Offline tests for HistoryRuntime: state, retries, recovery order and cleanup.

The engine, store and lock are fakes that record every call in one shared log, and time comes
from an injected clock, so the tests are deterministic and never sleep. One test uses the real
engine, store and lock with a connection that always fails; no database is contacted.
"""

import logging
import os
import threading

import psycopg
import pytest

from researchpilot.api.history import HistoryRuntime, HistoryState, HistoryStatus
from researchpilot.history.database import create_history_engine
from researchpilot.history.settings import DatabaseSettings
from researchpilot.history.store import (
    HistoryEncodingError,
    HistoryLockError,
    HistorySchemaError,
    HistoryUnavailableError,
)

SECRET = "hunter2-SECRET"
HOST = "db.invalid"
URL = f"postgresql+psycopg://rp_admin:{SECRET}@{HOST}:5432/researchpilot"
INSTANCE = "0123456789abcdef0123456789abcdef"
TIMEOUT = 5.0
RECOVERY = ["is_held", "acquire", "check_schema", "interrupt_unfinished", "purge_expired"]


@pytest.fixture(autouse=True)
def no_database_environment(monkeypatch):
    for name in list(os.environ):
        if name.startswith("RESEARCHPILOT_DATABASE_"):
            monkeypatch.delenv(name)


def settings(url: str | None = URL) -> DatabaseSettings:
    return DatabaseSettings(_env_file=None, url=url)  # type: ignore[call-arg]


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class FakeEngine:
    def __init__(self, log: list[str]) -> None:
        self.log = log

    def dispose(self) -> None:
        self.log.append("dispose")


class FakeLock:
    """Scripted instance lock: ``acquire`` answers in turn; ``held`` mirrors the server."""

    def __init__(self, log: list[str]) -> None:
        self.log = log
        self.held = False
        self.acquire_answers: list[bool | Exception] = []

    def acquire(self) -> bool:
        self.log.append("acquire")
        answer = self.acquire_answers.pop(0) if self.acquire_answers else True
        if isinstance(answer, Exception):
            raise answer
        self.held = self.held or answer
        return answer

    def is_held(self) -> bool:
        self.log.append("is_held")
        return self.held

    def release(self) -> None:
        self.log.append("release")
        self.held = False


class FakeStore:
    """Each operation raises the next scripted error, if any, else succeeds."""

    def __init__(self, log: list[str], lock: FakeLock) -> None:
        self.log = log
        self.lock = lock
        self.errors: dict[str, list[Exception]] = {}

    def _call(self, name: str) -> None:
        self.log.append(name)
        pending = self.errors.get(name)
        if pending:
            raise pending.pop(0)

    def check_schema(self) -> None:
        self._call("check_schema")

    def interrupt_unfinished(self, lock) -> int:
        assert lock is self.lock
        self._call("interrupt_unfinished")
        return 0

    def purge_expired(self) -> int:
        self._call("purge_expired")
        return 0


class Harness:
    def __init__(self, *, url: str | None = URL) -> None:
        self.log: list[str] = []
        self.clock = Clock()
        self.engine = FakeEngine(self.log)
        self.lock = FakeLock(self.log)
        self.store = FakeStore(self.log, self.lock)
        self.created: list[str] = []

        def engine_factory(_settings):
            self.created.append("engine")
            return self.engine

        def store_factory(engine, instance_id):
            assert engine is self.engine and instance_id == INSTANCE
            self.created.append("store")
            return self.store

        def lock_factory(engine):
            assert engine is self.engine
            self.created.append("lock")
            return self.lock

        self.runtime = HistoryRuntime(
            settings(url),
            instance_id=INSTANCE,
            retry_seconds=5.0,
            monotonic=self.clock,
            engine_factory=engine_factory,  # type: ignore[arg-type]
            store_factory=store_factory,  # type: ignore[arg-type]
            lock_factory=lock_factory,  # type: ignore[arg-type]
        )

    def ensure(self) -> HistoryStatus:
        return self.runtime.ensure_ready()

    def after(self, seconds: float) -> HistoryStatus:
        self.clock.now += seconds
        return self.ensure()


def unavailable(reason: str) -> HistoryStatus:
    return HistoryStatus(HistoryState.UNAVAILABLE, reason)  # type: ignore[arg-type]


READY = HistoryStatus(HistoryState.READY)


# --- Configuration ---


def test_a_missing_url_is_unconfigured_and_never_creates_an_engine() -> None:
    harness = Harness(url=None)

    assert harness.runtime.status == HistoryStatus(HistoryState.UNCONFIGURED, "not_configured")
    for seconds in (0, 10, 100):
        assert harness.after(seconds).state is HistoryState.UNCONFIGURED
    assert harness.created == [] and harness.log == []
    assert harness.runtime.store is None
    assert harness.runtime.status.problem().code == "history_unconfigured"


def test_an_unsupported_url_is_unconfigured_permanently() -> None:
    attempts: list[str] = []

    def engine_factory(database_settings):
        attempts.append("engine")
        return create_history_engine(database_settings)

    clock = Clock()
    runtime = HistoryRuntime(
        settings(f"sqlite:///{SECRET}.db"), monotonic=clock, engine_factory=engine_factory
    )

    assert runtime.ensure_ready() == HistoryStatus(HistoryState.UNCONFIGURED, "not_configured")
    clock.now += 60
    runtime.ensure_ready()
    assert attempts == ["engine"]


def test_retry_seconds_must_be_positive() -> None:
    with pytest.raises(ValueError):
        HistoryRuntime(settings(), retry_seconds=0)


def test_a_generated_instance_id_is_32_hex_characters() -> None:
    instance_id = HistoryRuntime(settings()).instance_id
    assert len(instance_id) == 32 and set(instance_id) <= set("0123456789abcdef")


# --- Successful initialization ---


def test_first_initialization_runs_the_recovery_sequence_in_order() -> None:
    harness = Harness()
    assert harness.runtime.status == unavailable("not_checked")
    assert harness.runtime.store is None

    assert harness.ensure() == READY

    assert harness.created == ["engine", "store", "lock"]
    assert harness.log == RECOVERY
    assert harness.runtime.store is harness.store


def test_repeated_checks_only_confirm_the_lock() -> None:
    harness = Harness()
    harness.ensure()

    for seconds in (0, 1, 60, 3600):
        assert harness.after(seconds) == READY

    assert harness.log == RECOVERY + ["is_held"] * 4
    assert harness.created == ["engine", "store", "lock"]


# --- Retries ---


def test_an_unavailable_database_recovers_on_a_later_attempt() -> None:
    harness = Harness()
    harness.lock.acquire_answers = [HistoryUnavailableError()]

    assert harness.ensure() == unavailable("database_unavailable")
    assert harness.log == ["is_held", "acquire"]
    assert harness.runtime.store is None

    assert harness.after(5) == READY
    assert harness.log == ["is_held", "acquire"] + RECOVERY
    assert harness.created == ["engine", "store", "lock"]  # resources were reused


def test_retries_wait_for_the_interval() -> None:
    harness = Harness()
    harness.lock.acquire_answers = [HistoryUnavailableError(), HistoryUnavailableError()]

    harness.ensure()
    for seconds in (0, 1, 2, 1.9):  # 4.9 seconds after the first attempt
        assert harness.after(seconds) == unavailable("database_unavailable")
    assert harness.log.count("acquire") == 1

    harness.after(0.1)  # exactly 5 seconds
    assert harness.log.count("acquire") == 2


# --- Lock ownership ---


def test_a_lock_held_elsewhere_prevents_recovery_and_purge() -> None:
    harness = Harness()
    harness.lock.acquire_answers = [False, False]

    assert harness.ensure() == unavailable("lock_held_elsewhere")
    assert harness.after(5) == unavailable("lock_held_elsewhere")

    assert "check_schema" not in harness.log
    assert "interrupt_unfinished" not in harness.log
    assert "purge_expired" not in harness.log
    assert harness.runtime.store is None
    assert harness.runtime.status.problem().code == "history_unavailable"


def test_a_lost_lock_makes_the_runtime_unavailable_and_reruns_recovery() -> None:
    harness = Harness()
    harness.ensure()
    harness.lock.held = False  # e.g. the lock's connection was terminated

    assert harness.after(1) == unavailable("lock_lost")
    assert harness.runtime.store is None
    assert harness.after(4.9) == unavailable("lock_lost")  # within the interval

    assert harness.after(0.1) == READY
    assert harness.log == RECOVERY + ["is_held"] + RECOVERY


def test_a_lock_lost_during_recovery_is_reported_as_such() -> None:
    harness = Harness()
    harness.store.errors["interrupt_unfinished"] = [HistoryLockError()]

    assert harness.ensure() == unavailable("lock_lost")
    assert "purge_expired" not in harness.log


# --- Schema, recovery and purge failures ---


@pytest.mark.parametrize(
    ("error", "reason"),
    [(HistorySchemaError(), "schema_outdated"), (HistoryEncodingError(), "encoding_unsupported")],
)
def test_schema_failures_are_reported_distinctly_and_keep_the_lock(error, reason) -> None:
    harness = Harness()
    harness.store.errors["check_schema"] = [error]

    status = harness.ensure()

    assert status == unavailable(reason)
    assert status.problem().code == "history_schema_outdated"
    assert harness.log == ["is_held", "acquire", "check_schema"]
    assert harness.lock.held  # kept, so no other instance recovers jobs meanwhile

    # Once migrated, the retry resumes without acquiring the lock again.
    assert harness.after(5) == READY
    assert harness.log[3:] == ["is_held", "check_schema", "interrupt_unfinished", "purge_expired"]


def test_a_schema_check_that_cannot_connect_is_database_unavailable() -> None:
    harness = Harness()
    harness.store.errors["check_schema"] = [HistoryUnavailableError()]

    status = harness.ensure()

    assert status == unavailable("database_unavailable")
    assert status.problem().code == "history_unavailable"


def test_a_recovery_failure_is_never_ready_and_recovery_is_retried() -> None:
    harness = Harness()
    harness.store.errors["interrupt_unfinished"] = [HistoryUnavailableError()]

    assert harness.ensure() == unavailable("recovery_failed")
    assert "purge_expired" not in harness.log
    assert harness.runtime.store is None

    assert harness.after(5) == READY
    assert harness.log.count("interrupt_unfinished") == 2
    assert harness.log.count("acquire") == 1


def test_a_purge_failure_is_never_ready_and_recovery_is_not_repeated() -> None:
    harness = Harness()
    harness.store.errors["purge_expired"] = [HistoryUnavailableError()]

    assert harness.ensure() == unavailable("purge_failed")
    assert harness.runtime.store is None

    assert harness.after(5) == READY
    assert harness.log.count("interrupt_unfinished") == 1
    assert harness.log.count("purge_expired") == 2


# --- Cleanup ---


def test_close_releases_the_lock_and_disposes_the_engine_once() -> None:
    harness = Harness()
    harness.ensure()

    harness.runtime.close()
    harness.runtime.close()

    assert harness.log[len(RECOVERY) :] == ["release", "dispose"]
    assert harness.runtime.status == unavailable("closed")
    assert harness.runtime.store is None
    assert harness.after(60) == unavailable("closed")
    assert harness.log[len(RECOVERY) :] == ["release", "dispose"]  # no attempts after close


def test_close_before_any_attempt_or_when_unconfigured_is_safe() -> None:
    untried = Harness()
    untried.runtime.close()
    assert untried.log == [] and untried.runtime.status == unavailable("closed")

    unconfigured = Harness(url=None)
    unconfigured.runtime.close()
    assert unconfigured.runtime.status.state is HistoryState.UNCONFIGURED


def test_failing_to_build_the_store_disposes_the_new_engine() -> None:
    harness = Harness()

    def broken_store(_engine, _instance_id):
        raise RuntimeError("store construction failed")

    harness.runtime._store_factory = broken_store

    with pytest.raises(RuntimeError):
        harness.ensure()
    assert harness.log == ["dispose"]
    assert harness.runtime.store is None


# --- Secrets ---


def test_real_components_fail_closed_without_exposing_connection_details(caplog) -> None:
    caplog.set_level(logging.DEBUG)

    def refuse(*_args, **_kwargs):
        raise psycopg.OperationalError(
            f'connection to server at "{HOST}" (10.0.0.5), port 5432 failed: '
            f'password authentication failed for user "rp_admin" {SECRET}'
        )

    clock = Clock()
    runtime = HistoryRuntime(
        settings(),
        monotonic=clock,
        engine_factory=lambda database_settings: create_history_engine(
            database_settings, creator=refuse
        ),
    )

    status = runtime.ensure_ready()
    problem = status.problem()
    runtime.close()

    assert status == unavailable("database_unavailable")
    exposed = " ".join(
        [repr(status), str(status), problem.code, problem.title, problem.detail, caplog.text]
    )
    for leaked in (SECRET, HOST, "10.0.0.5", "rp_admin", URL):
        assert leaked not in exposed


# --- Writes: only under the lock, and drained before the lock is released ---


def test_writes_are_refused_unless_ready() -> None:
    harness = Harness()

    with pytest.raises(HistoryUnavailableError), harness.runtime.writing():
        pass

    harness.ensure()
    with harness.runtime.writing() as store:
        assert store is harness.store
    assert harness.log == RECOVERY + ["is_held"]  # each write confirms the lock first


def test_a_write_after_losing_the_lock_is_refused_and_reported() -> None:
    harness = Harness()
    harness.ensure()
    harness.lock.held = False

    with pytest.raises(HistoryUnavailableError), harness.runtime.writing():
        pytest.fail("a write must not start without the lock")

    assert harness.runtime.status == unavailable("lock_lost")
    assert harness.runtime._active_writes == 0


def test_close_waits_for_a_write_in_progress_before_releasing_the_lock() -> None:
    harness = Harness()
    harness.ensure()
    writing = threading.Event()
    finish_write = threading.Event()

    def write() -> None:
        with harness.runtime.writing():
            writing.set()
            assert finish_write.wait(TIMEOUT)

    writer = threading.Thread(target=write)
    writer.start()
    assert writing.wait(TIMEOUT)
    closer = threading.Thread(target=harness.runtime.close)
    closer.start()

    closer.join(0.2)  # bounded: close must still be waiting for the write
    assert closer.is_alive()
    assert "release" not in harness.log and "dispose" not in harness.log
    with pytest.raises(HistoryUnavailableError), harness.runtime.writing():
        pass  # no new write may start once closing has begun

    finish_write.set()
    writer.join(TIMEOUT)
    closer.join(TIMEOUT)
    assert not closer.is_alive()
    assert harness.log[-2:] == ["release", "dispose"]
    assert harness.runtime.status == unavailable("closed")


def test_close_gives_up_on_a_stuck_write_after_the_drain_time(caplog) -> None:
    caplog.set_level(logging.WARNING)
    harness = Harness()
    harness.ensure()
    harness.runtime._drain_seconds = 0.05
    stuck = harness.runtime.writing()
    stuck.__enter__()

    harness.runtime.close()

    assert harness.log[-2:] == ["release", "dispose"]
    assert "write still in progress" in caplog.text
    stuck.__exit__(None, None, None)


def test_drain_seconds_must_not_be_negative() -> None:
    with pytest.raises(ValueError):
        HistoryRuntime(settings(), drain_seconds=-1)
