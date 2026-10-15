"""End-to-end PostgreSQL tests: API -> HistoryRecorder -> HistoryStore -> PostgreSQL; opt-in.

Skipped unless RESEARCHPILOT_TEST_DATABASE_URL names a dedicated, disposable database, with
the same rules as tests/history/test_history_postgres.py: a postgresql+psycopg:// URL whose
database name contains "test", never the application's own database. Each run creates one
randomly named schema, migrates it and drops only that schema afterwards; every test uses
its own random advisory-lock key, so a running instance's lock is never touched. Research is
a double, so no model or MCP server is contacted.
"""

import os
import random
import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from fakes import ScriptedModel, complete
from fastapi.testclient import TestClient
from sqlalchemy import Engine, event, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError

from researchpilot.api import create_app
from researchpilot.api.history import HistoryRuntime, HistoryState, HistoryStatus
from researchpilot.api.jobs import JobInput, JobManager, JobState
from researchpilot.api.settings import ApiSettings
from researchpilot.config import Settings
from researchpilot.history.database import InstanceLock, create_history_engine, session_factory
from researchpilot.history.records import FinishedJob, QueuedJob, StartedJob
from researchpilot.history.settings import DatabaseSettings
from researchpilot.history.store import (
    INTERRUPTED_BY_RESTART,
    HistoryStore,
    HistoryUnavailableError,
)
from researchpilot.pipeline import prepare_request, run_research
from researchpilot.results import ResearchResult, build_result

TEST_DATABASE_URL = os.environ.get("RESEARCHPILOT_TEST_DATABASE_URL")
pytestmark = [
    pytest.mark.postgres,
    pytest.mark.skipif(
        not TEST_DATABASE_URL,
        reason="set RESEARCHPILOT_TEST_DATABASE_URL to a disposable database to run these",
    ),
]

MIGRATIONS = (
    Path(__file__).resolve().parents[2] / "src" / "researchpilot" / "history" / "migrations"
)
TIMEOUT = 10.0
JOBS = "/api/v1/jobs"
QUESTION = "Compare System Alpha and System Beta"
SOURCES = "# Source A: Tasks\n\nSystem Alpha completed 85 of 100 tasks in a fictional test."
SUBMISSION = {"question": QUESTION, "source_material": SOURCES}


# --- Disposable database, as in tests/history/test_history_postgres.py ---


def _location(url) -> tuple[str, int, str]:
    return ((url.host or "localhost").lower(), url.port or 5432, url.database or "")


def disposable_url() -> str:
    """The test URL, refused unless it is clearly a dedicated test database."""
    assert TEST_DATABASE_URL is not None
    try:
        url = make_url(TEST_DATABASE_URL)
    except ArgumentError:
        pytest.fail("RESEARCHPILOT_TEST_DATABASE_URL is not a valid database URL.")
    if url.drivername != "postgresql+psycopg":
        pytest.fail("RESEARCHPILOT_TEST_DATABASE_URL must use postgresql+psycopg://.")
    if "test" not in (url.database or "").lower():
        pytest.fail("RESEARCHPILOT_TEST_DATABASE_URL must name a database containing 'test'.")
    application = DatabaseSettings().url  # environment and .env, as the application reads them
    if application is not None:
        try:
            same = _location(make_url(application.get_secret_value())) == _location(url)
        except ArgumentError:
            same = False
        if same:
            pytest.fail("RESEARCHPILOT_TEST_DATABASE_URL must not be the application's database.")
    return TEST_DATABASE_URL


def database_settings() -> DatabaseSettings:
    return DatabaseSettings(_env_file=None, url=disposable_url())  # type: ignore[call-arg]


def engine_for(schema: str | None = None) -> Engine:
    engine = create_history_engine(database_settings())
    if schema is not None:

        @event.listens_for(engine, "connect")
        def use_schema(dbapi_connection, _record) -> None:
            with dbapi_connection.cursor() as cursor:
                cursor.execute(f'SET search_path TO "{schema}"')
            dbapi_connection.commit()

    return engine


def migrate_to_head(engine: Engine) -> None:
    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS))
    config.set_main_option("path_separator", "os")
    with engine.connect() as connection:
        config.attributes["connection"] = connection
        command.upgrade(config, "head")
        connection.commit()


@dataclass
class Database:
    admin: Engine
    engine: Engine
    schema: str

    def row(self, job_id: str) -> dict:
        with self.engine.connect() as connection:
            found = connection.execute(
                text(
                    "SELECT state, label, failure_code, cancel_reason, result IS NULL AS no_result,"
                    " result FROM research_jobs WHERE id = :id"
                ),
                {"id": job_id},
            ).mappings()
            return dict(found.one())

    def runtime(self, lock_key: int) -> HistoryRuntime:
        """A real HistoryRuntime on this schema, with its own advisory-lock key."""
        return HistoryRuntime(
            database_settings(),
            engine_factory=lambda _settings: engine_for(self.schema),
            lock_factory=lambda engine: InstanceLock(engine, key=lock_key),
            drain_seconds=5.0,
        )


@pytest.fixture(scope="module")
def database():
    admin = engine_for()
    schema = f"rp_test_{uuid.uuid4().hex[:12]}"
    with admin.begin() as connection:
        connection.execute(text(f'CREATE SCHEMA "{schema}"'))
    engine = engine_for(schema)
    try:
        migrate_to_head(engine)
        yield Database(admin=admin, engine=engine, schema=schema)
    finally:
        engine.dispose()
        try:
            assert schema.startswith("rp_test_")  # never drop anything this run did not create
            with admin.begin() as connection:
                connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        finally:
            admin.dispose()


@pytest.fixture
def db(database):
    with database.engine.begin() as connection:
        connection.execute(text("TRUNCATE research_jobs"))
    return database


@pytest.fixture
def lock_key() -> int:
    return random.randrange(1, 2**62)


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


class Research:
    """Research double; with ``block``, each call waits until ``release`` is set."""

    def __init__(self, *, error: Exception | None = None, block: bool = False) -> None:
        self.error = error
        self.started = threading.Event()
        self.release = threading.Event()
        if not block:
            self.release.set()
        self.calls = 0

    def __call__(self, job_input: JobInput) -> object:
        self.calls += 1
        self.started.set()
        assert self.release.wait(TIMEOUT), "test never released the job"
        if self.error is not None:
            raise self.error
        return object()


def app_for(manager: JobManager, history: HistoryRuntime, **api_overrides):
    return create_app(
        settings=Settings(_env_file=None),  # type: ignore[call-arg]
        api_settings=ApiSettings(_env_file=None, **api_overrides),  # type: ignore[call-arg]
        ollama_probe=lambda: True,
        job_manager=manager,
        history=history,
    )


# --- Recorded outcomes ---


def test_a_succeeded_job_is_recorded_with_its_label_and_result(db, lock_key, result) -> None:
    manager = JobManager(Research(), lambda _outcome: result)

    with TestClient(app_for(manager, db.runtime(lock_key))) as client:
        job_id = client.post(JOBS, json={**SUBMISSION, "label": "  Report A "}).json()["id"]
        assert manager.wait(job_id, TIMEOUT).state is JobState.SUCCEEDED
        status = client.get(f"{JOBS}/{job_id}")
        fetched = client.get(f"{JOBS}/{job_id}/result")

    row = db.row(job_id)
    assert (row["state"], row["label"], row["no_result"]) == ("succeeded", "Report A", False)
    assert ResearchResult.model_validate(row["result"]) == result
    assert status.json()["state"] == "succeeded"
    assert fetched.json() == result.model_dump(mode="json")


def test_failed_and_cancelled_jobs_are_recorded_without_results(db, lock_key, result) -> None:
    research = Research(error=RuntimeError("model broke"), block=True)
    manager = JobManager(research, lambda _outcome: result)

    with TestClient(app_for(manager, db.runtime(lock_key))) as client:
        failed = client.post(JOBS, json=SUBMISSION).json()["id"]
        assert research.started.wait(TIMEOUT)
        cancelled = client.post(JOBS, json=SUBMISSION).json()["id"]
        assert client.post(f"{JOBS}/{cancelled}/cancel").status_code == 200
        research.release.set()
        manager.wait(failed, TIMEOUT)

    failed_row, cancelled_row = db.row(failed), db.row(cancelled)
    assert (failed_row["state"], failed_row["failure_code"], failed_row["no_result"]) == (
        "failed",
        "internal_error",
        True,
    )
    assert (cancelled_row["state"], cancelled_row["cancel_reason"], cancelled_row["no_result"]) == (
        "cancelled",
        "client",
        True,
    )
    assert research.calls == 1


# --- Failed writes are never served ---


def test_a_failed_finished_write_is_never_served(db, lock_key, result, monkeypatch) -> None:
    def refuse(_store, _job):
        raise HistoryUnavailableError

    monkeypatch.setattr(HistoryStore, "record_finished", refuse)
    manager = JobManager(Research(), lambda _outcome: result)

    with TestClient(app_for(manager, db.runtime(lock_key))) as client:
        job_id = client.post(JOBS, json=SUBMISSION).json()["id"]
        assert manager.wait(job_id, TIMEOUT).unrecorded
        status = client.get(f"{JOBS}/{job_id}")
        fetched = client.get(f"{JOBS}/{job_id}/result")

    for response in (status, fetched):
        assert response.status_code == 503 and response.json()["code"] == "history_unavailable"
    assert db.row(job_id)["state"] == "running"  # no outcome is claimed in the database


def test_a_job_finishing_after_shutdown_is_not_written_and_is_recovered_later(
    db, lock_key, result
) -> None:
    research = Research(block=True)
    manager = JobManager(research, lambda _outcome: result)
    first = db.runtime(lock_key)

    with TestClient(app_for(manager, first, shutdown_timeout_seconds=0.1)) as client:
        job_id = client.post(JOBS, json=SUBMISSION).json()["id"]
        assert research.started.wait(TIMEOUT)
    # The app has shut down: history is closed and the lock released while the job runs on.
    research.release.set()
    assert manager.shutdown(TIMEOUT) is True
    assert manager.get(job_id).unrecorded
    assert db.row(job_id)["state"] == "running"  # the late outcome was not written

    successor = db.runtime(lock_key)  # the next instance takes the lock and recovers
    try:
        assert successor.ensure_ready().ready
    finally:
        successor.close()
    row = db.row(job_id)
    assert (row["state"], row["failure_code"]) == ("failed", INTERRUPTED_BY_RESTART)


# --- Recovery and single ownership ---


def test_startup_recovery_interrupts_jobs_left_by_a_previous_instance(db, lock_key) -> None:
    previous = HistoryStore(session_factory(db.engine), instance_id=uuid.uuid4().hex)
    now = datetime.now(UTC)
    waiting, running = uuid.uuid4().hex, uuid.uuid4().hex
    previous.record_queued(QueuedJob(waiting, "local", now))
    previous.record_queued(QueuedJob(running, "web", now))
    previous.record_started(StartedJob(running, now))
    manager = JobManager(Research(), lambda _outcome: object())  # type: ignore[arg-type]

    with TestClient(app_for(manager, db.runtime(lock_key))) as client:
        assert client.get("/api/v1/health/ready").status_code == 200
        assert manager.started

    for job_id in (waiting, running):
        row = db.row(job_id)
        assert (row["state"], row["failure_code"]) == ("failed", INTERRUPTED_BY_RESTART)


def test_a_second_instance_cannot_take_over_while_the_first_holds_the_lock(db, lock_key) -> None:
    manager = JobManager(Research(), lambda _outcome: object())  # type: ignore[arg-type]
    second = db.runtime(lock_key)
    try:
        with TestClient(app_for(manager, db.runtime(lock_key))):
            assert second.ensure_ready() == HistoryStatus(
                HistoryState.UNAVAILABLE, "lock_held_elsewhere"
            )
    finally:
        second.close()


# --- Reads after a restart (Phase 4B.5) ---


def test_a_completed_job_and_its_result_are_read_from_history_after_a_restart(
    db, lock_key, result
) -> None:
    before = JobManager(Research(), lambda _outcome: result)
    with TestClient(app_for(before, db.runtime(lock_key))) as client:
        job_id = client.post(JOBS, json=SUBMISSION).json()["id"]
        assert before.wait(job_id, TIMEOUT).state is JobState.SUCCEEDED

    after = JobManager(Research(), lambda _outcome: result)  # a new process: memory is empty
    with TestClient(app_for(after, db.runtime(lock_key))) as client:
        status = client.get(f"{JOBS}/{job_id}")
        fetched = client.get(f"{JOBS}/{job_id}/result")

    assert status.status_code == 200
    assert (status.json()["state"], status.json()["queue_position"]) == ("succeeded", None)
    assert fetched.status_code == 200 and fetched.json() == result.model_dump(mode="json")


def test_a_job_interrupted_by_a_restart_is_read_as_failed_without_a_result(
    db, lock_key, result
) -> None:
    research = Research(block=True)
    before = JobManager(research, lambda _outcome: result)
    with TestClient(app_for(before, db.runtime(lock_key), shutdown_timeout_seconds=0.1)) as client:
        job_id = client.post(JOBS, json=SUBMISSION).json()["id"]
        assert research.started.wait(TIMEOUT)
    research.release.set()
    before.shutdown(TIMEOUT)

    after = JobManager(Research(), lambda _outcome: result)
    with TestClient(app_for(after, db.runtime(lock_key))) as client:  # recovery runs at startup
        status = client.get(f"{JOBS}/{job_id}").json()
        no_result = client.get(f"{JOBS}/{job_id}/result")

    assert status["state"] == "failed"
    assert status["failure"]["code"] == INTERRUPTED_BY_RESTART
    assert no_result.status_code == 409 and no_result.json()["code"] == "job_failed"


def test_an_expired_job_is_not_read_from_history(db, lock_key) -> None:
    manager = JobManager(Research(), lambda _outcome: object())  # type: ignore[arg-type]
    with TestClient(app_for(manager, db.runtime(lock_key))) as client:
        # Written after startup, so the startup purge has not removed it: only hidden.
        long_ago = datetime(2020, 1, 1, tzinfo=UTC)
        old = HistoryStore(
            session_factory(db.engine), instance_id=uuid.uuid4().hex, clock=lambda: long_ago
        )
        job_id = uuid.uuid4().hex
        old.record_queued(QueuedJob(job_id, "local", long_ago))
        old.record_finished(FinishedJob(job_id, "cancelled", long_ago, cancel_reason="client"))

        response = client.get(f"{JOBS}/{job_id}")

    assert db.row(job_id)["state"] == "cancelled"  # still stored, but past its retention
    assert response.status_code == 404 and response.json()["code"] == "job_not_found"
