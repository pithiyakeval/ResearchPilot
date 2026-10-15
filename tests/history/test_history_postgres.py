"""PostgreSQL integration tests for the research history layer; opt-in only.

Skipped unless RESEARCHPILOT_TEST_DATABASE_URL names a dedicated, disposable database (a
postgresql+psycopg:// URL whose database name contains "test"). The application's own
RESEARCHPILOT_DATABASE_URL and .env are never used for testing, and a test URL that points at
the application's database is refused. The database must use UTF8 encoding.

Every run creates one randomly named schema, migrates it, and drops only that schema at the
end, even after a failure. Advisory-lock tests use their own random key, so a running
instance's lock is never touched.
"""

import logging
import os
import random
import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from itertools import product
from pathlib import Path

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from fakes import ScriptedModel, complete
from sqlalchemy import Engine, event, insert, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError, DataError, IntegrityError

from researchpilot.config import Settings
from researchpilot.history.database import InstanceLock, create_history_engine, session_factory
from researchpilot.history.models import Base, ResearchJobRow
from researchpilot.history.records import (
    JOB_STATES,
    STATE_FIELDS,
    FinishedJob,
    QueuedJob,
    StartedJob,
    fields_allowed,
    label_is_valid,
)
from researchpilot.history.settings import DatabaseSettings
from researchpilot.history.store import (
    INTERRUPTED_BY_RESTART,
    HistoryConflictError,
    HistoryLockError,
    HistorySchemaError,
    HistoryStore,
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
SOURCE_SECRET = "SOURCE-SENTINEL-NEVER-STORED"
LABEL_SECRET = "LABEL-SENTINEL"
T0 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
TIMEOUT = 10.0
# PostgreSQL rejects a value longer than its VARCHAR column with "string data right
# truncation" before any check constraint runs.
STRING_DATA_RIGHT_TRUNCATION = "22001"


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


def engine_for(schema: str | None = None) -> Engine:
    settings = DatabaseSettings(_env_file=None, url=disposable_url())  # type: ignore[call-arg]
    engine = create_history_engine(settings)
    if schema is not None:

        @event.listens_for(engine, "connect")
        def use_schema(dbapi_connection, _record) -> None:
            with dbapi_connection.cursor() as cursor:
                cursor.execute(f'SET search_path TO "{schema}"')
            dbapi_connection.commit()

    return engine


def alembic_config() -> Config:
    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS))
    config.set_main_option("path_separator", "os")
    return config


def migrate(engine: Engine, revision: str, *, down: bool = False) -> None:
    config = alembic_config()
    with engine.connect() as connection:
        config.attributes["connection"] = connection
        (command.downgrade if down else command.upgrade)(config, revision)
        connection.commit()


@dataclass
class Database:
    admin: Engine
    engine: Engine
    schema: str

    def store(self, *, clock=lambda: T0) -> HistoryStore:
        sessions = session_factory(self.engine)
        return HistoryStore(sessions, instance_id=uuid.uuid4().hex, clock=clock)


def temporary_schema(admin: Engine) -> str:
    schema = f"rp_test_{uuid.uuid4().hex[:12]}"
    with admin.begin() as connection:
        connection.execute(text(f'CREATE SCHEMA "{schema}"'))
    return schema


def drop_schema(admin: Engine, schema: str) -> None:
    assert schema.startswith("rp_test_")  # never drop anything this run did not create
    with admin.begin() as connection:
        connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))


@pytest.fixture(scope="module")
def database():
    admin = engine_for()
    schema = temporary_schema(admin)
    engine = engine_for(schema)
    try:
        migrate(engine, "head")
        yield Database(admin=admin, engine=engine, schema=schema)
    finally:
        engine.dispose()
        try:
            drop_schema(admin, schema)
        finally:
            admin.dispose()


@pytest.fixture
def db(database):
    with database.engine.begin() as connection:
        connection.execute(text("TRUNCATE research_jobs"))
    return database


@pytest.fixture(scope="module")
def result() -> ResearchResult:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    model = ScriptedModel(["notes", complete(), "Report [S1]."])
    outcome = run_research(
        prepare_request(
            "Compare System Alpha and System Beta",
            f"# Source A: Tasks\n\nSystem Alpha completed 85 of 100 tasks. {SOURCE_SECRET}",
        ),
        settings=settings,
        model_factory=lambda _settings, **_kwargs: model,
    )
    return build_result(outcome, settings)


def new_id() -> str:
    return uuid.uuid4().hex


def queued(store: HistoryStore, *, at: datetime = T0, label: str | None = None) -> str:
    job_id = new_id()
    store.record_queued(QueuedJob(job_id, "local", at, label=label))
    return job_id


def started(store: HistoryStore, *, at: datetime = T0) -> str:
    job_id = queued(store, at=at)
    store.record_started(StartedJob(job_id, at + timedelta(seconds=1)))
    return job_id


# --- Migrations ---


def test_migrations_round_trip_and_match_the_models(database) -> None:
    store = database.store()
    store.check_schema()

    migrate(database.engine, "base", down=True)
    with pytest.raises(HistorySchemaError):
        store.check_schema()
    migrate(database.engine, "head")
    store.check_schema()

    with database.engine.connect() as connection:
        context = MigrationContext.configure(connection, opts={"compare_type": True})
        assert compare_metadata(context, Base.metadata) == []


def test_an_unmigrated_schema_fails_the_schema_check(database) -> None:
    schema = temporary_schema(database.admin)
    engine = engine_for(schema)
    try:
        store = HistoryStore(session_factory(engine), instance_id=new_id())
        with pytest.raises(HistorySchemaError):
            store.check_schema()
    finally:
        engine.dispose()
        drop_schema(database.admin, schema)


# --- Constraints: PostgreSQL enforces exactly the Python rules ---

FIELD_VALUES = {
    "started_at": T0 + timedelta(seconds=1),
    "finished_at": T0 + timedelta(seconds=2),
    "result": {"stand-in": True},
    "failure_code": "model_failed",
    "cancel_reason": "client",
    "expires_at": T0 + timedelta(days=14),
}


def try_insert(engine: Engine, **values) -> bool:
    row = {
        "id": new_id(),
        "mode": "local",
        "state": "queued",
        "created_at": T0,
        "instance_id": new_id(),
        **values,
    }
    try:
        with engine.begin() as connection:
            connection.execute(insert(ResearchJobRow).values(**row))
    except IntegrityError:
        return False
    except DataError as exc:
        # Only an overlength value counts as a rejection; any other data error is a real
        # problem with the test and must fail it.
        if getattr(exc.orig, "sqlstate", None) == STRING_DATA_RIGHT_TRUNCATION:
            return False
        raise
    return True


def test_state_field_constraint_matches_the_python_rules(db) -> None:
    names = list(STATE_FIELDS["queued"])
    for state in JOB_STATES:
        for flags in product((False, True), repeat=len(names)):
            present = dict(zip(names, flags, strict=True))
            values = {name: FIELD_VALUES[name] for name in names if present[name]}
            if present["result"]:
                values["result_version"] = 1
            accepted = try_insert(db.engine, state=state, **values)
            assert accepted == fields_allowed(state, present), (state, present)


@pytest.mark.parametrize(
    "values",
    [
        {"id": "A" * 32},
        {"instance_id": "short"},
        {"label": " padded"},
        {"label": ""},
        {"label": "x" * 121},
        {"mode": "remote"},
        {"state": "paused"},
        {"state": "failed", "finished_at": T0, "failure_code": "Bad-Code", "expires_at": T0},
        {"state": "cancelled", "finished_at": T0, "cancel_reason": "later", "expires_at": T0},
        {"result_version": 1},
        {"state": "running", "started_at": T0 - timedelta(seconds=1)},
    ],
)
def test_other_constraints_reject_invalid_rows(db, values) -> None:
    assert try_insert(db.engine, **values) is False


def test_a_label_of_120_characters_is_accepted(db) -> None:
    assert try_insert(db.engine, label="x" * 120) is True


# The same cases as the offline label tests, except U+0000, which PostgreSQL text cannot hold.
LABEL_CASES = [
    "x",
    "x" * 120,
    "Café – naïve résumé 研究",
    "inner  double  spaces",
    "\u00a0no-break space edges\u00a0",
    "\u2028line separator inside",
    "",
    "x" * 121,
    " leading space",
    "trailing space ",
    "tab\tinside",
    "new\nline",
    "carriage\rreturn",
    "unit\x1fseparator",
    "delete\x7fchar",
    "next\x85line",
    "c1\x9fcontrol",
]


@pytest.mark.parametrize("label", LABEL_CASES)
def test_the_label_constraint_matches_the_python_rule(db, label) -> None:
    assert try_insert(db.engine, label=label) is label_is_valid(label)


def test_the_database_is_utf8(database) -> None:
    database.store().check_schema()  # raises HistoryEncodingError otherwise


def test_0002_refuses_to_apply_over_labels_that_break_the_new_rule(database) -> None:
    migrate(database.engine, "0001_research_jobs", down=True)
    try:
        assert try_insert(database.engine, label="tab\tinside") is True  # allowed by 0001
        with pytest.raises(IntegrityError):
            migrate(database.engine, "head")
        with database.engine.connect() as connection:
            revision = connection.execute(text("SELECT version_num FROM alembic_version"))
            assert list(revision.scalars()) == ["0001_research_jobs"]
    finally:
        with database.engine.begin() as connection:
            connection.execute(text("TRUNCATE research_jobs"))
        migrate(database.engine, "head")
    database.store().check_schema()


# --- Lifecycle ---


def test_a_job_moves_through_its_lifecycle_with_retention(db, result) -> None:
    store = db.store()
    job_id = queued(store, label="My research")
    assert store.get(job_id).state == "queued"  # type: ignore[union-attr]

    store.record_started(StartedJob(job_id, T0 + timedelta(seconds=1)))
    finished_at = T0 + timedelta(minutes=5)
    store.record_finished(FinishedJob(job_id, "succeeded", finished_at, result=result))

    record = store.get(job_id)
    assert record is not None
    assert (record.state, record.label, record.mode) == ("succeeded", "My research", "local")
    assert record.started_at == T0 + timedelta(seconds=1)
    assert record.finished_at == finished_at
    assert record.expires_at == finished_at + timedelta(days=30)
    assert record.result == result


def test_failed_and_cancelled_jobs_expire_after_14_days(db) -> None:
    store = db.store()
    failed = started(store)
    cancelled = queued(store)
    at = T0 + timedelta(minutes=1)

    store.record_finished(FinishedJob(failed, "failed", at, failure_code="model_failed"))
    store.record_finished(FinishedJob(cancelled, "cancelled", at, cancel_reason="client"))

    for job_id in (failed, cancelled):
        record = store.get(job_id)
        assert record is not None and record.expires_at == at + timedelta(days=14)
        assert record.result is None


def test_failed_and_cancelled_jobs_store_sql_null_results(db) -> None:
    """Regression: a finished job without a result stores SQL NULL, not the JSON value null."""
    store = db.store()
    failed = started(store)
    cancelled = queued(store)
    at = T0 + timedelta(minutes=1)

    store.record_finished(FinishedJob(failed, "failed", at, failure_code="model_failed"))
    store.record_finished(FinishedJob(cancelled, "cancelled", at, cancel_reason="client"))

    with db.engine.connect() as connection:
        rows = connection.execute(
            text(
                "SELECT id, state, result IS NULL, result_version IS NULL FROM research_jobs "
                "WHERE id IN (:failed, :cancelled)"
            ),
            {"failed": failed, "cancelled": cancelled},
        ).all()

    assert sorted(tuple(row) for row in rows) == sorted(
        [(failed, "failed", True, True), (cancelled, "cancelled", True, True)]
    )


def test_writes_that_skip_a_state_conflict(db, result, caplog) -> None:
    caplog.set_level(logging.DEBUG)
    store = db.store()
    job_id = queued(store, label=LABEL_SECRET)
    later = T0 + timedelta(minutes=1)

    with pytest.raises(HistoryConflictError):
        store.record_queued(QueuedJob(job_id, "local", T0))  # duplicate ID
    with pytest.raises(HistoryConflictError):
        store.record_finished(FinishedJob(job_id, "succeeded", later, result=result))
    store.record_started(StartedJob(job_id, T0 + timedelta(seconds=1)))
    with pytest.raises(HistoryConflictError):
        store.record_started(StartedJob(job_id, later))
    with pytest.raises(HistoryConflictError):
        store.record_finished(FinishedJob(job_id, "cancelled", later, cancel_reason="client"))
    with pytest.raises(HistoryConflictError) as raised:
        # Finishing before the job started violates a constraint whose details quote the row.
        store.record_finished(FinishedJob(job_id, "failed", T0, failure_code="model_failed"))

    assert raised.value.__cause__ is None
    assert LABEL_SECRET not in caplog.text
    assert store.get(job_id).state == "running"  # type: ignore[union-attr]


def test_only_queued_jobs_can_be_deleted(db) -> None:
    store = db.store()
    waiting = queued(store)
    running = started(store)

    store.delete_queued(waiting)

    assert store.get(waiting) is None
    with pytest.raises(HistoryConflictError):
        store.delete_queued(running)


def test_stored_results_never_contain_source_material(db, result) -> None:
    store = db.store()
    job_id = started(store)
    store.record_finished(
        FinishedJob(job_id, "succeeded", T0 + timedelta(minutes=1), result=result)
    )

    with db.engine.connect() as connection:
        rows = connection.execute(text("SELECT to_jsonb(r)::text FROM research_jobs r")).scalars()
        stored = "\n".join(rows)

    assert job_id in stored
    assert SOURCE_SECRET not in stored


def test_concurrent_finishes_of_different_jobs_all_succeed(db, result) -> None:
    store = db.store()
    job_ids = [started(store) for _ in range(4)]
    barrier = threading.Barrier(len(job_ids))
    errors: list[BaseException] = []

    def finish(job_id: str) -> None:
        try:
            barrier.wait(TIMEOUT)
            at = T0 + timedelta(minutes=1)
            store.record_finished(FinishedJob(job_id, "succeeded", at, result=result))
        except BaseException as exc:  # noqa: BLE001 - reported by the assertion below
            errors.append(exc)

    threads = [threading.Thread(target=finish, args=(job_id,)) for job_id in job_ids]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(TIMEOUT)

    assert errors == []
    assert {store.get(job_id).state for job_id in job_ids} == {"succeeded"}  # type: ignore[union-attr]


# --- Reads, retention and recovery ---


def test_expired_jobs_are_hidden_then_purged(db) -> None:
    store = db.store()
    job_id = started(store)
    store.record_finished(
        FinishedJob(job_id, "failed", T0 + timedelta(minutes=1), failure_code="model_failed")
    )
    active = queued(store)

    later = db.store(clock=lambda: T0 + timedelta(days=15))
    assert later.get(job_id) is None
    assert [summary.id for summary in later.list_recent()] == [active]
    assert later.purge_expired() == 1
    assert later.purge_expired() == 0
    assert later.get(active) is not None


def test_history_lists_newest_first_without_results(db) -> None:
    store = db.store()
    ids = [queued(store, at=T0 + timedelta(minutes=minute)) for minute in range(5)]

    recent = store.list_recent(limit=3)
    older = store.list_recent(limit=3, before=recent[-1].cursor)

    assert [summary.id for summary in recent] == ids[::-1][:3]
    assert [summary.id for summary in older] == ids[::-1][3:]
    assert not hasattr(recent[0], "result")


def test_paging_never_skips_or_repeats_jobs_created_at_the_same_instant(db) -> None:
    store = db.store()
    tied = [queued(store, at=T0) for _ in range(5)]
    older = queued(store, at=T0 - timedelta(seconds=1))

    seen: list[str] = []
    page = store.list_recent(limit=2)
    while page:
        seen += [summary.id for summary in page]
        page = store.list_recent(limit=2, before=page[-1].cursor)

    assert seen == sorted(tied, reverse=True) + [older]


def test_restart_recovery_needs_the_lock_and_fails_only_unfinished_jobs(db, result) -> None:
    store = db.store(clock=lambda: T0 + timedelta(hours=1))
    waiting = queued(store)
    running = started(store)
    done = started(store)
    store.record_finished(FinishedJob(done, "succeeded", T0 + timedelta(minutes=1), result=result))
    lock = InstanceLock(db.engine, key=random.randrange(1, 2**62))

    with pytest.raises(HistoryLockError):
        store.interrupt_unfinished(lock)
    assert lock.acquire() is True
    try:
        assert store.interrupt_unfinished(lock) == 2
    finally:
        lock.release()

    for job_id in (waiting, running):
        record = store.get(job_id)
        assert record is not None
        assert (record.state, record.failure_code) == ("failed", INTERRUPTED_BY_RESTART)
        assert record.finished_at == T0 + timedelta(hours=1)
        assert record.expires_at == T0 + timedelta(hours=1, days=14)
    assert store.get(waiting).started_at is None  # type: ignore[union-attr]
    assert store.get(done).state == "succeeded"  # type: ignore[union-attr]


# --- Advisory lock ---


def test_only_one_instance_holds_the_lock(database) -> None:
    key = random.randrange(1, 2**62)
    first, second = InstanceLock(database.engine, key=key), InstanceLock(database.engine, key=key)
    try:
        assert first.acquire() is True
        assert first.is_held() is True
        assert second.acquire() is False
        first.release()
        assert first.is_held() is False
        assert second.acquire() is True
    finally:
        first.release()
        second.release()


def test_a_lost_lock_connection_is_detected(database) -> None:
    key = random.randrange(1, 2**62)
    lock = InstanceLock(database.engine, key=key)
    try:
        assert lock.acquire() is True
        high, low = (key >> 32) & 0xFFFF_FFFF, key & 0xFFFF_FFFF
        with database.admin.begin() as connection:
            connection.execute(
                text(
                    "SELECT pg_terminate_backend(pid) FROM pg_locks WHERE locktype = 'advisory' "
                    "AND classid::bigint = :high AND objid::bigint = :low AND objsubid = 1"
                ),
                {"high": high, "low": low},
            )
        assert lock.is_held() is False
        successor = InstanceLock(database.engine, key=key)
        try:
            assert successor.acquire() is True
        finally:
            successor.release()
    finally:
        lock.release()
