"""Offline tests for the research history layer.

No database is contacted and no socket is opened: engines are created lazily, and wherever a
connection would be made, ``psycopg.connect`` or the engine's ``creator`` is a double that
raises. PostgreSQL behaviour itself is covered by the opt-in tests in
``test_history_postgres.py``.
"""

import contextlib
import importlib.util
import io
import logging
import os
import re
import subprocess
import sys
import traceback
import uuid
from datetime import UTC, datetime, timedelta
from itertools import product
from pathlib import Path

import psycopg
import pytest
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from fakes import ScriptedModel, complete
from pydantic import ValidationError
from sqlalchemy import CheckConstraint
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import IntegrityError, OperationalError, ProgrammingError
from sqlalchemy.schema import CreateIndex, CreateTable

from researchpilot.config import Settings
from researchpilot.history import EXPECTED_SCHEMA_REVISION
from researchpilot.history.database import (
    INSTANCE_LOCK_KEY,
    InstanceLock,
    create_history_engine,
    session_factory,
)
from researchpilot.history.models import ResearchJobRow, label_check, state_fields_check
from researchpilot.history.records import (
    FINISHED_STATES,
    STATE_FIELDS,
    FinishedJob,
    HistoryCursor,
    HistorySummary,
    QueuedJob,
    StartedJob,
    fields_allowed,
    label_is_valid,
)
from researchpilot.history.settings import DatabaseSettings
from researchpilot.history.store import (
    HistoryConfigError,
    HistoryEncodingError,
    HistoryLockError,
    HistorySchemaError,
    HistoryStore,
    HistoryUnavailableError,
    _recent_statement,
    describe_error,
)
from researchpilot.pipeline import prepare_request, run_research
from researchpilot.results import ResearchResult, build_result

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS = PROJECT_ROOT / "src" / "researchpilot" / "history" / "migrations"
SECRET = "hunter2-SECRET"
URL = f"postgresql+psycopg://rp:{SECRET}@db.invalid:5432/researchpilot"
NOW = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture(autouse=True)
def no_database_environment(monkeypatch):
    """Local database settings must never leak into these tests."""
    for name in list(os.environ):
        if name.startswith("RESEARCHPILOT_DATABASE_"):
            monkeypatch.delenv(name)


def db_settings(**overrides) -> DatabaseSettings:
    return DatabaseSettings(_env_file=None, **overrides)  # type: ignore[call-arg]


def job_id() -> str:
    return uuid.uuid4().hex


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


HOST_DETAILS = 'connection to server at "db.invalid" (10.0.0.5), port 5432 failed'


def refuse_connections(*_args, **_kwargs):
    # psycopg's real message names the host, address and user.
    raise psycopg.errors.ConnectionFailure(f'{HOST_DETAILS}: user "rp_admin" password {SECRET}')


def assert_nothing_leaks(error: BaseException, logs: str) -> None:
    """Neither the logs nor a fully formatted traceback reveal connection details."""
    formatted = "".join(traceback.format_exception(error))
    assert error.__cause__ is None and error.__suppress_context__ is True
    for leaked in (SECRET, "db.invalid", "10.0.0.5", "rp_admin"):
        assert leaked not in formatted
        assert leaked not in logs


@pytest.fixture
def unavailable_store() -> HistoryStore:
    engine = create_history_engine(db_settings(url=URL), creator=refuse_connections)
    return HistoryStore(session_factory(engine), instance_id=job_id(), clock=lambda: NOW)


def alembic_config(output: io.StringIO | None = None) -> Config:
    config = Config(output_buffer=output)
    config.set_main_option("script_location", str(MIGRATIONS))
    config.set_main_option("path_separator", "os")
    return config


def statements(sql: str) -> list[str]:
    """SQL split into statements, without comment lines and with whitespace collapsed."""
    code = "\n".join(line for line in sql.splitlines() if not line.lstrip().startswith("--"))
    return [" ".join(part.split()) for part in re.split(r";\s*\n", code) if part.strip()]


def normalized(sql: object) -> str:
    return " ".join(str(sql).split())


# --- Settings ---


def test_database_settings_defaults_bounds_and_environment(monkeypatch) -> None:
    defaults = db_settings()
    assert defaults.url is None
    assert (
        defaults.connect_timeout_seconds,
        defaults.statement_timeout_ms,
        defaults.lock_timeout_ms,
        defaults.pool_size,
        defaults.max_overflow,
        defaults.pool_timeout_seconds,
    ) == (3, 5_000, 2_000, 5, 5, 5.0)

    monkeypatch.setenv("RESEARCHPILOT_DATABASE_URL", URL)
    monkeypatch.setenv("RESEARCHPILOT_DATABASE_POOL_SIZE", "7")
    configured = db_settings()
    assert configured.url is not None and configured.url.get_secret_value() == URL
    assert configured.pool_size == 7
    assert SECRET not in repr(configured) and SECRET not in str(configured)

    for bad in (
        {"connect_timeout_seconds": 1},
        {"statement_timeout_ms": 99},
        {"lock_timeout_ms": 60_001},
        {"pool_size": 0},
        {"max_overflow": 21},
        {"pool_timeout_seconds": 0},
    ):
        with pytest.raises(ValidationError):
            db_settings(**bad)


# --- Engine ---


@pytest.mark.parametrize(
    "url",
    [
        None,
        f"postgresql://rp:{SECRET}@db.invalid/researchpilot",
        f"postgresql+psycopg2://rp:{SECRET}@db.invalid/researchpilot",
        f"sqlite:///{SECRET}.db",
        f"not a url {SECRET}",
    ],
)
def test_engines_need_a_postgresql_psycopg_url_and_never_echo_it(url) -> None:
    with pytest.raises(HistoryConfigError) as raised:
        create_history_engine(db_settings(url=url))

    assert str(raised.value) == HistoryConfigError.message
    assert raised.value.__cause__ is None
    assert SECRET not in str(raised.value)


def test_an_engine_is_lazy_hides_parameters_and_masks_its_password() -> None:
    engine = create_history_engine(db_settings(url=URL, pool_size=3, max_overflow=2))

    assert (engine.dialect.name, engine.dialect.driver) == ("postgresql", "psycopg")
    assert engine.hide_parameters is True
    assert engine.pool.size() == 3
    assert SECRET not in repr(engine)
    engine.dispose()


def test_connections_use_the_configured_timeouts(monkeypatch) -> None:
    attempts: list[dict] = []

    def fake_connect(*_args, **kwargs):
        attempts.append(kwargs)
        raise psycopg.OperationalError("refused")

    monkeypatch.setattr(psycopg, "connect", fake_connect)
    settings = db_settings(
        url=URL, connect_timeout_seconds=4, statement_timeout_ms=750, lock_timeout_ms=300
    )
    engine = create_history_engine(settings)

    with pytest.raises(Exception, match="refused"):
        engine.connect()

    assert len(attempts) == 1
    assert attempts[0]["connect_timeout"] == 4
    assert attempts[0]["application_name"] == "researchpilot"
    assert attempts[0]["options"] == "-c statement_timeout=750 -c lock_timeout=300"


# --- Store failures without a database ---


def store_operations(store: HistoryStore, result: ResearchResult):
    queued_id = job_id()
    return {
        "check_schema": store.check_schema,
        "record_queued": lambda: store.record_queued(QueuedJob(queued_id, "local", NOW)),
        "record_started": lambda: store.record_started(StartedJob(queued_id, NOW)),
        "record_finished": lambda: store.record_finished(
            FinishedJob(queued_id, "succeeded", NOW, result=result)
        ),
        "delete_queued": lambda: store.delete_queued(queued_id),
        "get": lambda: store.get(queued_id),
        "list_recent": store.list_recent,
        "purge_expired": store.purge_expired,
    }


def test_every_store_operation_fails_closed_without_leaking_details(
    unavailable_store, result, caplog
) -> None:
    caplog.set_level(logging.DEBUG)

    for name, operation in store_operations(unavailable_store, result).items():
        with pytest.raises(HistoryUnavailableError) as raised:
            operation()
        assert str(raised.value) == HistoryUnavailableError.message, name
        assert_nothing_leaks(raised.value, caplog.text)

    assert "OperationalError / ConnectionFailure / SQLSTATE 08006" in caplog.text


def test_the_lock_fails_closed_without_a_database(unavailable_store, caplog) -> None:
    caplog.set_level(logging.DEBUG)
    engine = create_history_engine(db_settings(url=URL), creator=refuse_connections)
    lock = InstanceLock(engine)

    with pytest.raises(HistoryUnavailableError) as raised:
        lock.acquire()

    assert lock.is_held() is False
    lock.release()  # nothing to release
    with pytest.raises(HistoryLockError):
        unavailable_store.interrupt_unfinished(lock)
    assert_nothing_leaks(raised.value, caplog.text)


def test_errors_are_described_by_type_and_sqlstate_only() -> None:
    violation = psycopg.errors.CheckViolation(f"Failing row contains ({SECRET}, ...)")
    wrapped = IntegrityError("INSERT INTO research_jobs VALUES (...)", {"label": SECRET}, violation)
    connection = OperationalError("SELECT 1", None, psycopg.OperationalError(HOST_DETAILS))

    assert describe_error(wrapped) == "IntegrityError / CheckViolation / SQLSTATE 23514"
    assert describe_error(connection) == "OperationalError / OperationalError"
    assert describe_error(RuntimeError(SECRET)) == "RuntimeError"


# --- Pagination ---


def compiled(statement) -> str:
    return normalized(statement.compile(dialect=postgresql.dialect()))


def test_history_pages_use_a_keyset_on_created_at_and_id() -> None:
    first = compiled(_recent_statement(NOW, 10, None))
    after = compiled(_recent_statement(NOW, 10, HistoryCursor(NOW, job_id())))

    order = "ORDER BY research_jobs.created_at DESC, research_jobs.id DESC LIMIT"
    assert order in first and order in after
    assert "(research_jobs.created_at, research_jobs.id) <" not in first
    assert re.search(
        r"\(research_jobs\.created_at, research_jobs\.id\) < \(%\(\w+\)s, %\(\w+\)s\)", after
    )


def test_cursors_validate_and_come_from_summaries() -> None:
    summary_id = job_id()
    summary = HistorySummary(summary_id, None, "local", "queued", NOW, None, None, None, None, None)
    assert summary.cursor == HistoryCursor(NOW, summary_id)
    for created_at, cursor_id in ((datetime(2026, 1, 1), summary_id), (NOW, "not-hex")):
        with pytest.raises(ValueError):
            HistoryCursor(created_at, cursor_id)


def test_lock_keys_must_be_signed_64_bit() -> None:
    engine = create_history_engine(db_settings(url=URL))
    assert -(2**63) <= INSTANCE_LOCK_KEY < 2**63
    for key in (2**63, -(2**63) - 1):
        with pytest.raises(ValueError):
            InstanceLock(engine, key=key)


def test_reads_of_impossible_ids_never_reach_the_database(unavailable_store) -> None:
    for bad in ("missing", "A" * 32, "0" * 33, f"{SECRET}"):
        assert unavailable_store.get(bad) is None


def test_store_arguments_are_validated_before_any_query(unavailable_store) -> None:
    sessions = unavailable_store._sessions
    with pytest.raises(ValueError):
        HistoryStore(sessions, instance_id="not-hex")
    with pytest.raises(ValueError):
        HistoryStore(sessions, instance_id=job_id(), failure_retention=timedelta(0))

    naive = HistoryStore(sessions, instance_id=job_id(), clock=lambda: datetime(2026, 1, 1))
    with pytest.raises(ValueError):
        naive.record_queued(QueuedJob(job_id(), "local", NOW))
    with pytest.raises(ValueError):
        unavailable_store.list_recent(limit=0)
    with pytest.raises(ValueError):
        unavailable_store.list_recent(limit=201)
    with pytest.raises(ValueError):
        unavailable_store.list_recent(before=NOW)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        unavailable_store.delete_queued("not-hex")


class ScriptedSession:
    """Answers check_schema's two queries; never touches a database."""

    def __init__(self, encoding: str, revisions: list[str] | Exception) -> None:
        self.encoding = encoding
        self.revisions = revisions
        self.queries: list[str] = []

    def execute(self, statement):
        sql = str(statement)
        self.queries.append(sql)
        if sql == "SHOW server_encoding":
            return ScriptedResult([self.encoding])
        if isinstance(self.revisions, Exception):
            raise self.revisions
        return ScriptedResult(self.revisions)


class ScriptedResult:
    def __init__(self, values: list[str]) -> None:
        self.values = values

    def scalar_one(self) -> str:
        (value,) = self.values
        return value

    def scalars(self):
        return iter(self.values)


class ScriptedSessions:
    def __init__(self, session: ScriptedSession) -> None:
        self.session = session

    def begin(self):
        return contextlib.nullcontext(self.session)


def scripted_store(session: ScriptedSession) -> HistoryStore:
    return HistoryStore(ScriptedSessions(session), instance_id=job_id())  # type: ignore[arg-type]


def test_the_schema_check_passes_for_utf8_at_the_expected_revision() -> None:
    session = ScriptedSession("UTF8", [EXPECTED_SCHEMA_REVISION])

    scripted_store(session).check_schema()

    assert session.queries == ["SHOW server_encoding", "SELECT version_num FROM alembic_version"]


@pytest.mark.parametrize("encoding", ["SQL_ASCII", "LATIN1", "WIN1252", "utf8"])
def test_the_schema_check_requires_utf8(encoding, caplog) -> None:
    caplog.set_level(logging.DEBUG)
    session = ScriptedSession(encoding, [EXPECTED_SCHEMA_REVISION])

    with pytest.raises(HistoryEncodingError) as raised:
        scripted_store(session).check_schema()

    assert isinstance(raised.value, HistorySchemaError)  # fails closed like a schema problem
    assert session.queries == ["SHOW server_encoding"]
    assert "encoding is not UTF8" in caplog.text


@pytest.mark.parametrize(
    "revisions",
    [[], ["0001_research_jobs"], ["9999_future"], [EXPECTED_SCHEMA_REVISION, "0001_research_jobs"]],
)
def test_the_schema_check_requires_exactly_the_expected_revision(revisions) -> None:
    with pytest.raises(HistorySchemaError) as raised:
        scripted_store(ScriptedSession("UTF8", revisions)).check_schema()

    assert not isinstance(raised.value, HistoryEncodingError)


def test_the_schema_check_fails_without_a_migration_table() -> None:
    missing = ProgrammingError("SELECT version_num FROM alembic_version", None, Exception())

    with pytest.raises(HistorySchemaError):
        scripted_store(ScriptedSession("UTF8", missing)).check_schema()


# --- Records ---


def test_queued_and_started_jobs_validate_without_echoing_values() -> None:
    QueuedJob(job_id(), "web", NOW, label="x" * 120)
    QueuedJob(job_id(), "local", NOW, label=None)
    StartedJob(job_id(), NOW)

    invalid = [
        lambda: QueuedJob("A" * 32, "local", NOW),
        lambda: QueuedJob(job_id(), "remote", NOW),  # type: ignore[arg-type]
        lambda: QueuedJob(job_id(), "local", datetime(2026, 1, 1)),
        lambda: QueuedJob(job_id(), "local", NOW, label=""),
        lambda: QueuedJob(job_id(), "local", NOW, label=f" {SECRET}"),
        lambda: QueuedJob(job_id(), "local", NOW, label=f"{SECRET}\n"),
        lambda: QueuedJob(job_id(), "local", NOW, label=SECRET * 20),
        lambda: StartedJob("", NOW),
        lambda: StartedJob(job_id(), datetime(2026, 1, 1)),
    ]
    for build in invalid:
        with pytest.raises(ValueError) as raised:
            build()
        assert SECRET not in str(raised.value)


VALID_LABELS = [
    "x",
    "x" * 120,
    "Alpha vs Beta: 2026 review",
    "Café – naïve résumé 研究",
    "inner  double  spaces",
    "\u00a0no-break space edges\u00a0",  # not ordinary spaces; the API strips these first
    "\u2028line separator inside",
]
INVALID_LABELS = [
    "",
    "x" * 121,
    " leading space",
    "trailing space ",
    "tab\tinside",
    "new\nline",
    "carriage\rreturn",
    "nul\x00inside",
    "unit\x1fseparator",
    "delete\x7fchar",
    "next\x85line",
    "c1\x9fcontrol",
]


@pytest.mark.parametrize("label", VALID_LABELS)
def test_valid_labels_follow_the_stored_label_rule(label) -> None:
    assert label_is_valid(label)
    QueuedJob(job_id(), "local", NOW, label=label)


@pytest.mark.parametrize("label", INVALID_LABELS)
def test_invalid_labels_are_rejected_without_echoing_them(label) -> None:
    assert not label_is_valid(label)
    with pytest.raises(ValueError) as raised:
        QueuedJob(job_id(), "local", NOW, label=label)
    assert str(raised.value) == (
        "A label must be 1-120 characters, without control characters "
        "or leading or trailing spaces."
    )


def test_the_label_rule_is_the_same_in_python_model_and_migration() -> None:
    expected = (
        "label IS NULL OR (char_length(label) BETWEEN 1 AND 120 "
        "AND label !~ '[\\x01-\\x1f\\x7f-\\x9f]' AND label = btrim(label, ' '))"
    )
    assert label_check() == expected
    assert load_migration("0002_label_format").LABEL_CHECK == expected
    checks = {
        c.name: str(c.sqltext)
        for c in ResearchJobRow.__table__.constraints
        if isinstance(c, CheckConstraint)
    }
    assert checks["ck_research_jobs_label_format"] == expected
    assert "ck_research_jobs_label_trimmed" not in checks


def test_finished_jobs_accept_exactly_the_fields_their_state_allows(result) -> None:
    for state, has_result, has_failure, has_cancel in product(
        FINISHED_STATES, *[(False, True)] * 3
    ):
        rules = STATE_FIELDS[state]
        present = {
            "started_at": rules["started_at"] is not False,
            "finished_at": True,
            "result": has_result,
            "failure_code": has_failure,
            "cancel_reason": has_cancel,
            "expires_at": True,
        }
        expected = fields_allowed(state, present)

        def build(
            state=state, has_result=has_result, has_failure=has_failure, has_cancel=has_cancel
        ):
            return FinishedJob(
                job_id(),
                state,
                NOW,
                result=result if has_result else None,
                failure_code="model_failed" if has_failure else None,
                cancel_reason="client" if has_cancel else None,
            )

        if expected:
            build()
        else:
            with pytest.raises(ValueError):
                build()


def test_finished_jobs_validate_their_values(result) -> None:
    invalid = [
        lambda: FinishedJob(job_id(), "running", NOW),  # type: ignore[arg-type]
        lambda: FinishedJob(job_id(), "failed", NOW, failure_code="Model-Failed"),
        lambda: FinishedJob(job_id(), "failed", NOW, failure_code="x" * 65),
        lambda: FinishedJob(job_id(), "cancelled", NOW, cancel_reason="later"),  # type: ignore[arg-type]
        lambda: FinishedJob(job_id(), "succeeded", NOW, result={"question": "x"}),  # type: ignore[arg-type]
        lambda: FinishedJob(job_id(), "succeeded", datetime(2026, 1, 1), result=result),
    ]
    for build in invalid:
        with pytest.raises(ValueError):
            build()


# --- Schema and migrations ---


def load_migration(name: str = "0001_research_jobs"):
    path = MIGRATIONS / "versions" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"migration_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_state_fields_constraint_is_the_python_rule_in_both_model_and_migration() -> None:
    generated = normalized(state_fields_check())

    assert normalized(load_migration().STATE_FIELDS_CHECK) == generated
    checks = {
        c.name: normalized(c.sqltext)
        for c in ResearchJobRow.__table__.constraints
        if isinstance(c, CheckConstraint)
    }
    assert checks["ck_research_jobs_state_fields"] == generated


def test_the_result_column_stores_python_none_as_sql_null() -> None:
    # With JSONB's default none_as_null=False, None is written as the JSON value null, which
    # breaks the "result IS NULL" constraints when a job finishes as failed or cancelled.
    result_type = ResearchJobRow.__table__.c.result.type

    assert isinstance(result_type, postgresql.JSONB)
    assert result_type.none_as_null is True
    assert ResearchJobRow.__table__.c.result.nullable is True


def test_the_migration_creates_exactly_the_modelled_table_and_indexes() -> None:
    output = io.StringIO()
    command.upgrade(alembic_config(output), "head", sql=True)
    emitted = statements(output.getvalue())

    dialect = postgresql.dialect()
    table = ResearchJobRow.__table__
    # 0001 creates the table with its original label constraint; 0002 replaces that one.
    model_table = normalized(CreateTable(table).compile(dialect=dialect))
    label_format = f"CONSTRAINT ck_research_jobs_label_format CHECK ({label_check()})"
    label_trimmed = (
        "CONSTRAINT ck_research_jobs_label_trimmed "
        f"CHECK ({load_migration('0002_label_format').PREVIOUS_LABEL_CHECK})"
    )
    assert label_format in model_table
    assert model_table.replace(label_format, label_trimmed) in emitted
    assert "ALTER TABLE research_jobs DROP CONSTRAINT ck_research_jobs_label_trimmed" in emitted
    assert f"ALTER TABLE research_jobs ADD {label_format}" in emitted
    for index in table.indexes:
        assert normalized(CreateIndex(index).compile(dialect=dialect)) in emitted
    assert len(table.indexes) == 3
    assert any(s.startswith("CREATE TABLE alembic_version") for s in emitted)
    insert = "INSERT INTO alembic_version (version_num) VALUES ('0001_research_jobs')"
    assert any(s.startswith(insert) for s in emitted)
    assert (
        f"UPDATE alembic_version SET version_num='{EXPECTED_SCHEMA_REVISION}' "
        "WHERE alembic_version.version_num = '0001_research_jobs'"
    ) in emitted


def test_the_downgrade_drops_the_table_and_its_indexes() -> None:
    output = io.StringIO()
    command.downgrade(alembic_config(output), f"{EXPECTED_SCHEMA_REVISION}:base", sql=True)
    emitted = statements(output.getvalue())

    for index in ("created_at", "expires_at", "unfinished"):
        assert f"DROP INDEX ix_research_jobs_{index}" in emitted
    assert "DROP TABLE research_jobs" in emitted
    assert "ALTER TABLE research_jobs DROP CONSTRAINT ck_research_jobs_label_format" in emitted
    restored = load_migration("0002_label_format").PREVIOUS_LABEL_CHECK
    assert (
        "ALTER TABLE research_jobs ADD CONSTRAINT ck_research_jobs_label_trimmed "
        f"CHECK ({restored})"
    ) in emitted
    first = "0001_research_jobs"
    assert f"DELETE FROM alembic_version WHERE alembic_version.version_num = '{first}'" in emitted


def test_the_expected_revision_is_the_single_migration_head() -> None:
    from_ini = Config(str(PROJECT_ROOT / "alembic.ini"))
    for config in (from_ini, alembic_config()):
        script = ScriptDirectory.from_config(config)
        assert script.get_heads() == [EXPECTED_SCHEMA_REVISION]
        assert Path(script.dir).resolve() == MIGRATIONS.resolve()

    assert from_ini.get_main_option("sqlalchemy.url") is None


def test_the_history_package_does_not_import_the_api() -> None:
    code = (
        "import sys\n"
        "import researchpilot.history.database, researchpilot.history.models\n"
        "import researchpilot.history.records, researchpilot.history.settings\n"
        "import researchpilot.history.store\n"
        "print(sorted(m for m in sys.modules if m.startswith('researchpilot.api')))\n"
    )
    environment = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")}
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=environment,
        timeout=120,
        check=True,
    )

    assert completed.stdout.strip() == "[]"
