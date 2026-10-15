"""Alembic environment for the research history schema.

Online, it uses a connection passed in ``config.attributes["connection"]`` (as tests do), or
else ``RESEARCHPILOT_DATABASE_URL``. Offline (``--sql``), it only renders SQL and connects to
nothing. The database URL is never logged.
"""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import Connection

from researchpilot.history.database import create_history_engine
from researchpilot.history.models import Base
from researchpilot.history.settings import DatabaseSettings

config = context.config
target_metadata = Base.metadata
# Offline mode needs a dialect, not a database; this URL is never connected to.
OFFLINE_URL = "postgresql+psycopg://offline/researchpilot"

if config.config_file_name is not None:
    fileConfig(config.config_file_name, disable_existing_loggers=False)


def run_migrations_offline() -> None:
    context.configure(
        url=OFFLINE_URL,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        transaction_per_migration=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def _run(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        transaction_per_migration=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connection = config.attributes.get("connection")
    if connection is not None:
        _run(connection)
        return
    engine = create_history_engine(DatabaseSettings())
    try:
        with engine.connect() as connection:
            _run(connection)
    finally:
        engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
