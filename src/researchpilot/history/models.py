"""SQLAlchemy model of the ``research_jobs`` table.

Migrations ``0001_research_jobs`` and ``0002_label_format`` together create exactly this
table, and offline tests compare their DDL with this model, so any change here needs a new
migration. The value lists, the label rule and the state-field rule come from ``records`` so
that Python and PostgreSQL enforce the same constraints.
"""

from collections.abc import Iterable
from datetime import datetime
from typing import Any

from sqlalchemy import CheckConstraint, DateTime, Index, MetaData, SmallInteger, String, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from researchpilot.history.records import (
    CANCEL_REASONS,
    FAILURE_CODE_PATTERN,
    JOB_ID_PATTERN,
    JOB_STATES,
    LABEL_CONTROL_SQL_PATTERN,
    MAX_LABEL_LENGTH,
    MODES,
    STATE_FIELDS,
)

NAMING_CONVENTION = {"ck": "ck_%(table_name)s_%(constraint_name)s", "pk": "pk_%(table_name)s"}


def one_of(column: str, values: Iterable[str]) -> str:
    return f"{column} IN ({', '.join(f"'{value}'" for value in values)})"


def label_check() -> str:
    """SQL for ``ck_research_jobs_label_format``; the same rule as ``records.label_is_valid``."""
    return (
        f"label IS NULL OR (char_length(label) BETWEEN 1 AND {MAX_LABEL_LENGTH} "
        f"AND label !~ '{LABEL_CONTROL_SQL_PATTERN}' AND label = btrim(label, ' '))"
    )


def state_fields_check() -> str:
    """SQL for ``ck_research_jobs_state_fields``, generated from ``STATE_FIELDS``."""
    clauses = []
    for state, rules in STATE_FIELDS.items():
        conditions = [f"state = '{state}'"]
        conditions += [
            f"{name} IS {'NOT NULL' if rule else 'NULL'}"
            for name, rule in rules.items()
            if rule is not None
        ]
        clauses.append(f"({' AND '.join(conditions)})")
    return " OR ".join(clauses)


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class ResearchJobRow(Base):
    """One research job; ``result`` is set for succeeded jobs only."""

    __tablename__ = "research_jobs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    label: Mapped[str | None] = mapped_column(String(MAX_LABEL_LENGTH))
    mode: Mapped[str] = mapped_column(String(8))
    state: Mapped[str] = mapped_column(String(16))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    failure_code: Mapped[str | None] = mapped_column(String(64))
    cancel_reason: Mapped[str | None] = mapped_column(String(16))
    # Python None must be stored as SQL NULL, not as the JSON value null; the result_version and
    # state_fields constraints test "result IS NULL". This does not change the table's DDL.
    result: Mapped[dict[str, Any] | None] = mapped_column(JSONB(none_as_null=True))
    result_version: Mapped[int | None] = mapped_column(SmallInteger)
    instance_id: Mapped[str] = mapped_column(String(32))
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()")
    )

    __table_args__ = (
        CheckConstraint(f"id ~ '{JOB_ID_PATTERN}'", name="id_format"),
        CheckConstraint(f"instance_id ~ '{JOB_ID_PATTERN}'", name="instance_format"),
        CheckConstraint(label_check(), name="label_format"),
        CheckConstraint(one_of("mode", MODES), name="mode"),
        CheckConstraint(one_of("state", JOB_STATES), name="state"),
        CheckConstraint(
            f"failure_code IS NULL OR failure_code ~ '{FAILURE_CODE_PATTERN}'",
            name="failure_code",
        ),
        CheckConstraint(
            f"cancel_reason IS NULL OR {one_of('cancel_reason', CANCEL_REASONS)}",
            name="cancel_reason",
        ),
        CheckConstraint("(result IS NULL) = (result_version IS NULL)", name="result_version"),
        CheckConstraint("started_at IS NULL OR started_at >= created_at", name="start_order"),
        CheckConstraint(
            "finished_at IS NULL OR finished_at >= COALESCE(started_at, created_at)",
            name="finish_order",
        ),
        CheckConstraint(state_fields_check(), name="state_fields"),
        Index(
            "ix_research_jobs_unfinished",
            "state",
            postgresql_where=text(one_of("state", ("queued", "running"))),
        ),
        Index(
            "ix_research_jobs_expires_at",
            "expires_at",
            postgresql_where=text("expires_at IS NOT NULL"),
        ),
    )


# Newest first, for listing history.
Index("ix_research_jobs_created_at", ResearchJobRow.created_at.desc(), ResearchJobRow.id.desc())
