"""Create the research_jobs table.

Revision ID: 0001_research_jobs
Revises:
Create Date: 2026-10-14

Stored reports and analysis notes may contain sensitive, source-derived text; raw input
material and exception messages are never stored. Downgrading drops the table and with it all
research history.

The SQL is written out here rather than imported, so this migration never changes. An offline
test checks that it creates exactly the table in ``researchpilot.history.models``.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001_research_jobs"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

STATE_FIELDS_CHECK = (
    "(state = 'queued' AND started_at IS NULL AND finished_at IS NULL AND result IS NULL "
    "AND failure_code IS NULL AND cancel_reason IS NULL AND expires_at IS NULL) "
    "OR (state = 'running' AND started_at IS NOT NULL AND finished_at IS NULL "
    "AND result IS NULL AND failure_code IS NULL AND cancel_reason IS NULL "
    "AND expires_at IS NULL) "
    "OR (state = 'succeeded' AND started_at IS NOT NULL AND finished_at IS NOT NULL "
    "AND result IS NOT NULL AND failure_code IS NULL AND cancel_reason IS NULL "
    "AND expires_at IS NOT NULL) "
    "OR (state = 'failed' AND finished_at IS NOT NULL AND result IS NULL "
    "AND failure_code IS NOT NULL AND cancel_reason IS NULL AND expires_at IS NOT NULL) "
    "OR (state = 'cancelled' AND started_at IS NULL AND finished_at IS NOT NULL "
    "AND result IS NULL AND failure_code IS NULL AND cancel_reason IS NOT NULL "
    "AND expires_at IS NOT NULL)"
)


def upgrade() -> None:
    op.create_table(
        "research_jobs",
        sa.Column("id", sa.String(32), nullable=False),
        sa.Column("label", sa.String(120), nullable=True),
        sa.Column("mode", sa.String(8), nullable=False),
        sa.Column("state", sa.String(16), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("failure_code", sa.String(64), nullable=True),
        sa.Column("cancel_reason", sa.String(16), nullable=True),
        sa.Column("result", postgresql.JSONB(), nullable=True),
        sa.Column("result_version", sa.SmallInteger(), nullable=True),
        sa.Column("instance_id", sa.String(32), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_research_jobs")),
        sa.CheckConstraint("id ~ '^[0-9a-f]{32}$'", name=op.f("ck_research_jobs_id_format")),
        sa.CheckConstraint(
            "instance_id ~ '^[0-9a-f]{32}$'", name=op.f("ck_research_jobs_instance_format")
        ),
        sa.CheckConstraint(
            "label IS NULL OR (label = btrim(label) AND char_length(label) BETWEEN 1 AND 120)",
            name=op.f("ck_research_jobs_label_trimmed"),
        ),
        sa.CheckConstraint("mode IN ('local', 'web')", name=op.f("ck_research_jobs_mode")),
        sa.CheckConstraint(
            "state IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')",
            name=op.f("ck_research_jobs_state"),
        ),
        sa.CheckConstraint(
            "failure_code IS NULL OR failure_code ~ '^[a-z_]{1,64}$'",
            name=op.f("ck_research_jobs_failure_code"),
        ),
        sa.CheckConstraint(
            "cancel_reason IS NULL OR cancel_reason IN ('client', 'shutdown', 'worker_stopped')",
            name=op.f("ck_research_jobs_cancel_reason"),
        ),
        sa.CheckConstraint(
            "(result IS NULL) = (result_version IS NULL)",
            name=op.f("ck_research_jobs_result_version"),
        ),
        sa.CheckConstraint(
            "started_at IS NULL OR started_at >= created_at",
            name=op.f("ck_research_jobs_start_order"),
        ),
        sa.CheckConstraint(
            "finished_at IS NULL OR finished_at >= COALESCE(started_at, created_at)",
            name=op.f("ck_research_jobs_finish_order"),
        ),
        sa.CheckConstraint(STATE_FIELDS_CHECK, name=op.f("ck_research_jobs_state_fields")),
    )
    op.create_index(
        "ix_research_jobs_unfinished",
        "research_jobs",
        ["state"],
        postgresql_where=sa.text("state IN ('queued', 'running')"),
    )
    op.create_index(
        "ix_research_jobs_expires_at",
        "research_jobs",
        ["expires_at"],
        postgresql_where=sa.text("expires_at IS NOT NULL"),
    )
    op.create_index(
        "ix_research_jobs_created_at",
        "research_jobs",
        [sa.text("created_at DESC"), sa.text("id DESC")],
    )


def downgrade() -> None:
    op.drop_index("ix_research_jobs_created_at", table_name="research_jobs")
    op.drop_index("ix_research_jobs_expires_at", table_name="research_jobs")
    op.drop_index("ix_research_jobs_unfinished", table_name="research_jobs")
    op.drop_table("research_jobs")
