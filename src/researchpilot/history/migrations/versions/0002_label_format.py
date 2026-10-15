"""Replace the label constraint with the full stored-label rule.

Revision ID: 0002_label_format
Revises: 0001_research_jobs
Create Date: 2026-10-14

A stored label must be 1-120 characters, contain no U+0000-U+001F or U+007F-U+009F control
characters, and have no leading or trailing ordinary space. The rule in 0001 checked only the
length and surrounding spaces.

If a database already holds a label that breaks the new rule, adding the constraint fails and
this migration is rolled back without changing anything; fix or clear those labels first.
Downgrading restores the 0001 constraint, which every label valid here also satisfies.

The SQL is written out here rather than imported, so this migration never changes. An offline
test checks that it matches ``researchpilot.history.models``.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0002_label_format"
down_revision: str | Sequence[str] | None = "0001_research_jobs"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

PREVIOUS_LABEL_CHECK = (
    "label IS NULL OR (label = btrim(label) AND char_length(label) BETWEEN 1 AND 120)"
)
LABEL_CHECK = (
    "label IS NULL OR (char_length(label) BETWEEN 1 AND 120 "
    "AND label !~ '[\\x01-\\x1f\\x7f-\\x9f]' AND label = btrim(label, ' '))"
)


def upgrade() -> None:
    op.drop_constraint(op.f("ck_research_jobs_label_trimmed"), "research_jobs", type_="check")
    op.create_check_constraint(op.f("ck_research_jobs_label_format"), "research_jobs", LABEL_CHECK)


def downgrade() -> None:
    op.drop_constraint(op.f("ck_research_jobs_label_format"), "research_jobs", type_="check")
    op.create_check_constraint(
        op.f("ck_research_jobs_label_trimmed"), "research_jobs", PREVIOUS_LABEL_CHECK
    )
