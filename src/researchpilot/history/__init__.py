"""Persistent research history in PostgreSQL.

What is stored: job metadata (ID, optional user label, mode, state, timestamps, failure code,
cancel reason) and, for succeeded jobs, the ``ResearchResult`` the API serves. That result
includes the question, the report and the analysis notes. The report and notes are model
output written from the sources and may contain sensitive or source-derived text, including
verbatim quotations, so the database must be protected like the sources themselves.

What is never stored: the submitted source material, web source text, internal request
objects and exception messages.

Succeeded jobs are kept for 30 days and failed or cancelled jobs for 14 days.

The schema changes only through explicit migrations, run from the project directory with
``uv run alembic upgrade head``; the application never migrates itself and refuses to use a
database whose schema is not at ``EXPECTED_SCHEMA_REVISION``.
"""

EXPECTED_SCHEMA_REVISION = "0002_label_format"
