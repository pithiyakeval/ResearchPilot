"""Plain records exchanged with the history store; no database or API types.

The store accepts ``QueuedJob``, ``StartedJob`` and ``FinishedJob`` and returns
``HistorySummary`` and ``HistoryRecord``. Inputs validate themselves with the same rules as
the ``research_jobs`` table's constraints, so an invalid write fails before any SQL is sent.
Error messages never include the rejected values.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from researchpilot.results import ResearchResult

HistoryState = Literal["queued", "running", "succeeded", "failed", "cancelled"]
HistoryMode = Literal["local", "web"]
FinishedState = Literal["succeeded", "failed", "cancelled"]
CancelReason = Literal["client", "shutdown", "worker_stopped"]

JOB_STATES: tuple[HistoryState, ...] = ("queued", "running", "succeeded", "failed", "cancelled")
FINISHED_STATES: tuple[FinishedState, ...] = ("succeeded", "failed", "cancelled")
MODES: tuple[HistoryMode, ...] = ("local", "web")
CANCEL_REASONS: tuple[CancelReason, ...] = ("client", "shutdown", "worker_stopped")
MAX_LABEL_LENGTH = 120
# PostgreSQL patterns; the Python checks use the same expressions without the anchors.
JOB_ID_PATTERN = "^[0-9a-f]{32}$"
FAILURE_CODE_PATTERN = "^[a-z_]{1,64}$"
# Control characters a label may not contain: U+0000-U+001F and U+007F-U+009F. PostgreSQL text
# cannot hold U+0000 at all, so its pattern starts at U+0001.
LABEL_CONTROL_PATTERN = r"[\x00-\x1f\x7f-\x9f]"
LABEL_CONTROL_SQL_PATTERN = r"[\x01-\x1f\x7f-\x9f]"

# For each state, whether an optional column must be set (True), must be null (False) or may
# be either (None). This is the single source of the rules: ``FinishedJob`` checks them and
# the table's ``ck_research_jobs_state_fields`` constraint is generated from them.
STATE_FIELDS: Mapping[HistoryState, Mapping[str, bool | None]] = {
    "queued": {
        "started_at": False,
        "finished_at": False,
        "result": False,
        "failure_code": False,
        "cancel_reason": False,
        "expires_at": False,
    },
    "running": {
        "started_at": True,
        "finished_at": False,
        "result": False,
        "failure_code": False,
        "cancel_reason": False,
        "expires_at": False,
    },
    "succeeded": {
        "started_at": True,
        "finished_at": True,
        "result": True,
        "failure_code": False,
        "cancel_reason": False,
        "expires_at": True,
    },
    # A job interrupted while still queued fails without ever starting.
    "failed": {
        "started_at": None,
        "finished_at": True,
        "result": False,
        "failure_code": True,
        "cancel_reason": False,
        "expires_at": True,
    },
    "cancelled": {
        "started_at": False,
        "finished_at": True,
        "result": False,
        "failure_code": False,
        "cancel_reason": True,
        "expires_at": True,
    },
}

_JOB_ID = re.compile(JOB_ID_PATTERN.strip("^$"))
_FAILURE_CODE = re.compile(FAILURE_CODE_PATTERN.strip("^$"))
_LABEL_CONTROL = re.compile(LABEL_CONTROL_PATTERN)


def fields_allowed(state: HistoryState, present: Mapping[str, bool]) -> bool:
    """Whether a row in ``state`` may have exactly the optional fields marked present."""
    return all(rule is None or present[name] is rule for name, rule in STATE_FIELDS[state].items())


def is_job_id(value: object) -> bool:
    return isinstance(value, str) and _JOB_ID.fullmatch(value) is not None


def require_job_id(value: object) -> None:
    if not is_job_id(value):
        raise ValueError("A job ID must be 32 lowercase hexadecimal characters.")


def require_aware(name: str, value: object) -> None:
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise ValueError(f"{name} must be a timezone-aware datetime.")


def label_is_valid(label: str) -> bool:
    """The stored-label rule, identical to the ``ck_research_jobs_label_format`` constraint.

    1-120 characters, no U+0000-U+001F or U+007F-U+009F control characters, and no leading
    or trailing ordinary space (U+0020). Callers accepting user input should first apply
    Unicode ``strip()``, which also removes other whitespace.
    """
    return (
        1 <= len(label) <= MAX_LABEL_LENGTH
        and _LABEL_CONTROL.search(label) is None
        and label == label.strip(" ")
    )


def _require_label(label: object) -> None:
    if label is None:
        return
    if not isinstance(label, str) or not label_is_valid(label):
        raise ValueError(
            f"A label must be 1-{MAX_LABEL_LENGTH} characters, without control characters "
            "or leading or trailing spaces."
        )


@dataclass(frozen=True)
class QueuedJob:
    """A job accepted for research; written before it may run."""

    id: str
    mode: HistoryMode
    created_at: datetime
    label: str | None = None

    def __post_init__(self) -> None:
        require_job_id(self.id)
        if self.mode not in MODES:
            raise ValueError("A job's mode must be local or web.")
        require_aware("created_at", self.created_at)
        _require_label(self.label)


@dataclass(frozen=True)
class StartedJob:
    id: str
    started_at: datetime

    def __post_init__(self) -> None:
        require_job_id(self.id)
        require_aware("started_at", self.started_at)


@dataclass(frozen=True)
class FinishedJob:
    """A job's terminal state; the store adds the retention expiry."""

    id: str
    state: FinishedState
    finished_at: datetime
    result: ResearchResult | None = None  # succeeded only
    failure_code: str | None = None  # failed only
    cancel_reason: CancelReason | None = None  # cancelled only

    def __post_init__(self) -> None:
        require_job_id(self.id)
        if self.state not in FINISHED_STATES:
            raise ValueError("A finished job's state must be succeeded, failed or cancelled.")
        require_aware("finished_at", self.finished_at)
        if self.result is not None and not isinstance(self.result, ResearchResult):
            raise ValueError("A result must be a ResearchResult.")
        if self.failure_code is not None and not (
            isinstance(self.failure_code, str) and _FAILURE_CODE.fullmatch(self.failure_code)
        ):
            raise ValueError("A failure code must be 1-64 lowercase letters or underscores.")
        if self.cancel_reason is not None and self.cancel_reason not in CANCEL_REASONS:
            raise ValueError("A cancel reason must be client, shutdown or worker_stopped.")
        present = {
            "result": self.result is not None,
            "failure_code": self.failure_code is not None,
            "cancel_reason": self.cancel_reason is not None,
        }
        rules = STATE_FIELDS[self.state]
        if any(rules[name] is not None and present[name] is not rules[name] for name in present):
            raise ValueError(f"A {self.state} job must have exactly the fields its state requires.")


@dataclass(frozen=True)
class HistoryCursor:
    """A position in the newest-first history listing: a job's ``(created_at, id)``.

    The listing is ordered by ``created_at`` descending, then ``id`` descending, so the pair
    is unique and jobs created at the same instant are neither skipped nor repeated.
    """

    created_at: datetime
    id: str

    def __post_init__(self) -> None:
        require_aware("A cursor's created_at", self.created_at)
        require_job_id(self.id)


@dataclass(frozen=True)
class HistorySummary:
    """A stored job without its result, for listing history."""

    id: str
    label: str | None
    mode: HistoryMode
    state: HistoryState
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    failure_code: str | None
    cancel_reason: CancelReason | None
    expires_at: datetime | None

    @property
    def cursor(self) -> HistoryCursor:
        """Pass the last summary's cursor as ``before`` to fetch the next page."""
        return HistoryCursor(self.created_at, self.id)


@dataclass(frozen=True)
class HistoryRecord(HistorySummary):
    """A stored job with its result, re-validated as a ``ResearchResult`` when read.

    Its label and result may contain sensitive text: never log a record.
    """

    result: ResearchResult | None
