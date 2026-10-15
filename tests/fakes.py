"""Deterministic test doubles; no Ollama server or network access is used."""

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Self

from langchain_core.messages import AIMessage
from mcp.types import CallToolResult, TextContent

from researchpilot.api.history import HistoryState, HistoryStatus
from researchpilot.api.jobs import JobInput
from researchpilot.history.records import FinishedJob, HistoryRecord, QueuedJob, StartedJob
from researchpilot.history.store import HistoryUnavailableError

# Upper bound for the shared research double's wait, so a test that never releases a job
# fails instead of hanging.
RESEARCH_TIMEOUT = 5.0


class ScriptedModel:
    """Chat-model stand-in that returns scripted replies in order and records each prompt."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.prompts: list[str] = []

    @property
    def call_count(self) -> int:
        return len(self.prompts)

    def invoke(self, input, config=None, **kwargs):
        if self.call_count >= len(self._responses):
            raise AssertionError(
                f"Model called {self.call_count + 1} times; "
                f"only {len(self._responses)} responses were scripted."
            )
        prompt = input.to_string()
        self.prompts.append(prompt)
        content = self._responses[self.call_count - 1]
        # Mimic the token counts Ollama reports in response metadata.
        metadata = {"prompt_eval_count": len(prompt) // 4, "eval_count": len(content) // 4}
        return AIMessage(content=content, response_metadata=metadata)


def complete(source_gaps: list[str] | None = None, conflicts: list[str] | None = None) -> str:
    return json.dumps(
        {"analysis_gaps": [], "source_gaps": source_gaps or [], "conflicts": conflicts or []}
    )


def incomplete(
    analysis_gaps: list[str],
    source_gaps: list[str] | None = None,
    conflicts: list[str] | None = None,
) -> str:
    return json.dumps(
        {
            "analysis_gaps": analysis_gaps,
            "source_gaps": source_gaps or [],
            "conflicts": conflicts or [],
        }
    )


def tool_result(payload: object) -> CallToolResult:
    """An MCP tool result whose text block holds ``payload`` as JSON, as Parallel returns it."""
    return CallToolResult(content=[TextContent(type="text", text=json.dumps(payload))])


def search_item(url: str, title: str = "Result title", excerpt: str | None = None) -> dict:
    return {
        "url": url,
        "title": title,
        "publish_date": None,
        "excerpts": [excerpt if excerpt is not None else f"Search excerpt text for {url}."],
    }


def page_text(url: str) -> str:
    """Default fetched-page body: long enough to pass the low-quality page check."""
    return f"Fetched page text for {url}. " + "The page explains the topic in detail. " * 8


def page_payload(url: str, title: str = "Page title", text: str | None = None) -> dict:
    return {
        "extract_id": "extract_test",
        "results": [
            {
                "url": url,
                "title": title,
                "publish_date": None,
                "excerpts": [text if text is not None else page_text(url)],
                "full_content": None,
            }
        ],
        "errors": [],
    }


class FakeParallelClient:
    """Stand-in for ParallelSearchMCPClient; records calls and never touches the network.

    ``search`` and each ``fetch_results`` value may be a result, an exception to raise, or an
    async callable to await. Fetches of URLs without a scripted result get a matching page.
    """

    def __init__(
        self,
        search: object,
        fetch_results: dict[str, object] | None = None,
        enter_error: Exception | None = None,
    ) -> None:
        self.search_result = search
        self.fetch_results = fetch_results or {}
        self.enter_error = enter_error
        self.search_calls: list[tuple[str, list[str]]] = []
        self.fetch_calls: list[dict] = []
        self.closed = False

    async def __aenter__(self) -> Self:
        if self.enter_error is not None:
            raise self.enter_error
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        self.closed = True

    @staticmethod
    async def _resolve(value: object) -> object:
        if isinstance(value, BaseException):
            raise value
        if callable(value):
            return await value()
        return value

    async def search(self, objective: str, search_queries: list[str]) -> object:
        self.search_calls.append((objective, list(search_queries)))
        return await self._resolve(self.search_result)

    async def fetch(self, url: str, **kwargs) -> object:
        self.fetch_calls.append({"url": url, **kwargs})
        default = tool_result(page_payload(url))
        return await self._resolve(self.fetch_results.get(url, default))


class FakeHistoryStore:
    """In-memory HistoryStore stand-in for job recording.

    ``rows`` maps job IDs to the latest record type written; ``writes`` lists every write in
    order. Add an operation name to ``failing`` to make it raise HistoryUnavailableError, and
    set ``before`` to a callable run at the start of each write, given the operation and
    record.
    """

    def __init__(self) -> None:
        self.rows: dict[str, str] = {}
        self.records: dict[str, list[object]] = {}
        self.writes: list[tuple[str, str]] = []
        self.failing: set[str] = set()
        self.before = None
        self.stored: dict[str, HistoryRecord] = {}
        self.get_error: Exception | None = None
        self.gets: list[str] = []

    def _write(self, operation: str, job_id: str, record: object) -> None:
        if self.before is not None:
            self.before(operation, record)
        if operation in self.failing:
            raise HistoryUnavailableError
        self.writes.append((operation, job_id))
        self.records.setdefault(job_id, []).append(record)

    def record_queued(self, job) -> None:
        self._write("record_queued", job.id, job)
        self.rows[job.id] = "queued"

    def record_started(self, job) -> None:
        self._write("record_started", job.id, job)
        self.rows[job.id] = "running"

    def record_finished(self, job) -> None:
        self._write("record_finished", job.id, job)
        self.rows[job.id] = job.state

    def delete_queued(self, job_id: str) -> None:
        self._write("delete_queued", job_id, job_id)
        self.rows.pop(job_id, None)

    def get(self, job_id: str) -> HistoryRecord | None:
        """A seeded record from ``stored``, else one rebuilt from the writes, like the store.

        Set ``get_error`` to make reads raise; ``gets`` lists the IDs read.
        """
        self.gets.append(job_id)
        if self.get_error is not None:
            raise self.get_error
        if job_id in self.stored:
            return self.stored[job_id]
        if job_id not in self.rows:
            return None
        written = self.records[job_id]
        queued = next(record for record in written if isinstance(record, QueuedJob))
        started = next((record for record in written if isinstance(record, StartedJob)), None)
        finished = next((record for record in written if isinstance(record, FinishedJob)), None)
        return HistoryRecord(
            id=job_id,
            label=queued.label,
            mode=queued.mode,
            state=self.rows[job_id],  # type: ignore[arg-type]
            created_at=queued.created_at,
            started_at=started.started_at if started else None,
            finished_at=finished.finished_at if finished else None,
            failure_code=finished.failure_code if finished else None,
            cancel_reason=finished.cancel_reason if finished else None,
            expires_at=None,
            result=finished.result if finished else None,
        )


class FakeHistory:
    """HistoryRuntime stand-in without a database: ready unless told otherwise.

    Set ``current`` to change what ``ensure_ready`` reports; ``events`` records the calls.
    ``backing_store`` records jobs and, like the real runtime, is offered only while ready.
    """

    def __init__(self, current: HistoryStatus | None = None) -> None:
        self.current = current or HistoryStatus(HistoryState.READY)
        self.events: list[str] = []
        self.backing_store = FakeHistoryStore()
        self.closed = False

    @property
    def status(self) -> HistoryStatus:
        return self.current

    @property
    def store(self) -> FakeHistoryStore | None:
        return self.backing_store if self.current.ready else None

    @contextmanager
    def writing(self) -> Iterator[FakeHistoryStore]:
        """Like the real runtime: no writes unless ready, and none after ``close``."""
        if self.closed or not self.current.ready:
            raise HistoryUnavailableError
        yield self.backing_store

    def ensure_ready(self) -> HistoryStatus:
        self.events.append("ensure_ready")
        return self.current

    def close(self) -> None:
        self.events.append("close")
        self.closed = True


class Research:
    """Research double; with ``block``, each call waits until ``release`` is set."""

    def __init__(self, *, error: BaseException | None = None, block: bool = False) -> None:
        self.error = error
        self.started = threading.Event()
        self.release = threading.Event()
        if not block:
            self.release.set()
        self.inputs: list[JobInput] = []

    @property
    def calls(self) -> int:
        return len(self.inputs)

    def __call__(self, job_input: JobInput) -> object:
        self.inputs.append(job_input)
        self.started.set()
        assert self.release.wait(RESEARCH_TIMEOUT), "test never released the job"
        if self.error is not None:
            raise self.error
        return object()


class Probe:
    """Readiness probe double: Ollama always answers."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> bool:
        self.calls += 1
        return True
