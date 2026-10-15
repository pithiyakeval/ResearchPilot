"""Dependencies injected into the API, so tests can replace them without monkeypatching."""

import threading
from collections.abc import Callable
from dataclasses import dataclass, field

import httpx
from fastapi import Request

from researchpilot.api.history import HistoryControl, HistoryStatus
from researchpilot.api.jobs import JobManager
from researchpilot.api.settings import ApiSettings
from researchpilot.config import Settings

# Returns True when Ollama answered in time; must not raise for an unavailable server.
ReadinessProbe = Callable[[], bool]


class ServingState:
    """Whether the app is between lifespan startup and shutdown.

    The job worker is only ever started while serving, so an app used without its lifespan
    never starts a thread, and nothing starts the worker once shutdown has begun.
    """

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._serving = False

    def begin(self) -> None:
        with self._guard:
            self._serving = True

    def end(self) -> None:
        with self._guard:
            self._serving = False

    def start_worker(self, jobs: JobManager) -> None:
        """Start the worker if serving and not yet started; safe to call repeatedly."""
        with self._guard:
            if self._serving and not jobs.started and not jobs.closed:
                jobs.start()


@dataclass(frozen=True)
class AppDependencies:
    settings: Settings
    api_settings: ApiSettings
    ollama_probe: ReadinessProbe
    jobs: JobManager
    history: HistoryControl
    serving: ServingState = field(default_factory=ServingState)


def ensure_history_and_worker(dependencies: AppDependencies) -> HistoryStatus:
    """Bring history to ready if possible, and start the worker once it is.

    The runtime limits how often it retries, so this is cheap to call per request.
    """
    status = dependencies.history.ensure_ready()
    if status.ready:
        dependencies.serving.start_worker(dependencies.jobs)
    return status


def check_ollama(base_url: str, timeout_seconds: float) -> bool:
    """Ask Ollama for its version with a short timeout; any failure means not ready."""
    try:
        response = httpx.get(f"{base_url.rstrip('/')}/api/version", timeout=timeout_seconds)
    except httpx.HTTPError:
        return False
    return response.is_success


def get_dependencies(request: Request) -> AppDependencies:
    """FastAPI dependency returning the objects ``create_app`` was given."""
    return request.app.state.dependencies
