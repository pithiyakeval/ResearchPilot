"""Version 1 health routes."""

import logging
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from researchpilot.api.dependencies import (
    AppDependencies,
    ensure_history_and_worker,
    get_dependencies,
)
from researchpilot.api.problems import PROBLEM_MEDIA_TYPE, Problem, ProblemError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["health"])
# Seconds a client should wait before checking readiness again.
READY_RETRY_AFTER_SECONDS = 5


class Health(BaseModel):
    status: Literal["ok"]
    service: Literal["researchpilot"]
    version: str


class Readiness(BaseModel):
    status: Literal["ready"]
    checks: dict[str, Literal["ok"]]


def jobs_unavailable() -> ProblemError:
    """The job worker is not running: not started yet, shut down, or stopped after a failure."""
    return ProblemError(
        503,
        "jobs_unavailable",
        "Research jobs are unavailable",
        "The research job worker is not running.",
    )


@router.get("/health", response_model=Health, summary="Liveness: the API process is running")
def health(request: Request) -> Health:
    """Always 200 while the process runs; never contacts Ollama."""
    return Health(status="ok", service="researchpilot", version=request.app.version)


@router.get(
    "/health/ready",
    response_model=Readiness,
    summary="Readiness: jobs and history are available and the local Ollama server answers",
    responses={503: {"model": Problem, "content": {PROBLEM_MEDIA_TYPE: {}}}},
)
def ready(dependencies: Annotated[AppDependencies, Depends(get_dependencies)]) -> Readiness:
    """200 when jobs are accepted, history is ready and Ollama answers; 503 otherwise.

    A closed job manager (shut down, or stopped after a fatal worker failure or a lost
    history lock) is reported first; only a restart makes the server ready again. History
    that is not ready is reported next with its own problem code; checking it may retry
    initialization and start the worker. Neither contacts Ollama.
    """
    if dependencies.jobs.closed:
        raise jobs_unavailable()
    history = ensure_history_and_worker(dependencies)
    if not history.ready:
        raise history.problem()
    try:
        available = dependencies.ollama_probe()
    except Exception as exc:  # noqa: BLE001 - a broken probe must still answer "not ready"
        logger.warning("Readiness probe raised %s", type(exc).__name__)
        available = False
    if not available:
        raise ProblemError(
            503,
            "ollama_unavailable",
            "Ollama is not available",
            "The local Ollama server did not answer the readiness check in time.",
            headers={"Retry-After": str(READY_RETRY_AFTER_SECONDS)},
        )
    return Readiness(status="ready", checks={"ollama": "ok", "jobs": "ok", "history": "ok"})
