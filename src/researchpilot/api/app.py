"""Application factory for the ResearchPilot HTTP API."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from functools import partial
from importlib.metadata import PackageNotFoundError, version

from fastapi import FastAPI
from starlette.concurrency import run_in_threadpool

from researchpilot.api.dependencies import (
    AppDependencies,
    ReadinessProbe,
    check_ollama,
    ensure_history_and_worker,
)
from researchpilot.api.history import (
    HistoryControl,
    HistoryRecorder,
    HistoryRuntime,
    quiet_driver_loggers,
)
from researchpilot.api.job_routes import router as job_router
from researchpilot.api.jobs import JobManager, ManagerClosedError
from researchpilot.api.problems import install_problem_handlers
from researchpilot.api.routes import router
from researchpilot.api.settings import ApiSettings
from researchpilot.config import Settings, get_settings
from researchpilot.history.settings import DatabaseSettings


def _package_version() -> str:
    try:
        return version("researchpilot")
    except PackageNotFoundError:
        return "unknown"


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Serve with the job worker running only while research history is ready.

    Startup tries to bring history to ready; if PostgreSQL is unavailable the app still
    starts, degraded, and readiness or a submission retries later. The worker starts once
    history is ready, and the run gate stops it if the instance lock is lost.

    Shutdown stops the worker first, then closes history, releasing the instance lock and the
    database engine. Blocking work runs in the threadpool, so the event loop is never blocked.
    A job manager runs once: a second lifespan on the same app fails at startup.
    """
    dependencies: AppDependencies = app.state.dependencies
    if dependencies.jobs.closed:
        raise ManagerClosedError
    quiet_driver_loggers()
    dependencies.serving.begin()
    try:
        await run_in_threadpool(ensure_history_and_worker, dependencies)
        yield
    finally:
        dependencies.serving.end()
        try:
            await run_in_threadpool(
                dependencies.jobs.shutdown, dependencies.api_settings.shutdown_timeout_seconds
            )
        finally:
            await run_in_threadpool(dependencies.history.close)


def create_app(
    *,
    settings: Settings | None = None,
    api_settings: ApiSettings | None = None,
    ollama_probe: ReadinessProbe | None = None,
    job_manager: JobManager | None = None,
    history: HistoryControl | None = None,
) -> FastAPI:
    """Build the API. Every dependency can be injected; nothing is contacted at creation.

    Without ``ollama_probe``, readiness asks Ollama at ``settings.ollama_base_url`` with
    ``api_settings.ready_timeout_seconds`` as the timeout. Without ``job_manager``, jobs run
    the real research pipeline. Without ``history``, a ``HistoryRuntime`` is built from
    ``RESEARCHPILOT_DATABASE_*`` settings. The manager's run gate is set to require ready
    history, every job is recorded through it, and the app's lifespan starts and stops both.
    """
    settings = settings if settings is not None else get_settings()
    api_settings = api_settings if api_settings is not None else ApiSettings()
    probe = ollama_probe or partial(
        check_ollama, str(settings.ollama_base_url), api_settings.ready_timeout_seconds
    )
    jobs = (
        job_manager if job_manager is not None else JobManager.from_settings(settings, api_settings)
    )
    history = history if history is not None else HistoryRuntime(DatabaseSettings())
    # Before each job: once ready, this only asks the server whether the lock is still held.
    jobs.set_run_gate(lambda: history.ensure_ready().ready)
    jobs.set_recorder(HistoryRecorder(history))
    app = FastAPI(title="ResearchPilot API", version=_package_version(), lifespan=_lifespan)
    app.state.dependencies = AppDependencies(
        settings=settings, api_settings=api_settings, ollama_probe=probe, jobs=jobs, history=history
    )
    install_problem_handlers(app, max_request_bytes=api_settings.max_request_bytes)
    app.include_router(router)
    app.include_router(job_router)
    return app
