# Maintainer notes

Settings and decisions that live outside the code. Nothing here is applied automatically:
each item must be configured or decided by a repository maintainer. Items are
recommendations unless they say they have been verified.

## Branch protection for `main`

Configure under **Settings → Rules → Rulesets** (or **Branches → Branch protection rules**):

- Require a pull request before merging, with at least one approving review if more than
  one person contributes; dismiss stale approvals when new commits are pushed.
- Require status checks to pass, with the branch up to date. Required check names, exactly
  as the CI workflow reports them:
  - `Lint and format (Ruff)`
  - `Offline tests (Linux, Python 3.12)`
  - `Offline tests (Linux, Python 3.13)`
  - `Offline tests (Windows, Python 3.13)`
  - `PostgreSQL integration tests`
  - `Package build`
- Block force pushes and deletion of `main`.

All six checks pass on `main`; confirm the latest run on the **Actions** tab before marking
them required. A check can only be selected as required after it has reported at least
once, so add any renamed or new job after its first run. Do **not** make the advisory
Security jobs required: they never fail by design, and making them required would add no
protection.

## Security settings

The repository is public (checked through the public GitHub API), so these features are
available without a paid plan. Whether each is currently enabled was **not** verified.

- **Secret scanning and push protection** (Settings → Security → Advanced Security). A scan
  of the tracked files with `detect-secrets` found only test placeholders, documentation
  examples and the CI-only database password in `ci.yml`; nothing that grants access.
- **Dependabot alerts and security updates.** Version updates are configured in
  `.github/dependabot.yml`; alerts and automatic security pull requests are separate
  settings.
- **CodeQL default setup** for Python and GitHub Actions (Settings → Security → Code
  scanning). Optional: it overlaps with the workflow lint, but also covers Python code.
- **Private vulnerability reporting**, so security issues are not filed as public issues.

## Deployment readiness

There is no deployment target, container image or release process yet, and nothing deploys
from CI. Before any of that is added, decide: the hosting target, how the database is
provisioned and backed up, where Ollama runs and whether a GPU is needed, how secrets are
stored and injected, the release and rollback policy, and who may approve a deployment.

What the code already provides:

- **Configuration** comes only from environment variables (or `.env` for local use):
  `OLLAMA_*` and `RESEARCHPILOT_MAX_ANALYSIS_PASSES` for research, `RESEARCHPILOT_API_*`
  (host, port, readiness timeout, request size, queue and retention limits, shutdown
  timeout) for the API, and `RESEARCHPILOT_DATABASE_URL` plus the pool and timeout settings
  under `RESEARCHPILOT_DATABASE_*` for history. Only the database URL is a secret.
- **Health checks:** `GET /api/v1/health` for liveness (never touches dependencies) and
  `GET /api/v1/health/ready` for readiness (history, job worker and Ollama).
- **Migrations are explicit:** `uv run alembic upgrade head` from a checkout of the
  repository. `alembic.ini` lives in the repository, not in the package, so a deployment
  built from the wheel alone needs its own Alembic configuration pointing at the installed
  `researchpilot/history/migrations` (the package-build job checks they are installed).
- **Single instance:** a PostgreSQL advisory lock allows one API process per database; a
  second instance stays not-ready. Do not run several replicas against one database.
- **Shutdown:** on stop, queued jobs are cancelled and recorded, the worker is given
  `RESEARCHPILOT_API_SHUTDOWN_TIMEOUT_SECONDS` (default 5) to finish, history writes in
  progress are drained, and the lock is released. A job still running is not interrupted;
  the next start records it as `interrupted_by_restart`. Give the process a grace period
  longer than these limits.
- **Logs:** job lifecycle events (`job_queued`, `job_started`, `job_finished`) with job ID,
  timings, model name, stop reason, pass and call counts, and failure or cancel codes. They
  never contain questions, sources, reports or exception messages.

A container image is a reasonable next step once a target is chosen. Proposed shape, for
review before anything is built: a slim Python 3.13 base; dependencies installed with
`uv sync --locked --no-dev` in a build stage; a non-root user; the health endpoint as the
container health check; secrets supplied only as environment variables at run time; and
migrations run as a separate one-off step, not on container start.
