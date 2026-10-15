# ResearchPilot — Stage 2 (LangGraph + local Ollama)

A small, source-grounded research synthesis CLI. It accepts a question and a Markdown/text file containing source material, then uses a locally running Ollama model to analyze the sources, check that analysis for coverage, and produce a report.

**Scope:** by default, ResearchPilot does not search the web, fetch URLs, or independently verify claims. It only synthesizes material supplied by the user. Extra analysis passes re-read the same supplied material; they never find new evidence. With `--web` (see [Web mode](#web-mode)), one bounded web search runs before analysis and its results are added as extra, untrusted sources.

## Requirements

- Python 3.12+
- `uv`
- [Ollama](https://ollama.com/download) installed and running locally

No API key or paid service is required; inference runs on your machine.

## Setup (Windows PowerShell)

Install Ollama, then pull the default model (about 3 GB):

```powershell
ollama pull gemma3:4b
```

The Ollama desktop app starts the server automatically. If it is not running, start it with `ollama serve`. Check that it is reachable:

```powershell
ollama list
```

Install the project dependencies:

```powershell
uv sync
```

Configuration is optional. To override the defaults, copy the example file and edit it:

```powershell
Copy-Item .env.example .env
```

| Setting | Default | Meaning |
|---|---|---|
| `OLLAMA_MODEL` | `gemma3:4b` | Ollama model name (`ollama pull` it first) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server address |
| `OLLAMA_TEMPERATURE` | `0` | Sampling temperature, 0–2 |
| `OLLAMA_NUM_CTX` | `16384` | Context window requested from Ollama, in tokens (2048–131072) |
| `OLLAMA_NUM_PREDICT` | `2048` | Maximum tokens per model response (256–8192, at most a quarter of `OLLAMA_NUM_CTX`) |
| `RESEARCHPILOT_MAX_ANALYSIS_PASSES` | `2` | Maximum analysis passes, 1–3 |

Settings can also be supplied as environment variables. Smaller models are faster but follow the source-grounding and JSON instructions less reliably.

### Input size

ResearchPilot never truncates sources. Before calling the model it estimates the size of the largest prompt (about 3 characters per token) and rejects input that would not fit, with exit code 2. With the defaults, a source file of roughly 27,000 characters fits; with `OLLAMA_NUM_CTX=32768`, roughly 76,000. A larger context window needs more memory; lower it (for example `OLLAMA_NUM_CTX=8192` with `OLLAMA_NUM_PREDICT=1024`, about 11,500 characters) on machines with little RAM or VRAM.

## How a run works

```text
START → analyze → evaluate ──(analysis gaps remain, passes < max)──→ analyze
                     │
                     └──(complete, invalid evaluation, or pass limit)──→ generate_report → END
```

1. **analyze** writes notes from the supplied sources. A repeat pass receives the previous notes and the evaluator's analysis gaps and writes a revised set.
2. **evaluate** returns three lists: *analysis gaps* (in the sources but missing from the notes, which another pass can fix), *source gaps* (not in the sources at all, which no pass can fix), and *conflicts* between sources. Ollama's structured-output mode constrains the reply to a JSON schema generated from the Pydantic model, and Pydantic then validates it. The analysis counts as complete when the validated analysis-gap list is empty; the model is not asked for a separate verdict. Only analysis gaps trigger another pass.
3. **generate_report** writes the report using the same headings as Stage 1, and lists source gaps and conflicts under "Gaps, Conflicts, and Limitations". Its source references are then checked in code (see below).

The run always stops within the pass limit. A run makes at most `2 × RESEARCHPILOT_MAX_ANALYSIS_PASSES + 1` model calls: 5 with the default, 7 at the maximum. If the evaluator's output cannot be validated, it is recorded as invalid (never as complete), the run continues to the report, and a note at the end of the report says coverage is unverified. A similar note is added when the pass limit is reached with gaps still open. In both cases the report also starts with an **Incomplete analysis** notice, so a stopped run cannot be read as a comprehensive review; at the pass limit it gives the number of unresolved analysis gaps, which are also returned in `ResearchReport.unresolved_analysis_gaps` and counted in the CLI's stderr summary. The default limit stays at 2 passes because each local model call takes minutes on CPU.

## Run

Create `sources.md` containing source notes and excerpts. Start each source with a top-level heading and keep its title and URL with the excerpt:

```markdown
# Source A: Task completion

https://example.com/report (excerpt) ...

# Source B: Evaluation limitations

...
```

Each top-level `#` heading starts a new source, numbered `[S1]`, `[S2]`, ... in order. `##` subheadings and headings inside code blocks stay within their source. Text before the first heading becomes its own source, and a file with no headings is a single source. Text inside a source that looks like a label, such as `[S2]`, is rewritten as `(S2)` so one source cannot impersonate another.

```powershell
uv run researchpilot "Compare the approaches described in these sources" --source-file .\sources.md
```

Save the output:

```powershell
uv run researchpilot "Summarize the evidence and limitations" --source-file .\sources.md --output .\reports\research-report.md
```

The CLI does not claim citations are independently verified. Check references against the supplied material.

### Citation check

Reports must cite sources by their IDs, for example `[S1]`. After the report is generated, ResearchPilot checks it in code and flags:

- IDs that are not supplied sources, such as `[S7]` when there are three sources;
- internal prompt-block labels written like a citation or tag, such as `[run_status]`, `<analysis_notes>` or `{source_material}` (bare names in prose or `code` are not flagged, since a report may legitimately discuss them);
- references to internal steps ("analysis notes", "run status") instead of sources;
- a report with no valid source ID at all.

Flagged problems are listed in a note appended to the report, together with the ID-to-title key, and in `ResearchReport.invalid_citations`. The report text itself is never rewritten. A valid ID only shows which supplied source a sentence points to; it does not prove that the source supports the claim.

### Review hints (heuristic)

After the citation check, two word-pattern checks may add **review hints**: prompts for a human reviewer, listed in a separate note and in `ResearchReport.quality_warnings`. They never change the report text, the citation results or the exit code.

- **Possible over-generalization:** a cited sentence uses a generalizing word (often, frequently, commonly, generally, typically, usually, widely) that none of its cited sources uses. This is a review signal, not an evidence-support check: a source can support a general claim in other words, and a source that uses the word may still not support the claim.
- **Possible contradiction:** a claim that the sources *as a whole* lack something ("the sources do not address X", "no source…", "none of the sources…") shares at least three content words with a cited statement. Statements that one source reports something missing, such as "[S1] did not report confidence intervals", are not compared, because they are often correct. It misses contradictions between two positive statements, absence claims worded differently, and pairs that use synonyms.

Hints can be wrong in either direction, and a report without hints is not thereby supported. The analysis and report prompts also ask the model to keep claims at their source's scope, to use technical terms only as the sources define them, and never to write internal block names; whether a small local model follows these instructions is not guaranteed.

After each run the CLI prints a one-line summary to stderr: analysis passes, model calls, total time, time per graph node, and why the analysis loop stopped. Add `--verbose` to also log each model call (node, pass, seconds, prompt size and the token counts Ollama reports). Logs contain counts and timings only, never source text.

```powershell
uv run researchpilot "Summarize the evidence and limitations" --source-file .\sources.md --verbose
```

### Saving the evidence (`--save-sources`)

To debug a report, save the exact evidence the run used to a new JSON file:

```powershell
uv run researchpilot "Summarize the evidence and limitations" --source-file .\sources.md --output .\reports\report.md --save-sources .\reports\report-evidence.json
```

The file (`schema_version` 2) holds every source, in ID order, with:

- `id`, `type` (`local` or `web`) and `text`: the exact text as numbered for the run;
- `numbered_title`: the title as it appears in the numbered text, the prompts and the citation notes (label-like text such as `[S2]` rewritten as `(S2)`; for web sources, `<evidence type>: <page title> — <url>`);
- for web sources also `evidence_type`, `url` and `page_title`: the page title after retrieval cleanup but before label rewriting, so it may differ from what the model saw.

`source_block_format` gives the rebuild rule, `"\n\n".join(f"[{id}]\n{text}")`, and `source_block_sha256` is the SHA-256 of that block after delimiter escaping, which is what every analysis, evaluation and report prompt contained. `text_handling` documents each transformation the stored texts went through: delimiter escaping (applied only in prompts, not to the stored texts), label rewriting, web-text cleanup and clipping, note trimming, and evaluator-list validation.

The file also holds each pass's analysis notes, each pass's structured evaluation (gap lists, conflicts, status, validation error), the stop reason, unresolved gaps, web retrieval counts and warnings, the non-secret model settings, and report metadata: the `--output` path, citation problems, review hints, timings and two hashes:

- `markdown_sha256`: the report Markdown string itself (UTF-8, LF line endings, no added newline);
- `output_file_sha256`: the exact bytes of the `--output` file, read back after writing (on Windows that file has CRLF line endings and a final newline, so the two hashes differ); `null` without `--output`.

The report text itself stays in its own file. Version 1 files, written before this change, used `title` (the page title for web sources), had a text-only `text_handling`, and had no `source_block_format` or `output_file_sha256`.

It never holds full prompts, raw evaluator replies, environment variables or the Ollama base URL. It does hold full source text and analysis notes, which may be private: it is written only when you ask for it, and its contents are never printed or logged; the CLI prints only the path and counts.

The destination is checked before any model or network call (exit code 2 on failure). It must end in `.json`; must not exist yet (it is never overwritten); its folder must already exist (folders are not created); and it must differ from the source file and the `--output` path, including through `\\?\` long-path aliases, 8.3 short names, case or `..` segments. On Windows it must also not be a device path (`\\.\...`), an alternate data stream (`notes.md:x.json`, which would write into another file) or a reserved device name (`NUL.json`, `CON.json`, `COM1.json`, ...). The file is written to a temporary file in the same folder and then published without overwriting, so a failure never leaves a partial file. If writing fails after the report is done, the report is still printed and saved, and the CLI exits with code 1. A run that fails part-way writes no evidence file.

## Tests and lint

```powershell
uv run pytest -v
uv run ruff check .
```

Tests use scripted fake models and do not contact Ollama, so they run without the server or any downloaded model.

`tests/live/` holds opt-in checks that run the real model against known failure cases (claiming the task count is unknown, wrong source attribution, calling a limitation a conflict). They are skipped by default and take several minutes on CPU:

```powershell
$env:RESEARCHPILOT_LIVE_TESTS = "1"; uv run pytest tests/live -v; Remove-Item Env:RESEARCHPILOT_LIVE_TESTS
```

Model output is not deterministic, so a live pass is evidence, not proof.

## Parallel Search MCP client

`researchpilot.parallel_mcp` is a standalone async client for the public [Parallel Search MCP server](https://search.parallel.ai/mcp), using the official `mcp` SDK over Streamable HTTP. No API key is needed. The CLI uses it only with `--web`.

On connection it lists the server's tools (currently `web_search` and `web_fetch`) and keeps their names, descriptions and input schemas. Calls send only arguments that the discovered schema declares, and fail with `ParallelMCPError` if a required argument is unsupported, a tool is missing, or the server reports an error, times out or returns a malformed or empty result. Error messages do not include server error text.

```python
import asyncio

from researchpilot.parallel_mcp import ParallelSearchMCPClient


async def main() -> None:
    async with ParallelSearchMCPClient(timeout_seconds=60) as client:
        for tool in await client.list_tools():
            print(tool.name, sorted(tool.input_schema["properties"]))
        found = await client.search(
            "Official LangGraph docs on conditional edges", ["LangGraph conditional edges"]
        )
        page = await client.fetch(
            "https://docs.langchain.com/oss/python/langgraph/graph-api",
            objective="How conditional edges route between nodes",
        )
        print(found.content[0].text[:500], page.content[0].text[:500], sep="\n\n")


asyncio.run(main())
```

The unit tests in `tests/test_parallel_mcp.py` use an in-memory fake and need no network. An opt-in smoke test calls the real server (one listing, one search, one fetch):

```powershell
$env:RESEARCHPILOT_LIVE_MCP_TESTS = "1"; uv run pytest tests/live/test_live_parallel_mcp.py -v; Remove-Item Env:RESEARCHPILOT_LIVE_MCP_TESTS
```

Use of the server is subject to Parallel's [terms](https://parallel.ai/customer-terms) and [privacy policy](https://parallel.ai/privacy-policy).

## Web mode

```powershell
uv run researchpilot "How do LangGraph conditional edges route between nodes?" --source-file .\sources.md --web
```

`--web` (`-w`) adds web evidence to the supplied source file, which is still required. Without it, nothing changes: no MCP connection or network request is made. Before analysis, `researchpilot.web_research` runs:

1. **One search.** The question is the search objective and its first 12 words are the keyword query. Up to 5 results are kept.
2. **Up to 2 fetches**, one per call, of the unique, safe results with the most substantive excerpts (excerpt size capped at the 1,200-character clip length; ties keep the search rank). A fetched page with fewer than 300 meaningful characters (after collapsing whitespace and dropping link-only, symbol-only and common login, cookie and navigation lines) is skipped as low quality; its search excerpt is kept, and no replacement fetch is made.

Search excerpts and fetched pages become separate sources, numbered after the local ones: local sources keep `[S1]…[Sk]` and web sources follow in retrieval order. Each web source's title holds its evidence type, page title and URL (for example `Search excerpt: Graph API overview — https://…`), so the existing citation check covers web IDs too, and the report ends with a note listing each web ID with its URL. If the report stops at the pass limit or with an invalid evaluation, its closing note names the web sources instead of saying that only the supplied material was consulted.

Limits: excerpts are clipped to 1,200 characters, pages to 3,000, and all web text to 12,000. Retrieval has a 120-second deadline on top of the client's 60-second per-call timeout, and it never retries or loops. Web text counts toward the context budget; local input that is already too large is rejected before any network request.

Safety:

- **URL checks.** Only `http`/`https` URLs are used. Rejected: credentials in the URL, `localhost` and `.local`/`.internal`-style hosts, single-label and numeric hosts, and IP addresses that are not globally routable (loopback, private, link-local such as `169.254.169.254`, shared, reserved, multicast, IPv4-mapped forms). URLs are normalized (lowercase scheme and host, ASCII host, no fragment, default port or `utm_*` tracking parameters, matched in any case; other query parameters keep their order and encoding) and deduplicated. Fetched content is accepted only when the URL the fetch reports normalizes to the URL that was requested, so redirects to other URLs are rejected. Parallel's servers do the fetching; this machine never connects to result URLs.
- **Untrusted content.** Retrieved text is evidence only. It is never split into further sources. Label-like text such as `[S1]` is rewritten, prompt delimiter tags are escaped, and leading `#` marks are escaped. Instructions in pages are not followed by the software, and the prompts tell the model to ignore them; a small local model may still be influenced.
- **Failures.** If connecting or searching fails, times out, returns malformed data or has no usable result, the run stops with exit code 1 before any model call. A failed, malformed, empty, low-quality or redirected fetch is skipped with a warning on stderr, and the search excerpts are still used. Nothing is invented for a failed call.

stderr shows any warnings and a one-line count (searches, fetches, sources added); retrieved text and settings are not logged. Tests in `tests/test_web_research.py` and `tests/test_cli.py` use an in-memory fake client and need no network.

## HTTP API

The API runs the same research as the CLI as background jobs, one at a time, and records every job in PostgreSQL so it can still be read after a restart. It binds to `127.0.0.1:8000` and has no authentication or CORS: keep it local.

### Prerequisites

- Ollama running with the configured model (see [Setup](#setup-windows-powershell)).
- A PostgreSQL database with UTF-8 encoding, for example in Docker:

  ```powershell
  docker run -d --name researchpilot-db -p 5432:5432 -e POSTGRES_USER=researchpilot -e POSTGRES_PASSWORD=<password> -e POSTGRES_DB=researchpilot postgres:17
  ```

### Configure, migrate and start

The API reads `RESEARCHPILOT_DATABASE_URL` from the environment or `.env`; it must use the `postgresql+psycopg://` scheme. Do not commit it.

```powershell
$env:RESEARCHPILOT_DATABASE_URL = "postgresql+psycopg://researchpilot:<password>@127.0.0.1:5432/researchpilot"
uv run alembic upgrade head        # explicit; the API never migrates the database itself
uv run python -m researchpilot.api
```

| Setting | Default | Meaning |
|---|---|---|
| `RESEARCHPILOT_DATABASE_URL` | none | Job history database; without it the API starts but refuses jobs (503 `history_unconfigured`) |
| `RESEARCHPILOT_API_HOST`, `RESEARCHPILOT_API_PORT` | `127.0.0.1`, `8000` | Listening address |
| `RESEARCHPILOT_API_MAX_QUEUED_JOBS` | `4` | Jobs waiting behind the running one |
| `RESEARCHPILOT_API_MAX_RETAINED_JOBS` | `50` | Finished jobs kept in memory (older ones are read from the database) |
| `RESEARCHPILOT_API_MAX_REQUEST_BYTES` | `1048576` | Largest accepted request body |

Interactive documentation is at `http://127.0.0.1:8000/docs` (Swagger UI), `/redoc`, and the schema at `/openapi.json`.

### Endpoints

Errors are `application/problem+json` bodies with a stable `code`; they never contain source text, exception messages or connection details.

| Request | Success | Common errors |
|---|---|---|
| `GET /api/v1/health` | 200 `{"status": "ok", ...}` | none: liveness only |
| `GET /api/v1/health/ready` | 200, checks `ollama`, `jobs`, `history` all `ok` | 503 `jobs_unavailable`, `history_unavailable`, `history_schema_outdated`, `history_unconfigured`, `ollama_unavailable` |
| `POST /api/v1/jobs` | 202, job status (`queued`) and a `Location` header | 422 `validation_failed` or `input_too_large`, 413 `content_too_large`, 503 `queue_full` (with `Retry-After`), `jobs_unavailable`, `history_*` |
| `GET /api/v1/jobs/{job_id}` | 200, job status | 404 `job_not_found`, 503 `history_*` |
| `GET /api/v1/jobs/{job_id}/result` | 200, the research result | 409 `job_not_finished`, `job_failed`, `job_cancelled`; 404; 503 |
| `POST /api/v1/jobs/{job_id}/cancel` | 200, job status (`cancelled`) | 409 `job_not_cancellable` (running or finished), 404, 503 |

A submission is `{"question": ..., "source_material": ..., "web": false, "label": ...}`: the same limits as the CLI apply, `web` adds one Parallel Search MCP retrieval, and the optional `label` (trimmed, at most 120 characters) is stored with the job but not returned.

Three different things are checked:

- **Liveness** (`/health`): the process is running.
- **Readiness** (`/health/ready`): jobs can be accepted and run now. History must be ready (database reachable, schema current, and this instance holding the single-instance lock) and Ollama must answer.
- **Job status** (`/jobs/{job_id}`): where one job is in its lifecycle: `queued`, `running`, then `succeeded`, `failed` or `cancelled`. A job left unfinished by a previous process is reported as `failed` with code `interrupted_by_restart`; it is never re-run.

### Manual test with Postman

Import `postman/ResearchPilot.postman_collection.json` and `postman/ResearchPilot.local.postman_environment.json` (File > Import; Bruno can import the same collection), then select the **ResearchPilot local** environment. Submitting a job runs the local model and can take several minutes on CPU; the web-mode request also contacts the public Parallel Search MCP server. Send requests by hand:

1. **Liveness**, then **Readiness**: expect 200 for both before going on.
2. **Submit research job**: 202; the job's id is saved as `jobId`.
3. **Job status**, repeatedly: `queued` and `running` are normal, not errors.
4. **Job result** once the status is `succeeded` (409 `job_not_finished` means poll again).
5. While a job is running, **Submit job to cancel** and then **Cancel job**: 200 `cancelled`. If the job had already started, the expected answer is 409 `job_not_cancellable`.
6. Restart the API (Ctrl+C, then start it again) and send **Job status** and **Job result** for the first job: they are now read from the database.

**Invalid submission** shows a 422 without running any research.

### Tests

The offline suite covers the API with fake research and fake history and never contacts Ollama, a database or the network: `uv run pytest`. PostgreSQL tests are skipped unless you point them at a **dedicated, disposable** database whose name contains `test` (they create and drop their own schema and never touch the application's database):

```powershell
$env:RESEARCHPILOT_TEST_DATABASE_URL = "postgresql+psycopg://researchpilot:<password>@127.0.0.1:5432/researchpilot_test"
uv run pytest tests/history/test_history_postgres.py tests/api/test_job_history_postgres.py -v
Remove-Item Env:RESEARCHPILOT_TEST_DATABASE_URL
```

### Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| 503 `history_unconfigured` | `RESEARCHPILOT_DATABASE_URL` is not set, or does not use `postgresql+psycopg://`. Set it and restart. |
| 503 `history_unavailable` | PostgreSQL is unreachable, or another API instance holds the lock. Start the database or stop the other instance; readiness retries every few seconds. |
| 503 `history_schema_outdated` | Migrations are missing, or the database is not UTF-8. Run `uv run alembic upgrade head`. |
| 503 `ollama_unavailable` | Start Ollama (`ollama serve`) and check `OLLAMA_BASE_URL`. |
| 503 `jobs_unavailable` | The worker stopped (a fatal failure or a lost database lock). Restart the API. |
| A job stays `running` | Local models are slow: allow several minutes. A job is never interrupted; if the API stops meanwhile, the job becomes `failed` with `interrupted_by_restart` at the next start. |
| 503 `history_unavailable` for one job | Its outcome could not be saved, so it is not shown as finished. After a restart it reads as `interrupted_by_restart`. |

## Structure

```text
src/researchpilot/
├── __init__.py
├── budget.py     # context-size estimates and limits
├── capture.py    # opt-in --save-sources evidence file
├── cli.py        # Typer CLI and Ollama model construction
├── config.py     # validated settings
├── graph.py      # LangGraph nodes, routing and compilation
├── parallel_mcp.py  # Parallel Search MCP client
├── web_research.py  # --web retrieval: URL checks, limits, web sources
├── prompts.py    # grounding rules and per-node prompts
├── report_checks.py  # heuristic review hints (never change the report)
├── schemas.py    # request and report models
├── service.py    # ResearchService: runs the graph
├── sources.py    # source numbering and citation-reference checks
├── state.py      # typed graph state and evaluator schema
├── api/          # FastAPI app, job manager, routes and history integration
└── history/      # PostgreSQL job history: models, store, lock and migrations
postman/          # Postman collection and local environment for manual API tests
tests/
├── fakes.py      # scripted model and fake Parallel client used by the tests
├── test_capture.py
├── test_cli.py
├── test_config.py
├── test_graph.py
├── test_parallel_mcp.py
├── test_prompts.py
├── test_report_checks.py
├── test_service.py
├── test_sources.py
├── test_web_research.py
└── live/
    ├── test_live_parallel_mcp.py  # opt-in, calls the Parallel MCP server
    └── test_live_quality.py  # opt-in, needs Ollama
```

## Known limitations

- Source retrieval is manual unless `--web` is used, and web retrieval is basic: one keyword query taken from the question, with no query planning or follow-up searches.
- Web content is untrusted. The defenses reduce prompt injection but cannot rule it out with a small local model, and a page that redirects is dropped even when the redirect is harmless.
- Output is Markdown, not yet a strict structured report schema.
- Citation IDs are checked, but whether a cited source actually supports a claim is not verified. Review hints catch only a few word patterns of over-generalization and contradiction; semantic errors (such as misdescribing a technical term) are not detected in code.
- The completeness check is a judgement by the same local model, not verification. The JSON schema guarantees the shape of its answer, not its accuracy: `gemma3:4b` often reports gaps that are not real or that belong under source gaps, so runs frequently stop at the pass limit with a note listing those gaps. Its gap lists can also vary between runs on identical input.
- Token counts are estimated from character length, not measured with the model's tokenizer, and may be off for non-English text or code.
- A run makes 3–5 model calls by default, so it is slower than a single call.
- No tracing or UI yet. The HTTP API is single-instance and local-only, with no authentication and no job listing.
