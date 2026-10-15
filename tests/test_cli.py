"""CLI tests with a scripted model; Ollama is never contacted."""

import hashlib
import json
import logging
import os
from pathlib import Path

import pytest
from fakes import (
    FakeParallelClient,
    ScriptedModel,
    complete,
    incomplete,
    search_item,
    tool_result,
)
from typer.testing import CliRunner

from researchpilot import capture, cli, parallel_mcp, web_research
from researchpilot.config import Settings
from researchpilot.parallel_mcp import ParallelMCPError
from researchpilot.state import EVALUATION_JSON_SCHEMA

REPORT = "# Research Report\n\n## Executive Summary\n\nScripted CLI report [S1]."
SOURCES = "Source A: In a fictional test, System Alpha completed 85 of 100 tasks successfully."

runner = CliRunner()


@pytest.fixture
def model(monkeypatch: pytest.MonkeyPatch) -> ScriptedModel:
    scripted = ScriptedModel(["- notes", complete(), REPORT])
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        ollama_num_ctx=16_384,
        ollama_num_predict=2_048,
        researchpilot_max_analysis_passes=2,
    )
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr(cli, "build_model", lambda _settings, **_kwargs: scripted)
    return scripted


@pytest.fixture
def source_file(tmp_path: Path) -> Path:
    path = tmp_path / "sources.md"
    path.write_text(SOURCES, encoding="utf-8")
    return path


def test_cli_prints_report(model: ScriptedModel, source_file: Path) -> None:
    result = runner.invoke(cli.app, ["Summarize the evidence", "--source-file", str(source_file)])

    assert result.exit_code == 0, result.output
    assert "Scripted CLI report [S1]." in result.output
    assert model.call_count == 3


def test_cli_saves_report(model: ScriptedModel, source_file: Path, tmp_path: Path) -> None:
    output = tmp_path / "reports" / "report.md"

    result = runner.invoke(
        cli.app,
        ["Summarize the evidence", "--source-file", str(source_file), "--output", str(output)],
    )

    assert result.exit_code == 0, result.output
    assert output.read_text(encoding="utf-8") == REPORT + "\n"


def test_cli_rejects_short_question(model: ScriptedModel, source_file: Path) -> None:
    result = runner.invoke(cli.app, ["Why?", "--source-file", str(source_file)])

    assert result.exit_code == 2
    assert "Input validation failed" in result.output
    assert model.call_count == 0


def test_cli_rejects_source_too_large_for_context(model: ScriptedModel, tmp_path: Path) -> None:
    large = tmp_path / "large.md"
    large.write_text("Source A: evidence sentence. " * 2_000, encoding="utf-8")

    result = runner.invoke(cli.app, ["Summarize the evidence", "--source-file", str(large)])

    assert result.exit_code == 2
    assert "Input too large for the model context" in result.output
    assert model.call_count == 0


def test_cli_gives_evaluator_the_json_schema(
    monkeypatch: pytest.MonkeyPatch, model: ScriptedModel, source_file: Path
) -> None:
    schemas: list[dict | None] = []

    def fake_build_model(_settings, *, json_schema=None):
        schemas.append(json_schema)
        return model

    monkeypatch.setattr(cli, "build_model", fake_build_model)

    result = runner.invoke(cli.app, ["Summarize the evidence", "--source-file", str(source_file)])

    assert result.exit_code == 0, result.output
    assert schemas == [None, EVALUATION_JSON_SCHEMA]


def test_cli_prints_run_summary(model: ScriptedModel, source_file: Path) -> None:
    result = runner.invoke(cli.app, ["Summarize the evidence", "--source-file", str(source_file)])

    assert result.exit_code == 0, result.output
    assert "1 analysis pass(es), 3 model call(s)" in result.output
    assert "stop reason: evaluation_complete" in result.output


def test_cli_verbose_enables_timing_logs(
    monkeypatch: pytest.MonkeyPatch, model: ScriptedModel, source_file: Path
) -> None:
    package_logger = logging.getLogger("researchpilot")
    monkeypatch.setattr(package_logger, "handlers", [])
    monkeypatch.setattr(package_logger, "level", logging.NOTSET)

    result = runner.invoke(
        cli.app, ["Summarize the evidence", "--source-file", str(source_file), "--verbose"]
    )

    assert result.exit_code == 0, result.output
    assert package_logger.level == logging.INFO
    assert len(package_logger.handlers) == 1


def test_cli_keeps_report_and_shows_citation_warning(
    monkeypatch: pytest.MonkeyPatch, model: ScriptedModel, source_file: Path, tmp_path: Path
) -> None:
    flagged = ScriptedModel(["- notes", complete(), "# Research Report\n\nClaim [S9]."])
    monkeypatch.setattr(cli, "build_model", lambda _settings, **_kwargs: flagged)
    output = tmp_path / "report.md"

    result = runner.invoke(
        cli.app,
        ["Summarize the evidence", "--source-file", str(source_file), "--output", str(output)],
    )

    assert result.exit_code == 0, result.output
    saved = output.read_text(encoding="utf-8")
    assert saved.startswith(
        "# Research Report\n\nClaim [S9].\n\n> **ResearchPilot citation check:**"
    )
    assert "[S9] is not a supplied source ID" in result.output


WEB_URL = "https://docs.langchain.com/oss/python/langgraph/graph-api"


def _no_mcp(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Make any MCP SDK client construction fail loudly, and record attempts."""
    attempts: list[str] = []

    def refuse(endpoint: str, *args, **kwargs):
        attempts.append(endpoint)
        raise AssertionError("an MCP client was constructed")

    monkeypatch.setattr(parallel_mcp, "Client", refuse)
    return attempts


def test_offline_run_makes_no_mcp_connection(
    monkeypatch: pytest.MonkeyPatch, model: ScriptedModel, source_file: Path
) -> None:
    attempts = _no_mcp(monkeypatch)

    result = runner.invoke(cli.app, ["Summarize the evidence", "--source-file", str(source_file)])

    assert result.exit_code == 0, result.output
    assert attempts == []
    assert "Web:" not in result.output
    assert "web sources" not in result.output.lower()


def test_web_run_adds_web_sources_to_analysis_and_report(
    monkeypatch: pytest.MonkeyPatch, source_file: Path, tmp_path: Path
) -> None:
    scripted = ScriptedModel(["- notes [S2]", complete(), "# Research Report\n\nRouting [S2]."])
    monkeypatch.setattr(cli, "build_model", lambda _settings, **_kwargs: scripted)
    fake = FakeParallelClient(
        tool_result({"results": [search_item(WEB_URL, "Graph API overview", "Route with edges.")]})
    )
    monkeypatch.setattr(web_research, "ParallelSearchMCPClient", lambda: fake)
    output = tmp_path / "report.md"

    result = runner.invoke(
        cli.app,
        ["Summarize the evidence", "-s", str(source_file), "--web", "--output", str(output)],
    )

    assert result.exit_code == 0, result.output
    excerpt = f"[S2]\n# Search excerpt: Graph API overview — {WEB_URL}\nRoute with edges."
    assert excerpt in scripted.prompts[0]
    # The fetched page keeps the title reported by the fetch result.
    assert f"[S3]\n# Fetched page: Page title — {WEB_URL}" in scripted.prompts[0]
    saved = output.read_text(encoding="utf-8")
    assert saved.startswith(
        "# Research Report\n\nRouting [S2].\n\n> **ResearchPilot web sources:**"
    )
    assert WEB_URL in saved
    assert "ResearchPilot citation check" not in saved
    assert "Web: 1 search call(s), 1 fetch call(s); 2 source(s) added" in result.output
    assert fake.closed


def test_web_search_failure_exits_before_any_model_call(
    monkeypatch: pytest.MonkeyPatch, model: ScriptedModel, source_file: Path
) -> None:
    fake = FakeParallelClient(ParallelMCPError("Parallel MCP tool 'web_search' timed out."))
    monkeypatch.setattr(web_research, "ParallelSearchMCPClient", lambda: fake)

    result = runner.invoke(
        cli.app, ["Summarize the evidence", "--source-file", str(source_file), "--web"]
    )

    assert result.exit_code == 1
    assert "Web retrieval failed" in result.output
    assert "No model was called" in result.output
    assert "Research run failed" not in result.output
    assert model.call_count == 0


def test_web_run_rejects_oversized_local_input_before_retrieval(
    monkeypatch: pytest.MonkeyPatch, model: ScriptedModel, tmp_path: Path
) -> None:
    attempts = _no_mcp(monkeypatch)
    large = tmp_path / "large.md"
    large.write_text("Source A: evidence sentence. " * 2_000, encoding="utf-8")

    result = runner.invoke(cli.app, ["Summarize the evidence", "-s", str(large), "--web"])

    assert result.exit_code == 2
    assert "Input too large for the model context" in result.output
    assert attempts == []
    assert model.call_count == 0


def test_cli_summary_counts_unresolved_gaps_only_when_present(
    monkeypatch: pytest.MonkeyPatch, model: ScriptedModel, source_file: Path
) -> None:
    complete_run = runner.invoke(cli.app, ["Summarize the evidence", "-s", str(source_file)])
    assert complete_run.exit_code == 0, complete_run.output
    assert "unresolved analysis gap" not in complete_run.output

    gaps = incomplete(["Source A sample size is not covered."])
    limited = ScriptedModel(["- notes", gaps, "- notes 2", gaps, REPORT])
    monkeypatch.setattr(cli, "build_model", lambda _settings, **_kwargs: limited)

    result = runner.invoke(cli.app, ["Summarize the evidence", "-s", str(source_file)])

    assert result.exit_code == 0, result.output
    # Rich wraps stderr at the terminal width, so compare with whitespace collapsed.
    output = " ".join(result.output.split())
    assert "stop reason: pass_limit_reached; 1 unresolved analysis gap(s)" in output


def test_review_hints_do_not_change_exit_code_or_citation_results(
    monkeypatch: pytest.MonkeyPatch, source_file: Path, tmp_path: Path
) -> None:
    model_text = "# Research Report\n\nSystems often complete most tasks [S1]."
    hinted = ScriptedModel(["- notes", complete(), model_text])
    monkeypatch.setattr(cli, "build_model", lambda _settings, **_kwargs: hinted)
    output = tmp_path / "report.md"

    result = runner.invoke(
        cli.app, ["Summarize the evidence", "-s", str(source_file), "--output", str(output)]
    )

    assert result.exit_code == 0, result.output
    saved = output.read_text(encoding="utf-8")
    assert saved.startswith(model_text + "\n\n> **ResearchPilot review hints (heuristic):**")
    assert "ResearchPilot citation check" not in saved


# --- Opt-in evidence capture (--save-sources) ---


def files_in(folder: Path) -> list[str]:
    return sorted(path.name for path in folder.iterdir())


def flat(output: str) -> str:
    """CLI output with Rich's line wrapping collapsed."""
    return " ".join(output.split())


def test_no_evidence_file_without_the_flag(model: ScriptedModel, source_file: Path) -> None:
    before = files_in(source_file.parent)

    result = runner.invoke(cli.app, ["Summarize the evidence", "-s", str(source_file)])

    assert result.exit_code == 0, result.output
    assert files_in(source_file.parent) == before
    assert "Evidence saved" not in result.output


def test_local_run_saves_evidence_without_printing_source_text(
    model: ScriptedModel, source_file: Path, tmp_path: Path
) -> None:
    evidence = tmp_path / "evidence.json"
    args = ["Summarize the evidence", "-s", str(source_file), "--save-sources", str(evidence)]

    result = runner.invoke(cli.app, args)

    assert result.exit_code == 0, result.output
    data = json.loads(evidence.read_text(encoding="utf-8"))
    assert data["schema_version"] == 2
    assert data["source_mode"] == "local"
    assert [(s["id"], s["type"], s["text"]) for s in data["sources"]] == [("S1", "local", SOURCES)]
    assert data["analysis_passes"] == [{"pass": 1, "notes": "- notes"}]
    assert data["report_metadata"]["model_calls"] == model.call_count == 3
    assert "Evidence saved to" in flat(result.output)
    assert "(1 source(s), 1 analysis pass(es))" in flat(result.output)
    assert SOURCES not in result.output
    assert files_in(tmp_path) == ["evidence.json", "sources.md"]


def test_web_run_saves_web_sources_and_retrieval_metadata(
    monkeypatch: pytest.MonkeyPatch, source_file: Path, tmp_path: Path
) -> None:
    scripted = ScriptedModel(["- notes [S2]", complete(), "# Research Report\n\nRouting [S2]."])
    monkeypatch.setattr(cli, "build_model", lambda _settings, **_kwargs: scripted)
    fake = FakeParallelClient(
        tool_result({"results": [search_item(WEB_URL, "Graph API overview", "Route with edges.")]})
    )
    monkeypatch.setattr(web_research, "ParallelSearchMCPClient", lambda: fake)
    evidence = tmp_path / "evidence.json"
    args = ["Summarize the evidence", "-s", str(source_file), "--web"]

    result = runner.invoke(cli.app, [*args, "--save-sources", str(evidence)])

    assert result.exit_code == 0, result.output
    data = json.loads(evidence.read_text(encoding="utf-8"))
    assert data["source_mode"] == "local+web"
    web = [s for s in data["sources"] if s["type"] == "web"]
    assert [(s["id"], s["evidence_type"], s["url"]) for s in web] == [
        ("S2", "search_excerpt", WEB_URL),
        ("S3", "fetched_page", WEB_URL),
    ]
    assert web[0]["page_title"] == "Graph API overview"
    assert web[0]["numbered_title"] == f"Search excerpt: Graph API overview — {WEB_URL}"
    assert web[0]["text"].endswith("Route with edges.")
    assert web[0]["text"] in scripted.prompts[0]
    assert data["retrieval"] == {"search_calls": 1, "fetch_calls": 1, "warnings": []}
    # The excerpt reaches the terminal only through the model's report, never from capture.
    assert "Route with edges." not in result.output


@pytest.mark.parametrize("web", [False, True])
def test_existing_destination_is_refused_before_any_model_or_network_call(
    monkeypatch: pytest.MonkeyPatch, model: ScriptedModel, source_file: Path, web: bool
) -> None:
    attempts = _no_mcp(monkeypatch)
    existing = source_file.parent / "evidence.json"
    existing.write_text("keep me", encoding="utf-8")
    args = ["Summarize the evidence", "-s", str(source_file), "--save-sources", str(existing)]

    result = runner.invoke(cli.app, [*args, "--web"] if web else args)

    assert result.exit_code == 2
    assert "already exists; refusing to overwrite it" in flat(result.output)
    assert existing.read_text(encoding="utf-8") == "keep me"
    assert model.call_count == 0
    assert attempts == []


def test_destination_cannot_be_the_source_file_or_report_output(
    model: ScriptedModel, tmp_path: Path
) -> None:
    source = tmp_path / "sources.json"
    source.write_text(SOURCES, encoding="utf-8")
    report = tmp_path / "report.json"
    base = ["Summarize the evidence", "-s", str(source)]

    same_as_source = runner.invoke(cli.app, [*base, "--save-sources", str(source)])
    same_as_output = runner.invoke(
        cli.app, [*base, "-o", str(report), "--save-sources", str(report)]
    )

    assert same_as_source.exit_code == 2
    assert "is the source file" in flat(same_as_source.output)
    assert same_as_output.exit_code == 2
    assert "is the report output path" in flat(same_as_output.output)
    assert source.read_text(encoding="utf-8") == SOURCES
    assert not report.exists()
    assert model.call_count == 0


@pytest.mark.parametrize(
    ("relative", "message"),
    [
        ("missing/evidence.json", "does not exist; create it first"),
        ("evidence.md", "must end in .json"),
    ],
)
def test_invalid_destinations_exit_2_without_model_calls(
    model: ScriptedModel, source_file: Path, tmp_path: Path, relative: str, message: str
) -> None:
    destination = tmp_path / relative
    args = ["Summarize the evidence", "-s", str(source_file), "--save-sources", str(destination)]

    result = runner.invoke(cli.app, args)

    assert result.exit_code == 2
    assert message in flat(result.output)
    assert model.call_count == 0
    assert not (tmp_path / "missing").exists()


def test_write_failure_exits_1_after_saving_the_report(
    monkeypatch: pytest.MonkeyPatch, model: ScriptedModel, source_file: Path, tmp_path: Path
) -> None:
    def failing_fsync(_fd: int) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(capture.os, "fsync", failing_fsync)
    report = tmp_path / "report.md"
    evidence = tmp_path / "evidence.json"
    args = ["Summarize the evidence", "-s", str(source_file), "-o", str(report)]

    result = runner.invoke(cli.app, [*args, "--save-sources", str(evidence)])

    assert result.exit_code == 1
    assert "Evidence file not saved:" in flat(result.output)
    assert "No space left on device" in flat(result.output)
    assert report.read_text(encoding="utf-8") == REPORT + "\n"
    assert files_in(tmp_path) == ["report.md", "sources.md"]


def test_capture_does_not_change_report_prompts_or_exit_code(
    monkeypatch: pytest.MonkeyPatch, source_file: Path, tmp_path: Path
) -> None:
    results = {}
    for name, extra in (("plain", []), ("captured", ["--save-sources", str(tmp_path / "e.json")])):
        scripted = ScriptedModel(["- notes", incomplete(["gap"]), "- notes 2", complete(), REPORT])
        monkeypatch.setattr(cli, "build_model", lambda _settings, _m=scripted, **_kwargs: _m)
        report = tmp_path / f"{name}.md"
        args = ["Summarize the evidence", "-s", str(source_file), "-o", str(report), *extra]
        result = runner.invoke(cli.app, args)
        results[name] = (result.exit_code, report.read_text(encoding="utf-8"), scripted.prompts)

    assert results["plain"] == results["captured"]
    data = json.loads((tmp_path / "e.json").read_text(encoding="utf-8"))
    assert [s["id"] for s in data["sources"]] == ["S1"]
    assert [p["notes"] for p in data["analysis_passes"]] == ["- notes", "- notes 2"]


def test_failed_run_writes_no_evidence_file(
    monkeypatch: pytest.MonkeyPatch, source_file: Path, tmp_path: Path
) -> None:
    short = ScriptedModel(["- notes"])  # the evaluator call has no scripted reply and raises
    monkeypatch.setattr(cli, "build_model", lambda _settings, **_kwargs: short)
    evidence = tmp_path / "evidence.json"
    args = ["Summarize the evidence", "-s", str(source_file), "--save-sources", str(evidence)]

    result = runner.invoke(cli.app, args)

    assert result.exit_code == 1
    assert "Research run failed" in result.output
    assert files_in(tmp_path) == ["sources.md"]


def test_evidence_file_excludes_settings_secrets_and_environment(
    monkeypatch: pytest.MonkeyPatch, source_file: Path, tmp_path: Path
) -> None:
    monkeypatch.setenv("RESEARCHPILOT_TEST_TOKEN", "tok-should-not-appear")
    secret_settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        ollama_base_url="http://user:hunter2@ollama.internal:11434",
    )
    monkeypatch.setattr(cli, "get_settings", lambda: secret_settings)
    scripted = ScriptedModel(["- notes", complete(), REPORT])
    monkeypatch.setattr(cli, "build_model", lambda _settings, **_kwargs: scripted)
    evidence = tmp_path / "evidence.json"
    args = ["Summarize the evidence", "-s", str(source_file), "--save-sources", str(evidence)]

    result = runner.invoke(cli.app, args)

    assert result.exit_code == 0, result.output
    text = evidence.read_text(encoding="utf-8").lower()
    for secret in ("hunter2", "ollama.internal", "tok-should-not-appear", "authorization"):
        assert secret not in text


def test_output_file_hash_matches_the_report_bytes_actually_written(
    model: ScriptedModel, source_file: Path, tmp_path: Path
) -> None:
    report = tmp_path / "report.md"
    evidence = tmp_path / "evidence.json"
    args = ["Summarize the evidence", "-s", str(source_file), "-o", str(report)]

    result = runner.invoke(cli.app, [*args, "--save-sources", str(evidence)])

    assert result.exit_code == 0, result.output
    metadata = json.loads(evidence.read_text(encoding="utf-8"))["report_metadata"]
    written = report.read_bytes()
    assert metadata["output_file_sha256"] == hashlib.sha256(written).hexdigest()
    assert metadata["markdown_sha256"] == hashlib.sha256(REPORT.encode("utf-8")).hexdigest()
    # The file holds the Markdown plus one newline, in the platform's line endings.
    assert written.decode("utf-8").replace("\r\n", "\n") == REPORT + "\n"
    assert written == (REPORT + "\n").replace("\n", os.linesep).encode("utf-8")
    if os.name == "nt":
        assert written.count(b"\r\n") == REPORT.count("\n") + 1
        assert metadata["output_file_sha256"] != metadata["markdown_sha256"]


def test_output_file_hash_is_null_without_a_report_file(
    model: ScriptedModel, source_file: Path, tmp_path: Path
) -> None:
    evidence = tmp_path / "evidence.json"
    args = ["Summarize the evidence", "-s", str(source_file), "--save-sources", str(evidence)]

    result = runner.invoke(cli.app, args)

    assert result.exit_code == 0, result.output
    metadata = json.loads(evidence.read_text(encoding="utf-8"))["report_metadata"]
    assert metadata["output_path"] is None
    assert metadata["output_file_sha256"] is None


def test_report_read_back_failure_exits_1_and_writes_no_evidence(
    monkeypatch: pytest.MonkeyPatch, model: ScriptedModel, source_file: Path, tmp_path: Path
) -> None:
    def unreadable(path: Path) -> str:
        raise capture.CaptureError(f"Could not read back the report file {path}: denied")

    monkeypatch.setattr(cli, "file_sha256", unreadable)
    report = tmp_path / "report.md"
    evidence = tmp_path / "evidence.json"
    args = ["Summarize the evidence", "-s", str(source_file), "-o", str(report)]

    result = runner.invoke(cli.app, [*args, "--save-sources", str(evidence)])

    assert result.exit_code == 1
    assert "Evidence file not saved:" in flat(result.output)
    assert "Could not read back the report file" in flat(result.output)
    assert report.read_text(encoding="utf-8") == REPORT + "\n"
    assert files_in(tmp_path) == ["report.md", "sources.md"]
