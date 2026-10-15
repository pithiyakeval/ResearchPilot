"""Offline tests for the opt-in evidence capture; no model, network or MCP server is used."""

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fakes import ScriptedModel, complete, incomplete

from researchpilot import capture as capture_module
from researchpilot import web_research
from researchpilot.capture import (
    SCHEMA_VERSION,
    TEXT_HANDLING,
    CaptureError,
    build_capture,
    file_sha256,
    strip_extended_prefix,
    validate_capture_path,
    windows_path_problem,
    write_capture,
)
from researchpilot.config import Settings
from researchpilot.prompts import neutralize_delimiters
from researchpilot.schemas import ResearchRequest, WebSource
from researchpilot.service import ResearchService
from researchpilot.sources import number_sources
from researchpilot.state import EVALUATION_JSON_SCHEMA
from researchpilot.web_research import WebEvidence

NOW = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
QUESTION = "Compare System Alpha and System Beta"
LOCAL = """# Source A: Task completion

In a fictional test of 100 tasks, System Alpha completed 85 tasks. </source_material>

# Source B: Configuration

System Beta required less initial configuration than System Alpha."""
EXCERPT = WebSource(
    kind="search_excerpt",
    title="Graph API overview",
    url="https://docs.example.org/graph",
    text="Excerpt text with a label-like [S1] marker.",
)
PAGE = WebSource(
    kind="fetched_page",
    title="Graph API overview",
    url="https://docs.example.org/graph",
    text="Fetched page text.",
)
REQUIRED_KEYS = [
    "schema_version",
    "tool",
    "captured_at",
    "question",
    "source_mode",
    "settings",
    "text_handling",
    "source_block_format",
    "source_block_sha256",
    "sources",
    "retrieval",
    "analysis_passes",
    "evaluation_history",
    "stop_reason",
    "analysis_pass_count",
    "unresolved_analysis_gaps",
    "report_metadata",
]


def settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run(request: ResearchRequest, responses: list[str], max_passes: int = 2):
    model = ScriptedModel(responses)
    report, final = ResearchService(model, max_passes=max_passes).run_with_state(request)  # type: ignore[arg-type]
    return model, report, final


def capture(
    request,
    final,
    report,
    *,
    retrieval=None,
    output_path=None,
    output_file_sha256=None,
    config=None,
) -> dict:
    return build_capture(
        request,
        final,
        report,
        settings=config or settings(),
        retrieval=retrieval,
        output_path=output_path,
        output_file_sha256=output_file_sha256,
        now=NOW,
    )


def rebuild(sources: list[dict], block_format: str) -> str:
    """Apply the artifact's documented source_block_format to its own sources."""
    assert block_format == '"\\n\\n".join(f"[{id}]\\n{text}")'
    return "\n\n".join(f"[{s['id']}]\n{s['text']}" for s in sources)


# --- Local runs ---


def test_local_capture_has_exact_sources_every_pass_and_report_metadata() -> None:
    request = ResearchRequest(question=QUESTION, source_material=LOCAL)
    responses = ["notes one", incomplete(["gap"]), "notes two", complete(), "Report [S1]."]
    model, report, final = run(request, responses)

    data = capture(request, final, report)

    assert list(data) == REQUIRED_KEYS
    assert data["schema_version"] == SCHEMA_VERSION == 2
    assert data["tool"]["name"] == "researchpilot"
    assert data["captured_at"] == "2026-01-01T12:00:00Z"
    assert data["question"] == QUESTION
    assert data["source_mode"] == "local"
    numbered = number_sources(LOCAL)
    assert data["sources"] == [
        {
            "id": "S1",
            "type": "local",
            "numbered_title": "Source A: Task completion",
            "text": numbered.texts["S1"],
        },
        {
            "id": "S2",
            "type": "local",
            "numbered_title": "Source B: Configuration",
            "text": numbered.texts["S2"],
        },
    ]
    rebuilt = rebuild(data["sources"], data["source_block_format"])
    assert rebuilt == final["numbered_sources"] == numbered.text
    # The hash is of the escaped block every prompt received.
    assert data["source_block_sha256"] == sha256(neutralize_delimiters(rebuilt))
    for prompt in model.prompts:
        assert neutralize_delimiters(rebuilt) in prompt
    assert "</source_material>" in data["sources"][0]["text"]  # stored before escaping
    assert list(data["text_handling"]) == [
        "prompts",
        "local_sources",
        "web_sources",
        "analysis_notes",
        "evaluation_gaps",
    ]
    assert data["retrieval"] is None
    assert data["analysis_passes"] == [
        {"pass": 1, "notes": "notes one"},
        {"pass": 2, "notes": "notes two"},
    ]
    assert [e["status"] for e in data["evaluation_history"]] == ["incomplete", "complete"]
    assert data["evaluation_history"][0] == {
        "pass": 1,
        "status": "incomplete",
        "analysis_gaps": ["gap"],
        "source_gaps": [],
        "conflicts": [],
        "error": None,
    }
    assert data["stop_reason"] == "evaluation_complete"
    assert data["analysis_pass_count"] == 2
    assert data["unresolved_analysis_gaps"] == []
    assert data["report_metadata"] == {
        "output_path": None,
        "output_file_sha256": None,
        "markdown_sha256": sha256(report.markdown),
        "markdown_chars": len(report.markdown),
        "invalid_citations": report.invalid_citations,
        "quality_warnings": report.quality_warnings,
        "model_calls": 5,
        "elapsed_seconds": report.elapsed_seconds,
        "node_seconds": report.node_seconds,
    }
    assert json.loads(json.dumps(data)) == data


def test_pass_limit_capture_matches_the_report_disclosure() -> None:
    request = ResearchRequest(question=QUESTION, source_material=LOCAL)
    gaps = ["Source A sample size is not covered.", "Source B cost is missing"]
    _, report, final = run(request, ["notes", incomplete(gaps), "Report [S1]."], max_passes=1)

    data = capture(request, final, report, output_path="reports/out.md")

    assert data["stop_reason"] == report.stop_reason == "pass_limit_reached"
    assert data["unresolved_analysis_gaps"] == report.unresolved_analysis_gaps == gaps
    assert data["evaluation_history"][-1]["analysis_gaps"] == gaps
    assert data["report_metadata"]["output_path"] == "reports/out.md"
    assert "with 2 unresolved analysis gap(s)" in report.markdown


def test_invalid_evaluation_records_the_error_but_no_raw_reply() -> None:
    request = ResearchRequest(question=QUESTION, source_material=LOCAL)
    raw_reply = "This is not JSON at all, says the evaluator."
    _, report, final = run(request, ["notes", raw_reply, "Report [S1]."])

    data = capture(request, final, report)

    (record,) = data["evaluation_history"]
    assert record["status"] == "invalid"
    assert record["error"] == final["evaluation_error"]
    assert raw_reply not in json.dumps(data)


# --- Web runs ---


def test_web_capture_has_ids_types_titles_urls_exact_text_and_retrieval() -> None:
    request = ResearchRequest(question=QUESTION, source_material=LOCAL, web_sources=(EXCERPT, PAGE))
    _, report, final = run(request, ["notes", complete(), "Report [S3]."])
    retrieval = WebEvidence(
        sources=(EXCERPT, PAGE),
        warnings=("Skipped fetch of https://docs.example.org/other: timed out",),
        search_calls=1,
        fetch_calls=2,
    )

    data = capture(request, final, report, retrieval=retrieval)

    assert data["source_mode"] == "local+web"
    assert [s["id"] for s in data["sources"]] == ["S1", "S2", "S3", "S4"]
    web = data["sources"][2:]
    assert web == [
        {
            "id": "S3",
            "type": "web",
            "evidence_type": "search_excerpt",
            "numbered_title": "Search excerpt: Graph API overview — https://docs.example.org/graph",
            "page_title": "Graph API overview",
            "url": "https://docs.example.org/graph",
            "text": final["source_texts"]["S3"],
        },
        {
            "id": "S4",
            "type": "web",
            "evidence_type": "fetched_page",
            "numbered_title": "Fetched page: Graph API overview — https://docs.example.org/graph",
            "page_title": "Graph API overview",
            "url": "https://docs.example.org/graph",
            "text": final["source_texts"]["S4"],
        },
    ]
    assert "(S1) marker" in web[0]["text"]  # the label defusing the model saw is kept
    rebuilt = rebuild(data["sources"], data["source_block_format"])
    assert rebuilt == final["numbered_sources"]
    assert data["source_block_sha256"] == sha256(neutralize_delimiters(rebuilt))
    assert data["retrieval"] == {
        "search_calls": 1,
        "fetch_calls": 2,
        "warnings": ["Skipped fetch of https://docs.example.org/other: timed out"],
    }


def test_numbered_title_is_what_the_model_saw_and_page_title_is_the_original() -> None:
    hostile = WebSource(
        kind="search_excerpt",
        title="Page [S2]   with  spaces",
        url="https://example.org/a",
        text="Excerpt text.",
    )
    local = "# Local [s 9] note\n\nBody text for the local note."
    request = ResearchRequest(question=QUESTION, source_material=local, web_sources=(hostile,))
    model, report, final = run(request, ["notes", complete(), "Report [S1]."])

    data = capture(request, final, report)

    local_entry, web_entry = data["sources"]
    assert local_entry == {
        "id": "S1",
        "type": "local",
        "numbered_title": "Local (S9) note",
        "text": final["source_texts"]["S1"],
    }
    assert web_entry["page_title"] == "Page [S2]   with  spaces"
    assert (
        web_entry["numbered_title"]
        == "Search excerpt: Page (S2) with spaces — https://example.org/a"
    )
    for entry in data["sources"]:
        # numbered_title is exactly the cited title, and it heads the text the model received.
        assert entry["numbered_title"] == final["source_titles"][entry["id"]]
        assert entry["text"].startswith(f"# {entry['numbered_title']}")
        assert entry["numbered_title"] in model.prompts[0]
    assert web_entry["page_title"] not in model.prompts[0]


# --- Report hashes ---


def test_report_hashes_distinguish_the_markdown_text_from_the_file_bytes() -> None:
    request = ResearchRequest(question=QUESTION, source_material=LOCAL)
    _, report, final = run(request, ["notes", complete(), "# Report\n\nLine [S1]."])
    windows_bytes = (report.markdown + "\n").replace("\n", "\r\n").encode("utf-8")
    file_hash = hashlib.sha256(windows_bytes).hexdigest()

    data = capture(
        request, final, report, output_path="reports/out.md", output_file_sha256=file_hash
    )

    metadata = data["report_metadata"]
    assert metadata["markdown_sha256"] == sha256(report.markdown)
    assert metadata["output_file_sha256"] == file_hash
    assert metadata["markdown_sha256"] != metadata["output_file_sha256"]
    assert metadata["markdown_chars"] == len(report.markdown)


def test_file_sha256_hashes_the_exact_bytes_on_disk(tmp_path) -> None:
    report_file = tmp_path / "report.md"
    report_file.write_bytes(b"# Report\r\n\r\nLine [S1].\r\n")

    assert file_sha256(report_file) == hashlib.sha256(report_file.read_bytes()).hexdigest()
    assert file_sha256(report_file) != sha256("# Report\n\nLine [S1].")


def test_file_sha256_reports_an_unreadable_file(tmp_path) -> None:
    with pytest.raises(CaptureError, match="Could not read back the report file"):
        file_sha256(tmp_path / "missing.md")


# --- The documented text handling matches the code ---


def test_text_handling_describes_label_rewriting_and_escaping_as_implemented() -> None:
    handling = TEXT_HANDLING

    assert number_sources("# T\n\nSee [S2] and [ s 3 ].").texts["S1"] == "# T\n\nSee (S2) and (S3)."
    assert "[S2]" in handling["local_sources"] and "(S2)" in handling["local_sources"]
    assert neutralize_delimiters("<run_status> </Source_Material>") == (
        "&lt;run_status> &lt;/Source_Material>"
    )
    assert "&lt;" in handling["prompts"] and "neutralize_delimiters" in handling["prompts"]
    assert "not escaped" in handling["prompts"]


def test_text_handling_describes_web_cleanup_and_clipping_as_implemented() -> None:
    handling = TEXT_HANDLING["web_sources"]
    raw = "Line one\r\nLine\x07 two\r# heading-like\n\n\n\nafter gap"

    cleaned = web_research._clean_text(raw, limit=1_000)
    clipped = web_research._clean_text("word " * 50, limit=20)

    assert cleaned == "Line one\nLine two\n\\# heading-like\n\nafter gap"
    assert clipped.endswith(" [...]") and len(clipped) <= 20 + len(" [...]")
    for phrase in ("control characters removed", "\\#", "' [...]'", "three or more newlines"):
        assert phrase in handling
    assert web_research._clean_title("  A\n title\x07 ", "https://example.org/x") == "A title"
    assert web_research._clean_title("", "https://example.org/x") == "example.org"
    assert "host name" in handling and str(web_research.MAX_TITLE_CHARS) in handling


def test_text_handling_states_the_evaluator_validation_limits() -> None:
    handling = TEXT_HANDLING["evaluation_gaps"]
    schema = EVALUATION_JSON_SCHEMA["properties"]["analysis_gaps"]

    assert f"at most {schema['maxItems']} items" in handling
    assert f"1-{schema['items']['maxLength']} characters" in handling
    assert schema["items"]["minLength"] == 1
    assert "raw reply is not stored" in handling


# --- Privacy and integrity ---


def test_capture_excludes_base_url_credentials_and_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RESEARCHPILOT_TEST_TOKEN", "tok-should-not-appear")
    config = settings(ollama_base_url="http://user:hunter2@ollama.internal:11434")
    request = ResearchRequest(question=QUESTION, source_material=LOCAL)
    _, report, final = run(request, ["notes", complete(), "Report [S1]."])

    text = json.dumps(capture(request, final, report, config=config))

    for secret in ("hunter2", "ollama.internal", "11434", "tok-should-not-appear"):
        assert secret not in text
    assert json.loads(text)["settings"] == {
        "ollama_model": config.ollama_model,
        "ollama_temperature": config.ollama_temperature,
        "ollama_num_ctx": config.ollama_num_ctx,
        "ollama_num_predict": config.ollama_num_predict,
        "max_analysis_passes": config.researchpilot_max_analysis_passes,
    }


def test_capture_refuses_sources_that_do_not_rebuild_the_prompt_block() -> None:
    request = ResearchRequest(question=QUESTION, source_material=LOCAL)
    _, report, final = run(request, ["notes", complete(), "Report [S1]."])
    final["source_texts"]["S1"] = "tampered"

    with pytest.raises(CaptureError, match="do not match"):
        capture(request, final, report)


def test_run_with_state_returns_the_same_report_as_run() -> None:
    request = ResearchRequest(question=QUESTION, source_material=LOCAL)
    responses = ["notes", incomplete(["gap"]), "notes 2", complete(), "Report [S1]."]

    plain = ResearchService(ScriptedModel(responses)).run(request)  # type: ignore[arg-type]
    with_state, final = ResearchService(ScriptedModel(responses)).run_with_state(request)  # type: ignore[arg-type]

    ignore = {"elapsed_seconds", "node_seconds"}
    assert plain.model_dump(exclude=ignore) == with_state.model_dump(exclude=ignore)
    assert final["report_markdown"] == with_state.markdown


# --- Destination validation (runs before any model or network call) ---


def test_valid_destination_is_returned_as_an_absolute_path(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    assert validate_capture_path(Path("evidence.json")) == tmp_path / "evidence.json"
    assert validate_capture_path(Path("UPPER.JSON")) == tmp_path / "UPPER.JSON"


@pytest.mark.parametrize("name", ["evidence.md", "evidence.jsonl", "evidence", "evidence.json.txt"])
def test_destination_must_end_in_json(tmp_path, name: str) -> None:
    with pytest.raises(CaptureError, match=r"must end in \.json"):
        validate_capture_path(tmp_path / name)


def test_existing_file_or_folder_is_never_a_destination(tmp_path) -> None:
    existing = tmp_path / "evidence.json"
    existing.write_text("keep me", encoding="utf-8")
    folder = tmp_path / "folder.json"
    folder.mkdir()

    for path in (existing, folder):
        with pytest.raises(CaptureError, match="already exists"):
            validate_capture_path(path)
    assert existing.read_text(encoding="utf-8") == "keep me"


def test_dangling_symlink_counts_as_existing(tmp_path) -> None:
    link = tmp_path / "evidence.json"
    try:
        os.symlink(tmp_path / "missing-target.json", link)
    except (OSError, NotImplementedError):
        pytest.skip("creating symlinks is not permitted here")

    with pytest.raises(CaptureError, match="already exists"):
        validate_capture_path(link)


def test_missing_or_non_folder_parent_is_an_error_and_nothing_is_created(tmp_path) -> None:
    with pytest.raises(CaptureError, match="does not exist; create it first"):
        validate_capture_path(tmp_path / "missing" / "evidence.json")
    assert not (tmp_path / "missing").exists()

    not_a_folder = tmp_path / "file.txt"
    not_a_folder.write_text("x", encoding="utf-8")
    with pytest.raises(CaptureError, match="is not a folder"):
        validate_capture_path(not_a_folder / "evidence.json")


def test_destination_cannot_be_the_source_file_or_report_output(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "sources.json"
    source.write_text("{}", encoding="utf-8")
    output = Path("report.json")  # not created yet, given relative

    with pytest.raises(CaptureError, match="source file"):
        validate_capture_path(Path("sources.json"), source_file=source, report_output=output)
    with pytest.raises(CaptureError, match="report output"):
        validate_capture_path(tmp_path / "report.json", source_file=source, report_output=output)
    assert source.read_text(encoding="utf-8") == "{}"


@pytest.mark.skipif(os.name != "nt", reason="Windows paths are case-insensitive")
def test_destination_comparison_ignores_case_on_windows(tmp_path) -> None:
    output = tmp_path / "Report.json"

    with pytest.raises(CaptureError, match="report output"):
        validate_capture_path(tmp_path / "REPORT.JSON", report_output=output)


# --- Windows path aliases (checked only; nothing is ever written to these paths) ---


@pytest.mark.parametrize(
    ("path", "problem"),
    [
        (r"C:\work\evidence.json", None),
        (r"C:evidence.json", None),  # drive-relative: the drive colon is not a stream
        (r"\\server\share\evidence.json", None),
        (r"C:\work\sources.md:evil.json", "alternate data stream"),
        (r"C:\work\evidence.json:hidden.json", "alternate data stream"),
        (r"\\server\share\notes:x.json", "alternate data stream"),
        (r"\\?\C:\work\sources.md:evil.json", "alternate data stream"),
        (r"C:\work\NUL.json", "reserved Windows device name"),
        (r"C:\work\con.JSON", "reserved Windows device name"),
        (r"C:\work\COM1.json", "reserved Windows device name"),
        (r"C:\work\LPT9.json", "reserved Windows device name"),
        (r"C:\work\CONIN$.json", "reserved Windows device name"),
        (r"C:\work\nullable.json", None),
        (r"C:\work\COM10.json", None),
        (r"\\.\NUL", "device path"),
        (r"\\.\C:\work\evidence.json", "device path"),
        ("C:\\work\\evidence.json.", "ends with a dot or space"),
        ("C:\\work\\evidence.json ", "ends with a dot or space"),
        ("C:\\work\\evi*dence.json", "character Windows does not allow"),
        ("C:\\work\\evi|dence.json", "character Windows does not allow"),
        ("C:\\work\\evi\x01dence.json", "character Windows does not allow"),
    ],
)
def test_windows_path_problems_are_detected_on_any_platform(path: str, problem: str | None) -> None:
    found = windows_path_problem(path)

    if problem is None:
        assert found is None
    else:
        assert found is not None and problem in found


def test_reserved_names_fall_back_without_ntpath_isreserved(monkeypatch) -> None:
    monkeypatch.setattr(capture_module, "_ISRESERVED", None)

    for name in ("NUL.json", "nul.json", "Aux.json", "COM9.json", "LPT1.json", "CONOUT$.json"):
        assert "reserved Windows device name" in (windows_path_problem(rf"C:\w\{name}") or "")
    for name in ("null.json", "COM10.json", "console.json", "auxiliary.json"):
        assert windows_path_problem(rf"C:\w\{name}") is None


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (r"\\?\C:\work\out.json", r"C:\work\out.json"),
        (r"\\?\UNC\server\share\out.json", r"\\server\share\out.json"),
        (r"\\?\unc\server\share\out.json", r"\\server\share\out.json"),
        (r"C:\work\out.json", r"C:\work\out.json"),
        (r"\\server\share\out.json", r"\\server\share\out.json"),
    ],
)
def test_extended_length_prefixes_are_removed_for_comparison(path: str, expected: str) -> None:
    assert strip_extended_prefix(path) == expected


@pytest.mark.skipif(os.name != "nt", reason="Windows-only path forms")
@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("sources.md:evil.json", "alternate data stream"),
        ("NUL.json", "reserved Windows device name"),
        ("CON.json", "reserved Windows device name"),
        ("COM1.json", "reserved Windows device name"),
    ],
)
def test_dangerous_windows_destinations_are_rejected_before_any_write(
    tmp_path, name: str, message: str
) -> None:
    source = tmp_path / "sources.md"
    source.write_text("Source text", encoding="utf-8")

    with pytest.raises(CaptureError, match=message):
        validate_capture_path(tmp_path / name, source_file=source)
    assert leftovers(tmp_path) == ["sources.md"]
    assert source.read_text(encoding="utf-8") == "Source text"


@pytest.mark.skipif(os.name != "nt", reason="Windows extended-length paths")
def test_extended_length_alias_of_the_report_output_is_rejected(tmp_path) -> None:
    output = tmp_path / "out.json"
    alias = Path("\\\\?\\" + str(output))

    with pytest.raises(CaptureError, match="report output"):
        validate_capture_path(alias, report_output=output)
    with pytest.raises(CaptureError, match="report output"):
        validate_capture_path(output, report_output=alias)


# --- Atomic, no-overwrite write ---


def leftovers(folder: Path) -> list[str]:
    return sorted(path.name for path in folder.iterdir())


def test_write_creates_parseable_utf8_json_and_leaves_no_temporary_file(tmp_path) -> None:
    data = {"schema_version": 1, "question": "Ünïcödé — test", "sources": [{"id": "S1"}]}
    destination = tmp_path / "evidence.json"

    write_capture(destination, data)

    assert json.loads(destination.read_text(encoding="utf-8")) == data
    assert "Ünïcödé — test" in destination.read_text(encoding="utf-8")
    assert leftovers(tmp_path) == ["evidence.json"]


def test_write_refuses_a_file_created_after_validation(tmp_path) -> None:
    destination = tmp_path / "evidence.json"
    destination.write_text("created by someone else", encoding="utf-8")

    with pytest.raises(CaptureError, match="already exists"):
        write_capture(destination, {"schema_version": 1})
    assert destination.read_text(encoding="utf-8") == "created by someone else"
    assert leftovers(tmp_path) == ["evidence.json"]


def test_unserializable_data_writes_nothing(tmp_path) -> None:
    with pytest.raises(CaptureError, match="could not be serialized"):
        write_capture(tmp_path / "evidence.json", {"bad": object()})
    assert leftovers(tmp_path) == []


def test_disk_write_failure_cleans_up(tmp_path, monkeypatch) -> None:
    def failing_fsync(_fd: int) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(capture_module.os, "fsync", failing_fsync)

    with pytest.raises(CaptureError, match="No space left on device"):
        write_capture(tmp_path / "evidence.json", {"schema_version": 1})
    assert leftovers(tmp_path) == []


def test_without_hard_links_the_exclusive_fallback_is_used(tmp_path, monkeypatch) -> None:
    def no_links(_src, _dst) -> None:
        raise OSError(1, "Operation not permitted")

    monkeypatch.setattr(capture_module.os, "link", no_links)
    destination = tmp_path / "evidence.json"

    write_capture(destination, {"schema_version": 1})

    assert json.loads(destination.read_text(encoding="utf-8")) == {"schema_version": 1}
    assert leftovers(tmp_path) == ["evidence.json"]


def test_failed_fallback_write_removes_the_partial_file(tmp_path, monkeypatch) -> None:
    def no_links(_src, _dst) -> None:
        raise OSError(1, "Operation not permitted")

    real_fsync = capture_module.os.fsync
    calls = {"count": 0}

    def fail_second_fsync(fd: int) -> None:
        calls["count"] += 1
        if calls["count"] == 2:  # the fallback's own write
            raise OSError(5, "Input/output error")
        real_fsync(fd)

    monkeypatch.setattr(capture_module.os, "link", no_links)
    monkeypatch.setattr(capture_module.os, "fsync", fail_second_fsync)

    with pytest.raises(CaptureError, match="Input/output error"):
        write_capture(tmp_path / "evidence.json", {"schema_version": 1})
    assert leftovers(tmp_path) == []
