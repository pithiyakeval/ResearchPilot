"""Offline tests for the structured research result; no model, network or MCP server is used."""

import json

from fakes import ScriptedModel, complete, incomplete

from researchpilot.config import Settings
from researchpilot.pipeline import prepare_request, run_research
from researchpilot.results import ResearchResult, build_result
from researchpilot.schemas import WebSource
from researchpilot.sources import cited_source_ids
from researchpilot.web_research import WebEvidence

QUESTION = "Compare System Alpha and System Beta"
SOURCES = (
    "# Source A: Task completion\n\nIn a fictional test, System Alpha completed 85 of 100 tasks.\n\n"
    "# Source B: Configuration\n\nSystem Beta required less initial configuration."
)
TITLES = {"S1": "A", "S2": "B", "S3": "C"}
EXCERPT = WebSource(
    kind="search_excerpt",
    title="Benchmark page",
    url="https://docs.example.org/benchmarks",
    text="Excerpt about benchmark tasks.",
)


def settings() -> Settings:
    return Settings(_env_file=None)  # type: ignore[call-arg]


def outcome_for(responses: list[str], *, max_passes: int = 2, web_sources=()):
    model = ScriptedModel(responses)
    gatherer = None
    if web_sources:
        evidence = WebEvidence(
            sources=tuple(web_sources),
            warnings=("Skipped fetch: timed out",),
            search_calls=1,
            fetch_calls=1,
        )
        gatherer = lambda _question: evidence
    config = Settings(_env_file=None, researchpilot_max_analysis_passes=max_passes)  # type: ignore[call-arg]
    return run_research(
        prepare_request(QUESTION, SOURCES),
        settings=config,
        web=bool(web_sources),
        model_factory=lambda _settings, **_kwargs: model,
        web_gatherer=gatherer,
    )


# --- cited_source_ids ---


def test_cited_source_ids_are_in_order_without_duplicates() -> None:
    report = "First [S2]. Both [S1, S2]. Again [s1]; and [S3; S1]."

    assert cited_source_ids(report, TITLES) == ["S2", "S1", "S3"]


def test_cited_source_ids_ignore_unknown_ids_and_non_citations() -> None:
    report = "Known [S1]. Unknown [S9]. Label (S2). Link [S3 docs](https://x). Text S2."

    assert cited_source_ids(report, TITLES) == ["S1"]
    assert cited_source_ids("No citations here.", TITLES) == []


# --- ResearchResult ---


def test_local_result_matches_the_report_and_captured_history() -> None:
    outcome = outcome_for(
        ["notes one", incomplete(["gap"]), "notes two", complete(), "Claims [S2] [S1]."]
    )

    result = build_result(outcome, settings())

    report = outcome.report
    assert isinstance(result, ResearchResult)
    assert result.question == QUESTION
    assert result.mode == "local"
    assert result.report.markdown == report.markdown
    assert result.report.stop_reason == report.stop_reason == "evaluation_complete"
    assert result.report.analysis_passes == 2
    assert result.report.model_calls == 5
    assert result.report.elapsed_seconds == report.elapsed_seconds
    assert result.report.node_seconds == report.node_seconds
    assert [(s.id, s.type, s.numbered_title) for s in result.sources] == [
        ("S1", "local", "Source A: Task completion"),
        ("S2", "local", "Source B: Configuration"),
    ]
    assert all(
        s.url is None and s.page_title is None and s.evidence_type is None for s in result.sources
    )
    assert result.citations.cited_source_ids == ["S2", "S1"]
    assert result.citations.invalid == report.invalid_citations == []
    assert result.quality_warnings == report.quality_warnings
    assert result.unresolved_analysis_gaps == []
    assert [(p.pass_number, p.notes) for p in result.analysis_history] == [
        (1, "notes one"),
        (2, "notes two"),
    ]
    assert [(e.pass_number, e.status, e.analysis_gaps) for e in result.evaluation_history] == [
        (1, "incomplete", ["gap"]),
        (2, "complete", []),
    ]
    assert result.retrieval is None


def test_result_has_no_source_text_and_serializes_to_json() -> None:
    outcome = outcome_for(["notes", complete(), "Claim [S1]."])

    result = build_result(outcome, settings())
    dumped = json.loads(result.model_dump_json())

    assert "System Alpha completed 85 of 100 tasks" not in json.dumps(dumped)
    assert "text" not in dumped["sources"][0]
    assert ResearchResult.model_validate(dumped) == result


def test_cited_ids_come_from_the_model_text_not_the_appended_notes() -> None:
    # The citation warning lists every source ID; only the model's own citation counts.
    outcome = outcome_for(["notes", complete(), "Claim [S1]. Bad [S9]."])

    result = build_result(outcome, settings())

    assert "[S2] Source B: Configuration" in outcome.report.markdown  # in the warning's key
    assert result.citations.cited_source_ids == ["S1"]
    assert result.citations.invalid == ["[S9] is not a supplied source ID"]


def test_pass_limit_result_reports_unresolved_gaps() -> None:
    gaps = ["Source A sample size is not covered."]
    outcome = outcome_for(["notes", incomplete(gaps), "Claim [S1]."], max_passes=1)

    result = build_result(outcome, settings())

    assert result.report.stop_reason == "pass_limit_reached"
    assert result.unresolved_analysis_gaps == gaps == outcome.report.unresolved_analysis_gaps


def test_web_result_has_web_source_fields_retrieval_and_web_mode() -> None:
    outcome = outcome_for(["notes", complete(), "Claim [S3]."], web_sources=[EXCERPT])

    result = build_result(outcome, settings())

    assert result.mode == "web"
    web = result.sources[-1]
    assert (web.id, web.type, web.evidence_type, web.page_title, web.url) == (
        "S3",
        "web",
        "search_excerpt",
        "Benchmark page",
        "https://docs.example.org/benchmarks",
    )
    assert (
        web.numbered_title == "Search excerpt: Benchmark page — https://docs.example.org/benchmarks"
    )
    assert result.citations.cited_source_ids == ["S3"]  # not the IDs in the web-sources note
    assert result.retrieval is not None
    assert result.retrieval.model_dump() == {
        "search_calls": 1,
        "fetch_calls": 1,
        "warnings": ["Skipped fetch: timed out"],
    }
