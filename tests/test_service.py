"""Unit tests for the research service; no API calls are made."""

import pytest
from fakes import ScriptedModel, complete, incomplete
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from researchpilot.schemas import ResearchReport, ResearchRequest
from researchpilot.service import ResearchService


class FakeModel:
    def __init__(self, content: str) -> None:
        self.content = content
        self.last_input = None

    def invoke(self, input, config=None, **kwargs):
        self.last_input = input
        return AIMessage(content=self.content)


def test_service_returns_report_and_preserves_question() -> None:
    model = FakeModel("# Research Report\n\nEvidence-based summary.")
    service = ResearchService(model)  # type: ignore[arg-type]
    request = ResearchRequest(
        question="Compare the supplied approaches",
        source_material="Source A: Approach one has lower latency. Source B: Approach two costs less.",
    )

    report = service.run(request)

    assert report.question == request.question
    # FakeModel answers the evaluator with report text too, so the evaluation is invalid and
    # the incomplete-analysis notice comes before the unchanged report.
    notice, rest = report.markdown.split("\n\n", 1)
    assert notice.startswith("> **Incomplete analysis:**")
    assert rest.startswith("# Research Report")
    assert model.last_input is not None


def test_request_strips_whitespace() -> None:
    request = ResearchRequest(
        question="  Compare these sources carefully  ",
        source_material="  Source title: Example. This is enough source material for the test.  ",
    )

    assert request.question == "Compare these sources carefully"
    assert request.source_material.startswith("Source title:")


def test_request_rejects_blank_question() -> None:
    with pytest.raises(ValidationError):
        ResearchRequest(
            question="        ",
            source_material="This is a long enough source text for the validation test.",
        )


def test_service_rejects_empty_model_response() -> None:
    service = ResearchService(FakeModel("   "))  # type: ignore[arg-type]
    request = ResearchRequest(
        question="Summarize the supplied evidence",
        source_material="Source A states that the system was tested in a controlled environment.",
    )

    with pytest.raises(RuntimeError, match="empty"):
        service.run(request)


def test_service_reports_passes_and_stop_reason() -> None:
    model = ScriptedModel(["- notes", complete(), "# Research Report\n\nGraph summary [S1]."])
    request = ResearchRequest(
        question="Summarize the supplied evidence",
        source_material="Source A states that the system was tested in a controlled environment.",
    )

    report = ResearchService(model, max_passes=3).run(request)  # type: ignore[arg-type]

    assert report.markdown == "# Research Report\n\nGraph summary [S1]."
    assert report.analysis_passes == 1
    assert report.stop_reason == "evaluation_complete"


@pytest.mark.parametrize("max_passes", [0, 4])
def test_service_rejects_out_of_range_max_passes(max_passes: int) -> None:
    service = ResearchService(ScriptedModel([]), max_passes=max_passes)  # type: ignore[arg-type]
    request = ResearchRequest(
        question="Summarize the supplied evidence",
        source_material="Source A states that the system was tested in a controlled environment.",
    )

    with pytest.raises(ValueError, match="between 1 and 3"):
        service.run(request)


def test_service_reports_model_calls_and_timing() -> None:
    model = ScriptedModel(["- notes", complete(), "# Research Report\n\nGraph summary [S1]."])
    request = ResearchRequest(
        question="Summarize the supplied evidence",
        source_material="Source A states that the system was tested in a controlled environment.",
    )

    report = ResearchService(model).run(request)  # type: ignore[arg-type]

    assert report.model_calls == 3
    assert report.elapsed_seconds is not None and report.elapsed_seconds >= 0
    assert set(report.node_seconds or {}) == {"analyze", "evaluate", "generate_report"}


def test_service_handles_models_without_token_metadata() -> None:
    model = FakeModel("# Research Report\n\nEvidence-based summary.")
    request = ResearchRequest(
        question="Summarize the supplied evidence",
        source_material="Source A states that the system was tested in a controlled environment.",
    )

    report = ResearchService(model).run(request)  # type: ignore[arg-type]

    assert report.model_calls == 3


def test_service_reports_unresolved_gaps_at_the_pass_limit() -> None:
    model = ScriptedModel(
        ["- notes", incomplete(["Source A sample size is not covered."]), "# Research Report"]
    )
    request = ResearchRequest(
        question="Summarize the supplied evidence",
        source_material="Source A states that the system was tested in a controlled environment.",
    )

    report = ResearchService(model, max_passes=1).run(request)  # type: ignore[arg-type]

    assert report.stop_reason == "pass_limit_reached"
    assert report.unresolved_analysis_gaps == ["Source A sample size is not covered."]
    assert report.markdown.startswith("> **Incomplete analysis:**")
    assert "with 1 unresolved analysis gap(s)" in report.markdown


def test_completed_run_has_no_unresolved_gaps() -> None:
    model = ScriptedModel(["- notes", complete(), "# Research Report\n\nGraph summary [S1]."])
    request = ResearchRequest(
        question="Summarize the supplied evidence",
        source_material="Source A states that the system was tested in a controlled environment.",
    )

    report = ResearchService(model).run(request)  # type: ignore[arg-type]

    assert report.unresolved_analysis_gaps == []


def test_new_report_fields_default_to_empty_lists() -> None:
    report = ResearchReport(question="Q", markdown="# Report")

    assert report.unresolved_analysis_gaps == []
    assert report.quality_warnings == []
    assert {"unresolved_analysis_gaps", "quality_warnings"} <= set(report.model_dump())
