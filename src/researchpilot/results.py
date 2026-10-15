"""Structured result of one research run, for API and frontend consumers.

Built from a ``ResearchOutcome``. Sources are listed without their text; the full texts,
notes and evaluations stay available through the evidence capture (``capture.py``).
"""

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

from researchpilot.capture import build_capture
from researchpilot.config import Settings
from researchpilot.pipeline import ResearchOutcome
from researchpilot.schemas import StopReason, WebSourceKind
from researchpilot.state import EvaluationStatus


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ReportSummary(_Frozen):
    markdown: str
    stop_reason: StopReason | None
    analysis_passes: int | None
    model_calls: int | None
    elapsed_seconds: float | None
    node_seconds: dict[str, float] | None


class SourceSummary(_Frozen):
    id: str
    type: Literal["local", "web"]
    # The title as numbered, shown to the model and used in citation notes.
    numbered_title: str
    # Web sources only.
    evidence_type: WebSourceKind | None = None
    page_title: str | None = None
    url: str | None = None


class Citations(_Frozen):
    # Supplied source IDs the model's own report text cites, in first-cited order.
    cited_source_ids: list[str]
    # Problems found by the deterministic citation check.
    invalid: list[str]


class AnalysisPass(_Frozen):
    pass_number: int
    notes: str


class EvaluationPass(_Frozen):
    pass_number: int
    status: EvaluationStatus
    analysis_gaps: list[str]
    source_gaps: list[str]
    conflicts: list[str]
    error: str | None


class RetrievalSummary(_Frozen):
    search_calls: int
    fetch_calls: int
    warnings: list[str]


class ResearchResult(_Frozen):
    question: str
    mode: Literal["local", "web"]
    report: ReportSummary
    sources: list[SourceSummary]
    citations: Citations
    quality_warnings: list[str]
    unresolved_analysis_gaps: list[str]
    analysis_history: list[AnalysisPass]
    evaluation_history: list[EvaluationPass]
    retrieval: RetrievalSummary | None


def build_result(outcome: ResearchOutcome, settings: Settings) -> ResearchResult:
    """Build the structured result, reusing the evidence capture's view of the run."""
    report = outcome.report
    evidence = build_capture(
        outcome.request,
        outcome.final_state,
        report,
        settings=settings,
        retrieval=outcome.retrieval,
        output_path=None,
        now=datetime.now(UTC),
    )
    return ResearchResult(
        question=report.question,
        mode="web" if outcome.retrieval is not None else "local",
        report=ReportSummary(
            markdown=report.markdown,
            stop_reason=report.stop_reason,
            analysis_passes=report.analysis_passes,
            model_calls=report.model_calls,
            elapsed_seconds=report.elapsed_seconds,
            node_seconds=report.node_seconds,
        ),
        sources=[
            SourceSummary(**{key: value for key, value in source.items() if key != "text"})
            for source in evidence["sources"]
        ],
        citations=Citations(
            cited_source_ids=list(outcome.final_state["cited_source_ids"]),
            invalid=list(report.invalid_citations),
        ),
        quality_warnings=list(report.quality_warnings),
        unresolved_analysis_gaps=list(report.unresolved_analysis_gaps),
        analysis_history=[
            AnalysisPass(pass_number=record["pass"], notes=record["notes"])
            for record in evidence["analysis_passes"]
        ],
        evaluation_history=[
            EvaluationPass(
                pass_number=record["pass"],
                status=record["status"],
                analysis_gaps=record["analysis_gaps"],
                source_gaps=record["source_gaps"],
                conflicts=record["conflicts"],
                error=record["error"],
            )
            for record in evidence["evaluation_history"]
        ],
        retrieval=None
        if evidence["retrieval"] is None
        else RetrievalSummary(**evidence["retrieval"]),
    )
