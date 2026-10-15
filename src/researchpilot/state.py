"""Typed shared state for the research graph and the evaluator's validated output."""

import json
import operator
import re
from typing import Annotated, Literal, TypedDict

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
)

from researchpilot.schemas import ResearchRequest, StopReason
from researchpilot.sources import NumberedSources, number_sources

MIN_PASSES = 1
MAX_PASSES = 3

EvaluationStatus = Literal["complete", "incomplete", "invalid"]
GapItem = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=300)]


class CompletenessEvaluation(BaseModel):
    """Evaluator verdict on whether the analysis notes cover the supplied sources."""

    model_config = ConfigDict(extra="forbid", strict=True)

    # Present in the sources but missing from the notes; another pass can fix these.
    analysis_gaps: list[GapItem] = Field(max_length=8)
    # Not present in the sources at all; another pass cannot fix these.
    source_gaps: list[GapItem] = Field(max_length=8)
    # Disagreements between supplied sources; reported, never a reason to retry.
    conflicts: list[GapItem] = Field(max_length=8)

    @property
    def analysis_complete(self) -> bool:
        """Derived from the validated gap list; the model is never asked for a verdict."""
        return not self.analysis_gaps


# Passed to Ollama as its structured-output format so the evaluator can only emit this shape.
EVALUATION_JSON_SCHEMA = CompletenessEvaluation.model_json_schema()


class ModelCall(TypedDict):
    """Timing and size of one model call; never holds prompt or source text."""

    node: str
    pass_number: int
    seconds: float
    prompt_chars: int
    # Token counts as reported by Ollama; None when the model does not report them.
    prompt_tokens: int | None
    output_tokens: int | None


class AnalysisRecord(TypedDict):
    """The analysis notes one pass produced, exactly as later prompts receive them."""

    pass_number: int
    notes: str


class EvaluationRecord(TypedDict):
    """Structured result of one completeness check; never holds the raw model reply."""

    pass_number: int
    status: EvaluationStatus
    analysis_gaps: list[str]
    source_gaps: list[str]
    conflicts: list[str]
    error: str | None


class ResearchState(TypedDict):
    question: str
    source_material: str
    # Source material with [S1], [S2], ... ID lines; this is what prompts receive.
    numbered_sources: str
    # Source ID ("S1") to title.
    source_titles: dict[str, str]
    # IDs of retrieved web sources, after the local ones; empty in offline runs.
    web_source_ids: list[str]
    # Source ID to that source's text, for the heuristic report review hints.
    source_texts: dict[str, str]
    max_passes: int
    pass_count: int
    analysis_notes: str | None
    evaluation: CompletenessEvaluation | None
    evaluation_status: EvaluationStatus | None
    evaluation_error: str | None
    stop_reason: StopReason | None
    report_markdown: str | None
    # Problems found by the deterministic citation check; empty when none were found.
    invalid_citations: list[str]
    # Heuristic review hints; they never change the report text or the citation check.
    quality_warnings: list[str]
    # Supplied source IDs the model's own report text cites, in order (notes excluded).
    cited_source_ids: list[str]
    model_calls: Annotated[list[ModelCall], operator.add]
    # One record per completed pass, in order. Prompts never read these; they exist so an
    # opt-in evidence capture can show every pass, not only the last one.
    analysis_history: Annotated[list[AnalysisRecord], operator.add]
    evaluation_history: Annotated[list[EvaluationRecord], operator.add]


def number_request_sources(request: ResearchRequest) -> NumberedSources:
    """Number the supplied material, then any web sources, in the form prompts receive."""
    return number_sources(
        request.source_material,
        [(source.heading, source.text) for source in request.web_sources],
    )


def initial_state(request: ResearchRequest, max_passes: int) -> ResearchState:
    """Build the starting state for one research run."""
    if not MIN_PASSES <= max_passes <= MAX_PASSES:
        raise ValueError(f"max_passes must be between {MIN_PASSES} and {MAX_PASSES}.")
    numbered = number_request_sources(request)
    return ResearchState(
        question=request.question,
        source_material=request.source_material,
        numbered_sources=numbered.text,
        source_titles=numbered.titles,
        web_source_ids=list(numbered.extra_ids),
        source_texts=numbered.texts,
        max_passes=max_passes,
        pass_count=0,
        analysis_notes=None,
        evaluation=None,
        evaluation_status=None,
        evaluation_error=None,
        stop_reason=None,
        report_markdown=None,
        invalid_citations=[],
        quality_warnings=[],
        cited_source_ids=[],
        model_calls=[],
        analysis_history=[],
        evaluation_history=[],
    )


_CODE_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)


def parse_evaluation(raw: str) -> tuple[CompletenessEvaluation | None, str | None]:
    """Validate evaluator output; return (evaluation, None) or (None, reason it is invalid)."""
    text = raw.strip()
    fenced = _CODE_FENCE.match(text)
    if fenced:
        text = fenced.group(1)
    if not text:
        return None, "the evaluator returned an empty response"
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, f"the evaluator response was not valid JSON ({exc.msg})"
    try:
        return CompletenessEvaluation.model_validate(data), None
    except ValidationError as exc:
        problems = "; ".join(error["msg"] for error in exc.errors())
        return None, f"the evaluator response failed validation ({problems})"
