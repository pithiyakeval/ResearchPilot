"""Deterministic run metrics for regression evaluation of research runs.

Metrics are computed only from a finished run's ``ResearchResult`` (or its typed error), so
they cost no model call and are reproducible. They describe structure and process, not
answer quality:

* ``citation_problems``: problems found by the citation-reference check (unknown IDs,
  internal labels, or no citation at all). Zero means every reference names a supplied
  source; it does not mean the sources support the claims, which is never checked.
* ``source_coverage``: supplied sources cited at least once, divided by sources supplied.
  A traceability signal; a high value says nothing about correct use of the sources.
* ``pass_limit_respected``: analysis passes did not exceed the configured maximum.
* ``valid_evaluations`` and ``invalid_evaluations``: evaluator replies that did or did not
  validate against the evaluation schema. Validity is about shape, not correctness; the
  evaluator is the same local model that wrote the analysis, not an independent judge.
* ``error_category``: the stable code of a failed run (``researchpilot.errors``).

Live-model evaluation would compute the same metrics from real runs; it is deliberately not
part of ordinary CI.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from researchpilot.errors import ResearchPilotError
from researchpilot.results import ResearchResult
from researchpilot.schemas import StopReason


@dataclass(frozen=True)
class RunMetrics:
    status: Literal["completed", "failed"]
    error_category: str | None = None
    stop_reason: StopReason | None = None
    analysis_passes: int | None = None
    pass_limit_respected: bool | None = None
    model_calls: int | None = None
    supplied_sources: int = 0
    web_sources: int = 0
    cited_sources: tuple[str, ...] = ()
    source_coverage: float | None = None
    citation_problems: int | None = None
    valid_evaluations: int = 0
    invalid_evaluations: int = 0
    unresolved_analysis_gaps: int = 0
    quality_warnings: int = 0


def metrics_from_result(result: ResearchResult, *, max_passes: int) -> RunMetrics:
    supplied = [source.id for source in result.sources]
    cited = tuple(result.citations.cited_source_ids)
    passes = result.report.analysis_passes
    statuses = [record.status for record in result.evaluation_history]
    return RunMetrics(
        status="completed",
        stop_reason=result.report.stop_reason,
        analysis_passes=passes,
        pass_limit_respected=None if passes is None else passes <= max_passes,
        model_calls=result.report.model_calls,
        supplied_sources=len(supplied),
        web_sources=sum(source.type == "web" for source in result.sources),
        cited_sources=cited,
        source_coverage=round(len(set(cited)) / len(supplied), 3) if supplied else None,
        citation_problems=len(result.citations.invalid),
        valid_evaluations=sum(status != "invalid" for status in statuses),
        invalid_evaluations=sum(status == "invalid" for status in statuses),
        unresolved_analysis_gaps=len(result.unresolved_analysis_gaps),
        quality_warnings=len(result.quality_warnings),
    )


def metrics_from_error(error: ResearchPilotError) -> RunMetrics:
    return RunMetrics(status="failed", error_category=error.code)


@dataclass(frozen=True)
class SuiteSummary:
    """Aggregates over a set of evaluated cases; rates are over the cases they apply to."""

    cases: int
    expectations_met: int
    completed_runs: int
    runs_without_citation_problems: int
    pass_limit_violations: int

    @property
    def expectation_rate(self) -> float:
        return self.expectations_met / self.cases if self.cases else 0.0


def summarize(outcomes: Iterable[tuple[RunMetrics, bool]]) -> SuiteSummary:
    """Summarize ``(metrics, expectations_met)`` pairs from an evaluation run."""
    items = list(outcomes)
    completed = [metrics for metrics, _ in items if metrics.status == "completed"]
    return SuiteSummary(
        cases=len(items),
        expectations_met=sum(met for _, met in items),
        completed_runs=len(completed),
        runs_without_citation_problems=sum(m.citation_problems == 0 for m in completed),
        pass_limit_violations=sum(m.pass_limit_respected is False for m in completed),
    )
