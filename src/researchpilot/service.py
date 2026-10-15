"""Application service for source-grounded research synthesis."""

import logging
import time

from langchain_core.language_models import BaseChatModel

from researchpilot.budget import ContextBudget
from researchpilot.graph import (
    build_research_graph,
    check_request_fits,
    recursion_limit_for,
    unresolved_analysis_gaps,
)
from researchpilot.schemas import ResearchReport, ResearchRequest
from researchpilot.sources import web_source_note
from researchpilot.state import ResearchState, initial_state

logger = logging.getLogger(__name__)

DEFAULT_MAX_PASSES = 2


class ResearchService:
    """Generate a report from a question and user-supplied evidence."""

    def __init__(
        self,
        model: BaseChatModel,
        max_passes: int = DEFAULT_MAX_PASSES,
        budget: ContextBudget | None = None,
        evaluator_model: BaseChatModel | None = None,
    ) -> None:
        self._max_passes = max_passes
        self._budget = budget
        self._graph = build_research_graph(model, budget, evaluator_model)

    def run(self, request: ResearchRequest) -> ResearchReport:
        return self.run_with_state(request)[0]

    def run_with_state(self, request: ResearchRequest) -> tuple[ResearchReport, ResearchState]:
        """Run once and also return the final graph state, for opt-in evidence capture."""
        state = initial_state(request, self._max_passes)
        if self._budget is not None:
            check_request_fits(request, self._budget)
        start = time.perf_counter()
        final = self._graph.invoke(
            state, config={"recursion_limit": recursion_limit_for(self._max_passes)}
        )
        elapsed = round(time.perf_counter() - start, 3)

        node_seconds: dict[str, float] = {}
        for call in final["model_calls"]:
            total = node_seconds.get(call["node"], 0.0) + call["seconds"]
            node_seconds[call["node"]] = round(total, 3)
        logger.info(
            "run complete passes=%d model_calls=%d elapsed_seconds=%.2f stop_reason=%s "
            "citation_problems=%d",
            final["pass_count"],
            len(final["model_calls"]),
            elapsed,
            final["stop_reason"],
            len(final["invalid_citations"]),
        )
        markdown = final["report_markdown"]
        if final["web_source_ids"]:
            # Appended after the citation check, so it never counts as a citation.
            note = web_source_note(final["web_source_ids"], final["source_titles"])
            markdown = f"{markdown}\n\n{note}"
        report = ResearchReport(
            question=request.question,
            markdown=markdown,
            analysis_passes=final["pass_count"],
            stop_reason=final["stop_reason"],
            model_calls=len(final["model_calls"]),
            elapsed_seconds=elapsed,
            node_seconds=node_seconds,
            invalid_citations=final["invalid_citations"],
            unresolved_analysis_gaps=unresolved_analysis_gaps(final),
            quality_warnings=final["quality_warnings"],
        )
        return report, final
