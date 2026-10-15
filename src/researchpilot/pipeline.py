"""One research run, shared by the CLI and future API adapters.

``prepare_request`` validates the input; ``run_research`` runs the optional web retrieval and
the research graph and returns everything a caller may need. Neither prints, logs source
text or writes files. Failures are raised as the typed errors in ``researchpilot.errors``,
each carrying the original exception's message and keeping it as ``__cause__``.
"""

import asyncio
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from langchain_core.language_models import BaseChatModel
from pydantic import ValidationError

from researchpilot.budget import ContextBudget, ContextBudgetExceededError
from researchpilot.config import Settings
from researchpilot.errors import (
    InputError,
    InputTooLargeError,
    ModelRunError,
    ResearchPilotError,
    RetrievalError,
)
from researchpilot.graph import check_request_fits
from researchpilot.models import build_model
from researchpilot.schemas import ResearchReport, ResearchRequest
from researchpilot.service import ResearchService
from researchpilot.state import EVALUATION_JSON_SCHEMA, ResearchState

if TYPE_CHECKING:
    # Type-only: local runs never import the MCP client.
    from researchpilot.web_research import WebEvidence

# Called as ``factory(settings)`` for the main model, then
# ``factory(settings, json_schema=EVALUATION_JSON_SCHEMA)`` for the evaluator.
ModelFactory = Callable[..., BaseChatModel]
# Takes the question and returns the retrieved evidence; raises RetrievalError on failure.
WebGatherer = Callable[[str], "WebEvidence"]


@dataclass(frozen=True)
class ResearchOutcome:
    """A completed research run."""

    request: ResearchRequest
    """The request the graph ran on, including any retrieved web sources."""
    report: ResearchReport
    final_state: ResearchState
    retrieval: "WebEvidence | None"
    """Web retrieval counts and warnings; None for a local-only run."""


@contextmanager
def _typed_errors() -> Iterator[None]:
    """Turn failures into typed errors, in the order the CLI has always classified them."""
    try:
        yield
    except ResearchPilotError:
        raise
    except (ValidationError, UnicodeDecodeError) as exc:
        raise InputError(str(exc)) from exc
    except ContextBudgetExceededError as exc:
        raise InputTooLargeError(str(exc)) from exc
    except Exception as exc:
        raise ModelRunError(str(exc)) from exc


def prepare_request(question: str, source_material: str) -> ResearchRequest:
    """Validate a question and its source material; raise ``InputError`` if they are invalid."""
    with _typed_errors():
        return ResearchRequest(question=question, source_material=source_material)


def _context_budget(settings: Settings) -> ContextBudget:
    return ContextBudget(
        context_tokens=settings.ollama_num_ctx,
        response_tokens=settings.ollama_num_predict,
    )


def check_input_fits(request: ResearchRequest, settings: Settings) -> None:
    """Raise ``InputTooLargeError`` if the request cannot fit the model's context budget.

    The same check ``run_research`` makes before calling any model, so callers can reject
    oversized input up front. Web sources retrieved later are checked again by the run.
    """
    with _typed_errors():
        check_request_fits(request, _context_budget(settings))


def gather_web_sources(question: str) -> "WebEvidence":
    """Default web gatherer: one bounded Parallel Search MCP retrieval for the question."""
    # Imported here so local runs never load the MCP client.
    from researchpilot.web_research import WebRetrievalError, gather_web_evidence

    try:
        return asyncio.run(gather_web_evidence(question))
    except WebRetrievalError as exc:
        raise RetrievalError(str(exc)) from exc


def run_research(
    request: ResearchRequest,
    *,
    settings: Settings,
    web: bool = False,
    model_factory: ModelFactory = build_model,
    web_gatherer: WebGatherer | None = None,
    on_retrieval: Callable[["WebEvidence"], Any] | None = None,
) -> ResearchOutcome:
    """Run one research job: optional web retrieval, then the analysis and report graph.

    With ``web``, oversized local input is rejected before any network request, and
    ``on_retrieval`` is called with the evidence before any model is built. Blocks until
    the run finishes; it can take many minutes with a local model.
    """
    budget = _context_budget(settings)
    retrieval = None
    with _typed_errors():
        if web:
            check_request_fits(request, budget)
            retrieval = (web_gatherer or gather_web_sources)(request.question)
            if on_retrieval is not None:
                on_retrieval(retrieval)
            request = ResearchRequest(
                question=request.question,
                source_material=request.source_material,
                web_sources=retrieval.sources,
            )
        service = ResearchService(
            model_factory(settings),
            max_passes=settings.researchpilot_max_analysis_passes,
            budget=budget,
            evaluator_model=model_factory(settings, json_schema=EVALUATION_JSON_SCHEMA),
        )
        report, final_state = service.run_with_state(request)
    return ResearchOutcome(
        request=request, report=report, final_state=final_state, retrieval=retrieval
    )
