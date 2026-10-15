"""LangGraph workflow: analyze the supplied sources, check coverage, then write the report.

The graph receives an already-constructed chat model; it never builds one.
"""

import logging
import time
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.prompt_values import PromptValue
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from researchpilot.budget import ContextBudget
from researchpilot.prompts import (
    ANALYSIS_PROMPT,
    EVALUATION_PROMPT,
    REANALYSIS_PROMPT,
    REPORT_PROMPT,
    neutralize_delimiters,
)
from researchpilot.report_checks import review_hints, review_note
from researchpilot.schemas import ResearchRequest, StopReason
from researchpilot.sources import check_citations, citation_warning, cited_source_ids
from researchpilot.state import (
    AnalysisRecord,
    EvaluationRecord,
    EvaluationStatus,
    ModelCall,
    ResearchState,
    number_request_sources,
    parse_evaluation,
)

logger = logging.getLogger(__name__)

ANALYZE = "analyze"
EVALUATE = "evaluate"
GENERATE_REPORT = "generate_report"


def recursion_limit_for(max_passes: int) -> int:
    """LangGraph step limit: the input step, two steps per analysis pass, and the report step."""
    return 2 * max_passes + 2


def decide_stop_reason(
    status: EvaluationStatus, pass_count: int, max_passes: int
) -> StopReason | None:
    """Return why analysis should stop, or None if another pass is allowed and useful."""
    if status == "invalid":
        return "evaluation_invalid"
    if status == "complete":
        return "evaluation_complete"
    if pass_count >= max_passes:
        return "pass_limit_reached"
    return None


def route_after_evaluation(state: ResearchState) -> Literal["analyze", "generate_report"]:
    if state["stop_reason"] is not None or state["pass_count"] >= state["max_passes"]:
        return GENERATE_REPORT
    return ANALYZE


def check_request_fits(request: ResearchRequest, budget: ContextBudget) -> None:
    """Reject input up front if the largest prompt (the report) cannot fit the budget.

    The report prompt also carries the analysis notes and the run status, each produced
    from at most one model response, so space for both is reserved. Sources, including any
    web sources, are measured in the same numbered form the prompts receive.
    """
    numbered = number_request_sources(request)
    prompt = REPORT_PROMPT.invoke(
        {
            "question": request.question,
            "source_material": neutralize_delimiters(numbered.text),
            "analysis_notes": "",
            "run_status": "",
        }
    )
    budget.check(prompt.to_string(), "report", reserved_tokens=2 * budget.response_tokens)


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {neutralize_delimiters(item)}" for item in items) or "- (none)"


def describe_run_status(state: ResearchState) -> str:
    """Plain-text summary of the analysis loop for the report prompt."""
    evaluation = state["evaluation"]
    lines = [f"Analysis passes completed: {state['pass_count']} of at most {state['max_passes']}."]
    match state["stop_reason"]:
        case "evaluation_complete":
            lines.append(
                "Completeness check: the evaluator judged the notes complete relative to the "
                "supplied sources. This is a model judgement, not independent verification."
            )
        case "pass_limit_reached":
            lines.append(
                "Completeness check: the pass limit was reached while these analysis gaps "
                "were still reported. Treat them as unresolved:"
            )
            lines.append(_bullets(evaluation.analysis_gaps if evaluation else []))
        case _:
            lines.append(
                "Completeness check: INVALID. The evaluator output could not be validated "
                f"({state['evaluation_error']}). Coverage of the supplied sources is "
                "unverified; say so in the report."
            )
    if evaluation is not None:
        lines.append("Gaps in the supplied sources (no other sources are available):")
        lines.append(_bullets(evaluation.source_gaps))
        lines.append("Conflicts between the supplied sources:")
        lines.append(_bullets(evaluation.conflicts))
    return "\n".join(lines)


def unresolved_analysis_gaps(state: ResearchState) -> list[str]:
    """Analysis gaps still open when the pass limit stopped the loop; empty otherwise.

    Taken from the final evaluation, which each evaluate step replaces. This one value feeds
    the opening notice, the closing note and ``ResearchReport.unresolved_analysis_gaps``.
    """
    evaluation = state["evaluation"]
    if state["stop_reason"] != "pass_limit_reached" or evaluation is None:
        return []
    return list(evaluation.analysis_gaps)


def incomplete_analysis_notice(state: ResearchState) -> str | None:
    """Deterministic notice placed before the report when coverage was not confirmed."""
    match state["stop_reason"]:
        case "pass_limit_reached":
            count = len(unresolved_analysis_gaps(state))
            return (
                "> **Incomplete analysis:** ResearchPilot stopped at its configured limit of "
                f"{state['max_passes']} analysis pass(es) with {count} unresolved analysis "
                "gap(s), listed in the note at the end of this report. Parts of the sources may "
                "not be reflected; this is not a comprehensive review."
            )
        case "evaluation_invalid":
            return (
                "> **Incomplete analysis:** the completeness check could not be validated, so "
                "coverage of the sources is unverified. See the note at the end of this report."
            )
        case _:
            return None


def uncertainty_note(state: ResearchState) -> str | None:
    """Deterministic note appended to the report when coverage was not confirmed."""
    match state["stop_reason"]:
        case "pass_limit_reached":
            open_gaps = unresolved_analysis_gaps(state)
            gaps = "; ".join(gap.rstrip(".") for gap in open_gaps) if open_gaps else "unspecified"
            reason = (
                f"Analysis stopped at the configured limit of {state['max_passes']} pass(es) "
                f"while the completeness check still reported analysis gaps: {gaps}. Parts of "
                "the supplied sources may not be reflected in this report."
            )
        case "evaluation_invalid":
            reason = (
                f"The completeness check could not be validated ({state['evaluation_error']}), "
                "so coverage of the supplied sources is unverified."
            )
        case _:
            return None
    web_ids = state.get("web_source_ids") or []
    if web_ids:
        cited = ", ".join(f"[{sid}]" for sid in web_ids)
        scope = (
            "Sources were the supplied material and the web evidence retrieved before "
            f"analysis ({cited}); no other sources were consulted."
        )
    else:
        scope = "No sources beyond the supplied material were consulted."
    return f"> **ResearchPilot note:** {reason} {scope}"


def timed_invoke(
    model: BaseChatModel, prompt: PromptValue, node: str, pass_number: int
) -> tuple[BaseMessage, ModelCall]:
    """Invoke the model and record how long it took, without keeping any prompt text."""
    prompt_chars = len(prompt.to_string())
    start = time.perf_counter()
    response = model.invoke(prompt)
    seconds = time.perf_counter() - start
    metadata = getattr(response, "response_metadata", None) or {}
    call = ModelCall(
        node=node,
        pass_number=pass_number,
        seconds=round(seconds, 3),
        prompt_chars=prompt_chars,
        prompt_tokens=metadata.get("prompt_eval_count"),
        output_tokens=metadata.get("eval_count"),
    )
    logger.info(
        "model call node=%s pass=%d seconds=%.2f prompt_chars=%d prompt_tokens=%s output_tokens=%s",
        node,
        pass_number,
        seconds,
        prompt_chars,
        call["prompt_tokens"],
        call["output_tokens"],
    )
    return response, call


def build_research_graph(
    model: BaseChatModel,
    budget: ContextBudget | None = None,
    evaluator_model: BaseChatModel | None = None,
) -> CompiledStateGraph:
    """Compile the analysis graph around already-constructed chat models.

    ``evaluator_model`` is used for the completeness check, typically the same model
    configured to emit JSON matching ``EVALUATION_JSON_SCHEMA``. Defaults to ``model``.
    """
    evaluator = evaluator_model if evaluator_model is not None else model

    def invoke_text(
        prompt: PromptValue, stage: str, node: str, pass_number: int
    ) -> tuple[str, ModelCall]:
        if budget is not None:
            budget.check(prompt.to_string(), stage)
        response, call = timed_invoke(model, prompt, node, pass_number)
        content = response.content
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(
                f"The model returned an empty or unsupported response during {stage}."
            )
        return content.strip(), call

    def analyze(state: ResearchState) -> dict:
        inputs = {
            "question": state["question"],
            "source_material": neutralize_delimiters(state["numbered_sources"]),
        }
        evaluation = state["evaluation"]
        if state["analysis_notes"] is None or evaluation is None:
            prompt = ANALYSIS_PROMPT.invoke(inputs)
        else:
            prompt = REANALYSIS_PROMPT.invoke(
                {
                    **inputs,
                    "previous_notes": neutralize_delimiters(state["analysis_notes"]),
                    "analysis_gaps": _bullets(evaluation.analysis_gaps),
                }
            )
        pass_number = state["pass_count"] + 1
        notes, call = invoke_text(prompt, "analysis", ANALYZE, pass_number)
        return {
            "analysis_notes": notes,
            "pass_count": pass_number,
            "model_calls": [call],
            "analysis_history": [AnalysisRecord(pass_number=pass_number, notes=notes)],
        }

    def evaluate(state: ResearchState) -> dict:
        prompt = EVALUATION_PROMPT.invoke(
            {
                "question": state["question"],
                "source_material": neutralize_delimiters(state["numbered_sources"]),
                "analysis_notes": neutralize_delimiters(state["analysis_notes"] or ""),
            }
        )
        if budget is not None:
            budget.check(prompt.to_string(), "evaluation")
        response, call = timed_invoke(evaluator, prompt, EVALUATE, state["pass_count"])
        content = response.content
        if isinstance(content, str):
            evaluation, error = parse_evaluation(content)
        else:
            evaluation, error = None, "the evaluator returned a non-text response"

        if evaluation is None:
            status: EvaluationStatus = "invalid"
        elif evaluation.analysis_complete:
            status = "complete"
        else:
            status = "incomplete"
        return {
            "evaluation": evaluation,
            "evaluation_status": status,
            "evaluation_error": error,
            "stop_reason": decide_stop_reason(status, state["pass_count"], state["max_passes"]),
            "model_calls": [call],
            "evaluation_history": [
                EvaluationRecord(
                    pass_number=state["pass_count"],
                    status=status,
                    analysis_gaps=list(evaluation.analysis_gaps) if evaluation else [],
                    source_gaps=list(evaluation.source_gaps) if evaluation else [],
                    conflicts=list(evaluation.conflicts) if evaluation else [],
                    error=error,
                )
            ],
        }

    def generate_report(state: ResearchState) -> dict:
        prompt = REPORT_PROMPT.invoke(
            {
                "question": state["question"],
                "source_material": neutralize_delimiters(state["numbered_sources"]),
                "analysis_notes": neutralize_delimiters(state["analysis_notes"] or ""),
                "run_status": describe_run_status(state),
            }
        )
        markdown, call = invoke_text(
            prompt, "report generation", GENERATE_REPORT, state["pass_count"]
        )
        # Check only the model's own text, before any code-generated notes are appended.
        problems = check_citations(markdown, state["source_titles"])
        cited = cited_source_ids(markdown, state["source_titles"])
        hints = review_hints(markdown, state["source_texts"])
        if problems:
            markdown = f"{markdown}\n\n{citation_warning(problems, state['source_titles'])}"
        if hints:
            markdown = f"{markdown}\n\n{review_note(hints)}"
        note = uncertainty_note(state)
        if note is not None:
            markdown = f"{markdown}\n\n{note}"
        notice = incomplete_analysis_notice(state)
        if notice is not None:
            markdown = f"{notice}\n\n{markdown}"
        return {
            "report_markdown": markdown,
            "invalid_citations": problems,
            "quality_warnings": hints,
            "cited_source_ids": cited,
            "model_calls": [call],
        }

    builder = StateGraph(ResearchState)
    builder.add_node(ANALYZE, analyze)
    builder.add_node(EVALUATE, evaluate)
    builder.add_node(GENERATE_REPORT, generate_report)
    builder.add_edge(START, ANALYZE)
    builder.add_edge(ANALYZE, EVALUATE)
    builder.add_conditional_edges(EVALUATE, route_after_evaluation, [ANALYZE, GENERATE_REPORT])
    builder.add_edge(GENERATE_REPORT, END)
    return builder.compile()
