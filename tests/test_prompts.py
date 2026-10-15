"""Prompt regression tests for known report errors; no model is called.

These check what the model is told and given. They cannot prove what a real model writes;
tests/live/ covers that against a running Ollama server.
"""

import pytest
from fakes import ScriptedModel, complete, incomplete

from researchpilot.budget import ContextBudget
from researchpilot.config import Settings
from researchpilot.graph import (
    build_research_graph,
    check_request_fits,
    describe_run_status,
    recursion_limit_for,
)
from researchpilot.prompts import (
    ANALYSIS_PROMPT,
    CLAIM_SCOPE,
    EVALUATION_PROMPT,
    EVIDENCE_ACCURACY,
    REANALYSIS_PROMPT,
    REPORT_PROMPT,
)
from researchpilot.schemas import ResearchRequest, WebSource
from researchpilot.state import CompletenessEvaluation, initial_state
from researchpilot.web_research import MAX_EXCERPT_CHARS

# Mirrors the manual test case where a report wrongly said the task count was unspecified.
TASK_COUNT_SOURCES = """# Source A: Task completion

In a fictional controlled test of 100 tasks, System Alpha completed 85 tasks successfully.
System Beta completed 80 tasks successfully.

# Source B: Evaluation limitations

The test did not report confidence intervals, repeated trials, latency, or operating costs."""

REQUEST = ResearchRequest(
    question="Compare System Alpha and System Beta", source_material=TASK_COUNT_SOURCES
)
INPUTS = {
    "question": REQUEST.question,
    "source_material": TASK_COUNT_SOURCES,
    "previous_notes": "- notes",
    "analysis_gaps": "- gap",
    "analysis_notes": "- notes",
    "run_status": "status",
}


@pytest.mark.parametrize(
    "prompt", [ANALYSIS_PROMPT, REANALYSIS_PROMPT, EVALUATION_PROMPT, REPORT_PROMPT]
)
def test_stage2_prompts_require_checking_sources_before_claiming_missing(prompt) -> None:
    variables = {name: INPUTS[name] for name in prompt.input_variables}

    rendered = prompt.invoke(variables).to_string()

    assert EVIDENCE_ACCURACY in rendered
    assert "check that\nthe supplied source material really does not state it" in rendered


def test_explicit_task_count_reaches_every_model_call_unmodified() -> None:
    model = ScriptedModel(["- notes", complete(), "# Research Report"])
    graph = build_research_graph(model)  # type: ignore[arg-type]

    graph.invoke(initial_state(REQUEST, 2), config={"recursion_limit": recursion_limit_for(2)})

    assert model.call_count == 3
    for prompt in model.prompts:
        assert "In a fictional controlled test of 100 tasks, System Alpha completed 85" in prompt


def test_report_prompt_forbids_citing_notes_and_restating_unresolved_gaps() -> None:
    rendered = REPORT_PROMPT.invoke(
        {name: INPUTS[name] for name in REPORT_PROMPT.input_variables}
    ).to_string()

    assert "never the analysis notes or run status" in rendered
    assert "do not restate them as facts about the sources" in rendered
    assert "a limitation or a missing detail is not a conflict" in rendered


def test_evaluation_prompt_classifies_unanswerable_requests_as_source_gaps() -> None:
    rendered = EVALUATION_PROMPT.invoke(
        {name: INPUTS[name] for name in EVALUATION_PROMPT.input_variables}
    ).to_string()

    assert "detail that no supplied source gives is\na source gap, not an analysis gap" in rendered


def test_run_status_keeps_missing_information_and_conflicts_apart() -> None:
    missing = "Operating costs were not reported by any source"
    conflict = "Source A and Source C report different task counts for System Alpha"
    state = {
        **initial_state(REQUEST, 2),
        "pass_count": 1,
        "stop_reason": "evaluation_complete",
        "evaluation": CompletenessEvaluation(
            analysis_gaps=[], source_gaps=[missing], conflicts=[conflict]
        ),
    }

    status = describe_run_status(state)

    gaps_header = status.index("Gaps in the supplied sources")
    conflicts_header = status.index("Conflicts between the supplied sources:")
    assert gaps_header < status.index(missing) < conflicts_header
    assert status.index(conflict) > conflicts_header
    assert status.count(missing) == 1
    assert status.count(conflict) == 1


# --- Claim scope and term fidelity (Stage 3C) ---
# These show what the model is told and given. They cannot show that it complies: a small
# local model may still over-generalize or misdescribe a term such as benchmark saturation.

SATURATION_DEFINITION = (
    "A benchmark is saturated when leading systems score near its maximum, so it no longer "
    "distinguishes between them."
)


def test_claim_scope_reaches_analysis_reanalysis_and_report_prompts_only() -> None:
    for prompt in (ANALYSIS_PROMPT, REANALYSIS_PROMPT, REPORT_PROMPT):
        rendered = prompt.invoke({name: INPUTS[name] for name in prompt.input_variables})
        assert CLAIM_SCOPE in rendered.to_string()
    evaluation = EVALUATION_PROMPT.invoke(
        {name: INPUTS[name] for name in EVALUATION_PROMPT.input_variables}
    )
    assert CLAIM_SCOPE not in evaluation.to_string()


def test_claim_scope_covers_the_observed_failure_modes() -> None:
    assert 'Use words\nsuch as "often", "generally" or "typically" only when' in CLAIM_SCOPE
    assert "fictional or\nhypothetical is an illustration, not real-world evidence" in CLAIM_SCOPE
    assert "Use technical terms only as the\nsources define them" in CLAIM_SCOPE
    assert "say that it is not\ndefined instead of explaining it" in CLAIM_SCOPE
    assert "Do not say that the sources do not cover a topic" in CLAIM_SCOPE
    assert "Never write internal block names" in CLAIM_SCOPE


def test_defining_web_excerpt_reaches_every_prompt_unchanged_and_within_budget() -> None:
    # A full-length excerpt with the definition at its very end, so any clipping would drop it.
    filler = "Benchmark scores are compared across systems and tasks. "
    padding = (filler * 40)[: MAX_EXCERPT_CHARS - len(SATURATION_DEFINITION) - 1]
    excerpt = f"{padding.rstrip()} {SATURATION_DEFINITION}"
    assert len(excerpt) <= MAX_EXCERPT_CHARS
    request = ResearchRequest(
        question="What does benchmark saturation mean for comparing AI systems?",
        source_material=TASK_COUNT_SOURCES,
        web_sources=(
            WebSource(
                kind="search_excerpt",
                title="Benchmark saturation",
                url="https://example.org/saturation",
                text=excerpt,
            ),
        ),
    )
    defaults = Settings(_env_file=None)  # type: ignore[call-arg]
    check_request_fits(request, ContextBudget(defaults.ollama_num_ctx, defaults.ollama_num_predict))
    model = ScriptedModel(["- notes", incomplete(["gap"]), "- notes 2", complete(), "# Report"])

    graph = build_research_graph(model)  # type: ignore[arg-type]
    graph.invoke(initial_state(request, 2), config={"recursion_limit": recursion_limit_for(2)})

    assert len(model.prompts) == 5
    for prompt in model.prompts:
        assert excerpt in prompt
        assert prompt.count(SATURATION_DEFINITION) == 1
