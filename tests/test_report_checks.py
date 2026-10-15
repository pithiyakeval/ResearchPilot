"""Tests for the heuristic report review hints; no model or network is used.

These check the detectors on known text. They cannot show that a model will not
over-generalize or contradict itself, and a report without hints is not thereby supported.
"""

from fakes import ScriptedModel, complete, incomplete

from researchpilot.graph import build_research_graph, recursion_limit_for
from researchpilot.report_checks import (
    find_possible_contradictions,
    find_unscoped_generalizations,
    review_hints,
    review_note,
)
from researchpilot.schemas import ResearchRequest
from researchpilot.service import ResearchService
from researchpilot.sources import check_citations, number_sources
from researchpilot.state import initial_state

# Same content as examples/sources.md, used in the live run.
LOCAL = """# Source A: Hypothetical performance test

In a fictional test involving 100 tasks, System Alpha completed 85 tasks successfully. The test did not report confidence intervals.

# Source B: Hypothetical engineering assessment

In a fictional assessment, System Beta required less initial configuration than System Alpha. Long-term maintenance costs were not measured."""
TEXTS = number_sources(LOCAL).texts

# Lines copied verbatim from reports/web-e2e-2026-10-13.md.
OFTEN = (
    "*   **Hypothetical Test Conditions:** Benchmark results are often based on fictional tests "
    "lacking crucial context, such as confidence intervals, making reliability assessments "
    "impossible [S1]."
)
FREQUENTLY = (
    "*   **Configuration Differences:** Long-term maintenance costs are frequently overlooked "
    "in benchmark assessments, introducing a significant variable that impacts system value "
    "and comparability [S2]."
)
COMMERCIAL = (
    "*   **Commercial Incentives:** Commercial incentives can influence evaluation outcomes, "
    "leading to biased results [S6]."
)
NO_PROVIDER_BIAS = (
    "*   **Lack of Provider Bias:** The sources do not address the potential biases of "
    "benchmark providers or their influence on evaluation outcomes [S6]."
)
# Legitimate negative statements from the same report: a single source reports something
# missing. These must not be treated as contradictions.
WITHOUT_CI = (
    "*   System Alpha completed 85 tasks successfully in a fictional test without confidence "
    "intervals [S1]."
)
NOT_MEASURED = (
    "*   System Beta required less initial configuration than System Alpha, but long-term "
    "maintenance costs were not measured [S2]."
)
MISSING_CI = (
    "*   **Missing Confidence Intervals:** The absence of confidence intervals in the provided "
    "source material significantly limits the ability to assess the statistical significance "
    "of benchmark results [S1]."
)


def lines(*parts: str) -> str:
    return "\n".join(parts)


# --- Generalization hints ---


def test_live_overgeneralizations_get_hints_with_uncertain_wording() -> None:
    hints = find_unscoped_generalizations(f"{OFTEN}\n{FREQUENTLY}", TEXTS)

    assert len(hints) == 2
    assert hints[0].startswith('"Hypothetical Test Conditions: Benchmark results are often')
    assert '[S1] may generalize beyond its cited source(s): the word "often" does not' in hints[0]
    assert 'the word "frequently" does not appear in them' in hints[1]
    for hint in hints:
        assert "Check whether the sources support a general claim." in hint


def test_generalizing_word_present_in_a_cited_source_gets_no_hint() -> None:
    texts = {**TEXTS, "S3": "Benchmark results are often reported without error bars."}

    assert find_unscoped_generalizations("Results are often reported alone [S3].", texts) == []
    # The word only needs to appear in one of the cited sources.
    assert find_unscoped_generalizations("Results are often reported [S1, S3].", texts) == []


def test_sentences_without_generalizing_words_or_citations_get_no_hint() -> None:
    report = lines(
        WITHOUT_CI,
        "Benchmarks are often criticized.",  # no citation: nothing to compare against
        "Results are often unclear [S9].",  # unknown ID: left to the citation check
    )

    assert find_unscoped_generalizations(report, TEXTS) == []


def test_one_hint_per_sentence_and_whole_words_only() -> None:
    report = (
        "Costs are often and usually ignored [S1]. Oftentimes, results vary [S1]. "
        "Generally, maintenance matters [S2]."
    )

    hints = find_unscoped_generalizations(report, TEXTS)

    assert len(hints) == 2
    assert '"often"' in hints[0] and '"usually"' not in hints[0]
    assert '"generally"' in hints[1]


# --- Contradiction hints ---


def test_live_provider_bias_contradiction_gets_one_hint() -> None:
    report = lines(COMMERCIAL, NO_PROVIDER_BIAS)

    hints = find_possible_contradictions(report)

    assert len(hints) == 1
    hint = hints[0]
    assert hint.startswith('Possible contradiction: "Lack of Provider Bias: The sources do not')
    assert '" vs "Commercial Incentives: Commercial incentives can influence' in hint
    assert "(shared terms: bias, evaluation, influence, outcomes)" in hint
    assert hint.endswith("Check both statements against the sources.")


def test_legitimate_negative_statements_get_no_contradiction_hint() -> None:
    report = lines(
        OFTEN,
        FREQUENTLY,
        WITHOUT_CI,
        NOT_MEASURED,
        MISSING_CI,
        "*   [S1] did not report confidence intervals for the 85 completed tasks [S1].",
        "*   The test did not report latency [S1].",
    )

    assert find_possible_contradictions(report) == []


def test_unrelated_corpus_level_absence_gets_no_hint() -> None:
    report = lines(COMMERCIAL, "The sources do not cover pricing or licensing terms.")

    assert find_possible_contradictions(report) == []


def test_absence_claims_are_not_compared_with_each_other_or_with_uncited_text() -> None:
    report = lines(
        NO_PROVIDER_BIAS,
        "None of the sources discuss provider bias or evaluation outcomes.",
        "Commercial incentives can influence evaluation outcomes and bias.",  # no citation
    )

    assert find_possible_contradictions(report) == []


# --- Note and wiring: warning-only ---


def test_review_note_states_its_limits() -> None:
    note = review_note(["hint one", "hint two"])

    assert note.startswith("> **ResearchPilot review hints (heuristic):**")
    assert "not findings" in note
    assert "can be wrong in either direction" in note
    assert "the absence of hints does not mean claims are supported" in note
    assert note.endswith(">\n> - hint one\n> - hint two")


def run_graph(model_text: str, evaluation: str, max_passes: int = 2) -> dict:
    model = ScriptedModel(["- notes", evaluation, model_text])
    request = ResearchRequest(
        question="Compare System Alpha and System Beta", source_material=LOCAL
    )
    graph = build_research_graph(model)  # type: ignore[arg-type]
    return graph.invoke(
        initial_state(request, max_passes),
        config={"recursion_limit": recursion_limit_for(max_passes)},
    )


def test_hints_are_appended_without_changing_text_or_citation_results() -> None:
    # [S6] is not a supplied source here, so the citation check also flags it.
    model_text = f"# Research Report\n\n{OFTEN}\n{COMMERCIAL}\n{NO_PROVIDER_BIAS}"

    state = run_graph(model_text, complete())

    assert state["report_markdown"].startswith(model_text + "\n\n")
    assert state["invalid_citations"] == check_citations(model_text, state["source_titles"])
    assert state["invalid_citations"] == ["[S6] is not a supplied source ID"]
    assert state["quality_warnings"] == review_hints(model_text, state["source_texts"])
    assert len(state["quality_warnings"]) == 2
    assert state["report_markdown"].endswith(review_note(state["quality_warnings"]))


def test_clean_report_gets_no_hints_and_no_note() -> None:
    model_text = f"# Research Report\n\n{WITHOUT_CI}"

    state = run_graph(model_text, complete())

    assert state["quality_warnings"] == []
    assert state["report_markdown"] == model_text


def test_note_order_is_notice_text_citation_check_hints_then_pass_limit_note() -> None:
    model_text = f"# Research Report\n\n{OFTEN} Unknown source [S9]."

    markdown = run_graph(model_text, incomplete(["gap"]), max_passes=1)["report_markdown"]

    positions = [
        markdown.index("> **Incomplete analysis:**"),
        markdown.index(model_text),
        markdown.index("> **ResearchPilot citation check:**"),
        markdown.index("> **ResearchPilot review hints (heuristic):**"),
        markdown.index("> **ResearchPilot note:**"),
    ]
    assert positions == sorted(positions)


def test_service_exposes_hints_and_keeps_citation_results() -> None:
    model = ScriptedModel(["- notes", complete(), f"# Research Report\n\n{OFTEN}"])
    request = ResearchRequest(
        question="Compare System Alpha and System Beta", source_material=LOCAL
    )

    report = ResearchService(model).run(request)  # type: ignore[arg-type]

    assert len(report.quality_warnings) == 1
    assert report.invalid_citations == []
