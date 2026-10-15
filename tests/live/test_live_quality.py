"""Opt-in quality checks against the real local model.

Skipped unless RESEARCHPILOT_LIVE_TESTS=1 is set and the Ollama server is reachable.
Model output is not deterministic, so these are regression checks for known failure
modes rather than proofs of correctness. A full run takes several minutes on CPU.
"""

import os
import re
from functools import cache

import httpx
import pytest

from researchpilot.budget import ContextBudget
from researchpilot.cli import build_model
from researchpilot.config import Settings
from researchpilot.prompts import EVALUATION_PROMPT
from researchpilot.schemas import ResearchReport, ResearchRequest
from researchpilot.service import ResearchService
from researchpilot.state import EVALUATION_JSON_SCHEMA, CompletenessEvaluation, parse_evaluation


@cache
def _settings() -> Settings:
    """Real settings from .env and the environment, loaded only when a live test runs."""
    settings = Settings()
    try:
        reachable = httpx.get(f"{settings.ollama_base_url}api/version", timeout=3).is_success
    except httpx.HTTPError:
        reachable = False
    if not reachable:
        pytest.skip("Ollama server is not reachable")
    return settings


pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.environ.get("RESEARCHPILOT_LIVE_TESTS") != "1",
        reason="set RESEARCHPILOT_LIVE_TESTS=1 to run live model checks",
    ),
]

# Same content as the manual case whose report wrongly said the task count was unspecified.
TASK_COUNT_SOURCES = """# Source A: Task completion

In a fictional controlled test of 100 tasks, System Alpha completed 85 tasks successfully. System Beta completed 80 tasks successfully. Both systems were evaluated using the same tasks and model.

# Source B: Evaluation limitations

The test did not report confidence intervals, repeated trials, latency, or operating costs. These results are fictional examples for software testing, not real benchmark results."""

CONFLICTING_SOURCES = """# Source A: Vendor test

In a fictional test of 100 tasks, System Alpha completed 85 tasks successfully.

# Source C: Independent rerun

In a fictional rerun of the same 100 tasks under the same conditions, System Alpha completed 70 tasks successfully."""

MISSING_SOURCES = """# Source A: Product performance

A fictional product test reported 92 successful tasks out of 100 attempts.

# Evidence limitation

No information about long-term reliability, operating costs, or maintenance requirements was supplied."""

# Matches "unspecified", "not stated", "do not specify", "not provided", and similar.
MISSING_TASK_COUNT = re.compile(
    r"\b(unspecified|unknown|unclear|unstated|"
    r"not (specif|stat|report|provid|giv|defin|mention|indicat|clear)\w*)",
    re.IGNORECASE,
)
TASK_COUNT_SUBJECT = re.compile(r"number of tasks|task count|total tasks", re.IGNORECASE)
CITES_INTERNAL_STEP = re.compile(r"analysis notes|run status", re.IGNORECASE)
# Lines that only say there is no conflict.
NO_CONFLICT = re.compile(
    r"\bno (direct |apparent |explicit )?conflicts?\b|conflicts?:?\s*none|\bnot conflict",
    re.IGNORECASE,
)


def _lines(report: ResearchReport) -> list[str]:
    return [line for line in report.markdown.splitlines() if line.strip()]


@pytest.fixture(scope="module")
def task_count_report() -> ResearchReport:
    settings = _settings()
    service = ResearchService(
        build_model(settings),
        max_passes=settings.researchpilot_max_analysis_passes,
        budget=ContextBudget(settings.ollama_num_ctx, settings.ollama_num_predict),
        evaluator_model=build_model(settings, json_schema=EVALUATION_JSON_SCHEMA),
    )
    request = ResearchRequest(
        question="Compare System Alpha and System Beta and summarize the limitations",
        source_material=TASK_COUNT_SOURCES,
    )
    return service.run(request)


def _evaluate(question: str, sources: str, notes: str) -> CompletenessEvaluation:
    evaluator = build_model(_settings(), json_schema=EVALUATION_JSON_SCHEMA)
    prompt = EVALUATION_PROMPT.invoke(
        {"question": question, "source_material": sources, "analysis_notes": notes}
    )
    evaluation, error = parse_evaluation(str(evaluator.invoke(prompt).content))
    assert evaluation is not None, error
    return evaluation


def test_report_does_not_claim_task_count_is_unknown(task_count_report: ResearchReport) -> None:
    offending = [
        line
        for line in _lines(task_count_report)
        if TASK_COUNT_SUBJECT.search(line) and MISSING_TASK_COUNT.search(line)
    ]

    assert not offending, offending


def test_report_attributes_findings_and_limitations_to_the_right_source(
    task_count_report: ResearchReport,
) -> None:
    lines = _lines(task_count_report)

    # In TASK_COUNT_SOURCES, Source A is [S1] and Source B is [S2].
    def cited_ids(line: str) -> set[str]:
        return {
            sid.upper()
            for group in re.findall(r"\[([^\]]*)\]", line)
            for sid in re.findall(r"\b[Ss]\d+\b", group)
        }

    def cites_a(line: str) -> bool:
        return "Source A" in line or "S1" in cited_ids(line)

    def cites_b(line: str) -> bool:
        return "Source B" in line or "S2" in cited_ids(line)

    misattributed = [
        line
        for line in lines
        if "confidence interval" in line.lower() and cites_a(line) and not cites_b(line)
    ] + [
        line
        for line in lines
        if re.search(r"\b8[05] tasks\b", line) and cites_b(line) and not cites_a(line)
    ]
    cites_internal_step = [line for line in lines if CITES_INTERNAL_STEP.search(line)]

    assert not misattributed, misattributed
    assert not cites_internal_step, cites_internal_step
    assert task_count_report.invalid_citations == [], task_count_report.invalid_citations


def test_report_does_not_call_a_limitation_a_conflict(task_count_report: ResearchReport) -> None:
    # The sources agree with each other, so any line naming a conflict is a mislabel.
    mislabelled = [
        line
        for line in _lines(task_count_report)
        if not line.startswith("#")
        and re.search(r"\bconflict", line, re.IGNORECASE)
        and not NO_CONFLICT.search(line)
    ]

    assert not mislabelled, mislabelled


def test_evaluator_reports_contradictory_evidence_as_a_conflict() -> None:
    evaluation = _evaluate(
        "How many tasks did System Alpha complete?",
        CONFLICTING_SOURCES,
        "- Source A: System Alpha completed 85 of 100 tasks.\n"
        "- Source C: System Alpha completed 70 of the same 100 tasks.",
    )

    assert evaluation.conflicts, evaluation


def test_evaluator_reports_missing_information_as_a_source_gap_not_a_conflict() -> None:
    evaluation = _evaluate(
        "Compare performance, operating cost, and maintenance requirements",
        MISSING_SOURCES,
        "- Source A: 92 of 100 tasks succeeded.\n"
        "- No source covers reliability, operating costs, or maintenance.",
    )

    assert evaluation.source_gaps, evaluation
    assert not evaluation.conflicts, evaluation


# Stage 3C: the live web run described benchmark saturation as benchmarks "becoming harder".
# The topic-specific pattern lives only in this opt-in test, never in product code.
SATURATION_SOURCES = """# Source A: Benchmark saturation

A benchmark is saturated when leading systems score near its maximum, so it no longer
distinguishes between them. Saturated benchmarks are often replaced by newer, harder ones.

# Source B: Task completion

In a fictional test of 100 tasks, System Alpha completed 85 tasks successfully."""
SATURATION_LINE = re.compile(r"saturat", re.IGNORECASE)
SATURATION_INVERTED = re.compile(
    r"\b(?:become|becomes|becoming|grow|grows|growing|get|gets|getting)\s+"
    r"(?:harder|more\s+difficult)\b|\bdifficulty\b[^.]*\bincreas",
    re.IGNORECASE,
)


def test_report_does_not_describe_saturation_as_benchmarks_getting_harder() -> None:
    settings = _settings()
    service = ResearchService(
        build_model(settings),
        max_passes=settings.researchpilot_max_analysis_passes,
        budget=ContextBudget(settings.ollama_num_ctx, settings.ollama_num_predict),
        evaluator_model=build_model(settings, json_schema=EVALUATION_JSON_SCHEMA),
    )
    request = ResearchRequest(
        question="What does benchmark saturation mean when comparing AI systems?",
        source_material=SATURATION_SOURCES,
    )

    report = service.run(request)

    inverted = [
        line
        for line in _lines(report)
        if SATURATION_LINE.search(line) and SATURATION_INVERTED.search(line)
    ]
    assert not inverted, inverted
