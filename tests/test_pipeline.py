"""Offline tests for the shared research pipeline; no model, network or MCP server is used."""

import pytest
from fakes import FakeParallelClient, ScriptedModel, complete, incomplete, search_item, tool_result
from pydantic import ValidationError

from researchpilot import cli, models, parallel_mcp, web_research
from researchpilot.budget import ContextBudget, ContextBudgetExceededError
from researchpilot.config import Settings
from researchpilot.errors import (
    InputError,
    InputTooLargeError,
    ModelRunError,
    ResearchPilotError,
    RetrievalError,
)
from researchpilot.parallel_mcp import ParallelMCPError
from researchpilot.pipeline import ResearchOutcome, prepare_request, run_research
from researchpilot.schemas import ResearchRequest
from researchpilot.service import ResearchService
from researchpilot.state import EVALUATION_JSON_SCHEMA

QUESTION = "Compare System Alpha and System Beta"
SOURCES = (
    "# Source A: Task completion\n\nIn a fictional test, System Alpha completed 85 of 100 tasks.\n\n"
    "# Source B: Configuration\n\nSystem Beta required less initial configuration."
)
WEB_URL = "https://docs.example.org/benchmarks"
RESPONSES = ["notes one", incomplete(["gap"]), "notes two", complete(), "Report [S1] [S2]."]


def settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


class RecordingFactory:
    """Model factory that hands out one scripted model and records each call."""

    def __init__(self, responses: list[str], events: list[str] | None = None) -> None:
        self.model = ScriptedModel(responses)
        self.schemas: list[dict | None] = []
        self.events = events if events is not None else []

    def __call__(self, _settings: Settings, *, json_schema: dict | None = None) -> ScriptedModel:
        self.schemas.append(json_schema)
        self.events.append("model")
        return self.model


def refuse_mcp(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    attempts: list[str] = []

    def refuse(endpoint: str, *args, **kwargs):
        attempts.append(endpoint)
        raise AssertionError("an MCP client was constructed")

    monkeypatch.setattr(parallel_mcp, "Client", refuse)
    return attempts


# --- Request preparation ---


def test_prepare_request_validates_like_research_request() -> None:
    request = prepare_request(f"  {QUESTION}  ", SOURCES)

    assert request == ResearchRequest(question=QUESTION, source_material=SOURCES)


def test_prepare_request_raises_input_error_with_the_original_message() -> None:
    with pytest.raises(ValidationError) as original:
        ResearchRequest(question="Why?", source_material=SOURCES)

    with pytest.raises(InputError) as error:
        prepare_request("Why?", SOURCES)

    assert str(error.value) == str(original.value)
    assert isinstance(error.value.__cause__, ValidationError)
    assert error.value.code == "validation_failed"


# --- Local runs ---


def test_local_run_matches_running_the_service_directly(monkeypatch) -> None:
    attempts = refuse_mcp(monkeypatch)
    factory = RecordingFactory(RESPONSES)
    request = prepare_request(QUESTION, SOURCES)
    config = settings()

    outcome = run_research(request, settings=config, model_factory=factory)

    direct_model = ScriptedModel(RESPONSES)
    budget = ContextBudget(config.ollama_num_ctx, config.ollama_num_predict)
    direct_report, direct_state = ResearchService(
        direct_model,  # type: ignore[arg-type]
        max_passes=config.researchpilot_max_analysis_passes,
        budget=budget,
        evaluator_model=direct_model,  # type: ignore[arg-type]
    ).run_with_state(request)
    assert isinstance(outcome, ResearchOutcome)
    assert outcome.retrieval is None
    assert outcome.request == request
    assert factory.model.prompts == direct_model.prompts
    assert factory.model.call_count == 5
    ignore = {"elapsed_seconds", "node_seconds"}
    assert outcome.report.model_dump(exclude=ignore) == direct_report.model_dump(exclude=ignore)
    assert outcome.final_state["numbered_sources"] == direct_state["numbered_sources"]
    assert factory.schemas == [None, EVALUATION_JSON_SCHEMA]  # main model first, then evaluator
    assert attempts == []


def test_cli_keeps_build_model_as_the_shared_factory() -> None:
    assert cli.build_model is models.build_model


# --- Web runs ---


def test_web_run_retrieves_before_building_models_and_numbers_web_sources(monkeypatch) -> None:
    fake = FakeParallelClient(
        tool_result({"results": [search_item(WEB_URL, "Benchmarks", "Excerpt about tasks.")]})
    )
    monkeypatch.setattr(web_research, "ParallelSearchMCPClient", lambda: fake)
    events: list[str] = []
    factory = RecordingFactory(["notes", complete(), "Report [S3]."], events)
    request = prepare_request(QUESTION, SOURCES)

    outcome = run_research(
        request,
        settings=settings(),
        web=True,
        model_factory=factory,
        on_retrieval=lambda evidence: events.append(f"retrieval:{len(evidence.sources)}"),
    )

    assert events == ["retrieval:2", "model", "model"]
    assert outcome.retrieval is not None and outcome.retrieval.search_calls == 1
    assert [s.url for s in outcome.request.web_sources] == [WEB_URL, WEB_URL]
    assert outcome.final_state["web_source_ids"] == ["S3", "S4"]
    assert "Excerpt about tasks." in factory.model.prompts[0]
    assert fake.closed


def test_injected_web_gatherer_is_used_instead_of_mcp(monkeypatch) -> None:
    attempts = refuse_mcp(monkeypatch)
    evidence = web_research.WebEvidence(sources=(), warnings=(), search_calls=1, fetch_calls=0)
    asked: list[str] = []

    def gatherer(question: str) -> web_research.WebEvidence:
        asked.append(question)
        return evidence

    outcome = run_research(
        prepare_request(QUESTION, SOURCES),
        settings=settings(),
        web=True,
        model_factory=RecordingFactory(["notes", complete(), "Report [S1]."]),
        web_gatherer=gatherer,
    )

    assert asked == [QUESTION]
    assert outcome.retrieval is evidence
    assert attempts == []


# --- Errors: typed, with the original message and cause ---


def test_oversized_input_fails_before_any_web_call_or_model(monkeypatch) -> None:
    attempts = refuse_mcp(monkeypatch)
    factory = RecordingFactory([])
    gathered: list[str] = []
    request = prepare_request(QUESTION, "Source A: evidence sentence. " * 2_000)

    for web in (False, True):
        with pytest.raises(InputTooLargeError) as error:
            run_research(
                request,
                settings=settings(),
                web=web,
                model_factory=factory,
                web_gatherer=lambda question: gathered.append(question),  # type: ignore[arg-type,return-value]
            )
        assert isinstance(error.value.__cause__, ContextBudgetExceededError)
        assert str(error.value) == str(error.value.__cause__)
        assert error.value.code == "input_too_large"
    assert gathered == []
    assert factory.model.call_count == 0
    assert attempts == []


def test_web_retrieval_failure_is_a_retrieval_error_and_no_model_is_built(monkeypatch) -> None:
    fake = FakeParallelClient(ParallelMCPError("Parallel MCP tool 'web_search' timed out."))
    monkeypatch.setattr(web_research, "ParallelSearchMCPClient", lambda: fake)
    factory = RecordingFactory([])

    with pytest.raises(RetrievalError) as error:
        run_research(
            prepare_request(QUESTION, SOURCES), settings=settings(), web=True, model_factory=factory
        )

    assert str(error.value) == "Web search failed: Parallel MCP tool 'web_search' timed out."
    assert isinstance(error.value.__cause__, web_research.WebRetrievalError)
    assert error.value.code == "web_retrieval_failed"
    assert factory.schemas == []


@pytest.mark.parametrize(
    "failure",
    [
        RuntimeError("The model returned an empty or unsupported response during analysis."),
        ConnectionError("Failed to connect to Ollama."),
    ],
)
def test_model_failures_are_model_run_errors_keeping_message_and_cause(failure) -> None:
    class FailingModel:
        def invoke(self, *_args, **_kwargs):
            raise failure

    with pytest.raises(ModelRunError) as error:
        run_research(
            prepare_request(QUESTION, SOURCES),
            settings=settings(),
            model_factory=lambda _settings, **_kwargs: FailingModel(),
        )

    assert str(error.value) == str(failure)
    assert error.value.__cause__ is failure
    assert error.value.code == "model_failed"


def test_model_construction_failure_is_a_model_run_error() -> None:
    def broken_factory(_settings, **_kwargs):
        raise ValueError("bad model configuration")

    with pytest.raises(ModelRunError, match="bad model configuration"):
        run_research(
            prepare_request(QUESTION, SOURCES), settings=settings(), model_factory=broken_factory
        )


def test_all_errors_share_a_base_class_and_distinct_codes() -> None:
    classes = [InputError, InputTooLargeError, RetrievalError, ModelRunError]

    assert all(issubclass(cls, ResearchPilotError) for cls in classes)
    assert len({cls.code for cls in classes}) == len(classes)
