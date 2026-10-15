"""Tests for settings and model construction; Ollama does not need to be running."""

import pytest
from langchain_ollama import ChatOllama
from pydantic import ValidationError

from researchpilot.cli import build_model
from researchpilot.config import Settings
from researchpilot.state import EVALUATION_JSON_SCHEMA

OLLAMA_ENV_VARS = (
    "OLLAMA_MODEL",
    "OLLAMA_BASE_URL",
    "OLLAMA_TEMPERATURE",
    "OLLAMA_NUM_CTX",
    "OLLAMA_NUM_PREDICT",
    "RESEARCHPILOT_MAX_ANALYSIS_PASSES",
)


@pytest.fixture(autouse=True)
def clear_ollama_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in OLLAMA_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_settings_defaults_without_env_file() -> None:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.ollama_model == "gemma3:4b"
    assert str(settings.ollama_base_url) == "http://localhost:11434/"
    assert settings.ollama_temperature == 0.0
    assert settings.ollama_num_ctx == 16_384
    assert settings.ollama_num_predict == 2_048
    assert settings.researchpilot_max_analysis_passes == 2


def test_settings_read_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.2:3b")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11500")
    monkeypatch.setenv("OLLAMA_TEMPERATURE", "0.3")

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.ollama_model == "llama3.2:3b"
    assert settings.ollama_base_url.port == 11500
    assert settings.ollama_temperature == 0.3


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("OLLAMA_TEMPERATURE", "3"),
        ("OLLAMA_BASE_URL", "not a url"),
        ("OLLAMA_MODEL", ""),
        ("RESEARCHPILOT_MAX_ANALYSIS_PASSES", "0"),
        ("RESEARCHPILOT_MAX_ANALYSIS_PASSES", "4"),
        ("OLLAMA_NUM_CTX", "1024"),
        ("OLLAMA_NUM_PREDICT", "100"),
    ],
)
def test_settings_reject_invalid_values(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    monkeypatch.setenv(name, value)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_build_model_uses_settings() -> None:
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        ollama_model="gemma3:4b",
        ollama_base_url="http://localhost:11434",  # type: ignore[arg-type]
        ollama_temperature=0.2,
        ollama_num_ctx=8_192,
        ollama_num_predict=1_024,
    )

    model = build_model(settings)

    assert isinstance(model, ChatOllama)
    assert model.model == "gemma3:4b"
    assert model.base_url == "http://localhost:11434"
    assert model.temperature == 0.2
    assert model.num_ctx == 8_192
    assert model.num_predict == 1_024
    assert model.format is None


def test_build_model_applies_json_schema() -> None:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    model = build_model(settings, json_schema=EVALUATION_JSON_SCHEMA)

    assert model.format == EVALUATION_JSON_SCHEMA
    assert model.model == settings.ollama_model


@pytest.mark.parametrize("passes", ["1", "3"])
def test_max_analysis_passes_accepts_bounds(monkeypatch: pytest.MonkeyPatch, passes: str) -> None:
    monkeypatch.setenv("RESEARCHPILOT_MAX_ANALYSIS_PASSES", passes)

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.researchpilot_max_analysis_passes == int(passes)


def test_response_cap_must_leave_room_in_context(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_NUM_CTX", "8192")
    monkeypatch.setenv("OLLAMA_NUM_PREDICT", "4096")

    with pytest.raises(ValidationError, match="quarter of OLLAMA_NUM_CTX"):
        Settings(_env_file=None)  # type: ignore[call-arg]
