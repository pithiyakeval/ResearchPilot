"""Construction of the local Ollama chat models."""

from typing import Any

from langchain_ollama import ChatOllama

from researchpilot.config import Settings


def build_model(settings: Settings, *, json_schema: dict[str, Any] | None = None) -> ChatOllama:
    """Create the local Ollama chat model from validated settings.

    With ``json_schema``, Ollama's structured-output mode constrains replies to that schema.
    """
    return ChatOllama(
        model=settings.ollama_model,
        base_url=str(settings.ollama_base_url).rstrip("/"),
        temperature=settings.ollama_temperature,
        num_ctx=settings.ollama_num_ctx,
        num_predict=settings.ollama_num_predict,
        format=json_schema,
    )
