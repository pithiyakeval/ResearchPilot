"""Typed errors for a research run, shared by the CLI and future API adapters.

Each error's message is the original exception's message, unchanged, and the original
exception is kept as ``__cause__``. ``code`` is a stable, machine-readable identifier.
"""


class ResearchPilotError(Exception):
    """Base class for errors a research run reports to its caller."""

    code = "research_error"


class InputError(ResearchPilotError):
    """The question, source material or settings failed validation."""

    code = "validation_failed"


class InputTooLargeError(ResearchPilotError):
    """A prompt would not fit the model's context window.

    Usually raised before any model call; it can also come from an intermediate prompt,
    such as one carrying long analysis notes.
    """

    code = "input_too_large"


class RetrievalError(ResearchPilotError):
    """Web search failed or returned nothing usable; no model was called."""

    code = "web_retrieval_failed"


class ModelRunError(ResearchPilotError):
    """Building the models or running the research graph failed.

    Includes an unreachable Ollama server, which its client reports as ``ConnectionError``.
    """

    code = "model_failed"
