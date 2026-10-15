"""Context-window budget checks so oversized input is rejected instead of truncated."""

import math
from dataclasses import dataclass

# Conservative for English prose (Gemma averages closer to 4 characters per token).
CHARS_PER_TOKEN_ESTIMATE = 3


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN_ESTIMATE)


class ContextBudgetExceededError(ValueError):
    """A prompt would not fit in the model's context window."""


@dataclass(frozen=True)
class ContextBudget:
    """Token limits for one model: the context window and the cap on each response."""

    context_tokens: int
    response_tokens: int

    def check(self, prompt_text: str, stage: str, *, reserved_tokens: int = 0) -> None:
        """Raise if the prompt, any reserved space, and one response exceed the context window."""
        needed = estimate_tokens(prompt_text) + reserved_tokens + self.response_tokens
        if needed > self.context_tokens:
            raise ContextBudgetExceededError(
                f"The {stage} prompt needs about {needed} tokens, including "
                f"{self.response_tokens + reserved_tokens} reserved for model output, but the "
                f"context budget is {self.context_tokens} tokens. Shorten the source file or "
                "raise OLLAMA_NUM_CTX (the model and your hardware must support it)."
            )
