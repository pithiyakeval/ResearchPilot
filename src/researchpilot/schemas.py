"""Validated inputs and outputs for research synthesis."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Why the analysis loop stopped before the report was generated.
StopReason = Literal["evaluation_complete", "pass_limit_reached", "evaluation_invalid"]
# A search-result excerpt and a fetched page are kept as separate sources.
WebSourceKind = Literal["search_excerpt", "fetched_page"]

_WEB_SOURCE_LABELS = {"search_excerpt": "Search excerpt", "fetched_page": "Fetched page"}


class WebSource(BaseModel):
    """Evidence retrieved from the web before analysis; its text is untrusted."""

    model_config = ConfigDict(frozen=True)

    kind: WebSourceKind
    title: str = Field(min_length=1, max_length=300)
    url: str = Field(min_length=1, max_length=2_048)
    text: str = Field(min_length=1, max_length=4_000)

    @property
    def heading(self) -> str:
        """Source title as numbered and cited: evidence type, page title and URL."""
        return f"{_WEB_SOURCE_LABELS[self.kind]}: {self.title} — {self.url}"


class ResearchRequest(BaseModel):
    question: str = Field(min_length=8, max_length=2_000)
    source_material: str = Field(min_length=40, max_length=80_000)
    # Numbered after the local sources, in this order. Empty unless web mode was requested.
    web_sources: tuple[WebSource, ...] = Field(default=(), max_length=10)

    @field_validator("question", "source_material")
    @classmethod
    def reject_whitespace_only(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Value must not be blank.")
        return value.strip()


class ResearchReport(BaseModel):
    question: str
    markdown: str = Field(min_length=1)
    analysis_passes: int | None = None
    stop_reason: StopReason | None = None
    model_calls: int | None = None
    elapsed_seconds: float | None = None
    # Total model time per graph node, e.g. {"analyze": 41.2, "evaluate": 20.5}.
    node_seconds: dict[str, float] | None = None
    # Problems found by the deterministic citation-reference check; empty when none.
    invalid_citations: list[str] = Field(default_factory=list)
    # Analysis gaps still open when the pass limit stopped analysis; empty otherwise.
    unresolved_analysis_gaps: list[str] = Field(default_factory=list)
    # Heuristic review hints (possible over-generalization or contradiction). They never
    # change the report text or the citation check, and an empty list proves nothing.
    quality_warnings: list[str] = Field(default_factory=list)
